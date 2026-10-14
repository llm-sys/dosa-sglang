"""Single-request PAGED hook before SGLang's ordinary full cache gather."""

import os
import threading
import torch
from .native_binding import FEW_PREFILL_MIN_KEYS
from .query_slots import (execution_rows, service_execution_rows,
                          configured_query_capacity, service_rows_eligible)

_SLOTS = {}
_SLOT_LOCK = threading.Lock()


def _mode():
    mode = os.environ.get("SGLANG_DOSA_MODE", "off").lower()
    if mode == "few-decode":
        return "few"
    if mode not in ("off", "full", "few"):
        raise ValueError("SGLANG_DOSA_MODE must be off, full, few or few-decode")
    return mode


def _query_capacity(mode):
    capacity = configured_query_capacity()
    # Keep Full's existing reservation for smaller or unsupported buffers.
    return max(16384, capacity) if mode == "full" else capacity


def prepare_indexer_backend(indexer):
    """Allocate a qualified slot during model construction, before KV sizing."""
    mode = _mode()
    indexer._dosa_decode_enabled = (
        os.environ.get("SGLANG_DOSA_MODE", "off").lower() == "few-decode"
    )
    if mode == "off":
        return False
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (10, 0)
        or indexer.n_heads not in ((32, 64) if mode == "few" else (32,))
        or indexer.head_dim != 128
        or indexer.index_topk != 2048
        or indexer.num_init_tokens != 0
        or indexer.num_local_tokens != 0
        or indexer.dsa_enable_prefill_cp
        or indexer.logits_with_pp_recv
    ):
        return False
    device = torch.device("cuda", torch.cuda.current_device())
    capacity = _query_capacity(mode)
    if mode == "few" and not service_rows_eligible(capacity, capacity):
        return False
    key = (mode, str(device), indexer.n_heads)
    with _SLOT_LOCK:
        backend = _SLOTS.get(key)
        if backend is None:
            from .assembly import create_service_backend

            backend = create_service_backend(
                mode, device, max_queries=capacity, max_keys=1048576,
                score_heads=indexer.n_heads
            )
        if getattr(backend, 'logical_query_capacity', backend.max_q) != capacity:
            raise RuntimeError("DoSA prefill configuration changed after workspace reservation")
        # Publish either mode only after its first reservation succeeds.
        backend.reserve_for_model_load(indexer.layer_id)
        _SLOTS[key] = backend
    return True


def try_paged_prefill(
    indexer,
    forward_batch,
    layer_id,
    q,
    weights,
    metadata,
    starts,
    ends,
    table64,
    result,
    pool,
):
    """Use existing quantized operands and CPU visibility metadata.

    Unsupported shapes retain the Indexer's original path. Selected-mode setup
    errors propagate; missing source never silently binds an older implementation.
    This runs in the ordinary prefill body, including its existing split-op.
    """
    mode = _mode()
    if mode == "off" or torch.cuda.is_current_stream_capturing():
        return False
    if (
        q.device.type != "cuda"
        or torch.cuda.get_device_capability(q.device) != (10, 0)
        or indexer.n_heads not in ((32, 64) if mode == "few" else (32,))
        or indexer.head_dim != 128
        or indexer.index_topk != 2048
        or indexer.num_init_tokens != 0
        or indexer.num_local_tokens != 0
        or indexer.dsa_enable_prefill_cp
        or indexer.logits_with_pp_recv
    ):
        return False
    if (
        getattr(metadata.topk_transform_method, "name", None) != "PAGED"
        or getattr(metadata.topk_backend, "name", None)
        not in ("SGL_KERNEL", "FLASHINFER")
        or metadata.force_unfused_topk
    ):
        return False
    # Respect the original fused/unordered-FI selection policy.
    from sglang.srt.environ import envs

    if (
        not envs.SGLANG_DSA_FUSE_TOPK.get()
        or envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get()
    ):
        return False
    rows = starts.numel()
    # Reject outside the configured interval before requiring a prepared slot.
    # Unsupported calls retain stock fallback even when no Few slot exists.
    capacity = _query_capacity(mode)
    if mode == "few":
        if not service_rows_eligible(rows, capacity):
            return False
    elif rows <= 0 or rows > capacity:
        return False
    lengths = metadata.get_indexer_seq_len_cpu()
    extend = forward_batch.extend_seq_lens_cpu
    request_ids = forward_batch.rids
    if (
        lengths is None
        or lengths.device.type != "cpu"
        or lengths.numel() != 1
        or extend is None
        or len(extend) != 1
        or int(extend[0]) != rows
        or request_ids is None
        or len(request_ids) != 1
        or table64.shape[0] != 1
        or rows <= 0
    ):
        return False
    nk = int(lengths[0])
    common_end = nk - rows + 1
    min_keys = FEW_PREFILL_MIN_KEYS if mode == "few" else 196608
    if not (min_keys <= nk <= 1048576 and common_end >= 12288):
        return False
    if (
        q.dtype != torch.float8_e4m3fn
        or weights.dtype != torch.float32
        or tuple(q.shape[1:]) != (indexer.n_heads, 128)
        or tuple(weights.shape[1:]) != (indexer.n_heads,)
        or q.shape[0] < rows
        or weights.shape[0] < rows
    ):
        return False
    key = (mode, str(q.device), indexer.n_heads)
    with _SLOT_LOCK:
        backend = _SLOTS.get(key)
        if backend is None:
            raise RuntimeError(
                "DoSA workspace must be prepared during model construction before KV sizing"
            )
    if getattr(backend, 'logical_query_capacity', backend.max_q) != capacity:
        raise RuntimeError("DoSA prefill configuration changed after workspace reservation")
    row_geometry = service_execution_rows if mode == "few" else execution_rows
    if row_geometry(backend, rows) == 0:
        return False
    table1 = metadata.get_page_table_1()
    cache = indexer._get_index_k_read_buffer(pool, layer_id)
    interval = ({"causal_query_start": nk - rows}
                if mode == "few" and getattr(backend, "supports_causal_query_interval", False)
                else {})
    if (indexer._dosa_decode_enabled and indexer.n_heads == 64
            and metadata.topk_backend.should_use_topk_v2()):
        from sglang.srt.model_executor.forward_context import get_req_to_token_pool
        from .decode_witness import DecodeWitness

        if not hasattr(indexer, '_dosa_decode_witness'):
            indexer._dosa_decode_witness = DecodeWitness(get_req_to_token_pool(), q.device)
        interval['decode_seed'] = (indexer._dosa_decode_witness,
                                   forward_batch.req_pool_indices)
    return backend.run_paged(
        q[:rows],
        weights[:rows],
        cache,
        table64,
        table1,
        starts,
        ends,
        result,
        nk=nk,
        common_end=common_end,
        last_visible=nk,
        hot_key=(layer_id, request_ids[0]),
        window_start=0,
        **interval,
    )
