"""Publish only a completed original stock PAGED result to the Full caller state."""
import torch

from . import sglang_entry


def publish_stock_paged_prior(
    indexer, forward_batch, layer_id, metadata, table64, paged_result, *, rows
):
    """Host admission only; the Full publisher owns snapshot/inverse/votes/event.

    A True return means work was enqueued, not that asynchronous GPU validation
    has completed. Do not inspect GPU values or convert this result into a claim
    that every voted HOT key appeared among the selected stock winners.
    """
    mode = sglang_entry._mode()
    if mode not in ("full", "few") or torch.cuda.is_current_stream_capturing():
        return False
    if (
        paged_result.device.type != "cuda"
        or torch.cuda.get_device_capability(paged_result.device) != (10, 0)
        or indexer.n_heads not in ((32, 64) if mode == "few" else (32,))
        or indexer.head_dim != 128
        or indexer.index_topk != 2048
        or indexer.num_init_tokens != 0
        or indexer.num_local_tokens != 0
        or indexer.dsa_enable_prefill_cp
        or indexer.logits_with_pp_recv
        or getattr(forward_batch, "attn_cp_metadata", None) is not None
        or not forward_batch.forward_mode.is_extend_without_speculative()
    ):
        return False
    if (
        getattr(metadata.topk_transform_method, "name", None) != "PAGED"
        or getattr(metadata.topk_backend, "name", None) != "SGL_KERNEL"
        or metadata.force_unfused_topk
        or metadata.attn_metadata.topk_indices_offset is not None
    ):
        return False
    from sglang.srt.environ import envs

    if (
        not envs.SGLANG_DSA_FUSE_TOPK.get()
        or envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get()
    ):
        return False
    lengths = metadata.get_indexer_seq_len_cpu()
    batch_lengths = forward_batch.seq_lens_cpu
    extend = forward_batch.extend_seq_lens_cpu
    rids = forward_batch.rids
    if (
        lengths is None or lengths.device.type != "cpu" or lengths.numel() != 1
        or batch_lengths is None or batch_lengths.device.type != "cpu"
        or batch_lengths.numel() != 1
        or extend is None or len(extend) != 1 or int(extend[0]) != rows
        or rids is None or len(rids) != 1
        or not isinstance(rids[0], str) or not rids[0]
        or rows <= 0 or len(paged_result.shape) != 2
        or paged_result.shape[0] < rows or paged_result.shape[1] != 2048
        or paged_result.dtype != torch.int32
        or len(table64.shape) != 2 or table64.shape[0] != 1
        or table64.device != paged_result.device or table64.dtype != torch.int32
    ):
        return False
    visible_end = int(lengths[0])
    if (
        visible_end != int(batch_lengths[0])
        or visible_end < rows or visible_end < 12288 or visible_end > 1048576
        or table64.shape[1] < (visible_end + 63) // 64
    ):
        return False
    key = (mode, str(paged_result.device), indexer.n_heads)
    with sglang_entry._SLOT_LOCK:
        backend = sglang_entry._SLOTS.get(key)
        if backend is None:
            raise RuntimeError("Stock prior requires its pre-KV prepared slot")
    # Keep the caller's exact layer/RID identity and active rows. The output and
    # page table may both be overwritten later; the publisher must snapshot on
    # this call's CUDA stream before returning and hand off through its event.
    return backend.publish_stock_paged(
        paged_result[:rows], table64,
        hot_key=(layer_id, rids[0]), visible_end=visible_end, window_start=0,
    )
