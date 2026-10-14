"""The shared Full/Few paged-call transaction and completion boundary."""

import threading
import torch

from .cold_prefix import initialize_for_call
from .query_slots import execution_rows, pad_input_tail


def valid_paged_inputs(device, q, w, cache, table64, table1, starts, ends, result, *, score_heads=32):
    """Shared tensor contract for one scratch block or a larger slabbed call."""
    rows = len(q)
    inputs = (q, w, cache, table64, table1, starts, ends, result)
    if any(value.device != device for value in inputs):
        return False
    if (
        q.dtype != torch.float8_e4m3fn or tuple(q.shape) != (rows, score_heads, 128)
        or w.dtype != torch.float32 or tuple(w.shape) != (rows, score_heads)
        or not q.is_contiguous() or not w.is_contiguous()
        or cache.dtype != torch.uint8
        or any(value.dtype != torch.int32 for value in (table64, table1, starts, ends, result))
    ):
        return False
    if table64.ndim != 2 or table1.ndim != 2:
        return False
    return (
        len(table64) == len(table1) == 1
        and starts.numel() == ends.numel() == rows
        and result.ndim == 2 and result.shape[0] >= rows and result.shape[1] == 2048
    )


class PagedBackend:
    """Own one request from input staging through carry and output publication.

    Full maps its final page IDs in the vote kernel. Few first maps candidate
    IDs, then resolves pages after carry publication. Those numerical/output
    policies stay explicit; slot locking, input lifetimes and fences are shared.
    """

    score_heads = 32

    def _initialize_slot(self):
        self.lock = threading.Lock()
        self.last_done = None
        self.hot_graph = None
        self.graph_kind = None
        self.shape = None
        self.generation = 0
        self.setup_calls = 0
        self.replay_calls = 0
        self.startup_ready = None
        self.capture_memory_reserve = None
        self.capture_reserved_bytes = 0
        self.capture_reserve_released = False
        self.model_buffers_reserved = False

    def _allocate_query_inputs(self, rows):
        score_heads = self.score_heads
        self.q_input = torch.empty(
            (rows, score_heads, 128), device=self.device, dtype=torch.float8_e4m3fn
        )
        self.w_input = torch.empty(
            (rows, score_heads), device=self.device, dtype=torch.float32
        )
        self.start_input = torch.empty(
            rows, device=self.device, dtype=torch.int32
        )
        self.end_input = torch.empty_like(self.start_input)

    def _allocate_key_storage(self, keys):
        self.pk_storage = torch.empty(
            (keys, 128), device=self.device, dtype=torch.float8_e4m3fn
        )
        self.ps_storage = torch.empty(
            ((keys + 3) // 4 * 4, 4), device=self.device, dtype=torch.uint8
        )

    def _bind_gather_views(self, nk):
        self.pk = self.pk_storage[:nk]
        self.ps_bytes = self.ps_storage[:nk]
        self.ps = self.ps_bytes.view(torch.float32).view(-1)
        self.ps_storage[nk : (nk + 3) // 4 * 4].zero_()

    @staticmethod
    def tensor_buffers(owner):
        return {name: value for name, value in vars(owner).items()
                if isinstance(value, torch.Tensor)}

    def reserve_for_model_load(self, layer_id):
        """Reserve real model workspaces and carry before sizing the KV pool.

        This loads FI and owns its storage without fabricating numerical inputs.
        Full and Few provide their original mode-specific seed allocations.
        """
        from flashinfer.utils import _get_cache_buf

        with self.lock, torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            if self.last_done is not None:
                stream.wait_event(self.last_done)
            if not self.model_buffers_reserved:
                from flashinfer.topk import get_topk_module

                get_topk_module()
                self._ensure_model_buffers()
                _get_cache_buf(
                    f"radix_topk_row_states_{self.device}",
                    1024 * 1024,
                    self.device,
                    zero_init=True,
                )
                self.lt._pair_plan_bufs(self.max_k, self.device)
                self.model_buffers_reserved = True
            self.lt.reserve_layer(self.device, layer_id, self.max_k)
            self.stock_carry.reserve(layer_id)
            if self.startup_ready is None:
                self.startup_ready = torch.cuda.Event()
            self.startup_ready.record(stream)


    def _reserve_uncaptured_capacity(self):
        if self.capture_reserved_bytes == 0:
            return
        # Keep the unconsumed capacity live even if SGLang re-sizes KV after
        # capturing a smaller prefill shape. Later larger shapes can use it.
        rows = 0 if self.hot_graph is None else self.shape[0]
        used = rows * self._capture_bytes_per_row()
        quantum = 2 * 1024 * 1024
        used = ((used + quantum - 1) // quantum) * quantum
        remaining = max(0, self.capture_reserved_bytes - used)
        if (
            self.capture_memory_reserve is not None
            and self.capture_memory_reserve.numel() == remaining
        ):
            return
        self.capture_memory_reserve = (
            torch.empty(remaining, device=self.device, dtype=torch.uint8)
            if remaining
            else None
        )


    @staticmethod
    def flat_view(value, shape):
        n = 1
        for dim in shape:
            n *= dim
        return value.view(-1)[:n].view(shape)


    def run_paged(
        self, q, w, cache, table64, table1, starts, ends, result, *,
        nk, common_end, last_visible, hot_key, window_start=0, ready_event=None,
        causal_query_start=None, decode_seed=None,
    ):
        inputs = (q, w, cache, table64, table1, starts, ends, result)
        options = dict(nk=nk, common_end=common_end, last_visible=last_visible,
                       hot_key=hot_key, window_start=window_start, ready_event=ready_event)
        if decode_seed is not None:
            options['decode_seed'] = decode_seed
        if not hasattr(self, "logical_query_capacity"):
            return self._run_paged_block(*inputs, **options)
        if len(q) <= self.max_q:
            # Reset only after a preceding logical call has finished publishing.
            with self.lock:
                self.slab_last_call = None
                return self._run_paged_block(*inputs, **options)
        from .slabbed_few import run_slabbed_paged

        return run_slabbed_paged(
            self, self._run_paged_block, *inputs,
            causal_query_start=causal_query_start, **options,
        )

    def _run_paged_block(
        self, q, w, cache, table64, table1, starts, ends, result, *,
        nk, common_end, last_visible, hot_key, window_start=0, ready_event=None,
        decode_seed=None,
    ):
        full = self._full_paged
        if not full and self._record5_direct_storage and (
            not getattr(self, "record5_direct_seed", False)
            or getattr(self, "record_format", None) != "ordered20-after-H32-5B"
        ):
            raise RuntimeError("Install the direct record5 pipeline before using typed Few storage")
        rows = len(q)
        slot_rows = execution_rows(self, rows)
        if torch.cuda.is_current_stream_capturing():
            return False
        if full and (
            not self.lt.ENABLED or not self.lt.PAGED_CANDIDATES or hot_key is None
        ):
            return False
        inputs = (q, w, cache, table64, table1, starts, ends, result)
        if not valid_paged_inputs(self.device, *inputs, score_heads=self.score_heads):
            return False
        minimum_keys = 196608 if full else self.lt.PRODUCTION_MIN_S
        if not (slot_rows > 0 and minimum_keys <= nk <= self.max_k):
            return False
        if window_start != 0 or not (12288 <= common_end <= last_visible <= nk):
            return False
        if full:
            # Full's public API also validates its dense gather layout here.
            # Few retains the native gather's layout contract.
            if not cache.is_contiguous() or not (
                (cache.ndim == 2 and cache.shape[1] == 64 * 132)
                or (cache.ndim == 3 and tuple(cache.shape[1:]) == (64, 132))
            ):
                return False
            if table64.shape[1] < (nk + 63) // 64 or table1.shape[1] < nk:
                return False
            if not table64.is_contiguous() or not table1.is_contiguous():
                return False

        with self.lock:
            stream = torch.cuda.current_stream(self.device)
            try:
                self.bind_shape(slot_rows, nk)
                self.logical_rows = rows
                if self.last_done is not None:
                    stream.wait_event(self.last_done)
                if self.startup_ready is not None:
                    stream.wait_event(self.startup_ready)
                self._ensure_model_buffers()
                if self.hot_graph is None and self.capture_memory_reserve is not None:
                    self.capture_memory_reserve = None
                    torch.cuda.empty_cache()
                    self.capture_reserve_released = True
                if ready_event is not None:
                    stream.wait_event(ready_event)
                if not full and self.live_nk_enabled:
                    self.rt.live_nk.fill_(nk)
                for value in inputs:
                    value.record_stream(stream)
                self.lt.begin_request(self.device, hot_key)
                self.q_input[:rows].copy_(q)
                self.w_input[:rows].copy_(w)
                self.start_input[:rows].copy_(starts)
                self.end_input[:rows].copy_(ends)
                pad_input_tail(self, rows, slot_rows)

                key = (str(self.device), hot_key)
                self.stock_carry.adopt(hot_key, common_end, window_start)
                prior = self.lt._HOT_CARRY.get(key)
                if (
                    prior is None
                    or len(prior) != 4
                    or prior[1] > common_end
                    or prior[3] < window_start
                ):
                    initialize_for_call(
                        self, cache, table64, nk, common_end, rows,
                        self.cold_hot, None, window_start,
                    )
                    self.lt._HOT_CARRY[key] = (
                        self.cold_hot, common_end, None, window_start
                    )
                if full:
                    self.native.fused = False
                    keys, scales = self.pk, self.ps_bytes
                else:
                    self.rt.rho_from_gather = False
                    self.rt.rho_source_signature = None
                    keys = self.pk[:nk] if self.live_nk_enabled else self.pk
                    scales = self.ps_bytes[:nk] if self.live_nk_enabled else self.ps_bytes
                if cache.ndim == 2:
                    cache = cache.view(cache.shape[0], 64, 132)
                plan = self.lt.prepare_permuted_gather(
                    cache, keys, scales, table64,
                    sequence_length=nk, query_length=slot_rows, num_reqs=1,
                    common_end=common_end, window_start=window_start, hot_key=hot_key,
                )
                if plan is None:
                    self._reserve_uncaptured_capacity()
                    return False
                self.last_plan = plan
                graph_kind = self.native.fused if full else self.rt.rho_from_gather
                if self.hot_graph is None or self.graph_kind != graph_kind:
                    self._runtime_call()
                    self._capture_after_real_call()
                else:
                    self.hot_graph.replay()
                    self.replay_calls += 1

                recent = min(rows, self.lt.CARRY_RECENT_ROWS)
                votes = self.lt._carry_vote_hist(
                    last_visible, self.device, hot_key, plan.get("carry_event")
                )
                if full:
                    self.modules["caller"].map_vote_page_(
                        self.output[:rows], plan["permutation"],
                        self.refiner.map_status[:rows], votes, recent,
                        self.prep["cc"][:rows], self.run_max, self.over_events,
                        self.lt.OVF_WATERMARK, table1,
                    )
                else:
                    self.modules["caller"].map_topk_vote_stats_litetopk_(
                        self.rt.output[:rows], plan["permutation"],
                        self.map_status[:rows], votes, recent,
                        self.rt.counts_original[:rows], self.run_max, self.over_events,
                        self.lt.OVF_WATERMARK,
                    )
                self.lt._publish_carry(hot_key, votes, last_visible, 0, recent)
                carry = self.lt._HOT_CARRY[key]
                if carry[2] is not None:
                    stream.wait_event(carry[2])
                if full:
                    result[:rows].copy_(self.output[:rows])
                else:
                    self.mapper.transform_index_page_table_decode_fast(
                        table1.expand(rows, -1), self.rt.output[:rows],
                        result=result[:rows], page_size=1,
                        decode_seed=decode_seed,
                    )
                return True
            finally:
                # Fence all queued work, including rebinding and extent staging
                # before the request registry can decline or raise.
                if self.last_done is None:
                    self.last_done = torch.cuda.Event()
                self.last_done.record(stream)

    def publish_stock_paged(self, active_paged, table64, *, hot_key, visible_end, window_start=0):
        """True means enqueued; only first-use validation can adopt this prior."""
        if torch.cuda.is_current_stream_capturing() or not self.stock_carry.eligible(
            active_paged, table64, hot_key, visible_end, window_start
        ):
            return False
        with self.lock, torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            if self.last_done is not None:
                stream.wait_event(self.last_done)
            if self.startup_ready is not None:
                stream.wait_event(self.startup_ready)
            return self.stock_carry.publish(active_paged, table64, hot_key, visible_end)
