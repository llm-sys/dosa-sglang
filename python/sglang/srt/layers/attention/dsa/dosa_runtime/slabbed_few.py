"""One logical Few call with bounded query scratch and one carry transaction."""
import threading
import torch

from .query_slabs import query_slabs
from .paged_backend import valid_paged_inputs


def install_query_slabs(backend, logical_capacity):
    if (logical_capacity < backend.max_q or logical_capacity % 8
            or backend.shape is not None or backend.last_done is not None
            or backend.startup_ready is not None):
        raise ValueError("Install logical capacity on a fresh bounded backend")
    marker = getattr(backend.modules['caller'], 'supports_zero_recent_rows', None)
    if logical_capacity > backend.max_q and (not callable(marker) or not marker()):
        raise RuntimeError("Query slabs require the explicit zero-vote mapping ABI")
    if backend.lt.TIERED_SEED_12K:
        raise ValueError("Query slabs require the current ordinary B4 hint contract")
    # One logical call owns the slot across its original per-slab calls.
    backend.lock = threading.RLock()
    backend.logical_query_capacity = logical_capacity
    backend.supports_causal_query_interval = True
    recent = backend.lt.CARRY_RECENT_ROWS
    backend.slab_saved_hot = torch.empty(backend.lt.HOT_PREFIX, dtype=torch.int64, device=backend.device)
    backend.slab_restore_event = torch.cuda.Event()
    backend.slab_carry_rows = torch.empty((recent, 2048), dtype=torch.int32, device=backend.device)
    backend.slab_identity = torch.arange(backend.max_k, dtype=torch.int32, device=backend.device)
    backend.slab_zero_status = torch.zeros(recent, dtype=torch.int32, device=backend.device)
    backend.slab_zero_counts = torch.zeros(recent, dtype=torch.int32, device=backend.device)
    backend.slab_unused_stat = torch.zeros(1, dtype=torch.int32, device=backend.device)
    backend.slab_ledger = {
        name: torch.empty((logical_capacity, *backend.backing[name].shape[1:]),
                          dtype=backend.backing[name].dtype, device=backend.device)
        for name in ('counts_original', 'error_original', 'error_before_completion', 'replay_work')
    }
    backend.slab_map_status = torch.empty(logical_capacity, dtype=torch.int32, device=backend.device)
    backend.slab_original_scratch = torch.empty(backend.max_q, dtype=torch.int32, device=backend.device)
    backend.slab_last_call = None
    return backend


def copy_slab_results(backend, begin, end, physical_rows):
    """Restore row ownership before copying one slab's unchanged diagnostics."""
    logical_rows = end - begin
    for name, destination in backend.slab_ledger.items():
        source = getattr(backend.rt, name)
        if name in ('error_before_completion', 'replay_work'):
            backend.slab_original_scratch[:physical_rows].index_copy_(
                0, backend.rt.order, source)
            source = backend.slab_original_scratch
        destination[begin:end].copy_(source[:logical_rows])


def _eligible_slab_call(b, q, w, cache, table64, table1, starts, ends, result, *,
                      nk, common_end, last_visible, hot_key, window_start=0,
                      ready_event=None):
    rows = len(q)
    if torch.cuda.is_current_stream_capturing():
        return False
    if not (b.max_q < rows <= b.logical_query_capacity and b.lt.PRODUCTION_MIN_S <= nk <= b.max_k
            and window_start == 0 and 12288 <= common_end <= last_visible <= nk):
        return False
    inputs = (q, w, cache, table64, table1, starts, ends, result)
    if not valid_paged_inputs(b.device, *inputs, score_heads=b.score_heads):
        return False
    if b._record5_direct_storage and (not getattr(b, 'record5_direct_seed', False)
            or getattr(b, 'record_format', None) != 'ordered20-after-H32-5B'):
        raise RuntimeError("Direct record5 pipeline must be installed before execution")
    return True


def run_frozen_slabbed_paged(b, q, w, cache, table64, table1, starts, ends, result, *,
                      nk, common_end, last_visible, hot_key, window_start=0,
                      ready_event=None, decode_seed=None):
    if not _eligible_slab_call(b, q, w, cache, table64, table1, starts, ends, result,
                               nk=nk, common_end=common_end, last_visible=last_visible,
                               hot_key=hot_key, window_start=window_start, ready_event=ready_event):
        return False
    rows = len(q)
    inputs = (q, w, cache, table64, table1, starts, ends, result)
    slabs = tuple(query_slabs(rows, b.max_q, b.lt.CARRY_RECENT_ROWS))
    with b.lock:
        stream = torch.cuda.current_stream(b.device)
        b.bind_shape(slabs[0][2], nk)
        if b.last_done is not None:
            stream.wait_event(b.last_done)
        if b.startup_ready is not None:
            stream.wait_event(b.startup_ready)
        b._ensure_model_buffers()
        if b.hot_graph is None and b.capture_memory_reserve is not None:
            b.capture_memory_reserve = None
            torch.cuda.empty_cache()
            b.capture_reserve_released = True
        if ready_event is not None:
            stream.wait_event(ready_event)
        for value in inputs:
            value.record_stream(stream)
        b.lt.begin_request(b.device, hot_key)
        key = (str(b.device), hot_key)
        previous_prior = b.lt._HOT_CARRY.get(key)
        committed = False
        publishing = False
        b.slab_last_call = dict(rows=rows, scratch_rows=b.max_q, slabs=slabs, queued_complete=False)
        try:
            prior = previous_prior
            if (prior is None or len(prior) != 4 or prior[1] > common_end or prior[3] < window_start):
                # All four anchors belong to the original logical Q, not slab 0.
                b.bootstrap.cold_call(
                    cache, table64, nk, common_end, rows, b.cold_hot, None, window_start,
                    source_q=q, source_w=w,
                )
                b.lt._HOT_CARRY[key] = (b.cold_hot, common_end, None, window_start)
            b.rt.rho_from_gather = False
            b.rt.rho_source_signature = None
            raw_cache = cache.view(cache.shape[0], 64, 132) if cache.ndim == 2 else cache
            plan = b.lt.prepare_permuted_gather(
                raw_cache, b.pk[:nk] if b.live_nk_enabled else b.pk,
                b.ps_bytes[:nk] if b.live_nk_enabled else b.ps_bytes, table64,
                sequence_length=nk, query_length=slabs[0][2], num_reqs=1,
                common_end=common_end, window_start=window_start, hot_key=hot_key,
            )
            if plan is None:
                b._reserve_uncaptured_capacity()
                return False
            b.last_plan = plan
            votes = b.lt._carry_vote_hist(last_visible, b.device, hot_key, plan.get('carry_event'))
            for begin, end, physical, recent in slabs:
                logical = end - begin
                if begin:
                    # Shape retirement must wait for this call's earlier slab,
                    # including output mapping and diagnostics, not an older call.
                    if b.last_done is None:
                        b.last_done = torch.cuda.Event()
                    b.last_done.record(stream)
                    b.bind_shape(physical, nk)
                b.logical_rows = logical
                if b.live_nk_enabled:
                    b.rt.live_nk.fill_(nk)
                for slot, source in ((b.q_input, q), (b.w_input, w),
                                     (b.start_input, starts), (b.end_input, ends)):
                    slot[:logical].copy_(source[begin:end])
                    if physical != logical:
                        slot[logical:physical].copy_(slot[logical-1:logical].expand(
                            physical-logical, *slot.shape[1:]))
                if b.hot_graph is None or b.graph_kind != b.rt.rho_from_gather:
                    b._runtime_call()
                    b._capture_after_real_call()
                else:
                    b.hot_graph.replay()
                    b.replay_calls += 1
                copy_slab_results(b, begin, end, physical)
                b.modules['caller'].map_topk_vote_stats_litetopk_(
                    b.rt.output[:logical], plan['permutation'], b.map_status[:logical],
                    votes, recent, b.rt.counts_original[:logical], b.run_max,
                    b.over_events, b.lt.OVF_WATERMARK,
                )
                b.slab_map_status[begin:end].copy_(b.map_status[:logical])
                b.mapper.transform_index_page_table_decode_fast(
                    table1.expand(logical, -1), b.rt.output[:logical],
                    result=result[begin:end], page_size=1,
                    decode_seed=decode_seed if end == rows else None,
                )
            publishing = True
            b.lt._publish_carry(hot_key, votes, last_visible, 0, min(rows, b.lt.CARRY_RECENT_ROWS))
            carry = b.lt._HOT_CARRY[key]
            if carry[2] is not None:
                stream.wait_event(carry[2])
            committed = True
            b.logical_rows = rows
            b.slab_last_call['queued_complete'] = True
            return True
        finally:
            if not committed:
                # Publication writes reusable hot storage in place. Once it
                # starts, a failure cannot restore old bits by restoring a tuple.
                if publishing:
                    side = b.lt._HOT_STREAM.get(str(b.device))
                    if side is not None:
                        stream.wait_stream(side)
                if publishing or previous_prior is None:
                    b.lt._HOT_CARRY.pop(key, None)
                else:
                    b.lt._HOT_CARRY[key] = previous_prior
            if b.last_done is None:
                b.last_done = torch.cuda.Event()
            b.last_done.record(stream)


def run_slabbed_paged(b, original, q, w, cache, table64, table1, starts, ends,
                      result, *, nk, common_end, last_visible, hot_key,
                      window_start=0, ready_event=None, causal_query_start=None,
                      decode_seed=None):
    """Advance only a contiguous causal interval; preserve one public transaction.

    Each finished slab may improve the next slab's hint. This changes the
    amount of certified work, never the score arithmetic or visibility. A short
    final slab still publishes votes for the entire logical call's last window.
    """
    if not _eligible_slab_call(b, q, w, cache, table64, table1, starts, ends, result,
                               nk=nk, common_end=common_end, last_visible=last_visible,
                               hot_key=hot_key, window_start=window_start, ready_event=ready_event):
        return False
    rows = len(q)
    # Min/max endpoints alone do not prove row order. Only the service's
    # explicit host interval contract permits advancing the scan boundary.
    if (causal_query_start is None or common_end != causal_query_start + 1
            or last_visible != causal_query_start + rows):
        return run_frozen_slabbed_paged(
            b, q, w, cache, table64, table1, starts, ends, result, nk=nk,
            common_end=common_end, last_visible=last_visible, hot_key=hot_key,
            window_start=window_start, ready_event=ready_event, decode_seed=decode_seed)
    slabs = tuple(query_slabs(rows, b.max_q, b.lt.CARRY_RECENT_ROWS))
    recent = min(rows, b.lt.CARRY_RECENT_ROWS)
    repair_window = slabs[-1][1] - slabs[-1][0] < recent
    carry_begin = rows - recent
    with b.lock:
        stream = torch.cuda.current_stream(b.device)
        for event in (b.last_done, b.startup_ready, ready_event):
            if event is not None:
                stream.wait_event(event)
        b.lt.begin_request(b.device, hot_key)
        key = (str(b.device), hot_key)
        previous = b.lt._HOT_CARRY.get(key)
        saved = (previous is not None and len(previous) == 4
                 and isinstance(previous[0], torch.Tensor)
                 and previous[0].device == b.device
                 and previous[0].dtype == torch.int64
                 and previous[0].numel() == b.lt.HOT_PREFIX)
        if saved:
            if previous[2] is not None:
                stream.wait_event(previous[2])
            b.slab_saved_hot.copy_(previous[0])
        committed = False
        b.slab_last_call = dict(rows=rows, scratch_rows=b.max_q, slabs=slabs,
                                causal_progress=True, completed_slabs=0,
                                repaired_carry_window=repair_window, queued_complete=False)
        try:
            for begin, end, physical, overlap in slabs:
                logical = end - begin
                visible = common_end + end - 1
                # The first part can precede the native admission threshold.
                # Scanning its existing full prefix remains safe: ends[] masks
                # the not-yet-visible keys exactly as in the original call.
                scan_k = max(b.lt.PRODUCTION_MIN_S, visible)
                handled = original(
                    q[begin:end], w[begin:end], cache, table64, table1,
                    starts[begin:end], ends[begin:end], result[begin:end],
                    nk=scan_k, common_end=common_end + begin, last_visible=visible,
                    hot_key=hot_key, window_start=window_start,
                    **({'decode_seed': decode_seed}
                       if end == rows and decode_seed is not None else {}))
                if not handled:
                    return False
                copy_slab_results(b, begin, end, physical)
                b.slab_map_status[begin:end].copy_(b.map_status[:logical])
                if repair_window and overlap:
                    offset = end - overlap - carry_begin
                    b.slab_carry_rows[offset:offset + overlap].copy_(
                        b.rt.output[logical - overlap:logical])
                # Shape retirement or a subsequent stream must cover these
                # copies as well as the original per-slab call.
                b.last_done.record(stream)
                b.slab_last_call['completed_slabs'] += 1
            if repair_window:
                carry = b.lt._HOT_CARRY[key]
                votes = b.lt._carry_vote_hist(last_visible, b.device, hot_key, carry[2])
                # Rows already contain original logical IDs. Identity mapping
                # uses the original checked voting kernel; private zero stats
                # avoid counting these rows twice in runtime telemetry.
                b.modules['caller'].map_topk_vote_stats_litetopk_(
                    b.slab_carry_rows[:recent], b.slab_identity[:last_visible],
                    b.slab_zero_status[:recent], votes, recent,
                    b.slab_zero_counts[:recent], b.slab_unused_stat,
                    b.slab_unused_stat, b.lt.OVF_WATERMARK)
                b.lt._publish_carry(hot_key, votes, last_visible, 0, recent)
                stream.wait_event(b.lt._HOT_CARRY[key][2])
            b.logical_rows = rows
            b.slab_last_call['queued_complete'] = True
            committed = True
            return True
        finally:
            if not committed:
                # Publication reuses the caller-owned hot tensor in place.
                # Restore values, not merely the earlier Python tuple.
                side = b.lt._HOT_STREAM.get(str(b.device))
                if side is not None:
                    stream.wait_stream(side)
                if saved:
                    previous[0].copy_(b.slab_saved_hot)
                    b.slab_restore_event.record(stream)
                    b.lt._HOT_CARRY[key] = (previous[0], previous[1],
                                            b.slab_restore_event, previous[3])
                else:
                    b.lt._HOT_CARRY.pop(key, None)
            if b.last_done is None:
                b.last_done = torch.cuda.Event()
            b.last_done.record(stream)
