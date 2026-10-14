"""One Few prefill interval and eight-row physical execution geometry."""

MIN_FEW_QUERY_CAPACITY = 4096
MIN_FEW_PREFILL_ROWS = 1
MAX_FEW_QUERY_CAPACITY = 32768
LEGACY_NATIVE_QUERY_CAPACITY = 16384
FEW_RECORD_CAPACITY = 114688


def configured_query_capacity(server_args=None):
    """Use SGLang's real buffer ceiling; do not change its scheduling settings."""
    if server_args is None:
        from sglang.srt.runtime_context import get_server_args
        server_args = get_server_args()
    rows = int(server_args.max_prefill_buffer_tokens())
    if rows <= 0:
        rows = int(server_args.max_prefill_tokens or 0)
    if rows <= 0:
        return 0
    capacity = (rows + 7) // 8 * 8
    return capacity if capacity <= MAX_FEW_QUERY_CAPACITY else 0


def service_rows_eligible(logical_rows, capacity):
    # A small tail can reuse an enabled slot. It must not enable a service whose
    # configured workspace capacity is itself outside the reviewed range.
    return (MIN_FEW_QUERY_CAPACITY <= capacity <= MAX_FEW_QUERY_CAPACITY
            and capacity % 8 == 0
            and MIN_FEW_PREFILL_ROWS <= logical_rows <= capacity)


def validate_query_capacity(rows, modules=(), *, minimum=MIN_FEW_QUERY_CAPACITY):
    """Old binaries retain their 16K ceiling until they declare a wider host ABI."""
    limit = MAX_FEW_QUERY_CAPACITY
    for module in modules:
        getter = getattr(module, "max_query_rows", None)
        limit = min(limit, int(getter()) if callable(getter) else LEGACY_NATIVE_QUERY_CAPACITY)
    if not (minimum <= rows <= limit and rows % 8 == 0):
        raise ValueError(f"Few query capacity {rows} exceeds the aligned native capacity {limit}")
    return rows


def causal_id_page_bound(rows, keys, seed=12288):
    """Worst-case ID pages for zero-start, contiguous causal query rows.

    Within four queries' common interval, compaction stores each surviving key
    once. Beyond that interval at most three solo copies survive. The endpoint
    span is at most rows-1, even after query sorting. Five separate streams add
    at most four partially occupied pages per group.
    """
    per_group = (keys - seed + 2 * (rows - 1) + 127) // 128 + 4
    return (rows // 4) * per_group


def few_id_pool_pages(rows, keys):
    """Reserve the original row allowance or the full causal ID extent."""
    return max(256 * rows, causal_id_page_bound(rows, keys))


def service_query_storage(logical_rows, keys, score_heads):
    """Bound either head geometry by the existing per-slot ID memory budget.

    Logical chunk sizes stay unchanged. The public paged call splits them into
    physical query blocks when their complete causal ID extent does not fit.
    Record5 and all other query workspaces use the same smaller block geometry
    before model-load reservation and KV sizing.
    """
    rows = min(logical_rows, LEGACY_NATIVE_QUERY_CAPACITY)
    pages = 256 * rows
    while rows > 8 and few_id_pool_pages(rows, keys) > pages:
        rows = (rows // 16) * 8
    return rows, pages


def enable_service_prefill_interval(backend, *, min_rows=MIN_FEW_QUERY_CAPACITY):
    """Select paid cold-hint eligibility separately from row-padding capability."""
    if not (MIN_FEW_PREFILL_ROWS <= min_rows <= backend.max_q
            and 8 <= backend.max_q <= MAX_FEW_QUERY_CAPACITY
            and backend.max_q % 8 == 0):
        raise ValueError("service prefill interval is outside supported capacity")
    if (backend.shape is not None or backend.hot_graph is not None
            or backend.last_done is not None or backend.startup_ready is not None):
        raise RuntimeError("configure service interval before reserve, calls or capture")
    backend._dosa_service_prefill_min_rows = min_rows
    return backend


def enable_query_tail_padding(backend, *, min_rows=MIN_FEW_QUERY_CAPACITY):
    """Enable before slot reservation; service admission is owned by the caller."""
    if not (MIN_FEW_PREFILL_ROWS <= min_rows <= backend.max_q
            and 8 <= backend.max_q <= MAX_FEW_QUERY_CAPACITY
            and backend.max_q % 8 == 0):
        raise ValueError("tail padding requires a supported prefill interval within native capacity")
    if (backend.shape is not None or backend.hot_graph is not None
            or backend.last_done is not None or backend.startup_ready is not None):
        raise RuntimeError("configure padding before reserve, calls or capture")
    backend._dosa_query_padding_min_rows = min_rows
    backend.logical_rows = None


def padding_enabled_for(backend, rows):
    minimum = getattr(backend, "_dosa_query_padding_min_rows", None)
    return minimum is not None and minimum <= rows <= backend.max_q


def execution_rows(backend, logical_rows):
    if not 0 < logical_rows <= backend.max_q:
        return 0
    if logical_rows % 8 == 0:
        return logical_rows
    if not padding_enabled_for(backend, logical_rows):
        return 0
    physical_rows = (logical_rows + 7) // 8 * 8
    return physical_rows if physical_rows <= backend.max_q else 0


def service_execution_rows(backend, logical_rows):
    """Intersect the explicit service interval with the backend's row geometry.

    Full and Few share the service interval and eight-row padding policy.
    A backend with no explicit interval keeps max-only eligibility.
    """
    minimum = getattr(backend, "_dosa_service_prefill_min_rows", backend.max_q)
    capacity = getattr(backend, "logical_query_capacity", backend.max_q)
    if not minimum <= logical_rows <= capacity:
        return 0
    if logical_rows > backend.max_q:
        return (logical_rows + 7) // 8 * 8
    return execution_rows(backend, logical_rows)


def pad_input_tail(backend, logical_rows, physical_rows):
    """Duplicate only the final real row in already-owned slots, on their stream."""
    if physical_rows == logical_rows:
        return
    if not (0 < physical_rows - logical_rows < 8
            and physical_rows <= backend.max_q
            and padding_enabled_for(backend, logical_rows)):
        raise RuntimeError("invalid logical/physical query extent")
    for slot in (backend.q_input, backend.w_input,
                 backend.start_input, backend.end_input):
        slot[logical_rows:physical_rows].copy_(
            slot[logical_rows - 1:logical_rows].expand(
                physical_rows - logical_rows, *slot.shape[1:]
            )
        )
