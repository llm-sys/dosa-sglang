"""Same-call paid initialization of distinct, currently visible logical hints."""

import triton
import triton.language as tl

from .query_slots import service_execution_rows


@triton.jit
def _initialize(
    HOT, STRONG, START: tl.constexpr, H: tl.constexpr, S: tl.constexpr, B: tl.constexpr
):
    i = tl.arange(0, B)
    tl.store(HOT + i, START + i, i < H)
    if S > 0:
        tl.store(STRONG + i, START + i, i < S)


def initialize(hot, strong=None, start=0):
    # These are suggestions only. The complete chain still produces original
    # H32 seed scores and its safe threshold for every current query.
    _initialize[(1,)](
        hot,
        hot if strong is None else strong,
        start,
        hot.numel(),
        0 if strong is None else strong.numel(),
        triton.next_power_of_2(hot.numel()),
        num_warps=4,
    )


def initialize_for_call(backend, cache, table, nk, common_end, queries,
                        hot, strong=None, start=0):
    """Bind cold inputs only after the backend owns its slot and input buffers."""
    owner = backend.bootstrap
    if owner is not None and service_execution_rows(backend, queries):
        return owner.cold_call(
            cache, table, nk, common_end, queries, hot, strong, start,
        )
    return initialize(hot, strong, start)
