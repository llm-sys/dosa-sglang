"""Logical query partitions for a bounded, serial execution workspace."""


def query_slabs(rows, capacity, recent_rows):
    """Yield source interval, physical Q8 extent and global carry intersection.

    Partitioning never changes the logical final-row window used for carry.
    In particular, a short last slab may share that window with its predecessor.
    The caller owns one frozen prior and one completion transaction across them.
    """
    if rows <= 0 or capacity <= 0 or capacity % 8 or recent_rows <= 0:
        raise ValueError("Invalid query workspace or carry extent")
    carry_begin = max(0, rows - recent_rows)
    for begin in range(0, rows, capacity):
        end = min(begin + capacity, rows)
        logical = end - begin
        yield begin, end, (logical + 7) // 8 * 8, max(0, end - max(begin, carry_begin))
