"""Original graph-safe, unordered FI with an explicitly owned output buffer."""

import torch
from flashinfer.topk import get_topk_module
from flashinfer.utils import _get_cache_buf


def topk_into(scores, offsets, lengths, output):
    """Use the same FFI operation as top_k_ragged_transform(dsa_graph_safe=True).

    Only output allocation moves to startup. The row offsets/lengths, k=2048,
    unordered policy, row-state buffer and native dispatch remain unchanged.
    Callers consume each output before reusing its storage on their serial
    stream; captured graphs explicitly keep those storages alive.
    """
    if (output.dtype != torch.int32 or output.device != scores.device
            or tuple(output.shape) != (len(scores), 2048) or not output.is_contiguous()):
        raise ValueError("FI output must be contiguous int32[rows,2048] on the score device")
    row_states = _get_cache_buf(
        f"radix_topk_row_states_{scores.device}", 1024 * 1024, scores.device, zero_init=True
    )
    get_topk_module().radix_topk_ragged_transform(
        scores, output, offsets, lengths, row_states, 2048, False, 0, True,
        row_starts=None,
    )
    return output
