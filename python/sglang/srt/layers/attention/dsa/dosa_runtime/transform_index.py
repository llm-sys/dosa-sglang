from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def transform_index_page_table_decode_kernel(
    page_table_ptr: torch.Tensor,
    topk_indices_ptr: torch.Tensor,
    result_ptr: torch.Tensor,
    page_size: tl.constexpr,
    page_table_row_stride: tl.constexpr,
    history,
    seen,
    generations,
    request_slot,
    seed_rows: tl.constexpr,
    seed_slots: tl.constexpr,
    seed_decode: tl.constexpr,
):
    TOPK: tl.constexpr = 2048
    req_id = tl.program_id(0)
    page_table_ptr = page_table_ptr + req_id * page_table_row_stride
    topk_indices_ptr = topk_indices_ptr + req_id * TOPK
    result_ptr = result_ptr + req_id * TOPK

    offset = tl.arange(0, TOPK)  # topk should be 2048
    loaded_topk_indices = tl.load(topk_indices_ptr + offset)
    mask = loaded_topk_indices >= 0
    loaded_kv_indices = tl.load(page_table_ptr + loaded_topk_indices, mask=mask)
    tl.store(result_ptr + offset, loaded_kv_indices, mask=mask)
    tl.store(result_ptr + offset, -1, mask=~mask)
    if seed_decode:
        if req_id >= seed_rows - 2:
            request = tl.load(request_slot)
            active = (request >= 0) & (request < seed_slots)
            bank = seed_rows - 1 - req_id
            tl.store(history + request * 4096 + bank * TOPK + offset,
                     loaded_topk_indices, active)
            if seed_rows == 1:
                tl.store(history + request * 4096 + TOPK + offset, -1, active)
            if req_id == seed_rows - 1:
                epoch = tl.load(generations + request, active, other=0)
                tl.debug_barrier()
                tl.store(seen + request, epoch, active)
    # A later ordinary stream launch consumes the seed after both row CTAs
    # finish. No consumer polls the epoch inside this kernel.


def transform_index_page_table_decode_fast(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    result: Optional[torch.Tensor] = None,
    page_size: int = 1,
    decode_seed=None,
) -> torch.Tensor:
    """
    Transform the page table according to topk indices for sparse topk attention.
    Args:
        page_table: [qo_len, max_seqlen_k], the original page table
        topk_indices: [qo_len, topk], the topk indices for each query position
    Returns:
        transformed_page_table: [qo_len, topk], the transformed page table
        For out-of-bound indices in topk_indices, this should be filled with -1.
    """
    assert page_size == 1
    assert page_table.shape[0] == topk_indices.shape[0]
    assert topk_indices.shape[1] == 2048
    qo_len = topk_indices.shape[0]
    if result is None:
        result = torch.empty_like(topk_indices, dtype=torch.int32)
    if decode_seed is None:
        history = seen = generations = request_slot = None
        slots = 0
    else:
        state, request_slot = decode_seed
        assert request_slot.numel() == 1
        history, seen, generations = state.history, state.seen, state.generations
        slots = seen.numel()
    # Launch triton kernel
    grid = (qo_len,)
    transform_index_page_table_decode_kernel[grid](
        page_table,
        topk_indices,
        result,
        page_size,
        page_table_row_stride=page_table.stride(0),
        history=history,
        seen=seen,
        generations=generations,
        request_slot=request_slot,
        seed_rows=qo_len,
        seed_slots=slots,
        seed_decode=decode_seed is not None,
    )
    return result
