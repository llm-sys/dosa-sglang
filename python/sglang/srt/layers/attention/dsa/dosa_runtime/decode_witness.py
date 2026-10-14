"""Exact history witnesses reduce selection work after ordinary FP8 scoring."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _advance_generation(requests, generations, N: tl.constexpr, SLOTS: tl.constexpr,
                        BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    request = tl.load(requests + i, i < N, other=-1)
    valid = (i < N) & (request >= 0) & (request < SLOTS)
    tl.atomic_add(generations + request, 1, mask=valid)


def invalidate_prefill(pool, forward_batch):
    generations = getattr(pool, '_dosa_decode_generations', None)
    if generations is None or forward_batch.forward_mode.is_decode_or_idle():
        return
    requests = forward_batch.req_pool_indices
    if requests.numel():
        _advance_generation[(1,)](
            requests, generations, requests.numel(), generations.numel(),
            tr.next_power_of_2(requests.numel()),
        )


@tr.jit
def _witness(scores, lengths, requests, history, seen, generations, thresholds,
             counts, SCORE_STRIDE: tl.constexpr, SLOTS: tl.constexpr,
             K: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, K)
    request = tl.load(requests + row)
    length = tl.load(lengths + row)
    active = (request >= 0) & (request < SLOTS) & (length > 0)
    epoch = tl.load(generations + request, active, other=0)
    previous = tl.load(seen + request, active, other=-1)
    valid = active & (epoch == previous)
    first = tl.load(history + request * (2 * K) + i, valid, other=-1)
    second = tl.load(history + request * (2 * K) + K + i, valid, other=-1)
    aok = valid & (first >= 0) & (first < length)
    bok = valid & (second >= 0) & (second < length)
    a = tl.load(scores + row * SCORE_STRIDE + first, aok, other=float('-inf'))
    b = tl.load(scores + row * SCORE_STRIDE + second, bok, other=float('-inf'))
    aok = aok & (a == a) & (tl.abs(a) != float('inf'))
    bok = bok & (b == b) & (tl.abs(b) != float('inf'))
    amin = tl.min(tl.where(aok, a, float('-inf')), 0)
    bmin = tl.min(tl.where(bok, b, float('-inf')), 0)
    tl.store(thresholds + row, tl.maximum(amin, bmin))
    tl.store(counts + row, 0)


@tr.jit
def _compact(scores, lengths, thresholds, values, keys, counts,
             SCORE_STRIDE: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    start = tl.program_id(1) * BLOCK
    length = tl.load(lengths + row)
    # This CTA cannot produce a record beyond the visible sequence.
    if start < length:
        i = start + tl.arange(0, BLOCK)
        tau = tl.load(thresholds + row)
        valid = (i < length) & (i < WIDTH)
        score = tl.load(scores + row * SCORE_STRIDE + i, valid, other=float('-inf'))
        finite = (score == score) & (tl.abs(score) != float('inf'))
        keep = valid & ((score >= tau) | ~finite)
        n = tl.sum(keep.to(tl.int32), 0)
        if n > 0:
            offset = tl.atomic_add(counts + row, n)
            rank = tl.cumsum(keep.to(tl.int32), 0) - 1
            tl.store(values + row * WIDTH + offset + rank, score, keep)
            tl.store(keys + row * WIDTH + offset + rank, i, keep)


@tr.jit
def _commit(keys, selected, table, requests, lengths, history, seen, generations,
            output, WIDTH: tl.constexpr, TABLE_STRIDE: tl.constexpr,
            PAGE: tl.constexpr, SLOTS: tl.constexpr, K: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, K)
    request = tl.load(requests + row)
    length = tl.load(lengths + row)
    active = (request >= 0) & (request < SLOTS) & (length > 0)
    chosen = tl.load(selected + row * K + i)
    logical = tl.load(keys + row * WIDTH + chosen, chosen >= 0, other=-1)
    valid = active & (logical >= 0) & (logical < length)
    page = tl.load(table + row * TABLE_STRIDE + logical // PAGE, valid, other=0)
    physical = tl.where(valid, page * PAGE + logical % PAGE, -1)
    tl.store(output + row * K + i, physical)
    epoch = tl.load(generations + request, active, other=0)
    previous = tl.load(seen + request, active, other=-1)
    old = tl.load(history + request * (2 * K) + i,
                  active & (epoch == previous), other=-1)
    tl.store(history + request * (2 * K) + K + i, old, active)
    tl.store(history + request * (2 * K) + i, tl.where(valid, logical, -1), active)
    # Publish the epoch only after every warp has read and replaced its old bank.
    tl.debug_barrier()
    tl.store(seen + request, epoch, active)


class DecodeWitness:
    def __init__(self, pool, device):
        slots = pool.size + 1
        if not hasattr(pool, '_dosa_decode_generations'):
            pool._dosa_decode_generations = torch.zeros(slots, device=device, dtype=torch.int64)
        self.generations = pool._dosa_decode_generations
        self.history = torch.full((slots, 4096), -1, device=device, dtype=torch.int32)
        self.seen = torch.full((slots,), -1, device=device, dtype=torch.int64)

    def select(self, scores, lengths, requests, page_table, page_size):
        from sglang.kernels.ops.attention.dsv4.topk import (
            topk_transform_512_v2,
        )

        rows, width = scores.shape
        slots = self.seen.numel()
        values = torch.empty((rows, width), device=scores.device, dtype=torch.float32)
        keys = torch.empty((rows, width), device=scores.device, dtype=torch.int32)
        counts = torch.empty((rows,), device=scores.device, dtype=torch.int32)
        thresholds = torch.empty((rows,), device=scores.device, dtype=torch.float32)
        selected = torch.empty((rows, 2048), device=scores.device, dtype=torch.int32)
        output = torch.empty_like(selected)
        _witness[(rows,)](
            scores, lengths, requests, self.history, self.seen, self.generations,
            thresholds, counts, scores.stride(0), slots, 2048, num_warps=8,
        )
        _compact[(rows, tr.cdiv(width, 1024))](
            scores, lengths, thresholds, values, keys, counts,
            scores.stride(0), width, 1024, num_warps=4,
        )
        # The raw-index output supplies compact positions for the final mapping.
        topk_transform_512_v2(values, counts, keys, output, 1,
                              out_raw_indices=selected)
        _commit[(rows,)](
            keys, selected, page_table, requests, lengths, self.history, self.seen,
            self.generations, output, width, page_table.stride(0), page_size,
            slots, 2048, num_warps=8,
        )
        return output
