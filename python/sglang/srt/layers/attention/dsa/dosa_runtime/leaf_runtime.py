"""Only leaf production changes; the proved original-FI hierarchy is reused."""

import torch, flashinfer
from .fi_output import topk_into


class FusedLeafReplayFI:
    """Own the original hierarchy and its retained FI output buffers."""
    def __init__(self, nq, nk, device, module):
        assert 0 < nq <= 512 and 0 < nk <= 1048576
        self.core = module
        self.nq = nq
        self.nk = nk
        self.chunks = (nk + 32767) // 32768

        def i(n):
            return torch.empty(n, device=device, dtype=torch.int32)

        self.rows = i(8)
        self.state = i(1)
        self.finite = i(8)
        self.leaf_lengths = i(8 * self.chunks)
        self.merge_lengths = i(8)
        self.fallback_lengths = i(nq)
        self.leaf = torch.empty(
            (8 * self.chunks, 32768), device=device, dtype=torch.float32
        )
        self.merge = torch.empty(
            (8, self.chunks * 2048), device=device, dtype=torch.float32
        )
        self.ids = i((8, self.chunks * 2048))
        self.leaf_zero = torch.zeros_like(self.leaf_lengths)
        self.merge_zero = torch.zeros_like(self.merge_lengths)
        self.full_zero = torch.zeros(nq, device=device, dtype=torch.int32)
        self.leaf_selected = torch.empty(
            (8 * self.chunks, 2048), device=device, dtype=torch.int32)
        self.merge_selected = torch.empty((8, 2048), device=device, dtype=torch.int32)
        self.full_selected = torch.empty((nq, 2048), device=device, dtype=torch.int32)

    def replay_and_select(
        self, replay, q, k, scale, weights, starts, ends, error, dense, counts,
        *, logical_nk=None,
    ):
        args = (
            q,
            k,
            scale,
            weights,
            starts,
            ends,
            error,
            dense,
            counts,
            self.leaf,
            self.rows,
            self.state,
            self.finite,
            self.leaf_lengths,
        )
        if logical_nk is None:
            replay.replay_and_leaf(*args)
        else:
            replay.replay_and_leaf_capacity(*args, logical_nk)
        return self.select_prepared(dense, counts)

    def select_prepared(self, dense, counts):
        c = self.core
        leaf_ids = topk_into(self.leaf, self.leaf_zero, self.leaf_lengths,
                            self.leaf_selected[:len(self.leaf)])
        c.gather(
            self.leaf,
            leaf_ids,
            self.leaf_lengths,
            self.rows,
            self.state,
            self.finite,
            self.merge,
            self.ids,
        )
        c.final_meta(
            counts,
            self.rows,
            self.state,
            self.finite,
            self.merge_lengths,
            self.fallback_lengths,
            self.chunks,
        )
        selected = topk_into(self.merge, self.merge_zero, self.merge_lengths, self.merge_selected)
        output = topk_into(dense, self.full_zero, self.fallback_lengths,
                          self.full_selected[:len(dense)])
        c.publish(self.ids, selected, self.rows, self.state, self.finite, output)
        return output

    def nbytes(self):
        return sum(
            x.numel() * x.element_size()
            for x in vars(self).values()
            if isinstance(x, torch.Tensor)
        )
