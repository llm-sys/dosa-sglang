"""Original hybrid seed selection without normal-case score-table copying."""

import torch
import flashinfer
from .fi_output import topk_into


def seed_scores_view(scores):
    """Keep DG's aligned row-padded view; retain the old copy for other layouts.

    Call this around the DG result instead of .contiguous(), so later seed
    metadata observes the same supported layout as the initializer.
    """
    if (
        scores.ndim == 2
        and scores.stride(1) == 1
        and scores.stride(0) >= scores.shape[1]
        and scores.stride(0) % 4 == 0
        and scores.data_ptr() % 16 == 0
    ):
        return scores
    return scores.contiguous()


def fi_score_segments(scores):
    """Yield contiguous FI views without reading absent last-row padding.

    FI's unchanged FFI uses size(1) as its physical row pitch. The original
    lengths still restrict selection to logical seed columns, or zero.
    """
    rows, width = scores.shape
    stride = scores.stride(0)
    assert rows > 0 and scores.stride(1) == 1 and stride >= width
    storage_elements = scores.untyped_storage().nbytes() // scores.element_size()
    if scores.storage_offset() + rows * stride <= storage_elements:
        yield 0, rows, scores.as_strided((rows, stride), (stride, 1))
        return
    if rows > 1:
        yield 0, rows - 1, scores[:-1].as_strided((rows - 1, stride), (stride, 1))
    # A single row's stride(0) is irrelevant to its contiguous logical extent.
    last = scores[-1:]
    assert last.is_contiguous()
    yield rows - 1, rows, last


class ExactSeedInitializer:
    def __init__(self, rows, device, module):
        self.bind_core(module)
        self.selected = None
        self.zero = torch.zeros(rows, device=device, dtype=torch.int32)
        self.lengths = torch.empty_like(self.zero)
        self.diag = torch.empty((rows, 4), device=device, dtype=torch.int32)

    def bind_core(self, module, *, packed=False):
        """Choose the native storage interface before any graph capture."""
        self.core = module
        self.prepare = module.prepare5 if packed else module.prepare
        self.finish = module.finish5 if packed else module.finish

    def __call__(self, scores, values, ids, counts, tau, error):
        self.prepare(
            scores, values, ids, counts, tau, error, self.lengths, self.diag
        )
        for begin, end, view in fi_score_segments(scores):
            ix = slice(begin, end)
            if self.selected is None:
                selected = flashinfer.top_k_ragged_transform(
                    view, self.zero[: end - begin], self.lengths[ix], 2048,
                    deterministic=False, dsa_graph_safe=True,
                )
            else:
                selected = topk_into(view, self.zero[: end - begin], self.lengths[ix],
                                     self.selected[ix])
            self.finish(
                scores[ix],
                selected,
                values[ix],
                ids[ix],
                counts[ix],
                tau[ix],
                error[ix],
                self.lengths[ix],
                self.diag[ix],
            )
        return ids[:, :2048]
