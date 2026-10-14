"""Reuse the original DG padded output without replacing its scorer."""

import deep_gemm
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class _SeedOutputMode(TorchDispatchMode):
    """A thread-local override for the one observed DG output factory."""

    def __init__(self, output):
        super().__init__()
        self.output = output
        self.matches = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.ops.aten.empty.memory_format:
            device = kwargs.get("device")
            dtype = kwargs.get("dtype")
            if device is not None and torch.device(device).type == "cuda" and dtype == torch.float32:
                expected = (
                    tuple(args[0]) == tuple(self.output.shape)
                    and torch.device(device) == self.output.device
                    and kwargs.get("layout", torch.strided) == torch.strided
                    and kwargs.get("memory_format") in (None, torch.contiguous_format)
                )
                if not expected or self.matches:
                    raise RuntimeError("The installed DG padded-output allocation contract changed")
                self.matches += 1
                return self.output
        return func(*args, **kwargs)


def seed_logits(q, k, scale, weights, starts, ends, workspace=None):
    """Run the same DG operation; optional storage owns its padded F32 result."""
    if workspace is None:
        return deep_gemm.fp8_fp4_mqa_logits(
            (q, None), (k, scale), weights, starts, ends, clean_logits=False
        )
    rows, keys = len(q), len(k)
    if (
        workspace.dtype != torch.float32
        or workspace.device != q.device
        or workspace.ndim != 2
        or workspace.shape[0] < rows
        or workspace.shape[1] != keys + 256
        or not workspace.is_contiguous()
    ):
        raise ValueError("DG seed workspace must be contiguous F32[maxQ,seed+256]")
    padded = workspace[:rows]
    mode = _SeedOutputMode(padded)
    with mode:
        result = deep_gemm.fp8_fp4_mqa_logits(
            (q, None), (k, scale), weights, starts, ends, clean_logits=False
        )
    if (
        mode.matches != 1
        or result.data_ptr() != padded.data_ptr()
        or tuple(result.shape) != (rows, keys)
        or tuple(result.stride()) != (keys + 256, 1)
        or result.dtype != torch.float32
    ):
        raise RuntimeError("DG did not return the supplied original-layout output")
    return result
