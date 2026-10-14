"""Source JIT for the conservative SM100 FP8 prefill primary."""
import importlib.util
from pathlib import Path


def load_dosa_primary(build_directory: str | Path, *, verbose: bool = False):
    """Load candidate-generation primitives; final H32 refinement is separate."""
    from sglang.kernels.jit.utils import KERNEL_PATH

    path = KERNEL_PATH / "csrc/attention/dosa_primary/build.py"
    spec = importlib.util.spec_from_file_location("sglang_dosa_primary_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_primary(build_directory, verbose=verbose)
