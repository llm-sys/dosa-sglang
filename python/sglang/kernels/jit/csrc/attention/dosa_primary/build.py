"""Build the qualified full primary from packaged source and installed headers."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def load_primary(build_directory: str | Path, *, verbose: bool = False):
    """Return a PyTorch extension; this function launches no GPU work."""
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("A CUDA toolkit with the sm_100a target is required")
    spec = importlib.util.find_spec("deep_gemm")
    if spec is None or spec.origin is None:
        raise RuntimeError("Install DeepGEMM with its bundled include tree")
    include = Path(spec.origin).resolve().parent / "include"
    required = ("deep_gemm", "cute", "cutlass")
    if not all((include / name).is_dir() for name in required):
        raise RuntimeError(f"Incomplete DeepGEMM/CuTe/CUTLASS headers: {include}")

    source = Path(__file__).resolve().parent
    common = source.parents[4] / "srt/layers/attention/dsa/dosa_runtime/native/common"
    # torch's in-process JIT versioner otherwise sees only primary.cu. Include
    # the shared headers in its key so changing a common primitive cannot reuse
    # a previously loaded private implementation from the same build directory.
    digest = hashlib.sha256()
    for path in [
        source / name for name in
        ("primary.cu", "sm100_dsa_litetopk.cuh", "coarse_selector.cuh", "record5_codec.cuh", "dense_score_access.cuh")
    ] + [common / name for name in ("record5_codec.cuh", "record5_ops.cuh", "h32_record5_codec.cuh", "score_emit_loop.cuh", "score_emit_mma.cuh", "score_emit_body.inl")]:
        digest.update(path.read_bytes())
    source_key = digest.hexdigest()[:16]
    build = Path(build_directory).resolve()
    build.mkdir(parents=True, exist_ok=True)
    # Match the qualified native's CUDA arithmetic/code-generation flags.
    # No fast-math, FTZ, or arithmetic re-association option is added.
    return load(
        name="dosa_full_primary_sm100",
        sources=[str(source / "primary.cu")],
        extra_include_paths=[str(include)],
        extra_cflags=["-std=c++20"],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_100a,code=sm_100a",
            "-std=c++20",
            f"-DDOSA_SHARED_SOURCE_KEY=0x{source_key}ULL",
        ],
        extra_ldflags=[f"-L{CUDA_HOME}/lib64/stubs", "-lcuda"],
        build_directory=str(build),
        with_cuda=True,
        verbose=verbose,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    module = load_primary(args.build_dir, verbose=args.verbose)
    print(json.dumps({
        "module": module.__name__,
        "binary": module.__file__,
        "operations": ["seed_prep_checked", "surrogate_scan", "h2048_coarse_band_paged_"],
        "record_bytes": module.candidate_record_bytes(),
        "dynamic_shared_bytes": module.surrogate_dynamic_smem_bytes(),
    }), flush=True)
