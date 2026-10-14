"""Shared native build contract; no backend construction or GPU execution."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re


_QUOTED_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"\n]+)"', re.MULTILINE)


def source_closure(source):
    """Follow local includes, including both sides of compile-time branches.

    DoSA's quoted includes are relative to their including file. Third-party
    headers use angle brackets and remain the installed toolchain's dependency.
    Keeping this closure explicit avoids manually maintained header inventories.
    """
    pending, files = [Path(source).resolve()], {}
    while pending:
        path = pending.pop()
        if path in files:
            continue
        data = path.read_bytes()
        files[path] = data
        pending.extend((path.parent / name).resolve()
                       for name in _QUOTED_INCLUDE.findall(data.decode()))
    return files


def source_tag(source, recipe):
    root = Path(source).resolve().parent
    digest = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode())
    for path, data in sorted(source_closure(source).items()):
        # Relative identity survives moving the same source tree or making a
        # worktree. Contents and every compiler option remain part of the key.
        digest.update(os.path.relpath(path, root).encode() + b'\0')
        digest.update(data + b'\0')
    return digest.hexdigest()[:12]


def package_headers(package):
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"DoSA native build requires installed {package} headers")
    root = Path(spec.origin).resolve().parent
    include = root / ("include" if package == "deep_gemm" else "data/cutlass/include")
    required = ("deep_gemm", "cute", "cutlass") if package == "deep_gemm" else ("cutlass",)
    if not all((include / name).is_dir() for name in required):
        raise RuntimeError(f"Incomplete {package} include tree: {include}")
    return include


def load_extension(root, name, source, *, cuda_flags, cflags, build_directory=None,
                   headers=None, local_include=False, link_cuda=False,
                   with_cuda=None, operations=(), verbose=False):
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("DoSA native build requires a CUDA sm_100a toolkit")
    root = Path(root).resolve()
    recipe = dict(cuda_flags=cuda_flags, cflags=cflags, headers=headers,
                  local_include=local_include, link_cuda=link_cuda,
                  with_cuda=with_cuda)
    name = f"{name}_{source_tag(root / source, recipe)}_sm100"
    include = [str(root)] if local_include else []
    if headers is not None:
        include.append(str(package_headers(headers)))
    options = dict(name=name, sources=[str(root / source)],
                   extra_cflags=cflags, extra_cuda_cflags=cuda_flags,
                   build_directory=None, verbose=verbose)
    if include:
        options["extra_include_paths"] = include
    if link_cuda:
        options["extra_ldflags"] = [f"-L{CUDA_HOME}/lib64/stubs", "-lcuda"]
    if with_cuda is not None:
        options["with_cuda"] = with_cuda
    if build_directory is not None:
        directory = Path(build_directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        options["build_directory"] = str(directory)
    module = load(**options)
    missing = [op for op in operations if not callable(getattr(module, op, None))]
    if missing:
        raise RuntimeError(f"DoSA native {name} is missing operations: {missing}")
    return module


ARCH = "-gencode=arch=compute_100a,code=sm_100a"
SUPPORT = ("-DDOSA_SUPPORT_COLS=4", "-DDOSA_SUPPORT_MODE=1")


FEW_MODULES = {
    "planner": ("c++20", False, ("plan128",)),
    "audit": (
        "c++17",
        True,
        (
            "emit_trusted_candidates",
            "prepare_fp16_h22",
            "emit_fp16_candidates",
            "packed_shared_bytes",
        ),
    ),
    "h16": (
        "c++20",
        True,
        ("emit_h16_indexed_trusted", "emit_h16_indexed_trusted_grid"),
    ),
    "h18": ("c++20", True, ("emit_h18_indexed_trusted",)),
    "common": ("c++17", False, ("compact_candidates",)),
    "sample": ("c++20", False, ("seed_score", "rms")),
    "exact": ("c++20", False, ("exact_paged_solo", "exact_paged_common")),
    "seed": (
        "c++20",
        False,
        ("prepare", "finish", "prepare5", "finish5", "validate_seed5"),
    ),
    "publisher": ("c++20", False, ("normal", "recovered")),
    "leaf": ("c++20", False, ("replay_and_leaf", "replay_and_leaf_capacity")),
    "hierarchy": ("c++20", False, ("gather", "final_meta", "publish")),
}
LIVE_MODULES = {
    "h16": ("../few/h16.cu", "c++20", ("emit_h16_indexed_trusted_live_k",)),
    "h18": ("../f2_live_h18/h18.cu", "c++20", ("emit_h18_indexed_trusted_live_k",)),
    "audit": (
        "../f2_live_h22/audit.cu",
        "c++17",
        (
            "prepare_fp16_h22_live_k",
            "emit_trusted_candidates_live_k",
            "emit_fp16_candidates_live_k",
        ),
    ),
    "common": ("../live_common/common.cu", "c++17", ("compact_candidates_live_nk",)),
    "exact": (
        "record5_live.cu",
        "c++20",
        (
            "exact_paged_solo_live_nk",
            "exact_paged_common_live_nk",
            "validate_seed5_live_nk",
            "rms_live_nk",
        ),
    ),
}



def load_native(group, key=None, build_directory=None, verbose=False, *, source_root=None,
                score_heads=32):
    """Resolve one packaged recipe directly into the shared extension transaction."""
    if score_heads not in (32, 64):
        raise ValueError("DoSA score_heads must be32 or64")
    if score_heads == 64 and group == "primary":
        raise NotImplementedError("Full primary remains qualified for32 score heads")
    if group == "primary":
        from sglang.kernels.ops.attention.dosa_primary import load_dosa_primary

        return load_dosa_primary(build_directory, verbose=verbose)
    root = Path(source_root) if source_root is not None else Path(__file__).parent
    directory = Path(build_directory) if build_directory is not None else None
    options = {}
    if group in ("few", "live_nk", "record4"):
        if group == "few":
            standard, ftz, operations = FEW_MODULES[key]
            source = key + ".cu"
        elif group == "live_nk":
            source, standard, operations = LIVE_MODULES[key]
            ftz = True
        else:
            source = {"exact5": "../record5/exact.cu", "integer": "integer.cu"}[key]
            standard = "c++17" if key == "integer" else "c++20"
            ftz, operations = False, ()
        cuda = ["-O3", "-lineinfo", "--expt-relaxed-constexpr", ARCH, f"-std={standard}"]
        if ftz:
            cuda.append("--ftz=false")
        if key in ("h16", "h18"):
            cuda += ["--fmad=false", "--expt-extended-lambda"]
        if key == "planner":
            cuda.append("-DDOSA_NEGATIVE_FILL_MINORITY=1")
        if key in ("exact", "exact5", "seed", "publisher", "leaf", "hierarchy"):
            cuda.extend(SUPPORT)
        cflags = [f"-std={standard}"]
        name = f"dosa_{group}_{key}"
        if directory is not None:
            directory /= key
        options = dict(headers="deep_gemm", local_include=True, link_cuda=True,
                       operations=operations)
        if group != "record4":
            options["with_cuda"] = True
    else:
        standard = "c++17" if group == "caller" else "c++20"
        cflags = ["-O3", f"-std={standard}"]
        if group == "caller":
            name, source = "sglang_dosa_caller", "native/caller.cu"
            cuda = ["-O3", "-lineinfo", "-std=c++17", ARCH]
        elif group == "post":
            name, source = "dosa_post", "post.cu"
            cuda = ["-O3", "-lineinfo", "-std=c++20", "--expt-relaxed-constexpr", ARCH]
            options = dict(headers="deep_gemm", local_include=True, link_cuda=True)
        elif group == "direct_abs":
            name, source = "dosa_direct_abs_planner", "planner.cu"
            cuda = ["-O3", "-std=c++20", "--ftz=false", "--fmad=false",
                    "--expt-relaxed-constexpr", "-DDOSA_NEGATIVE_FILL_MINORITY=1",
                    ARCH, "-lineinfo"]
            options = dict(headers="flashinfer", with_cuda=True)
        elif group == "key_rho":
            name, source = "dosa_key_rho", "key_rms.cu"
            cuda = ["-O3", "-lineinfo", ARCH, "-std=c++20",
                    "--expt-relaxed-constexpr", "--ftz=false"]
            options = dict(link_cuda=True, with_cuda=True)
        else:
            raise KeyError(group)
    if score_heads != 32:
        name += f"_h{score_heads}"
        cuda.append(f"-DDOSA_SCORE_HEADS={score_heads}")
        if directory is not None:
            directory /= f"h{score_heads}"
    module = load_extension(root / group, name, source, cuda_flags=cuda, cflags=cflags,
                            build_directory=directory, verbose=verbose, **options)
    if group == "direct_abs" and module.field_contract() != "direct-abs-paid14-16-v1":
        raise RuntimeError("Direct planner field contract mismatch")
    return module
