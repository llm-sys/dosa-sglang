"""Native dependencies, ABI and immutable bindings for all DoSA factories."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading

from .query_slots import (
    FEW_CANDIDATE_BYTES_PER_QUERY,
    FEW_RECORD_CAPACITY,
    MAX_FEW_QUERY_CAPACITY,
    MIN_FEW_PREFILL_ROWS,
)
from .native.build import FEW_MODULES, LIVE_MODULES

_BUILD_LOCK = threading.Lock()
_MODULES = {}
_SERVICE_BINDING = None

FEW_BASE_KEYS = ("caller", "post", *FEW_MODULES)
FEW_RECORD_KEYS = ("record5_exact", "record5_selector", "record5_seed")
FEW_SERVICE_ABI = {
    "caller": (
        "plan_and_permuted_paged_gather_out",
        "plan_and_permuted_paged_gather_norm_out",
        "plan_and_permuted_paged_gather_rho_out",
        "carry_votes_topk_reset_", "carry_votes_topk_noreset_",
        "map_topk_vote_stats_litetopk_", "map_vote_page_", "stock_paged_to_votes_",
    ),
    "post": (
        "collect_band", "complete_h32_fused_large", "publish_replay",
        "replay", "attach_if_any",
    ),
    **{name: operations for name, (_, _, operations) in FEW_MODULES.items()},
    # Preserve the legacy service ABI when a freshly built library exports
    # additional entry points. Capacity upgrades remain explicitly declared.
    "audit": ("emit_trusted_candidates",),
    "seed": ("prepare", "finish"),
    "leaf": ("replay_and_leaf",),
    "record5_exact": ("pack_seed", "exact_paged_solo", "exact_paged_common"),
    "record5_selector": ("select_dense5",),
    "record5_seed": ("prepare5", "finish5", "validate_seed5"),
}

FEW_SERVICE_LIMITS = dict(
    # Supported call domain and service storage defaults.
    query_rows=MAX_FEW_QUERY_CAPACITY,
    min_query_rows=MIN_FEW_PREFILL_ROWS,
    # Internal scoring requires the original 12K seed to fit the visible prefix.
    min_keys=12288,
    max_keys=1048576,
    record_capacity=FEW_RECORD_CAPACITY,
    hint_keys=12288,
    candidate_bytes_per_query=FEW_CANDIDATE_BYTES_PER_QUERY,
)

# One service policy for every query size; internal causal slabs retain their
# smaller mathematical extent instead of inheriting this dispatch boundary.
FEW_PREFILL_MIN_KEYS = 131072


# All live roles are required together; legacy static manifests stay valid.
FEW_LIVE_ABI = {
    "live_" + name: operations for name, (_, _, operations) in LIVE_MODULES.items()
}


FULL_SERVICE_ABI = {
    "primary": ("seed_prep_checked", "surrogate_scan", "h2048_coarse_band_paged_"),
    "caller": ("plan_and_permuted_paged_gather_norm_out", "map_vote_page_",
               "carry_votes_topk_reset_", "carry_votes_topk_noreset_",
               "stock_paged_to_votes_"),
    "post": FEW_SERVICE_ABI["post"],
    "sample": ("seed_score",),
}
CAPACITY_METHODS = {
    "post": ("replay_capacity",),
    "leaf": ("replay_and_leaf_capacity",),
}


def required_service_methods(mode, *, live=False):
    """The legacy ABI remains valid for existing immutable native manifests."""
    if mode == "few":
        return {**FEW_SERVICE_ABI, **FEW_LIVE_ABI} if live else FEW_SERVICE_ABI
    if mode == "full":
        return FULL_SERVICE_ABI
    raise ValueError("Service native mode must be full or few")


def declared_service_methods(mode, key, methods):
    """Accept only the exact old ABI or its named capacity extension."""
    required = required_service_methods(mode, live=(mode == "few"))
    base = required[key]
    extended = base + CAPACITY_METHODS.get(key, ())
    if not isinstance(methods, (list, tuple)) or tuple(methods) not in (base, extended):
        raise ValueError(f"Manifest must declare the qualified ABI for {key}")
    return tuple(methods)


def native_service_methods(mode, key, module):
    """Declare and validate exposed capacity methods without requiring an upgrade."""
    required = required_service_methods(mode, live=(mode == "few"))
    base = required[key]
    extension = CAPACITY_METHODS.get(key, ())
    if any(hasattr(module, method) for method in extension):
        if not all(callable(getattr(module, method, None)) for method in extension):
            raise RuntimeError(f"Invalid fixed-capacity native ABI in {key}")
        return base + extension
    return base


def import_native(path, name=None):
    path = Path(path).resolve()
    name = path.stem if name is None else name
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path:
            raise RuntimeError(f"Different native module already loaded: {name}")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Publish only a successfully imported module, never a partial binding.
    sys.modules[name] = module
    return module

def load_pinned_or_build(pin_path, cached, build):
    """Return (resolved path, digest or None, module), preserving one binding."""
    pin_path = Path(pin_path)
    if not pin_path.exists():
        if cached is not None:
            return cached
        module = build()
        return str(Path(module.__file__).resolve()), None, module
    pin = json.loads(pin_path.read_text())
    path = Path(pin["binary"]).resolve()
    identity = (str(path), pin["sha256"])
    if cached is not None:
        if cached[:2] != identity:
            raise RuntimeError("Cannot change a native binding in a live process")
        return cached
    if hashlib.sha256(path.read_bytes()).hexdigest() != pin["sha256"]:
        raise RuntimeError(f"Qualified native file changed: {path}")
    return *identity, import_native(path, pin.get("module"))

def native_build_directory(directory=None):
    if directory is not None:
        return Path(directory)
    from torch.utils.cpp_extension import get_default_build_root

    return (
        Path(os.environ.get("TORCH_EXTENSIONS_DIR", get_default_build_root()))
        / "sglang_dosa"
    )

def load_modules(mode, *, source_root=None, build_directory=None, verbose=False,
                 service=False, score_heads=32):
    """Resolve one dependency plan; cache the same libraries for every factory."""
    from .native.build import FEW_MODULES, load_native

    if mode not in ("full", "few", "sample", "records") or (
        service and mode not in ("full", "few")
    ):
        raise ValueError("Invalid DoSA native dependency plan")
    root = Path(source_root) if source_root is not None else Path(__file__).parent / "native"
    build = native_build_directory(build_directory)
    key = (str(root.resolve()), str(build.resolve()), mode, service, score_heads)
    with _BUILD_LOCK:
        if key in _MODULES:
            return _MODULES[key]
        # Each entry retains its original recipe, build path and compiler flags.
        plans = {
            "caller": ("caller", None, "caller"),
            "post": ("post", None, "post"),
            "primary": ("primary", None, "primary"),
            **{name: ("few", name, "few") for name in FEW_MODULES},
            "record5_exact": ("record4", "exact5", "record5"),
            "record5_selector": ("record4", "integer", "record4"),
            "record5_seed": ("few", "seed", "few"),
        }
        roles = {
            "full": ("caller", "post", "primary"),
            "few": FEW_BASE_KEYS,
            "sample": ("sample",),
            "records": FEW_RECORD_KEYS,
        }[mode]
        if service:
            roles += (("sample",) if mode == "full" else FEW_RECORD_KEYS)
        modules = {}
        for role in roles:
            group, name, directory = plans[role]
            modules[role] = load_native(
                group, name, build / directory, verbose, source_root=root,
                score_heads=score_heads if role in ("sample", "planner", "audit", "h16", "h18", "common",
                                                   "exact", "leaf", "post", "record5_exact") else 32
            )
        if service and mode == "few":
            for live_key, methods in FEW_LIVE_ABI.items():
                role = live_key.removeprefix("live_")
                existing = modules[role]
                modules[live_key] = (existing if all(callable(getattr(existing, m, None))
                                               for m in methods) else
                                load_native("live_nk", role, build / "live_nk", verbose,
                                            source_root=root,
                                            score_heads=score_heads))
        if mode == "records":
            modules = {k.removeprefix("record5_"): v for k, v in modules.items()}
        _MODULES[key] = modules
        return modules


def load_service_modules(mode="full", *, score_heads=32):
    """Resolve once, before KV reservation, and prohibit in-process rebinding."""
    global _SERVICE_BINDING
    required = required_service_methods(mode)
    value = os.environ.get("SGLANG_DOSA_NATIVE_MANIFEST")
    path = Path(value).resolve() if value else None
    if _SERVICE_BINDING is not None:
        if _SERVICE_BINDING[:2] != (path, mode) or _SERVICE_BINDING[3] != score_heads:
            raise RuntimeError("Cannot change native bindings inside a live process")
        return _SERVICE_BINDING[2]
    if path is None:
        modules = load_modules(mode, service=True, score_heads=score_heads)
    else:
        manifest = json.loads(path.read_text())
        if manifest.get("score_heads", 32) != score_heads:
            raise ValueError("Native manifest score head geometry mismatch")
        live = manifest.get("live_nk_graph", False)
        if live:
            if mode != "few" or live is not True:
                raise ValueError("Explicit live-NK manifest requires Few and true boolean")
            required = required_service_methods(mode, live=True)
        entries = manifest.get("modules", {})
        if (manifest.get("schema") != 1 or set(entries) != set(required)
                or manifest.get("mode", "full") != mode):
            raise ValueError(f"Require exactly the qualified {mode} native manifest")
        # Validate the entire immutable manifest before importing any extension.
        declared = {}
        for role, entry in entries.items():
            declared[role] = declared_service_methods(mode, role, entry.get("required_methods"))
            source = Path(entry["path"])
            if not source.is_absolute() or source.stem != entry["module"]:
                raise ValueError("Native module identity must match its explicit file")
            if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Qualified native file changed: {source}")
        modules = {}
        for role, entry in entries.items():
            module = import_native(entry["path"], entry["module"])
            if not all(callable(getattr(module, m, None)) for m in declared[role]):
                raise RuntimeError(f"Missing qualified native ABI in {role}")
            native_service_methods(mode, role, module)
            modules[role] = module
    _SERVICE_BINDING = (path, mode, modules, score_heads)
    return modules


def manifest_from_modules(modules, *, mode="full", score_heads=32):
    if score_heads not in (32, 64) or (mode == "full" and score_heads != 32):
        raise ValueError("Supported service geometry is H32 Full/Few or H64 Few")
    live = mode == "few" and bool(set(modules) & set(FEW_LIVE_ABI))
    required = required_service_methods(mode, live=live)
    if set(modules) != set(required):
        raise ValueError(f"Require exactly the {mode} service native module keys")
    entries = {}
    for key in required:
        module = modules[key]
        operations = native_service_methods(mode, key, module)
        path = Path(module.__file__).resolve(strict=True)
        if path.stem != module.__name__:
            raise ValueError(f"Native module name does not match its file: {key}")
        missing = [name for name in operations if not callable(getattr(module, name, None))]
        if missing:
            raise ValueError(f"Native module {key} is missing required methods: {missing}")
        data = path.read_bytes()
        entries[key] = {
            "path": str(path), "module": module.__name__,
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
            "required_methods": list(operations),
        }
    result = {"schema": 1, "mode": mode, "modules": entries}
    if score_heads != 32:
        result["score_heads"] = score_heads
    if live:
        result["live_nk_graph"] = True
    return result
