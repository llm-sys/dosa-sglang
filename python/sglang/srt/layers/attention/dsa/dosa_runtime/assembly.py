"""Construct a complete Full/Few slot once, before reservation or capture."""

import torch

from . import create_backend
from .capture_workspace import install_outer_capture_workspace
from .native_binding import FEW_BASE_KEYS, FEW_LIVE_ABI, load_modules, load_service_modules
from .paid_bootstrap import PaidBootstrap
from .query_slots import (
    MIN_FEW_PREFILL_ROWS, MIN_FEW_QUERY_CAPACITY,
    enable_query_tail_padding, enable_service_prefill_interval, validate_query_capacity,
    service_query_storage,
)


def _assemble(mode, device, max_queries, max_keys, modules, records, sample,
              min_rows, score_heads=32, id_pool_pages=None):
    # No installer can observe a half-assembled slot: this object is fresh and
    # remains unpublished until records, cold policy and capture ownership exist.
    if not callable(getattr(sample, "seed_score", None)):
        raise ValueError("Live bootstrap requires the original-H32 sample module")
    pool_options = {} if id_pool_pages is None else {"id_pool_pages": id_pool_pages}
    backend = create_backend(
        mode, device, max_queries=max_queries, max_keys=max_keys, modules=modules,
        record5_direct=records is not None and records.get("seed") is not None,
        score_heads=score_heads,
        **pool_options,
    )
    if records is not None:
        from .record5 import install_record5

        install_record5(backend, records["exact"], records["selector"], modules["post"],
                        seed=records.get("seed"))
    enable_service_prefill_interval(backend, min_rows=min_rows)
    with torch.cuda.device(backend.device):
        backend.bootstrap = PaidBootstrap(backend, sample)
    install_outer_capture_workspace(backend)
    return backend


def create_live_backend(mode, device, *, max_queries, max_keys, modules=None,
                        record5_modules=None, bootstrap_sample_module=None, score_heads=32):
    """Explicit legacy assembly; service sizing is owned by create_service_backend."""
    if mode not in ("full", "few") or not 196608 <= max_keys <= 1048576:
        raise ValueError("Live assembly requires Full/Few and supported K capacity")
    if mode == "full" and (max_queries != 16384 or record5_modules is not None):
        raise ValueError("Explicit Full uses Q16384 and its own surrogate records")
    native = load_modules(mode, score_heads=score_heads) if modules is None else modules
    if mode == "few":
        validate_query_capacity(max_queries, (native["h16"], native["h18"]))
    sample = bootstrap_sample_module
    if sample is None:
        sample = native.get("sample")
    if sample is None:
        sample = load_modules("sample", score_heads=score_heads)["sample"]
    records = None
    if mode == "few":
        supplied = load_modules("records", score_heads=score_heads) if record5_modules is None else record5_modules
        if "record5_exact" in supplied:
            records = dict(exact=supplied.get("record5_exact"),
                           selector=supplied.get("integer"), seed=supplied.get("seed"))
        else:
            records = {key: supplied.get(key) for key in ("exact", "selector", "seed")}
        if records["exact"] is None or records["selector"] is None:
            raise ValueError("Few requires matching completed-H32 exact and selector modules")
    return _assemble(
        mode, device, max_queries, max_keys, native, records, sample,
        MIN_FEW_PREFILL_ROWS if mode == "few" else MIN_FEW_QUERY_CAPACITY, score_heads,
    )


def create_service_backend(mode, device, *, max_queries, max_keys, score_heads=32):
    """Build, assemble and reserve the original Full/Few policies in one direction."""
    modules = load_service_modules(mode, score_heads=score_heads)
    records = None
    scratch_queries = max_queries
    id_pool_pages = None
    if mode == "few":
        if not 196608 <= max_keys <= 1048576:
            raise ValueError("Live assembly requires supported K capacity")
        validate_query_capacity(max_queries)
        scratch_queries, id_pool_pages = service_query_storage(
            max_queries, max_keys, score_heads)
        validate_query_capacity(scratch_queries, (modules[k] for k in
            ("h16", "h18", "live_h16", "live_h18", "live_audit") if k in modules),
            minimum=8)
        native = {key: modules[key] for key in FEW_BASE_KEYS}
        records = {key: modules["record5_" + key] for key in ("exact", "selector", "seed")}
    else:
        native = {key: modules[key] for key in ("primary", "caller", "post")}
    backend = _assemble(mode, device, scratch_queries, max_keys, native, records,
                        modules["sample"], MIN_FEW_PREFILL_ROWS, score_heads,
                        id_pool_pages)
    if mode == "few":
        live = {key.removeprefix("live_"): modules[key]
                for key in FEW_LIVE_ABI if key in modules}
        if live:
            from .live_nk_runtime import install

            install(backend, live)
        backend.stock_carry.max_q = max_queries
    enable_query_tail_padding(backend, min_rows=MIN_FEW_PREFILL_ROWS)
    if max_queries > scratch_queries:
        from .slabbed_few import install_query_slabs

        install_query_slabs(backend, max_queries)
    return backend
