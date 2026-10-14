"""Exact projected-FP8 full/few prefill backends with owned private graphs."""


def create_backend(mode, device, *, max_queries, max_keys, modules=None,
                   record5_direct=False, score_heads=32, id_pool_pages=None):
    """Allocate one serial slot. Reuse run_paged across layers and requests."""
    if score_heads not in (32, 64) or (mode == "full" and score_heads != 32):
        raise ValueError("Unsupported score head geometry")
    if mode == "full":
        if record5_direct:
            raise ValueError("Direct completed-H32 record storage belongs to Few")
        from .full import FullBackend

        return FullBackend(
            device, max_queries=max_queries, max_keys=max_keys, modules=modules
        )
    if mode == "few":
        from .few import FewBackend

        return FewBackend(
            device, max_queries=max_queries, max_keys=max_keys, modules=modules,
            record5_direct=record5_direct, score_heads=score_heads,
            id_pool_pages=id_pool_pages,
        )
    raise ValueError("DoSA mode must be 'full' or 'few'")


def create_live_backend(mode, device, **kwargs):
    """Explicit dynamic Full/Few assembly, installed before model reservation."""
    from .assembly import create_live_backend as create

    return create(mode, device, **kwargs)
