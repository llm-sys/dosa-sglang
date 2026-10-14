"""Bind the shared certificate writer to a device-owned K extent."""
from .indexed import CertificateWriter


def bind_components(backend, rows):
    r = backend.rt
    # The original fixed-Q allocation capacity is unchanged. Changing Q creates
    # new schedule workspaces and a new graph; changing NK does neither.
    current = r.fused
    if (not isinstance(current, CertificateWriter) or current.live_nk is None
            or current.work.rows != rows):
        r.fused = CertificateWriter(
            backend.fp16_emitter, backend.live_components, rows, backend.max_k,
            backend.device, r.seed, owner=backend,
        )
    w = r.metadata_workspace
    w.live_nk = r.live_nk
    w.live_seed = r.seed
    w.live_workers = min(r.sms, len(w.ps))
    if w.live_workers <= 0:
        raise RuntimeError('No metadata worker capacity')


def install(backend, components):
    from .key_rho_runtime import validate_live_contract as validate_key_rho

    validate_key_rho(backend, components)
    from .direct_abs_runtime import validate_h18_contract
    validate_h18_contract(backend, components)
    from .query_slots import validate_query_capacity
    validate_query_capacity(backend.max_q, (components['h16'], components['h18'],
                                            components['audit']), minimum=8)
    if backend.hot_graph is not None or backend.last_done is not None:
        raise RuntimeError('Install liveNK before the first actual call/capture')
    if not backend.fixed_recovery_capacity or not backend.record5_direct_seed:
        raise RuntimeError('LiveNK requires current capacity recovery and direct record5')
    if backend.rt.record_output is None:
        raise RuntimeError('Install the original record5 adapter before liveNK')
    backend.live_components = components
    backend.live_nk_enabled = True
    r = backend.rt
    r.live_nk = r.recovery_logical_nk
    r.live_common, r.live_exact = components['common'], components['exact']
    bind_components(backend, backend.max_q)
    return backend
