"""Owner-local current-scale RMS dispatch; seed/H32 provider objects stay intact."""
from pathlib import Path

CONTRACT='rms128-ks-positive-le2p64-v1'
_AUXILIARY=None

def required_contract(module):
    getter=getattr(module,'required_key_rho_contract',None)
    return getter() if callable(getter) else None

def load_auxiliary():
    global _AUXILIARY
    from .native.build import load_native
    from .native_binding import load_pinned_or_build

    def build():
        return load_native("key_rho")

    _AUXILIARY = load_pinned_or_build(
        Path(__file__).with_name("KEY_RHO_NATIVE.json"), _AUXILIARY, build
    )
    return _AUXILIARY[2]

class CurrentScaleRMS:
    def __init__(self,owner,auxiliary):
        self.owner=owner;self.auxiliary=auxiliary

    def __call__(self,k,scale,rho,live_nk,sms,seed):
        # Scale is supplied by this FusedRuntime invocation, never closed over.
        if live_nk is None:
            return self.auxiliary.rms_qualified(k,scale,rho)
        return self.auxiliary.rms_live_nk_qualified(k,scale,rho,live_nk,sms,seed)

def install_if_required(backend,auxiliary=None):
    required=required_contract(backend.modules.get('h16'))
    if required is None:
        return backend
    if required!=CONTRACT:
        raise RuntimeError('Unsupported H16 key-rho contract')
    if backend.hot_graph is not None or backend.last_done is not None:
        raise RuntimeError('Install current-key producers before the first call/capture')
    r=backend.rt
    if r._required_key_rho_contract!=CONTRACT:
        raise RuntimeError('The runtime lacks current-scale RMS dispatch')
    if r.core is not backend.modules['sample'] or backend.gather_adapter.module is not backend.modules['caller']:
        raise RuntimeError('Original sample/gather producer identity changed')
    auxiliary=load_auxiliary() if auxiliary is None else auxiliary
    for module in (backend.modules['caller'],auxiliary):
        if getattr(module,'key_rho_contract',lambda:None)()!=CONTRACT:
            raise RuntimeError('Unqualified current-key rho producer')
    if not all(callable(getattr(auxiliary,name,None)) for name in ('rms_qualified','rms_live_nk_qualified')):
        raise RuntimeError('Incomplete current-scale RMS auxiliary')
    previous=getattr(backend,'_current_scale_rms',None)
    if previous is not None and previous.auxiliary is not auxiliary:
        raise RuntimeError('Cannot replace an owner auxiliary binding')
    if previous is None:
        backend._current_scale_rms=CurrentScaleRMS(backend,auxiliary)
    r._qualified_rms=backend._current_scale_rms
    r.key_rho_contract=CONTRACT
    r.rho_from_gather=False;r.rho_source_signature=None
    return backend

def validate_live_contract(backend,components):
    ordinary=required_contract(backend.modules.get('h16'))
    live=required_contract(components.get('h16'))
    if ordinary!=live:
        raise RuntimeError('Ordinary/live H16 key-rho contracts must agree')
    if live is not None and (live!=CONTRACT or backend.rt.key_rho_contract!=CONTRACT):
        raise RuntimeError('Install qualified current-key producers before live H16')
