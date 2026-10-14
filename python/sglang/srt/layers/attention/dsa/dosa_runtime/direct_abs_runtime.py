"""Bind the native ABS planner once, before any Few call or graph capture."""
from pathlib import Path

PLANNER_CONTRACT = 'direct-abs-paid14-16-v1'
FIELD_CONTRACT = 'abs-shift-dd2-v1'
H18_FIELD_CONTRACT = 'abs-shift-dd2-h18-v1'
_DIRECT = {}


def contract(module):
    getter = getattr(module, 'field_contract', None)
    return getter() if callable(getter) else None


def load_direct_planner(score_heads=32):
    global _DIRECT
    from .native.build import load_native
    from .native_binding import load_pinned_or_build

    def build():
        return load_native("direct_abs", score_heads=score_heads)

    binding = load_pinned_or_build(
        Path(__file__).with_name("DIRECT_ABS_PLANNER.json" if score_heads == 32
                                 else "DIRECT_ABS_PLANNER_H64.json"),
        _DIRECT.get(score_heads), build
    )
    module = binding[2]
    if module.field_contract() != PLANNER_CONTRACT or not callable(module.plan128):
        raise RuntimeError('Direct planner contract mismatch')
    _DIRECT[score_heads] = binding
    return module


class BudgetPlanner:
    """Native direct fields for paid14/16; the original planner for other budgets."""

    def __init__(self, direct, original):
        self.direct = direct
        self.original = original
        self.__file__ = str(Path(__file__).resolve())

    def field_contract(self):
        return PLANNER_CONTRACT

    def plan128(self, paid, q, w, capture, outputs, budget):
        module = self.direct if paid in (14, 16) else self.original
        return module.plan128(paid, q, w, capture, outputs, budget)

    def invoice(self):
        return dict(
            planner_contract=PLANNER_CONTRACT, field_contract=FIELD_CONTRACT,
            kind='python_fixed_budget_dispatch', source=self.__file__,
            paid14_and16_native=str(self.direct.__file__),
            other_budgets_native=str(self.original.__file__),
            launches_per_plan_epoch=0, extra_storage_bytes=0,
            mode='direct_native_fields',
        )


def validate_h18_contract(backend, components=None):
    ordinary = contract(backend.modules.get('h16')) == FIELD_CONTRACT
    h18 = contract(backend.modules.get('h18'))
    if components is not None:
        live = contract(components.get('h16')) == FIELD_CONTRACT
        if ordinary != live:
            raise RuntimeError('Ordinary and live H16 field contracts must agree')
        live_h18 = contract(components.get('h18'))
    else:
        live_h18 = None
    if not ordinary:
        if h18 == H18_FIELD_CONTRACT or live_h18 == H18_FIELD_CONTRACT:
            raise RuntimeError('ABS16 H18 requires matching ABS H16 and dual planner')
        return
    if h18 != H18_FIELD_CONTRACT:
        raise RuntimeError('Dual direct planner requires the ABS16 H18 consumer')
    if components is not None and live_h18 != H18_FIELD_CONTRACT:
        raise RuntimeError('Ordinary and live ABS16 H18 field contracts must agree')


def install_if_required(backend, planner=None):
    validate_h18_contract(backend)
    if contract(backend.modules.get('h16')) != FIELD_CONTRACT:
        return backend
    if backend.hot_graph is not None or backend.last_done is not None:
        raise RuntimeError('Install the direct planner before first call/capture')
    planner = load_direct_planner(backend.score_heads) if planner is None else planner
    if contract(planner) != PLANNER_CONTRACT or not callable(getattr(planner, 'plan128', None)):
        raise RuntimeError('Planner does not emit the qualified direct ABS fields')
    runtime = backend.rt
    if runtime.abs_planner is not None:
        current = runtime.abs_planner
        if (runtime.planner128 is not current or backend.modules['planner'] is not current
                or current.direct is not planner):
            raise RuntimeError('Direct planner owner or native binding changed')
        return backend
    original = runtime.planner128
    if (original is not backend.modules['planner'] or contract(original) == PLANNER_CONTRACT
            or not callable(getattr(original, 'plan128', None))):
        raise RuntimeError('Require the original owner planner before direct installation')
    dispatcher = BudgetPlanner(planner, original)
    backend.modules = dict(backend.modules, planner=dispatcher)
    runtime.planner128 = runtime.abs_planner = dispatcher
    return backend
