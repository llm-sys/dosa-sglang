"""Direct ABS contract and owner isolation; no Torch, model or CUDA execution."""
import ast
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[3] / "python/sglang/srt/layers/attention/dsa/dosa_runtime"


class AbsShiftOwnerTest(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("direct_abs_contract_test", ROOT / "direct_abs_runtime.py")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.events = []
        tree = ast.parse((ROOT / "few_core.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FusedRuntime")
        call = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__call__")
        args = ast.arguments(posonlyargs=[], args=[ast.arg(arg="self")], kwonlyargs=[],
                             kw_defaults=[], defaults=[])
        guard = ast.FunctionDef(name="guard", args=args, body=[call.body[0]], decorator_list=[])
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[])),
                     "actual_few_abs_guard", "exec"), namespace)
        self.guard = namespace["guard"]

    def native(self, name, tag=None):
        def plan(*args):
            self.events.append((name, args))
        return NS(__file__="/native/" + name, field_contract=lambda: tag, plan128=plan)

    def owner(self, name, abs_fields=True):
        original = self.native(name + "-original")
        return NS(hot_graph=None, last_done=None,
                  modules={"planner": original,
                           "h16": self.native("h16", self.module.FIELD_CONTRACT if abs_fields else None),
                           "h18": self.native("h18", self.module.H18_FIELD_CONTRACT if abs_fields else None)},
                  rt=NS(planner128=original, abs_planner=None, _requires_abs_shift=abs_fields))

    def direct(self):
        return self.native("direct", self.module.PLANNER_CONTRACT)

    def test_native_planner_routes_and_preserves_exact_arguments(self):
        owner = self.owner("A")
        self.module.install_if_required(owner, self.direct())
        self.guard(owner.rt)
        operands = tuple(object() for _ in range(5))
        for budget in (14, 16, 20):
            owner.rt.planner128.plan128(budget, *operands)
        self.assertEqual(self.events, [("direct", (14, *operands)),
                                     ("direct", (16, *operands)),
                                     ("A-original", (20, *operands))])
        self.assertEqual(owner.rt.abs_planner.invoice()["launches_per_plan_epoch"], 0)

    def test_two_owners_and_reinstall_do_not_share_or_replace_dispatch(self):
        direct = self.direct()
        first, second = self.owner("A"), self.owner("B")
        for owner in (first, second):
            self.module.install_if_required(owner, direct)
        saved = first.rt.abs_planner
        self.module.install_if_required(first, direct)
        self.assertIs(first.rt.planner128, saved)
        self.assertIsNot(first.rt.abs_planner, second.rt.abs_planner)
        self.assertIs(first.modules["planner"], saved)
        self.assertIsNot(saved.original, second.rt.abs_planner.original)
        with self.assertRaisesRegex(RuntimeError, "binding changed"):
            self.module.install_if_required(first, self.direct())

    def test_uninstalled_or_rebound_abs_planner_fails_before_compute(self):
        owner = self.owner("A")
        with self.assertRaisesRegex(RuntimeError, "owner-local"):
            self.guard(owner.rt)
        self.module.install_if_required(owner, self.direct())
        self.guard(owner.rt)
        owner.rt.planner128 = self.native("unrelated")
        with self.assertRaisesRegex(RuntimeError, "owner-local"):
            self.guard(owner.rt)
        self.assertEqual(self.events, [])
        self.guard(self.owner("ordinary", abs_fields=False).rt)

    def test_mixed_consumers_and_invalid_direct_planner_are_rejected(self):
        owner = self.owner("A")
        validate = self.module.validate_h18_contract
        validate(owner, {"h16": owner.modules["h16"], "h18": owner.modules["h18"]})
        with self.assertRaisesRegex(RuntimeError, "contracts"):
            validate(owner, {"h16": self.native("old"), "h18": owner.modules["h18"]})
        with self.assertRaisesRegex(RuntimeError, "contracts"):
            validate(owner, {"h16": owner.modules["h16"], "h18": self.native("old")})
        with self.assertRaisesRegex(RuntimeError, "qualified direct ABS"):
            self.module.install_if_required(owner, self.native("wrong"))
        owner.modules["h18"] = self.native("old")
        with self.assertRaisesRegex(RuntimeError, "ABS16 H18"):
            self.module.install_if_required(owner, self.direct())


if __name__ == "__main__":
    unittest.main()
