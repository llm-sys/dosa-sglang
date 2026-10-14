"""Native setup must preserve one identity and never cache a failed import."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "python/sglang").is_dir())
SOURCE = ROOT / "python/sglang/srt/layers/attention/dsa/dosa_runtime/native_binding.py"
PREFIX = "dosa_native_binding_test"
package = types.ModuleType(PREFIX)
package.__path__ = [str(SOURCE.parent)]
sys.modules[PREFIX] = package
SPEC = importlib.util.spec_from_file_location(PREFIX + ".native_binding", SOURCE)
binding = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(binding)


class NativeBindingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.source = self.root / "native_binding_fixture.py"
        self.source.write_text("VALUE = 7\n")
        self.pin = self.root / "pin.json"
        self.write_pin()

    def tearDown(self):
        sys.modules.pop(self.source.stem, None)
        self.directory.cleanup()

    def write_pin(self):
        self.pin.write_text(json.dumps({
            "binary": str(self.source),
            "sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
        }))

    def test_pin_reuses_identity_and_refuses_a_changed_binding(self):
        def forbidden_build():
            self.fail("A pinned module must not compile a replacement")

        first = binding.load_pinned_or_build(self.pin, None, forbidden_build)
        self.assertEqual(first[2].VALUE, 7)
        self.assertIs(binding.load_pinned_or_build(self.pin, first, forbidden_build), first)
        other = self.root / "other"
        other.mkdir()
        (other / self.source.name).write_text("VALUE = 8\n")
        with self.assertRaisesRegex(RuntimeError, "already loaded"):
            binding.import_native(other / self.source.name)
        self.source.write_text("VALUE = 9\n")
        self.write_pin()
        with self.assertRaisesRegex(RuntimeError, "Cannot change"):
            binding.load_pinned_or_build(self.pin, first, forbidden_build)

    def test_failed_import_is_not_published(self):
        self.source.write_text("raise RuntimeError('broken extension')\n")
        with self.assertRaisesRegex(RuntimeError, "broken extension"):
            binding.import_native(self.source)
        self.assertNotIn(self.source.stem, sys.modules)

    def test_default_build_is_once_and_digest_failure_does_not_import(self):
        self.pin.unlink()
        calls = []

        def build():
            calls.append(1)
            return binding.import_native(self.source)

        first = binding.load_pinned_or_build(self.pin, None, build)
        self.assertIs(binding.load_pinned_or_build(self.pin, first, build), first)
        self.assertEqual(calls, [1])
        self.write_pin()
        self.source.write_text("raise AssertionError('must not import')\n")
        sys.modules.pop(self.source.stem)
        with self.assertRaisesRegex(RuntimeError, "Qualified native file changed"):
            binding.load_pinned_or_build(self.pin, None, build)
        self.assertNotIn(self.source.stem, sys.modules)


class DependencySourceTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.reuse = set()

        def build(group, role=None, directory=None, verbose=False, *, source_root=None, score_heads=32):
            self.calls.append((group, role, directory, source_root, score_heads))
            result = types.SimpleNamespace()
            if group == "few" and role in self.reuse:
                for name in binding.FEW_LIVE_ABI["live_" + role]:
                    setattr(result, name, lambda: None)
            return result

        native = types.ModuleType(PREFIX + ".native")
        recipes = types.ModuleType(PREFIX + ".native.build")
        recipes.FEW_MODULES = {key: None for key in binding.FEW_BASE_KEYS[2:]}
        recipes.load_native = build
        self.modules = patch.dict(sys.modules, {
            native.__name__: native, recipes.__name__: recipes,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        binding._MODULES.clear()
        self.addCleanup(binding._MODULES.clear)
        self.root = ROOT / "alternate-native-source"
        self.build = ROOT / "alternate-build-cache"

    def load(self, source=None):
        return binding.load_modules("few", service=True,
                                    source_root=source or self.root,
                                    build_directory=self.build)

    def test_h64_binding_isolated_and_head_independent_roles_reused(self):
        first = self.load()
        self.assertTrue(all(call[-1] == 32 for call in self.calls))
        self.calls.clear()
        second = binding.load_modules("few", service=True, score_heads=64,
                                     source_root=self.root, build_directory=self.build)
        self.assertIsNot(second, first)
        expected = {("post", None), ("few", "planner"), ("few", "sample"),
                    ("few", "audit"), ("few", "common"),
                    ("few", "h16"), ("few", "h18"),
                    ("few", "exact"), ("few", "leaf"), ("record4", "exact5"),
                    ("live_nk", "audit"), ("live_nk", "common"), ("live_nk", "exact"),
                    ("live_nk", "h16"), ("live_nk", "h18")}
        actual = {(group, role) for group, role, _, _, heads in self.calls if heads == 64}
        self.assertEqual(actual, expected)

    def test_h64_reuses_matching_certificate_and_compactor_live_exports(self):
        self.reuse = {"audit", "common", "h16", "h18"}
        modules = binding.load_modules("few", service=True, score_heads=64,
                                       source_root=self.root, build_directory=self.build)
        for role in self.reuse:
            self.assertIs(modules[role], modules["live_" + role])
            calls = [(group, heads) for group, name, _, _, heads in self.calls if name == role]
            self.assertEqual(calls, [("few", 64)])

    def test_all_separate_live_builds_use_the_selected_source(self):
        modules = self.load()
        live = [call for call in self.calls if call[0] == "live_nk"]
        self.assertEqual({call[1] for call in live},
                         {key.removeprefix("live_") for key in binding.FEW_LIVE_ABI})
        self.assertTrue(all(call[3] == self.root for call in self.calls), self.calls)
        self.assertEqual(set(modules), set(binding.FEW_BASE_KEYS)
                         | set(binding.FEW_RECORD_KEYS) | set(binding.FEW_LIVE_ABI))

    def test_reused_live_exports_keep_identity_and_other_roles_keep_source(self):
        self.reuse = {"h16", "h18", "audit"}
        modules = self.load()
        for role in self.reuse:
            self.assertIs(modules[role], modules["live_" + role])
        self.assertEqual({call[1] for call in self.calls if call[0] == "live_nk"},
                         {"common", "exact"})
        self.assertTrue(all(call[3] == self.root for call in self.calls))

    def test_cache_isolated_by_source_and_successful_plan_reused(self):
        first = self.load()
        count = len(self.calls)
        self.assertIs(self.load(), first)
        self.assertEqual(len(self.calls), count)
        second_root = self.root.with_name("second-native-source")
        second = self.load(second_root)
        self.assertIsNot(second, first)
        self.assertTrue(all(call[3] == second_root for call in self.calls[count:]))


if __name__ == "__main__":
    unittest.main()
