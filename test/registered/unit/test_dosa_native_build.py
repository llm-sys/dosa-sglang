"""Local build identity must follow real dependencies, not a stale header list."""
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest


REPO = next(p for p in Path(__file__).resolve().parents if (p / "python/sglang").is_dir())
NATIVE = REPO / "python/sglang/srt/layers/attention/dsa/dosa_runtime/native"
spec = importlib.util.spec_from_file_location("dosa_native_build_test", NATIVE / "build.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


class BuildIdentityTests(unittest.TestCase):
    def test_transitive_header_and_flags_change_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, header = root / "main.cu", root / "inner.cuh"
            source.write_text('#include "middle.cuh"\n')
            (root / "middle.cuh").write_text('#include "inner.cuh"\n')
            header.write_text("constexpr int rows = 16;\n")
            recipe = {"cuda_flags": ["--ftz=false", "--fmad=false"]}
            original = build.source_tag(source, recipe)
            header.write_text("constexpr int rows = 32;\n")
            changed = build.source_tag(source, recipe)
            self.assertNotEqual(original, changed)
            self.assertNotEqual(changed, build.source_tag(source, {"cuda_flags": []}))
            (root / "unrelated.cuh").write_text("ignored\n")
            self.assertEqual(changed, build.source_tag(source, recipe))

    def test_relocation_and_include_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "first"
            root.mkdir()
            (root / "main.cu").write_text('#include "a.cuh"\n')
            (root / "a.cuh").write_text('#include "b.cuh"\n')
            (root / "b.cuh").write_text('#include "a.cuh"\n')
            self.assertEqual(len(build.source_closure(root / "main.cu")), 3)
            moved = root.with_name("second")
            shutil.copytree(root, moved)
            self.assertEqual(build.source_tag(root / "main.cu", {}),
                             build.source_tag(moved / "main.cu", {}))

    def test_missing_local_header_is_not_a_cache_hit(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "main.cu"
            source.write_text('#include "missing.cuh"\n')
            with self.assertRaises(FileNotFoundError):
                build.source_tag(source, {})

    def test_real_static_and_live_transitive_dependencies(self):
        for relative in ("few/h16.cu", "few/audit.cu", "f2_live_h22/audit.cu"):
            with self.subTest(source=relative):
                files = build.source_closure(NATIVE / relative)
                if relative != "few/audit.cu":
                    self.assertIn(NATIVE / "common/query_capacity.cuh", set(files))
                self.assertIn(NATIVE / "common/emit_types.inl", set(files))
        caller = build.source_closure(NATIVE / "caller/native/caller.cu")
        self.assertIn(NATIVE / "caller/native/gather_rho.cuh", set(caller))
        for relative in ("few/sample.cu", "live_nk/record5_live.cu",
                         "key_rho/key_rms.cu", "caller/native/caller.cu"):
            with self.subTest(rms_source=relative):
                self.assertIn(NATIVE / "common/rms_integer.inl",
                              build.source_closure(NATIVE / relative))


if __name__ == "__main__":
    unittest.main()
