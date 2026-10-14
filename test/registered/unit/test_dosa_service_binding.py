"""Exercise service admission and native-binding failures without CUDA imports."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

REPO = next(p for p in Path(__file__).resolve().parents if (p / "python/sglang").is_dir())
ROOT = REPO / "python/sglang/srt/layers/attention/dsa/dosa_runtime"
PREFIX = "isolated_few_service_test"


def load(name):
    package = sys.modules.setdefault(PREFIX, types.ModuleType(PREFIX))
    package.__path__ = [str(ROOT)]
    spec = importlib.util.spec_from_file_location(PREFIX + "." + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Tensor:
    def __init__(self, shape, dtype="float32", device="cuda", value=None):
        self.shape, self.dtype = shape, dtype
        self.device = types.SimpleNamespace(type=device)
        self.value = value

    def numel(self):
        return self.shape[0]

    def __getitem__(self, item):
        return self.value if isinstance(item, int) else self


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.torch = types.SimpleNamespace(
            float8_e4m3fn="fp8", float32="float32", device=lambda *a: "cuda:0",
            cuda=types.SimpleNamespace(is_available=lambda: True,
                get_device_capability=lambda *a: (10, 0), current_device=lambda: 0,
                is_current_stream_capturing=lambda: False),
        )
        flag = lambda value: types.SimpleNamespace(get=lambda: value)
        self.patch = patch.dict(sys.modules, {
            "torch": self.torch,
            "sglang.srt.environ": types.SimpleNamespace(envs=types.SimpleNamespace(
                SGLANG_DSA_FUSE_TOPK=flag(True),
                SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC=flag(False))),
            "sglang.srt.runtime_context": types.SimpleNamespace(
                get_server_args=lambda: types.SimpleNamespace(
                    chunked_prefill_size=16384,
                    max_prefill_tokens=16384)),
            "sglang.srt.arg_groups.overrides": types.SimpleNamespace(
                max_prefill_buffer_tokens=lambda args: args.chunked_prefill_size),
            "sglang.srt.model_executor.forward_context": types.SimpleNamespace(
                get_req_to_token_pool=lambda: "request-pool"),
            PREFIX + ".decode_witness": types.SimpleNamespace(
                DecodeWitness=lambda pool, device: self.witness),
        })
        self.patch.start()
        self.env = patch.dict(os.environ, {"SGLANG_DOSA_MODE": "few"})
        self.env.start()
        self.entry = load("sglang_entry")
        self.indexer = types.SimpleNamespace(n_heads=32, head_dim=128, index_topk=2048,
            num_init_tokens=0, num_local_tokens=0, dsa_enable_prefill_cp=False,
            logits_with_pp_recv=False, layer_id=6, _dosa_decode_enabled=False)
        self.witness = object()
        self.request_slots = object()
        self.calls = []
        self.backend = types.SimpleNamespace(
            max_q=16384, _dosa_query_padding_min_rows=1,
            _dosa_service_prefill_min_rows=1,
            reserve_for_model_load=lambda layer: self.calls.append(("reserve", layer)),
            run_paged=lambda *a, **kw: self.calls.append(("run", kw)) or True,
        )
        self.binding = types.SimpleNamespace(create_service_backend=lambda mode, *a, **kw:
            self.calls.append(("full" if mode == "full" else "factory", kw)) or self.backend)
        sys.modules[PREFIX + ".assembly"] = self.binding

    def tearDown(self):
        self.env.stop()
        self.patch.stop()

    def call(self, rows=16384, nk=196608, metadata_reads=True, topk_v2=True):
        def read_lengths():
            if not metadata_reads:
                raise AssertionError("unsupported Few Q touched visibility/cache")
            return Tensor((1,), device="cpu", value=nk)
        metadata = types.SimpleNamespace(topk_transform_method=types.SimpleNamespace(name="PAGED"),
            topk_backend=types.SimpleNamespace(name="SGL_KERNEL", should_use_topk_v2=lambda: topk_v2), force_unfused_topk=False,
            get_indexer_seq_len_cpu=read_lengths,
            get_page_table_1=lambda: "table1")
        self.indexer._get_index_k_read_buffer = lambda *a: "cache"
        batch = types.SimpleNamespace(extend_seq_lens_cpu=[rows], rids=["r1"],
                                      req_pool_indices=self.request_slots)
        q, w = Tensor((rows, self.indexer.n_heads, 128), "fp8"), Tensor((rows, self.indexer.n_heads))
        # Device string only identifies a prepared slot; no device access occurs.
        self.entry._SLOTS[(self.entry._mode(), str(q.device), self.indexer.n_heads)] = self.backend
        return self.entry.try_paged_prefill(self.indexer, batch, 6, q, w, metadata,
            Tensor((rows,)), Tensor((rows,)), Tensor((1, nk)), "result", "pool")

    def test_h64_reserves_its_own_geometry_and_dispatches_few(self):
        self.indexer.n_heads = 64
        self.assertTrue(self.entry.prepare_indexer_backend(self.indexer))
        self.assertEqual(self.calls[0][1]["score_heads"], 64)
        self.assertIs(self.entry._SLOTS[("few", "cuda:0", 64)], self.backend)
        self.assertTrue(self.call())
        with patch.dict(os.environ, {"SGLANG_DOSA_MODE": "full"}):
            self.assertFalse(self.entry.prepare_indexer_backend(self.indexer))

    def test_decode_opt_in_reuses_the_same_h64_prefill_backend(self):
        self.indexer.n_heads = 64
        self.assertTrue(self.entry.prepare_indexer_backend(self.indexer))
        self.assertFalse(self.indexer._dosa_decode_enabled)
        original = self.entry._SLOTS[("few", "cuda:0", 64)]
        with patch.dict(os.environ, {"SGLANG_DOSA_MODE": "few-decode"}):
            self.assertEqual(self.entry._mode(), "few")
            self.assertTrue(self.entry.prepare_indexer_backend(self.indexer))
            self.assertTrue(self.indexer._dosa_decode_enabled)
            self.assertIs(self.entry._SLOTS[("few", "cuda:0", 64)], original)
            self.assertTrue(self.call())
            self.assertEqual(self.calls[-1][1]['decode_seed'],
                             (self.witness, self.request_slots))
            self.assertIs(self.indexer._dosa_decode_witness, self.witness)
        self.assertEqual(sum(c[0] == "factory" for c in self.calls), 1)

    def test_prefill_seed_does_not_activate_a_different_decoder(self):
        for heads, mode, topk_v2 in ((32, 'few-decode', True),
                                    (64, 'few', True),
                                    (64, 'few-decode', False)):
            with self.subTest(heads=heads, mode=mode, topk_v2=topk_v2), patch.dict(
                os.environ, {'SGLANG_DOSA_MODE': mode}
            ):
                self.indexer.n_heads = heads
                self.assertTrue(self.entry.prepare_indexer_backend(self.indexer))
                self.assertTrue(self.call(topk_v2=topk_v2))
                self.assertNotIn('decode_seed', self.calls[-1][1])

    def test_existing_modes_do_not_enable_decode_witnesses(self):
        for mode in ("off", "full", "few"):
            with self.subTest(mode=mode), patch.dict(
                os.environ, {"SGLANG_DOSA_MODE": mode}
            ):
                self.indexer._dosa_decode_enabled = True
                self.entry.prepare_indexer_backend(self.indexer)
                self.assertFalse(self.indexer._dosa_decode_enabled)

    def test_few_factory_reserves_before_first_publication(self):
        def reserve(layer):
            self.assertFalse(self.entry._SLOTS)
            self.calls.append(("reserve", layer))
        self.backend.reserve_for_model_load = reserve
        self.assertTrue(self.entry.prepare_indexer_backend(self.indexer))
        self.assertEqual([v[0] for v in self.calls], ["factory", "reserve"])
        self.assertIs(self.entry._SLOTS[("few", "cuda:0", 32)], self.backend)

    def test_failed_first_reserve_does_not_publish(self):
        self.backend.reserve_for_model_load = lambda layer: (_ for _ in ()).throw(RuntimeError("reserve failed"))
        for mode in ("few", "full"):
            with self.subTest(mode=mode), patch.dict(os.environ, {"SGLANG_DOSA_MODE": mode}):
                with self.assertRaisesRegex(RuntimeError, "reserve failed"):
                    self.entry.prepare_indexer_backend(self.indexer)
                self.assertFalse(self.entry._SLOTS)

    def test_full_factory_reserves_before_first_publication(self):
        def reserve(layer):
            self.assertFalse(self.entry._SLOTS)
            self.calls.append(("reserve", layer))
        self.backend.reserve_for_model_load = reserve
        with patch.dict(os.environ, {"SGLANG_DOSA_MODE": "full"}):
            self.assertTrue(self.entry.prepare_indexer_backend(self.indexer))
        self.assertEqual([v[0] for v in self.calls], ["full", "reserve"])
        self.assertIs(self.entry._SLOTS[("full", "cuda:0", 32)], self.backend)

    def test_each_actual_layer_reserved_and_factory_not_repeated(self):
        for layer in (0, 6, 74):
            self.indexer.layer_id = layer
            self.assertTrue(self.entry.prepare_indexer_backend(self.indexer))
        self.assertEqual(sum(c[0] == "factory" for c in self.calls), 1)
        self.assertEqual([c[1] for c in self.calls if c[0] == "reserve"], [0, 6, 74])

    def test_off_and_unsupported_cp_decline_before_factory(self):
        with patch.dict(os.environ, {"SGLANG_DOSA_MODE": "off"}):
            self.assertFalse(self.entry.prepare_indexer_backend(self.indexer))
        self.indexer.dsa_enable_prefill_cp = True
        self.assertFalse(self.entry.prepare_indexer_backend(self.indexer))
        self.assertFalse(self.calls)

    def test_full_preserves_qualified_factory(self):
        with patch.dict(os.environ, {"SGLANG_DOSA_MODE": "full"}):
            self.assertTrue(self.entry.prepare_indexer_backend(self.indexer))
        self.assertEqual([v[0] for v in self.calls], ["full", "reserve"])

    def test_full_reserves_configured_q32_and_admits_real_tails(self):
        self.backend.max_q = 32768
        with patch.dict(os.environ, {"SGLANG_DOSA_MODE": "full"}), patch.object(
            self.entry, "configured_query_capacity", return_value=32768
        ):
            self.assertTrue(self.entry.prepare_indexer_backend(self.indexer))
            self.assertEqual(self.calls[0][1]["max_queries"], 32768)
            for rows in (16385, 26174, 29660, 32761, 32768):
                self.assertTrue(self.call(rows=rows, nk=1048569))
            self.assertFalse(self.call(rows=32769, metadata_reads=False))
            self.assertFalse(self.call(rows=32768, nk=196607))

    def test_full_keeps_legacy_reservation_for_other_buffer_sizes(self):
        with patch.dict(os.environ, {"SGLANG_DOSA_MODE": "full"}):
            for capacity in (0, 4096, 8192, 16384):
                with self.subTest(capacity=capacity), patch.object(
                    self.entry, "configured_query_capacity", return_value=capacity
                ):
                    self.assertTrue(self.entry.prepare_indexer_backend(self.indexer))
                    self.assertTrue(self.call(rows=16384))
                    self.assertFalse(self.call(rows=16385, metadata_reads=False))
        self.assertEqual(sum(c[0] == "full" for c in self.calls), 1)

    def test_full_rejects_configuration_change_after_reservation(self):
        with patch.dict(os.environ, {"SGLANG_DOSA_MODE": "full"}), patch.object(
            self.entry, "configured_query_capacity", return_value=32768
        ):
            with self.assertRaisesRegex(RuntimeError, "configuration changed"):
                self.call(rows=16384)

    def test_outside_reserved_q_capacity_declines_before_metadata(self):
        for rows in (0, 16385, 32768):
            self.assertFalse(self.call(rows=rows, metadata_reads=False))
        self.assertFalse(self.calls)

    def test_supported_chunks_and_natural_prefill_tails_are_admitted(self):
        for rows in (1, 8, 4096, 8192, 10386, 10432, 16384):
            self.assertTrue(self.call(rows=rows))
        self.assertEqual(len(self.calls), 7)

    def test_supported_q16k_uses_dynamic_actual_k_layer_rid(self):
        for nk in (131072, 196608, 917504, 393216):
            self.assertTrue(self.call(nk=nk))
        self.assertEqual([v[1]["nk"] for v in self.calls],
                         [131072, 196608, 917504, 393216])
        self.assertTrue(all(v[1]["hot_key"] == (6, "r1") for v in self.calls))

    def test_outside_k_range_declines(self):
        for nk in (131071, 1048577):
            self.assertFalse(self.call(nk=nk))
        self.assertFalse(self.calls)


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.abi = self.binding = load("native_binding")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.names = []
        self.env = patch.dict(os.environ, {"SGLANG_DOSA_NATIVE_MANIFEST": str(self.root / "manifest.json")})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()
        for name in self.names:
            sys.modules.pop(name, None)

    def manifest(self, mode="few", live=False):
        abi = self.abi.FEW_SERVICE_ABI if mode == "few" else self.binding.FULL_SERVICE_ABI
        if live:
            abi = {**abi, **self.abi.FEW_LIVE_ABI}
        entries = {}
        for key, methods in abi.items():
            name = "test_native_" + key
            self.names.append(name)
            path = self.root / (name + ".so")
            path.write_bytes((mode + key).encode())
            entries[key] = dict(path=str(path), module=name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), required_methods=methods)
        manifest = dict(schema=1, mode=mode, modules=entries)
        if live:
            manifest["live_nk_graph"] = True
        self.write(manifest)
        return manifest

    def write(self, manifest):
        (self.root / "manifest.json").write_text(json.dumps(manifest))

    def install_stubs(self, manifest):
        for entry in manifest["modules"].values():
            module = types.SimpleNamespace(__file__=entry["path"], __name__=entry["module"])
            for method in entry["required_methods"]:
                setattr(module, method, lambda: None)
            sys.modules[entry["module"]] = module

    def test_all_file_hashes_checked_before_any_native_import(self):
        manifest = self.manifest()
        Path(manifest["modules"]["record5_seed"]["path"]).write_bytes(b"changed")
        with patch.object(importlib.util, "spec_from_file_location", side_effect=AssertionError("imported")):
            with self.assertRaisesRegex(RuntimeError, "Qualified native file changed"):
                self.binding.load_service_modules("few")

    def test_wrong_mode_and_missing_keys_rejected(self):
        manifest = self.manifest()
        with self.assertRaises(ValueError):
            self.binding.load_service_modules("full")
        del manifest["modules"]["record5_seed"]
        self.write(manifest)
        with self.assertRaises(ValueError):
            self.binding.load_service_modules("few")

    def test_abi_and_live_mode_cannot_change(self):
        manifest = self.manifest()
        self.install_stubs(manifest)
        loaded = self.binding.load_service_modules("few")
        self.assertIs(loaded, self.binding.load_service_modules("few"))
        with self.assertRaisesRegex(RuntimeError, "Cannot change native bindings"):
            self.binding.load_service_modules("full")

    def test_packaged_build_is_default_and_reused(self):
        modules, calls = {}, []
        def build(mode, **options):
            calls.append((mode, options))
            return modules
        with patch.dict(os.environ, {"SGLANG_DOSA_NATIVE_MANIFEST": ""}), patch.object(
            self.binding, "load_modules", side_effect=build
        ):
            self.assertIs(self.binding.load_service_modules("few"), modules)
            self.assertIs(self.binding.load_service_modules("few"), modules)
            self.assertEqual(calls, [("few", {"service": True, "score_heads": 32})])
            with self.assertRaisesRegex(RuntimeError, "Cannot change native bindings"):
                self.binding.load_service_modules("full")
        with self.assertRaisesRegex(RuntimeError, "Cannot change native bindings"):
            self.binding.load_service_modules("few")

    def test_native_cache_uses_standard_location_or_explicit_override(self):
        builders = self.binding
        self.assertEqual(builders.native_build_directory(self.root / "explicit"),
                         self.root / "explicit")
        extension = types.SimpleNamespace(get_default_build_root=lambda: str(self.root / "torch"))
        with patch.dict(sys.modules, {"torch.utils.cpp_extension": extension}), patch.dict(
            os.environ, {"TORCH_EXTENSIONS_DIR": str(self.root / "cache")}
        ):
            self.assertEqual(builders.native_build_directory(), self.root / "cache/sglang_dosa")

    def test_failed_packaged_build_is_not_published(self):
        with patch.dict(os.environ, {"SGLANG_DOSA_NATIVE_MANIFEST": ""}), patch.object(
            self.binding, "load_modules", side_effect=RuntimeError("toolchain unavailable")
        ):
            with self.assertRaisesRegex(RuntimeError, "toolchain unavailable"):
                self.binding.load_service_modules("few")
        self.assertIsNone(self.binding._SERVICE_BINDING)
        self.assertFalse(self.binding._MODULES)

    def test_invalid_explicit_binding_does_not_fall_back_to_build(self):
        with patch.object(self.binding, "load_modules",
                          side_effect=AssertionError("explicit binding silently replaced")):
            with self.assertRaises(FileNotFoundError):
                self.binding.load_service_modules("few")

    def test_full_four_key_legacy_manifest_remains_supported(self):
        manifest = self.manifest("full")
        manifest.pop("mode")
        self.write(manifest)
        self.install_stubs(manifest)
        self.assertEqual(set(self.binding.load_service_modules()), set(self.binding.FULL_SERVICE_ABI))

    def add_capacity_methods(self, manifest):
        for key, method in (("post", "replay_capacity"), ("leaf", "replay_and_leaf_capacity")):
            if key in manifest["modules"]:
                entry = manifest["modules"][key]
                entry["required_methods"] = (*entry["required_methods"], method)
        return manifest

    def test_old_and_capacity_writer_loader_roundtrip(self):
        for mode in ("full", "few"):
            for capacity in (False, True):
                with self.subTest(mode=mode, capacity=capacity):
                    self.binding = load("native_binding")
                    manifest = self.manifest(mode)
                    if capacity:
                        self.add_capacity_methods(manifest)
                    self.install_stubs(manifest)
                    modules = {key: sys.modules[entry["module"]]
                               for key, entry in manifest["modules"].items()}
                    output = load("build_manifest").manifest_from_modules(modules, mode=mode)
                    self.assertEqual(output["modules"]["post"]["required_methods"],
                                     list(manifest["modules"]["post"]["required_methods"]))
                    self.write(output)
                    loaded = self.binding.load_service_modules(mode)
                    self.assertEqual(set(loaded), set(modules))
                    self.assertEqual(hasattr(loaded["post"], "replay_capacity"), capacity)
                    if mode == "few":
                        self.assertEqual(hasattr(loaded["leaf"], "replay_and_leaf_capacity"), capacity)

    def test_live_manifest_writer_loader_roundtrip(self):
        manifest = self.manifest(live=True)
        self.install_stubs(manifest)
        modules = {key: sys.modules[entry["module"]]
                   for key, entry in manifest["modules"].items()}
        output = load("build_manifest").manifest_from_modules(modules, mode="few")
        self.assertIs(output["live_nk_graph"], True)
        self.write(output)
        self.assertEqual(set(self.binding.load_service_modules("few")), set(modules))

    def native_recipe_stubs(self, modules, calls):
        # The real seed library exports both ordinary and record5 operations.
        for method in self.abi.FEW_SERVICE_ABI["record5_seed"]:
            setattr(modules["seed"], method, lambda: None)
        modules["record5_seed"] = modules["seed"]
        def build(group, role=None, directory=None, verbose=False, *, source_root=None, score_heads=32):
            if group == "live_nk":
                calls.append((role, directory, source_root))
                return modules["live_" + role]
            if group == "record4":
                return modules["record5_exact" if role == "exact5" else "record5_selector"]
            return modules[role if group == "few" else group]
        return types.SimpleNamespace(
            FEW_MODULES=dict.fromkeys(self.abi.FEW_BASE_KEYS[2:]), load_native=build)

    def test_default_few_build_appends_all_live_roles_without_replacing_base(self):
        manifest = self.manifest(live=True)
        self.install_stubs(manifest)
        modules = {key: sys.modules[value["module"]]
                   for key, value in manifest["modules"].items()}
        calls = []
        recipes = self.native_recipe_stubs(modules, calls)
        output = self.root / "built.json"
        with patch.dict(sys.modules, {PREFIX + ".native.build": recipes}):
            result = load("build_manifest").build_manifest(self.root / "build", output, mode="few")
        self.assertEqual([item[0] for item in calls],
                         [key.removeprefix("live_") for key in self.abi.FEW_LIVE_ABI])
        self.assertTrue(all(Path(item[1]).resolve() == (self.root / "build/live_nk").resolve()
                            and Path(item[2]) == ROOT / "native" for item in calls))
        self.assertIs(result["live_nk_graph"], True)
        self.assertEqual(set(result["modules"]), set(modules))
        for key, value in modules.items():
            self.assertEqual(result["modules"][key]["path"], str(Path(value.__file__).resolve()))

    def test_h64_manifest_records_and_builds_the_requested_geometry(self):
        manifest = self.manifest(live=True)
        self.install_stubs(manifest)
        modules = {key: sys.modules[value["module"]]
                   for key, value in manifest["modules"].items()}
        recipes = self.native_recipe_stubs(modules, [])
        original, geometries = recipes.load_native, []

        def build(group, role=None, *args, score_heads=32, **kwargs):
            geometries.append((group, role, score_heads))
            return original(group, role, *args, score_heads=score_heads, **kwargs)

        recipes.load_native = build
        with patch.dict(sys.modules, {PREFIX + ".native.build": recipes}):
            result = load("build_manifest").build_manifest(
                self.root / "build64", self.root / "h64.json",
                mode="few", score_heads=64)
        self.assertEqual(result["score_heads"], 64)
        self.assertIn(("few", "sample", 64), geometries)
        self.assertIn(("live_nk", "exact", 64), geometries)
        self.assertIn(("few", "h16", 64), geometries)
        self.assertIn(("caller", None, 32), geometries)

    def test_default_few_build_reuses_complete_static_live_libraries(self):
        manifest = self.manifest(live=True)
        self.install_stubs(manifest)
        modules = {key: sys.modules[value["module"]]
                   for key, value in manifest["modules"].items()}
        calls = []
        recipes = self.native_recipe_stubs(modules, calls)
        modules["audit"].prepare_fp16_h22_live_k = lambda *a: None
        roles = [key.removeprefix("live_") for key in self.abi.FEW_LIVE_ABI]
        for capable in (("h16",), ("h16", "h18", "audit")):
            with self.subTest(capable=capable):
                for role in capable:
                    for method in self.abi.FEW_LIVE_ABI["live_" + role]:
                        setattr(modules[role], method, lambda *a: None)
                calls.clear()
                self.binding._MODULES.clear()
                with patch.dict(sys.modules, {PREFIX + ".native.build": recipes}):
                    values = self.binding.load_modules(
                        "few", service=True, build_directory=self.root / "build")
                    result = self.binding.manifest_from_modules(values, mode="few")
                self.assertEqual([call[0] for call in calls],
                                 [role for role in roles if role not in capable])
                self.assertEqual(len(result["modules"]), 21)
                for role in capable:
                    self.assertIs(values[role], values["live_" + role])
                    self.assertEqual(result["modules"][role]["path"],
                                     result["modules"]["live_" + role]["path"])
                if "audit" not in capable:
                    self.assertIs(values["live_audit"], modules["live_audit"])

    def test_partial_live_extension_rejected_by_writer_and_loader(self):
        manifest = self.manifest(live=True)
        del manifest["modules"]["live_common"]
        self.install_stubs(manifest)
        modules = {key: sys.modules[entry["module"]]
                   for key, entry in manifest["modules"].items()}
        with self.assertRaises(ValueError):
            load("build_manifest").manifest_from_modules(modules, mode="few")
        self.write(manifest)
        with self.assertRaises(ValueError):
            self.binding.load_service_modules("few")

    def test_capacity_declaration_requires_callable_entry_points(self):
        for key, method in (("post", "replay_capacity"), ("leaf", "replay_and_leaf_capacity")):
            with self.subTest(key=key):
                self.binding = load("native_binding")
                manifest = self.add_capacity_methods(self.manifest())
                self.install_stubs(manifest)
                self.write(manifest)
                delattr(sys.modules[manifest["modules"][key]["module"]], method)
                with self.assertRaisesRegex(RuntimeError, "Missing qualified native ABI"):
                    self.binding.load_service_modules("few")
                self.assertIsNone(self.binding._SERVICE_BINDING)

    def test_unknown_or_reordered_capacity_abi_rejected_before_import(self):
        for suffix in (("replay_capacity", "unexpected"), ("replay_capacity", "replay_capacity")):
            manifest = self.manifest()
            manifest["modules"]["post"]["required_methods"] += suffix
            self.write(manifest)
            with patch.object(importlib.util, "spec_from_file_location", side_effect=AssertionError("imported")):
                with self.assertRaises(ValueError):
                    self.binding.load_service_modules("few")
        manifest = self.add_capacity_methods(self.manifest())
        manifest["modules"]["post"]["required_methods"] = tuple(reversed(manifest["modules"]["post"]["required_methods"]))
        self.write(manifest)
        with self.assertRaises(ValueError):
            self.binding.load_service_modules("few")

    def test_legacy_declaration_checks_exposed_capacity_attribute(self):
        for valid in (False, True):
            with self.subTest(valid=valid):
                self.binding = load("native_binding")
                manifest = self.manifest()
                self.install_stubs(manifest)
                module = sys.modules[manifest["modules"]["leaf"]["module"]]
                module.replay_and_leaf_capacity = (lambda: None) if valid else None
                if valid:
                    self.assertIn("leaf", self.binding.load_service_modules("few"))
                else:
                    with self.assertRaisesRegex(RuntimeError, "Invalid fixed-capacity native ABI"):
                        self.binding.load_service_modules("few")

    def test_service_assembly_splits_base_and_record_seed(self):
        manifest = self.manifest()
        self.install_stubs(manifest)
        seen = {}
        backend = types.SimpleNamespace(_dosa_live_bootstrap=object(), max_q=None,
                                        stock_carry=types.SimpleNamespace(max_q=16384),
                                        shape=None, hot_graph=None, last_done=None,
                                        startup_ready=None)
        def factory(mode, device, max_queries, max_keys, modules, records, sample,
                    min_rows, score_heads=32, id_pool_pages=None):
            seen.update(mode=mode, device=device, max_queries=max_queries, max_keys=max_keys,
                        modules=modules, records=records, sample=sample, id_pool_pages=id_pool_pages)
            backend.max_q = max_queries
            return backend
        def install_slabs(owner, logical_rows):
            self.assertIs(owner, backend)
            owner.logical_query_capacity = logical_rows
            seen['slabs'] = logical_rows
        with patch.object(sys.modules[PREFIX], "create_backend", create=True), patch.dict(
            sys.modules, {
                "torch": types.SimpleNamespace(),
                PREFIX + ".capture_workspace": types.SimpleNamespace(install_outer_capture_workspace=None),
                PREFIX + ".paid_bootstrap": types.SimpleNamespace(PaidBootstrap=None),
                PREFIX + ".slabbed_few": types.SimpleNamespace(install_query_slabs=install_slabs),
            }
        ):
            assembly = load("assembly")
            with patch.object(assembly, "_assemble", side_effect=factory):
                result = assembly.create_service_backend(
                    "few", "cuda:0", max_queries=16384, max_keys=1048576)
        self.assertIs(result, backend)
        self.assertEqual(set(seen["modules"]), set(self.abi.FEW_BASE_KEYS))
        self.assertEqual(set(seen["records"]), {"exact", "selector", "seed"})
        self.assertIsNot(seen["modules"]["seed"], seen["records"]["seed"])
        self.assertIs(seen["sample"], seen["modules"]["sample"])
        self.assertEqual((seen["mode"], seen["max_queries"], seen["max_keys"]), ("few", 4096, 1048576))
        self.assertEqual(seen["id_pool_pages"], 8359936)
        self.assertEqual(seen["slabs"], 16384)
        self.assertEqual(backend.logical_query_capacity, 16384)
        self.assertEqual(backend.stock_carry.max_q, 16384)


if __name__ == "__main__":
    unittest.main(verbosity=2)
