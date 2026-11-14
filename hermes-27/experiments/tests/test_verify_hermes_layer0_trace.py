#!/usr/bin/env python3
"""CPU negative tests for scratch capture verifier; fake fixture is never oracle."""
from __future__ import annotations

from array import array
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MOD = ROOT / "experiments" / "verify_hermes_layer0_trace.py"
spec = importlib.util.spec_from_file_location("h27_l0_verify", MOD)
assert spec and spec.loader
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


class VerifyLayer0Test(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.capture = root / "capture"
        self.capture.mkdir()
        self.model = root / "model.gguf"
        self.model.write_bytes(b"test-mock-gguf")
        (root / "original").mkdir()
        (root / "scratch").mkdir()
        self.src = root / "original" / "pp2_fwd.cu"
        # Minimal source containing the FOUR exact published patch anchors.
        # This is synthetic source, not a claim that a model was compiled.
        import importlib.util
        patch_spec = importlib.util.spec_from_file_location(
            "h27_fixture_patch_constants", MOD.with_name("pp2_layer0_trace_scratch_patch.py"))
        assert patch_spec and patch_spec.loader
        patch_mod = importlib.util.module_from_spec(patch_spec)
        patch_spec.loader.exec_module(patch_mod)
        self.src.write_text("\n".join((patch_mod.INCLUDE, patch_mod.BEGIN,
                                        patch_mod.POINT, patch_mod.ASYNC_POINT,
                                        patch_mod.DECODE_POINT)),
                            encoding="utf-8", newline="\n")
        self.patched = root / "scratch" / "pp2_fwd.cu"
        self.patched.write_bytes(v.reconstructed_patch(self.src.read_bytes()))
        self.binary = root / "hermes_generate"
        self.binary.write_bytes(b"test-mock-executable")
        self.arguments = {
            "capture_dir": self.capture, "model": self.model,
            "source": self.src, "patched_source": self.patched,
            "binary": self.binary,
            "expected_sha": hashlib.sha256(self.model.read_bytes()).hexdigest(),
            "expected_bytes": self.model.stat().st_size,
            "test_fixture": True,
        }
        for position in (0, 1):
            (self.capture / f"p{position}_context.txt").write_text(
                f"token_id={(6205, 161421)[position]}\n"
                f"absolute_position={position}\nsequence={position}\n",
                encoding="ascii",
            )
            for name, count in v.STAGES.items():
                (self.capture / f"p{position}_{name}.f32").write_bytes(
                    array("f", [0.5] * count).tobytes())

    def test_exact_mock_provenance_and_stage_shape(self):
        report = v.verify(**self.arguments)
        self.assertEqual(len(report["tensors"]), 22)
        self.assertTrue(report["not_release_gate"])
        self.assertEqual(v.SYNTHETIC_SCHEMA, report["schema"])
        self.assertEqual("hermes-27-synthetic-fixture", report["backend"])
        self.assertEqual("SYNTHETIC_CPU_FIXTURE_NOT_PHYSICAL_RUNTIME",
                         report["status"])
        self.assertEqual([6205, 161421], report["prompt_ids_runtime_reported"])
        self.assertEqual([0, 1], report["absolute_positions_runtime_reported"])
        self.assertEqual(len(report["runtime_decode_contexts"]), 2)
        self.assertFalse(report["independent_llama_tensor_identity_verified"])
        self.assertEqual("BLOCKED_NO_INDEPENDENT_LLAMA_CALLBACK",
                         report["cross_engine_comparison_status"])
        self.assertEqual("BLOCKED_UNVERIFIED_HEAD_REPETITION_AND_Q_SCALING",
                         report["qk_cross_engine_mapping"]["status"])

    def test_runtime_context_wrong_token_position_missing_and_symlink(self):
        context = self.capture / "p1_context.txt"
        valid = context.read_bytes()
        for corrupted in (
            valid.replace(b"161421", b"6205"),
            valid.replace(b"position=1", b"position=0"),
            valid.replace(b"sequence=1", b"sequence=0"),
            valid + b"unexpected=1\n",
            valid[:-1],
        ):
            with self.subTest(corrupted=corrupted):
                context.write_bytes(corrupted)
                with self.assertRaisesRegex(v.Blocked, "runtime decode token/position"):
                    v.verify(**self.arguments)
        context.unlink()
        with self.assertRaisesRegex(v.Blocked, "runtime context/tensor files"):
            v.verify(**self.arguments)
        context.symlink_to(self.capture / "p0_context.txt")
        with self.assertRaisesRegex(v.Blocked, "symlink or nonregular file"):
            v.verify(**self.arguments)

    def test_bad_model_bytes_or_hash_blocks(self):
        with self.assertRaises(v.Blocked):
            v.verify(**{**self.arguments, "expected_bytes": 1})
        with self.assertRaises(v.Blocked):
            v.verify(**{**self.arguments, "expected_sha": "0" * 64})

    def test_missing_extra_size_and_nan_fail_closed(self):
        stage = self.capture / "p0_beta_sigmoid.f32"
        raw = stage.read_bytes()
        stage.unlink()
        with self.assertRaises(v.Blocked):
            v.verify(**self.arguments)
        stage.write_bytes(raw)
        (self.capture / "p1_unknown.f32").write_bytes(b"x")
        with self.assertRaises(v.Blocked):
            v.verify(**self.arguments)
        (self.capture / "p1_unknown.f32").unlink()
        stage.write_bytes(raw[:-4])
        with self.assertRaises(v.Blocked):
            v.verify(**self.arguments)
        stage.write_bytes(raw)
        data = array("f", [float("nan")] + [0.0] * 47).tobytes()
        stage.write_bytes(data)
        with self.assertRaises(v.Blocked):
            v.verify(**self.arguments)

    def test_patched_and_source_identity_rejected(self):
        self.patched.write_bytes(self.src.read_bytes())
        with self.assertRaises(v.Blocked):
            v.verify(**self.arguments)
        self.patched.write_bytes(b"HERMES_SCRATCH_LAYER0_TRACE_DIR")
        with self.assertRaises(v.Blocked):
            v.verify(**self.arguments)
        self.src.write_bytes(b"HERMES_SCRATCH_LAYER0_TRACE_DIR")
        with self.assertRaises(v.Blocked):
            v.verify(**self.arguments)

    def test_fake_model_digest_cannot_create_genuine_v2(self):
        real = {**self.arguments, "test_fixture": False}
        with self.assertRaisesRegex(v.Blocked, "fixed GGUF SHA/size"):
            v.verify(**real)
        # Fixture mode cannot be enabled from the public CLI.
        self.assertNotIn("--test-fixture", MOD.read_text())

    def test_exact_directory_only_no_hidden_files_or_subdirectories(self):
        extra = self.capture / ".leftover"
        extra.write_text("not a tensor", encoding="ascii")
        with self.assertRaisesRegex(v.Blocked, "missing/unexpected"):
            v.verify(**self.arguments)
        extra.unlink()
        extra = self.capture / "nested"
        extra.mkdir()
        with self.assertRaisesRegex(v.Blocked, "nonregular file"):
            v.verify(**self.arguments)

    def test_missing_sidecar_and_swapped_file_names_rejected(self):
        p0 = self.capture / "p0_context.txt"
        p1 = self.capture / "p1_context.txt"
        p0bytes, p1bytes = p0.read_bytes(), p1.read_bytes()
        p0.write_bytes(p1bytes)
        p1.write_bytes(p0bytes)
        with self.assertRaisesRegex(v.Blocked, "runtime decode token/position"):
            v.verify(**self.arguments)
        p0.write_bytes(p0bytes)
        p1.write_bytes(p1bytes)
        p1.unlink()
        with self.assertRaisesRegex(v.Blocked, "runtime context/tensor files"):
            v.verify(**self.arguments)

    def test_three_markers_but_wrong_cuda_semantics_rejected(self):
        valid = self.patched.read_bytes()
        self.assertEqual(valid.count(b"HERMES_SCRATCH_LAYER0_TRACE_DIR"), 3)
        self.assertIn(b"cudaMemcpyDeviceToHost", valid)
        self.patched.write_bytes(valid.replace(b"cudaMemcpyDeviceToHost",
                                               b"cudaMemcpyHostToDevice"))
        with self.assertRaisesRegex(v.Blocked, "differs from exact frozen"):
            v.verify(**self.arguments)

    def test_gguf_hash_is_full_bytes_not_heading(self):
        self.model.write_bytes(self.model.read_bytes() + b"tampered suffix")
        with self.assertRaisesRegex(v.Blocked, "physical GGUF size/hash"):
            v.verify(**self.arguments)


if __name__ == "__main__":
    unittest.main()
