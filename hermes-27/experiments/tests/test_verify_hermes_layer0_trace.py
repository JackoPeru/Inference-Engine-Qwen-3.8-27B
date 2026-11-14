#!/usr/bin/env python3
"""CPU negative tests for scratch capture verifier; no physical GGUF loads."""
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
        self.src = root / "pp2_fwd.cu"
        self.src.write_bytes(b"untouched source")
        self.patched = root / "patched_pp2_fwd.cu"
        self.patched.write_bytes(
            b'HERMES_SCRATCH_LAYER0_TRACE_DIR\n'
            b'HERMES_SCRATCH_LAYER0_TRACE_DIR\n')
        self.binary = root / "hermes_generate"
        self.binary.write_bytes(b"test-mock-executable")
        self.arguments = {
            "capture_dir": self.capture, "model": self.model,
            "source": self.src, "patched_source": self.patched,
            "binary": self.binary,
            "expected_sha": hashlib.sha256(self.model.read_bytes()).hexdigest(),
            "expected_bytes": self.model.stat().st_size,
        }
        for position in (0, 1):
            for name, count in v.STAGES.items():
                (self.capture / f"p{position}_{name}.f32").write_bytes(
                    array("f", [0.5] * count).tobytes())

    def test_exact_mock_provenance_and_stage_shape(self):
        report = v.verify(**self.arguments)
        self.assertEqual(len(report["tensors"]), 22)
        self.assertTrue(report["not_release_gate"])
        self.assertIn("NOT_INDEPENDENTLY_PROVEN", report["status"])
        self.assertEqual([6205, 161421], report["prompt_ids_assumed_from_cli"])

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


if __name__ == "__main__":
    unittest.main()
