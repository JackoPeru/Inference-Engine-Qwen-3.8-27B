#!/usr/bin/env python3
"""Host-only parser and scratch-dispatch regression; no model or GPU."""
from __future__ import annotations

import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
PATCHER = ROOT / "experiments" / "pp2_layer0_trace_scratch_patch.py"
SOURCE = ROOT / "distributed" / "pp2_fwd.cu"
spec = importlib.util.spec_from_file_location("h27_l0_patch", PATCHER)
assert spec and spec.loader
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class ScratchLayer0Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = SOURCE.read_text(encoding="utf-8")

    def test_real_source_exact_one_anchor_and_no_side_effect(self):
        patched = p.patched_source(self.original)
        self.assertEqual(patched.count(p.TAG), 2)
        self.assertEqual(patched.count("static bool run_gdn_decode_layer"), 1)
        self.assertEqual(patched.count("l0_trace_scratch("), 2)
        self.assertIn("cudaStreamSynchronize(stream)", patched)
        self.assertIn('std::fopen(filename, "wbx")', patched)
        self.assertEqual(self.original, SOURCE.read_text(encoding="utf-8"))

    def test_all_required_shapes_and_names(self):
        patched = p.patched_source(self.original)
        for name, shape in [
            ("attn_norm", "HID"), ("linear_attn_qkv_mixed", "CC"),
            ("conv_output_silu", "CC"), ("beta_sigmoid", "HV"),
            ("gate", "HV"), ("q_norm_scaled", "VDD"),
            ("k_norm", "VDD"), ("attn_output", "VDD"),
            ("final_output", "VDD"), ("linear_attn_out", "HID"),
            ("attn_residual", "HID"),
        ]:
            self.assertIn('{"' + name + '",', patched)
            self.assertIn('{"' + name + '",', patched)
            self.assertIn(name + '", d_', patched)
        self.assertIn("stage.floats > 10240", patched)

    def test_missing_anchor_blocks(self):
        with self.assertRaisesRegex(p.Blocked, "exact anchors"):
            p.patched_source(self.original.replace(
                "static bool run_gdn_decode_layer", "static bool new_name", 1))
        with self.assertRaisesRegex(p.Blocked, "exact anchors"):
            p.patched_source(self.original.replace(
                p.POINT, '  if(false){', 1))

    def test_repatch_and_preexisting_output_rejected(self):
        patched = p.patched_source(self.original)
        with self.assertRaisesRegex(p.Blocked, "double patch"):
            p.patched_source(patched)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            src = root / "original" / "pp2_fwd.cu"
            dst = root / "candidate" / "pp2_fwd.cu"
            src.parent.mkdir()
            dst.parent.mkdir()
            src.write_text(self.original, encoding="utf-8")
            dst.write_text("do not overwrite", encoding="utf-8")
            with mock.patch.object(sys, "argv", [
                "patch", "--source", str(src), "--output", str(dst)]):
                with self.assertRaisesRegex(p.Blocked, "never overwrite"):
                    p.main()
            self.assertEqual(dst.read_text(), "do not overwrite")

    def test_output_is_new_temp_and_bytes_stable(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            src = root / "original" / "pp2_fwd.cu"
            dst = root / "candidate" / "pp2_fwd.cu"
            src.parent.mkdir()
            dst.parent.mkdir()
            src.write_text(self.original, encoding="utf-8", newline="\n")
            with mock.patch.object(sys, "argv", [
                "patch", "--source", str(src), "--output", str(dst)]):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(p.main(), 0)
            self.assertIn("l0_trace_scratch(", dst.read_text())
            self.assertEqual(src.read_text(), self.original)
            with mock.patch.object(sys, "argv", [
                "patch", "--source", str(src), "--output", str(dst)]):
                with self.assertRaises(p.Blocked):
                    p.main()

    def test_no_implict_trace_output_without_opt_in(self):
        patched = p.patched_source(self.original)
        self.assertIn('li == 0 && std::getenv("HERMES_SCRATCH_LAYER0_TRACE_DIR")', patched)
        self.assertIn("capture refuses async/multiuser mode", patched)
        self.assertIn("scratch layer0 trace limited to exactly two decode tokens", patched)
        self.assertIn("cudaMemcpyDeviceToHost", patched)
        self.assertNotIn("cudaDeviceReset(", patched)


if __name__ == "__main__":
    unittest.main()
