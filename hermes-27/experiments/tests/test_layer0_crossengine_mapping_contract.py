#!/usr/bin/env python3
"""No GPU; reject fake cross-engine equivalence and Q/K head-axis guesses."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hverify = load("h27_crossengine_hermes_verifier", ROOT / "experiments" /
               "verify_hermes_layer0_trace.py")
patch = load("h27_crossengine_frozen_patcher", ROOT / "experiments" /
             "pp2_layer0_trace_scratch_patch.py")
auth = load("h27_crossengine_frozen_comparator", ROOT / "tests" /
            "reliability_layer0_compare_authentic.py")


class CrossEngineMappingContract(unittest.TestCase):
    def test_source_labels_remain_distinct_and_non_equivalent(self):
        self.assertEqual((6205, 161421), hverify.RUNTIME_TOKENS)
        self.assertEqual(6144, hverify.STAGES["q_norm_scaled"])
        self.assertEqual(6144, hverify.STAGES["k_norm"])
        self.assertNotIn("q_norm_scaled", auth.COMPARABLE)
        self.assertNotIn("k_norm", auth.COMPARABLE)
        self.assertIn("q_conv_predelta", auth.LLAMA_EXTRA)
        self.assertIn("k_conv_predelta", auth.LLAMA_EXTRA)
        self.assertEqual(auth.HEAD_SHAPES["attn_output"], (128, 48))

    def test_llama_actual_time_axis_is_two_not_flattened_width(self):
        for name in ("q_conv_predelta-0", "k_conv_predelta-0"):
            for heads in (16, 48):
                rec = {
                    "name": name, "shape": [128, heads, 2, 1],
                    "elements": 128 * heads * 2, "bytes": 128 * heads * 2 * 4,
                    "source_strides_bytes": [4, 512, 512 * heads, 1024 * heads],
                    "source_type": "f32", "buffer": "CPU", "token_axis": 2,
                }
                self.assertEqual(auth._llama_entry(rec, batch=True)[0], name[:-2])
                invalid = copy.deepcopy(rec)
                invalid["shape"] = [128 * heads, 2, 1, 1]
                with self.assertRaises(auth.Blocked):
                    auth._llama_entry(invalid, batch=True)
                invalid = copy.deepcopy(rec)
                invalid["token_axis"] = 1
                with self.assertRaises(auth.Blocked):
                    auth._llama_entry(invalid, batch=True)

    def test_equal_flat_qk_buffers_do_not_override_missing_semantics(self):
        # Comparator API accepts only provenance-checked data in production.
        # This in-memory dummy deliberately provides NO real tensor bytes.
        # Even equal-shaped faux Q/K are semantically unsupported.
        llama = {"model_sha256": "f" * 64, "schema": auth.SCHEMA_LLAMA_BATCH,
                 "position_tokens": (6205, 161421),
                 "exporter_verified": False, "batch": True,
                 "captures": {"q_norm_scaled": {}, "k_norm": {}}}
        hermes = {"model_sha256": "f" * 64,
                  "schema": auth.SCHEMA_HERMES_V2,
                  "runtime_token_ids": [6205, 161421],
                  "captures": {(0, "q_norm_scaled"): [0.0] * 6144,
                               (0, "k_norm"): [0.0] * 6144}}
        for pos in (0, 1):
            report = auth.compare(llama, hermes, position=pos,
                                  stages=("q_norm_scaled", "k_norm"))
            self.assertEqual("BLOCKED_UNSUPPORTED_STAGE", report["status"])
            self.assertTrue(report["not_release_gate"])

    def test_genuine_manifest_cannot_be_built_from_mock_model(self):
        with tempfile.TemporaryDirectory(prefix="h27-crossengine-fake-") as tmp:
            root = Path(tmp)
            capture = root / "capture"
            original_dir = root / "original"
            scratch_dir = root / "scratch"
            for d in (capture, original_dir, scratch_dir):
                d.mkdir()
            model = root / "fake.gguf"
            model.write_bytes(b"GGUF fake CPU-only fixture")
            original = original_dir / "pp2_fwd.cu"
            original.write_text("\n".join((patch.INCLUDE, patch.BEGIN,
                                           patch.POINT, patch.ASYNC_POINT,
                                           patch.DECODE_POINT)),
                                encoding="utf-8", newline="\n")
            modified = scratch_dir / "pp2_fwd.cu"
            modified.write_bytes(hverify.reconstructed_patch(original.read_bytes()))
            binary = root / "hermes_generate"
            binary.write_bytes(b"NONEXECUTABLE fake CPU fixture")
            for pos, token in enumerate(hverify.RUNTIME_TOKENS):
                (capture / f"p{pos}_context.txt").write_bytes(
                    hverify.runtime_sidecar(pos, token))
                for stage, size in hverify.STAGES.items():
                    (capture / f"p{pos}_{stage}.f32").write_bytes(
                        struct.pack("<f", 0.25) * size)
            arguments = dict(capture_dir=capture, model=model, source=original,
                             patched_source=modified, binary=binary,
                             expected_sha=hashlib.sha256(model.read_bytes()).hexdigest(),
                             expected_bytes=model.stat().st_size)
            with self.assertRaisesRegex(hverify.Blocked, "fixed GGUF SHA/size"):
                hverify.verify(**arguments)
            synthetic = hverify.verify(**arguments, test_fixture=True)
            self.assertEqual(hverify.SYNTHETIC_SCHEMA, synthetic["schema"])
            self.assertEqual(hverify.SYNTHETIC_STATUS, synthetic["status"])
            self.assertFalse(synthetic["independent_llama_tensor_identity_verified"])
            self.assertEqual("BLOCKED_UNVERIFIED_HEAD_REPETITION_AND_Q_SCALING",
                             synthetic["qk_cross_engine_mapping"]["status"])
            output = root / "fake_manifest.json"
            output.write_text(json.dumps(synthetic), encoding="utf-8")
            with self.assertRaisesRegex(auth.Blocked, "HERMES runtime-token capture v2"):
                auth.hermes(output, capture, model, source=original,
                            patched_source=modified, binary=binary,
                            expected_sha=arguments["expected_sha"],
                            expected_bytes=arguments["expected_bytes"])


if __name__ == "__main__":
    unittest.main()
