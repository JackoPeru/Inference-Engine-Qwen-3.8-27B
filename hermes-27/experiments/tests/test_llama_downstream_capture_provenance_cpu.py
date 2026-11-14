#!/usr/bin/env python3
"""Offline negative proof of llama oracle admissibility. Never synthesizes an oracle."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / "experiments" / "llama_layer0_downstream_capture_guard.py"
SOURCE = Path("/opt/hermes/runtimes/llama-mainline/src/models/qwen35.cpp")
POS0 = Path("/tmp/h27-layer0-llama-pos0-prime3/manifest.json")
MODEL = Path("/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf")
LIB = Path("/opt/hermes/runtimes/llama-mainline/build-cuda/bin/libllama.so.0.5.0")
spec = importlib.util.spec_from_file_location("llama_l0_provenance_guard", GUARD)
assert spec and spec.loader
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def metadata():
    """Synthetic metadata only: never physical tensors or llama runtime."""
    records = []
    for name, (shape, axis) in guard.CAPTURE_STAGES.items():
        dims = [v[0] if isinstance(v, tuple) else v for v in shape]
        n = 1
        for size in dims:
            n *= size
        stride = [4]
        for v in dims[:-1]:
            stride.append(stride[-1] * v)
        records.append({
            "name": name, "shape": dims, "token_axis": axis,
            "source_type": "f32", "source_strides_bytes": stride,
            "buffer": "CPU_SYNTHETIC_METADATA_ONLY",
            "elements": n, "bytes": 4 * n, "file": name + ".f32",
        })
    return {
        "schema": "h27.reliability-layer0-downstream-cb.v1",
        "backend": "llama.cpp",
        "status": "CAPTURED_NOT_INDEPENDENTLY_COMPARED",
        "layer": 0, "prompt_ids": [6205, 161421],
        "absolute_positions": [0, 1],
        "one_contiguous_decode_batch": True,
        "tensor_data_format": "little-endian-fp32-ne0-fastest",
        "n_gpu_layers": 80,
        "greedy_argmax": [161421, 512],
        "requested_stages": sorted(guard.CAPTURE_STAGES),
        "callback_ask_count": len(records),
        "callback_ready_count": len(records),
        "tensors": records,
    }


class OracleAdmissionNegativeCPU(unittest.TestCase):
    def setUp(self):
        self.doc = metadata()

    def assert_blocked(self, modify, pattern=None):
        candidate = copy.deepcopy(self.doc)
        modify(candidate)
        with self.assertRaises(guard.Blocked) as error:
            guard.ensure_stage_layout(candidate, strict_source_contract=True)
        if pattern:
            self.assertIn(pattern, str(error.exception))

    def test_synthetic_geometry_is_never_oracle(self):
        # This tests a metadata predicate ONLY, no tensor files exist.
        guard.ensure_stage_layout(self.doc, strict_source_contract=True)
        self.assertEqual(8, len(self.doc["tensors"]))
        self.assertIn("ffn_out-0", guard.UNPROVEN_ORACLE_STAGES)
        self.assertNotIn("ffn_out-0", guard.CAPTURE_STAGES)
        self.assertIn("attn_post_norm-0", guard.UNPROVEN_ORACLE_STAGES)

    def test_two_token_ids_positions_and_same_batch_required(self):
        for field, value in (
            ("prompt_ids", [6205, 6205]),
            ("prompt_ids", [161421, 6205]),
            ("absolute_positions", [0, 0]),
            ("absolute_positions", [1, 2]),
            ("one_contiguous_decode_batch", False),
            ("schema", "h27.reliability-layer-cb.v1"),
            ("status", "VERIFIED_ORACLE"),
            ("greedy_argmax", [111, 512]),
        ):
            with self.subTest(field=field, value=value):
                self.assert_blocked(lambda m: m.update({field: value}))

    def test_head_axes_cannot_be_inferred_from_flat_lengths(self):
        for stage in ("attn_output-0", "beta_sigmoid-0", "q_conv_predelta-0"):
            def corrupt(candidate):
                item = next(v for v in candidate["tensors"] if v["name"] == stage)
                if stage == "attn_output-0":
                    item["shape"] = [6144, 2, 1, 1]  # identical FP32 length
                elif stage == "beta_sigmoid-0":
                    item["shape"] = [48, 2, 1, 1]  # identical FP32 length
                else:
                    item["shape"] = [128, 24, 2, 1]  # invalid head count
            with self.subTest(stage=stage):
                self.assert_blocked(corrupt, "GGML head/position layout")

    def test_source_scalar_stride_dtype_and_backend_must_agree(self):
        for key, value in (
            ("source_strides_bytes", [2, 10, 20, 40]),
            ("source_type", "quantized-q4"),
            ("source_strides_bytes", [0, 20, 40, 80]),
            ("buffer", ""),
        ):
            self.assert_blocked(lambda m: m["tensors"][0].update({key: value}),
                                "callback source scalar stride/type/backend")

    def test_nonphysical_manifest_metadata_blocks(self):
        for key, value in (
            ("tensor_data_format", "little-endian-fp16"),
            ("n_gpu_layers", True),
            ("n_gpu_layers", -1),
            ("n_gpu_layers", 129),
        ):
            self.assert_blocked(lambda m: m.update({key: value}),
                                "actual backend tensor format")

    def test_partial_callback_or_extra_label_blocks(self):
        self.assert_blocked(lambda m: m["tensors"].pop())
        self.assert_blocked(lambda m: m["requested_stages"].append("ffn_out-0"))
        self.assert_blocked(lambda m: m.update(callback_ready_count=0))

    def test_empty_inventory_then_stale_partial_file_blocks(self):
        with tempfile.TemporaryDirectory(prefix="llama-l0-inventory-cpu-") as root:
            folder = Path(root)
            guard.exact_file_inventory(folder, set())
            (folder / "manifest.json.partial").write_bytes(b"not a capture")
            with self.assertRaisesRegex(guard.Blocked, "stale, partial"):
                guard.exact_file_inventory(folder, set())

    def test_hidden_file_nested_dir_and_symlink_block(self):
        with tempfile.TemporaryDirectory(prefix="llama-l0-symlink-cpu-") as root:
            folder = Path(root)
            (folder / ".leftover").write_bytes(b"x")
            with self.assertRaisesRegex(guard.Blocked, "unexpected"):
                guard.exact_file_inventory(folder, set())
            (folder / ".leftover").unlink()
            (folder / "nested").mkdir()
            with self.assertRaisesRegex(guard.Blocked, "nonregular"):
                guard.exact_file_inventory(folder, set())
            (folder / "nested").rmdir()
            (folder / "alias").symlink_to(folder / "not-there")
            with self.assertRaisesRegex(guard.Blocked, "symlink"):
                guard.exact_file_inventory(folder, set())

    def test_short_fake_model_never_admitted(self):
        with tempfile.TemporaryDirectory(prefix="llama-l0-fake-model-cpu-") as root:
            folder = Path(root)
            model = folder / "fake.gguf"
            model.write_bytes(b"GGUF test bytes, not a model")
            args = types.SimpleNamespace(model=model)
            with self.assertRaisesRegex(guard.Blocked, "physical size mismatch"):
                guard.preflight(args)
            self.assertNotEqual(guard.sha256_file(model), guard.MODEL_SHA)

    def test_current_genuine_pos0_only_authenticates_pos0(self):
        if not (POS0.is_file() and MODEL.is_file() and LIB.is_file()):
            self.skipTest("genuine Ubuntu pos0 runtime files unavailable here")
        document = guard.prior_manifest(POS0, MODEL.resolve(), LIB.resolve())
        self.assertEqual([6205], document["prompt_ids"])
        self.assertEqual(0, document["absolute_position"])
        self.assertEqual(161421, document["greedy_argmax"])
        self.assertEqual(3, len(document["tensors"]))
        self.assertNotIn("beta_sigmoid-0", {rec["name"] for rec in document["tensors"]})
        self.assertNotIn("ffn_out-0", {rec["name"] for rec in document["tensors"]})

    def test_installed_source_has_named_callbacks_but_not_real_outputs(self):
        if not SOURCE.is_file():
            self.skipTest("installed llama graph source unavailable here")
        self.assertEqual(guard.SOURCE_SHA["src/models/qwen35.cpp"],
                         hashlib.sha256(SOURCE.read_bytes()).hexdigest())
        actual = SOURCE.read_text(encoding="utf-8")
        for label in ("beta_sigmoid", "gate", "q_conv_predelta",
                      "k_conv_predelta", "attn_post_norm", "ffn_out", "post_ffn",
                      "final_output", "linear_attn_out", "attn_residual"):
            self.assertIn('"' + label + '"', actual)
        # Two source call sites may fail the strict receiver duplicate check.
        self.assertEqual(2, actual.count('cb(cur, "ffn_out", il);'))

    def test_full_oracle_proof_is_explicitly_blocked_by_guard(self):
        script = GUARD.read_text(encoding="utf-8")
        self.assertIn('"BLOCKED_NO_AUTHENTIC_CROSSENGINE_PROOF"', script)
        self.assertIn('"BLOCKED_UNPROVEN_HEAD_REPETITION_AND_Q_SCALE"', script)
        self.assertIn('"VERIFIED_CAPTURE_STAGES_ONLY"', script)
        self.assertNotIn('"FULL_MODEL_PARITY_PASS"', script)


if __name__ == "__main__":
    unittest.main()
