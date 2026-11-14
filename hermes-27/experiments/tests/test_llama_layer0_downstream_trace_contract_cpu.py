"""No model inference: independent structural/negative CPU preflight tests.

The tiny fixtures below exercise rejection paths. They are never tensors,
physical llama libraries, exporter identities, or an independent oracle.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


GUARD = Path(__file__).resolve().parents[1] / "llama_layer0_downstream_capture_guard.py"
SPEC = importlib.util.spec_from_file_location("llama_l0_cpu_capture_guard", GUARD)
guard = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(guard)


def synthetic_metadata():
    tensors = []
    for name, (shape, axis) in guard.CAPTURE_STAGES.items():
        chosen_shape = [v[0] if isinstance(v, tuple) else v for v in shape]
        count = 1
        for v in chosen_shape:
            count *= v
        tensors.append({"name": name, "shape": chosen_shape, "token_axis": axis,
                        "elements": count, "bytes": count * 4,
                        "file": name + ".f32"})
    return {
        "schema": "h27.reliability-layer0-downstream-cb.v1",
        "status": "CAPTURED_NOT_INDEPENDENTLY_COMPARED", "backend": "llama.cpp",
        "layer": 0, "prompt_ids": [6205, 161421],
        "absolute_positions": [0, 1], "one_contiguous_decode_batch": True,
        "greedy_argmax": [161421, 500],
        "requested_stages": sorted(guard.CAPTURE_STAGES),
        "callback_ask_count": len(tensors), "callback_ready_count": len(tensors),
        "tensors": tensors,
    }


class GuardContractTests(unittest.TestCase):
    def setUp(self):
        self.meta = synthetic_metadata()

    def reject(self, mutation):
        manifest = copy.deepcopy(self.meta)
        mutation(manifest)
        with self.assertRaises(guard.Blocked):
            guard.ensure_stage_layout(manifest)

    def test_expected_static_callback_identity(self):
        self.assertEqual(len(guard.CAPTURE_STAGES), 8)
        self.assertEqual(guard.CAPTURE_STAGES["beta_sigmoid-0"], ((1, 48, 2, 1), 2))
        self.assertEqual(guard.CAPTURE_STAGES["attn_output-0"], ((128, 48, 2, 1), 2))
        guard.ensure_stage_layout(self.meta)  # metadata fixture ONLY

    def test_q_and_k_allow_exact_source_derived_repeat_layout(self):
        for name in ("q_conv_predelta-0", "k_conv_predelta-0"):
            rec = next(r for r in self.meta["tensors"] if r["name"] == name)
            rec["shape"] = [128, 48, 2, 1]
            rec["elements"] = 128 * 48 * 2
            rec["bytes"] = rec["elements"] * 4
        guard.ensure_stage_layout(self.meta)

    def test_flattened_gdn_head_layout_rejected(self):
        self.reject(lambda m: next(r for r in m["tensors"] if
                       r["name"] == "attn_output-0").update(shape=[6144, 2, 1, 1]))

    def test_beta_position_axis_rejected(self):
        self.reject(lambda m: next(r for r in m["tensors"] if
                       r["name"] == "beta_sigmoid-0").update(token_axis=1))

    def test_q_invalid_head_count_rejected(self):
        self.reject(lambda m: next(r for r in m["tensors"] if
                       r["name"] == "q_conv_predelta-0").update(shape=[128, 32, 2, 1]))

    def test_token_one_replay_rejected(self):
        self.reject(lambda m: m.update(prompt_ids=[6205, 6205]))

    def test_absolute_position_shift_rejected(self):
        self.reject(lambda m: m.update(absolute_positions=[1, 2]))

    def test_two_independent_contexts_rejected(self):
        self.reject(lambda m: m.update(one_contiguous_decode_batch=False))

    def test_pos0_greedy_anchor_rejected(self):
        self.reject(lambda m: m.update(greedy_argmax=[161420, 500]))

    def test_ready_count_rejected(self):
        self.reject(lambda m: m.update(callback_ready_count=7))

    def test_requested_stage_set_rejected(self):
        self.reject(lambda m: m.update(requested_stages=["attn_norm-0"]))

    def test_missing_tensor_rejected(self):
        self.reject(lambda m: m["tensors"].pop())

    def test_duplicate_tensor_rejected(self):
        self.reject(lambda m: m["tensors"].__setitem__(0, m["tensors"][-1]))

    def test_wrong_dtype_bytes_rejected(self):
        self.reject(lambda m: m["tensors"][0].update(bytes=1024))

    def test_unknown_schema_rejected(self):
        self.reject(lambda m: m.update(schema="h27.reliability-layer-cb.v1"))

    def test_runtime_lock_exact_pins(self):
        sample = "\n".join(f"{name} {sha}" for name, sha in guard.LIB_SHA.items())
        self.assertEqual(guard.parse_lock(sample), guard.LIB_SHA)
        self.assertEqual(guard.parse_loaded_runtime(
            "LOADED_RUNTIME_SHA_LOCK_TEMPLATE (no model initialized):\n" + sample),
            guard.LIB_SHA)
        with self.assertRaises(guard.Blocked):
            guard.parse_loaded_runtime("LOADED_RUNTIME_SHA_LOCK_TEMPLATE\n" + sample)

    def test_runtime_lock_tamper_missing_extra_duplicate_and_sha(self):
        sample = "\n".join(f"{name} {sha}" for name, sha in guard.LIB_SHA.items())
        rows = sample.splitlines()
        for corrupted in ("\n".join(rows[:-1]), sample + "\n" + rows[0],
                          sample + "\nlibggml-extra.so.1 " + "a" * 64,
                          sample.replace(rows[0].split()[1], "0" * 64)):
            with self.subTest(corrupted=corrupted[-100:]):
                with self.assertRaises(guard.Blocked):
                    guard.parse_lock(corrupted)

    def test_physical_hash_and_size_verified_in_full(self):
        with tempfile.TemporaryDirectory(prefix="l0-source-fixture-") as folder:
            path = Path(folder) / "fixture-not-llama"
            raw = b"tiny non-model fixture\x00"
            path.write_bytes(raw)
            digest = hashlib.sha256(raw).hexdigest()
            self.assertEqual(guard.checked_file(path, digest, len(raw)), path.resolve())
            with self.assertRaises(guard.Blocked):
                guard.checked_file(path, digest, len(raw) + 1)
            with self.assertRaises(guard.Blocked):
                guard.checked_file(path, "a" * 64, len(raw))
            path.write_bytes(raw[:-1] + b"X")
            with self.assertRaises(guard.Blocked):
                guard.checked_file(path, digest, len(raw))

    def test_capture_mode_never_admitted_by_cpu_wrapper(self):
        with self.assertRaises(guard.Blocked):
            guard.cpu_inspect(Path("/unused/test-exporter"), "--capture", "anything")

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(guard.Blocked):
            guard.json_no_duplicates([("prompt_ids", [6205]),
                                      ("prompt_ids", [161421])])


if __name__ == "__main__":
    unittest.main()
