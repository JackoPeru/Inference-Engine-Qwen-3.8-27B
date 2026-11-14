#!/usr/bin/env python3
"""Independent disjoint synthetic captures and strict authenticity negatives.

Uses no model, CUDA, network, installed library, service or GPU. The tiny
fixture's expected GGUF SHA is explicit test injection only; production CLI
pins the original model hash and bytes and cannot override them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import reliability_layer0_compare_authentic as auth


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return digest(path)


def pack(values) -> bytes:
    values = list(values)
    return struct.pack("<" + "f" * len(values), *values)


def nums(stage: str, position: int, count: int):
    # Intentionally independently constructed per backend, not copied.
    identity = sum(ord(char) for char in stage) % 97
    return [float((identity % 17) / 8.0 + position / 4.0 +
                  (index % 37) / 1024.0) for index in range(count)]


class Fixture:
    def __init__(self, folder: Path, salt: bytes = b"first"):
        self.folder = folder
        self.model = folder / "model.gguf"
        self.source = folder / "pp2_fwd_original.cu"
        self.patched = folder / "pp2_fwd_scratch.cu"
        self.binary = folder / "hermes_generate"
        self.lib = folder / "libllama.so.0.5.0"
        self.ggml = folder / "libggml-base.so.0.25.3"
        self.exporter = folder / "tracer"
        self.lock = folder / "runtime.lock"
        self.llama_v1_dir = folder / "llama_v1"
        self.llama_batch_dir = folder / "llama_batch"
        self.hermes_dir = folder / "hermes_v2"
        for d in (self.llama_v1_dir, self.llama_batch_dir, self.hermes_dir):
            d.mkdir()
        write(self.model, b"GGUF OFFLINE FIXTURE " + salt)
        write(self.source, b"unmodified scratch hook-free source\n" + salt)
        write(self.patched, b"patch" +
              b"HERMES_SCRATCH_LAYER0_TRACE_DIR " * 2 + salt)
        write(self.binary, b"binary made from scratch" + salt)
        write(self.lib, b"fake llama library " + salt)
        write(self.ggml, b"fake GGML library " + salt)
        write(self.exporter, b"test callback executable " + salt)
        self.lock.write_text(f"{self.lib.name} {digest(self.lib)}\n"
                             f"{self.ggml.name} {digest(self.ggml)}\n",
                             encoding="ascii")
        self.expected_sha = digest(self.model)
        self.model_bytes = self.model.stat().st_size
        self.hermes_manifest = self._make_hermes()
        self.llama_v1_manifest = self._make_llama_single()
        self.llama_batch_manifest = self._make_llama_batch()

    def _make_hermes(self):
        context = []
        tensors = []
        for position in (0, 1):
            token = auth.TOKENS[position]
            raw_context = (f"token_id={token}\nabsolute_position={position}\n"
                           f"sequence={position}\n").encode("ascii")
            context_path = self.hermes_dir / f"p{position}_context.txt"
            sha_context = write(context_path, raw_context)
            context.append({
                "runtime_token_id": token,
                "runtime_absolute_position": position,
                "decode_sequence": position, "context_sha256": sha_context})
            for stage, n in auth.HERMES_WIDTHS.items():
                path = self.hermes_dir / f"p{position}_{stage}.f32"
                h = write(path, pack(nums(stage, position, n)))
                tensors.append({
                    "name": stage, "runtime_reported_position": position,
                    "dtype": "little-endian-fp32", "shape": [n],
                    "bytes": n * 4, "sha256": h})
        m = {
            "schema": auth.SCHEMA_HERMES_V2,
            "backend": "hermes-27-scratch", "status": auth.HERMES_STATUS,
            "not_release_gate": True,
            "model_sha256": self.expected_sha, "model_bytes": self.model_bytes,
            "prompt_ids_runtime_reported": list(auth.TOKENS),
            "absolute_positions_runtime_reported": [0, 1],
            "runtime_decode_contexts": context,
            "source_sha256": digest(self.source),
            "patched_source_sha256": digest(self.patched),
            "binary_sha256": digest(self.binary), "tensors": tensors}
        path = self.hermes_dir / "manifest.json"
        path.write_text(json.dumps(m), encoding="utf-8")
        return path

    def _tensor_record(self, stage, path, dims, *, batch=False):
        count = 1
        for d in dims: count *= d
        strides = [4]
        for d in dims[:-1]:
            strides.append(strides[-1] * d)
        return {"name": stage + "-0", "file": path.name, "shape": dims,
                "source_strides_bytes": strides, "source_type": "f32",
                "buffer": "CUDA0", "elements": count, "bytes": count * 4,
                "sha256": digest(path),
                **({"token_axis": 2 if stage in auth.AXIS2 else 1}
                   if batch else {})}

    def _make_llama_single(self):
        records = []
        for stage in ("attn_norm", "linear_attn_qkv_mixed", "conv_output_silu"):
            n = auth.HERMES_WIDTHS[stage]
            path = self.llama_v1_dir / f"{stage}-0.f32"
            write(path, pack(nums(stage, 0, n)))
            records.append(self._tensor_record(stage, path, [n, 1, 1, 1]))
        m = {"schema": auth.SCHEMA_LLAMA_SINGLE,
             "backend": "llama.cpp", "status": auth.LLAMA_STATUS,
             "model_path": str(self.model), "model_sha256": self.expected_sha,
             "libllama_path": str(self.lib), "libllama_sha256": digest(self.lib),
             "exporter_sha256": digest(self.exporter),
             "prompt_ids": [6205], "absolute_position": 0,
             "layer": 0, "greedy_argmax": 161421,
             "callback_ask_count": 3, "callback_ready_count": 3,
             "tensor_data_format": "little-endian-fp32-ne0-fastest",
             "tensors": records}
        path = self.llama_v1_dir / "manifest.json"
        path.write_text(json.dumps(m), encoding="utf-8")
        return path

    def _make_llama_batch(self):
        records = []
        for stage in auth.COMPARABLE:
            width = auth.HERMES_WIDTHS[stage]
            if stage in auth.HEAD_SHAPES:
                a, b = auth.HEAD_SHAPES[stage]
                dims = [a, b, 2, 1]
            else:
                dims = [width, 2, 1, 1]
            path = self.llama_batch_dir / f"{stage}-0.f32"
            write(path, pack(nums(stage, 0, width) + nums(stage, 1, width)))
            records.append(self._tensor_record(stage, path, dims, batch=True))
        m = {"schema": auth.SCHEMA_LLAMA_BATCH, "backend": "llama.cpp",
             "status": auth.LLAMA_STATUS,
             "model_path": str(self.model), "model_sha256": self.expected_sha,
             "model_bytes": self.model_bytes,
             "exporter_path": str(self.exporter),
             "exporter_sha256": digest(self.exporter),
             "libllama_path": str(self.lib), "libllama_sha256": digest(self.lib),
             "runtime_lock_sha256": digest(self.lock),
             "runtime_libraries": [
                 {"name": p.name, "path": str(p), "sha256": digest(p)}
                 for p in (self.lib, self.ggml)],
             "prompt_ids": [6205, 161421],
             "absolute_positions": [0, 1],
             "one_contiguous_decode_batch": True,
             "layer": 0, "greedy_argmax": [161421, 303],
             "n_gpu_layers": 80, "callback_ask_count": len(records),
             "callback_ready_count": len(records),
             "requested_stages": [x["name"] for x in records],
             "tensor_data_format": "little-endian-fp32-ne0-fastest",
             "tensors": records}
        path = self.llama_batch_dir / "manifest.json"
        path.write_text(json.dumps(m), encoding="utf-8")
        return path

    def llama(self, batch: bool = False):
        return auth.llama(self.llama_batch_manifest if batch else
                          self.llama_v1_manifest, self.model,
                          expected_sha=self.expected_sha,
                          expected_bytes=self.model_bytes,
                          lib_sha=digest(self.lib), exporter=self.exporter,
                          runtime_lock=self.lock if batch else None)

    def hermes(self):
        return auth.hermes(self.hermes_manifest, self.hermes_dir, self.model,
                           source=self.source, patched_source=self.patched,
                           binary=self.binary, expected_sha=self.expected_sha,
                           expected_bytes=self.model_bytes)

    def update_hermes(self, stage: str, pos: int, new_values: list[float]):
        target = self.hermes_dir / f"p{pos}_{stage}.f32"
        write(target, pack(new_values))
        manifest = json.loads(self.hermes_manifest.read_text(encoding="utf-8"))
        for item in manifest["tensors"]:
            if (item["runtime_reported_position"], item["name"]) == (pos, stage):
                item["sha256"] = digest(target)
                break
        self.hermes_manifest.write_text(json.dumps(manifest), encoding="utf-8")


class TestAuthenticCompare(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h27-authentic-disjoint-")
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        one = root / "one"
        two = root / "two"
        one.mkdir()
        two.mkdir()
        self.f = Fixture(one, b"first")
        self.other = Fixture(two, b"second")

    def assertBlocked(self, fn, text: str = ""):
        with self.assertRaises(auth.Blocked) as cm:
            fn()
        if text:
            self.assertIn(text, str(cm.exception))

    def test_disjoint_model_and_files(self):
        self.assertNotEqual(self.f.expected_sha, self.other.expected_sha)
        self.assertNotEqual(digest(self.f.lib), digest(self.other.lib))
        self.assertBlocked(lambda: auth.llama(
            self.f.llama_v1_manifest, self.other.model,
            expected_sha=self.other.expected_sha,
            expected_bytes=self.other.model_bytes,
            lib_sha=digest(self.f.lib)), "model")

    def test_batch_head_axis_both_positions_match(self):
        l, h = self.f.llama(True), self.f.hermes()
        for pos in (0, 1):
            result = auth.compare(l, h, position=pos)
            self.assertEqual(result["status"], "CAPTURED_STAGES_MATCH_ONLY")
            self.assertEqual(len(result["stages"]), len(auth.COMPARABLE))
            self.assertTrue(all(s["max_abs"] == 0 for s in result["stages"]))
        b = auth.llama_token_vector(l, "beta_sigmoid", 1)
        a = auth.llama_token_vector(l, "attn_output", 1)
        self.assertEqual((len(b), len(a)), (48, 6144))
        self.assertEqual(a, h["captures"][(1, "attn_output")])

    def test_single_token_v1_missing_downstream_never_pass(self):
        result = auth.compare(self.f.llama(), self.f.hermes())
        self.assertEqual(result["status"], "BLOCKED_MISSING_LLAMA_DOWNSTREAM")
        self.assertIsNone(result["first_differing_verified_prefix_stage"])
        self.assertTrue(all(s["status"] == "MATCH" for s in result["stages"][:3]))
        self.assertEqual(result["blocker"]["stage"], "beta_sigmoid")

    def test_first_different_stage(self):
        stage = "linear_attn_qkv_mixed"
        vals = nums(stage, 0, auth.HERMES_WIDTHS[stage])
        vals[30] += 0.5
        self.f.update_hermes(stage, 0, vals)
        output = auth.compare(self.f.llama(True), self.f.hermes(), position=0)
        self.assertEqual(output["status"], "FIRST_DIFFERENCE")
        self.assertEqual(output["first_differing_verified_prefix_stage"], stage)
        self.assertEqual(output["stages"][1]["worst_index"], 30)
        self.assertAlmostEqual(output["stages"][1]["max_abs"], 0.5)

    def test_position1_does_not_use_position0_data(self):
        stage = "beta_sigmoid"
        vals = nums(stage, 1, 48)
        vals[5] += 0.25
        self.f.update_hermes(stage, 1, vals)
        ll, hh = self.f.llama(True), self.f.hermes()
        self.assertEqual(auth.compare(ll, hh, 0)["status"], "CAPTURED_STAGES_MATCH_ONLY")
        result = auth.compare(ll, hh, 1)
        self.assertEqual(result["first_differing_verified_prefix_stage"], stage)

    def test_sparse_stage_difference_not_false_full_prefix(self):
        stage = "attn_residual"
        vals = nums(stage, 0, 5120)
        vals[11] += 0.5
        self.f.update_hermes(stage, 0, vals)
        out = auth.compare(self.f.llama(True), self.f.hermes(),
                           stages=("attn_residual",))
        self.assertEqual(out["status"], "FIRST_DIFFERENCE")
        self.assertEqual(out["first_differing_captured_stage"], stage)
        self.assertIsNone(out["first_differing_verified_prefix_stage"])

    def test_reordered_stages_refused(self):
        self.assertBlocked(lambda: auth.compare(
            self.f.llama(True), self.f.hermes(),
            stages=("attn_residual", "attn_norm")), "causal graph order")

    def test_runtime_token_identity_tamper(self):
        path = self.f.hermes_manifest
        m = json.loads(path.read_text(encoding="utf-8"))
        m["prompt_ids_runtime_reported"] = [6205, 6205]
        path.write_text(json.dumps(m), encoding="utf-8")
        self.assertBlocked(self.f.hermes, "runtime token")

    def test_batch_llama_position_tamper(self):
        path = self.f.llama_batch_manifest
        m = json.loads(path.read_text(encoding="utf-8"))
        m["absolute_positions"] = [1, 0]
        path.write_text(json.dumps(m), encoding="utf-8")
        self.assertBlocked(lambda: self.f.llama(True), "contiguous tokens")

    def test_non_equivalent_stage_blocked_even_if_equal_shape(self):
        m = auth.compare(self.f.llama(True), self.f.hermes(),
                         stages=("q_norm_scaled",))
        self.assertEqual(m["status"], "BLOCKED_UNSUPPORTED_STAGE")

    def test_token_axis2_tamper(self):
        path = self.f.llama_batch_manifest
        m = json.loads(path.read_text(encoding="utf-8"))
        for item in m["tensors"]:
            if item["name"] == "attn_output-0":
                item["token_axis"] = 1
                break
        path.write_text(json.dumps(m), encoding="utf-8")
        self.assertBlocked(lambda: self.f.llama(True), "token_axis")

    def test_head_layout_tamper(self):
        path = self.f.llama_batch_manifest
        m = json.loads(path.read_text(encoding="utf-8"))
        for item in m["tensors"]:
            if item["name"] == "attn_output-0":
                item["shape"] = [6144, 2, 1, 1]
                break
        path.write_text(json.dumps(m), encoding="utf-8")
        self.assertBlocked(lambda: self.f.llama(True), "head")

    def test_hash_tamper_real_bytes(self):
        file = self.f.hermes_dir / "p0_attn_norm.f32"
        raw = bytearray(file.read_bytes())
        raw[0] ^= 1
        file.write_bytes(raw)
        self.assertBlocked(self.f.hermes, "SHA-256 mismatch")

    def test_bad_context_token_sidecar(self):
        file = self.f.hermes_dir / "p1_context.txt"
        file.write_text("token_id=6205\nabsolute_position=1\nsequence=1\n")
        self.assertBlocked(self.f.hermes, "SHA-256 mismatch")

    def test_missing_context_sidecar(self):
        (self.f.hermes_dir / "p1_context.txt").unlink()
        self.assertBlocked(self.f.hermes, "missing physical")

    def test_truncated_tensor(self):
        file = self.f.llama_v1_dir / "attn_norm-0.f32"
        file.write_bytes(file.read_bytes()[:-4])
        self.assertBlocked(self.f.llama, "byte count")

    def test_corrupted_model(self):
        self.f.model.write_bytes(b"GGUF MUTATED CONTENT")
        self.assertBlocked(lambda: auth.verify_model(
            self.f.model, self.f.expected_sha, self.f.model_bytes))

    def test_malformed_runtime_lock(self):
        self.f.lock.write_text("libllama.so.0.5.0 " + "0"*64 + "\n")
        self.assertBlocked(lambda: self.f.llama(True), "SHA-256 mismatch")

    def test_missing_exporter_on_batch(self):
        self.assertBlocked(lambda: auth.llama(
            self.f.llama_batch_manifest, self.f.model,
            expected_sha=self.f.expected_sha,
            expected_bytes=self.f.model_bytes,
            lib_sha=digest(self.f.lib), runtime_lock=self.f.lock),
            "exporter")

    def test_v1_partial_exporter_provenance_visible(self):
        captured = auth.llama(
            self.f.llama_v1_manifest, self.f.model,
            expected_sha=self.f.expected_sha,
            expected_bytes=self.f.model_bytes, lib_sha=digest(self.f.lib))
        self.assertFalse(auth.inspect_llama(captured)["exporter_verified"])

    def test_absent_provenance_not_silent(self):
        manifest = self.f.hermes_manifest
        m = json.loads(manifest.read_text(encoding="utf-8"))
        m.pop("runtime_decode_contexts")
        manifest.write_text(json.dumps(m), encoding="utf-8")
        self.assertBlocked(self.f.hermes, "runtime decode context")

    def test_nonfinite_data_with_rehashed_manifest(self):
        path = self.f.llama_v1_dir / "attn_norm-0.f32"
        raw = bytearray(path.read_bytes())
        raw[:4] = struct.pack("<f", float("nan"))
        path.write_bytes(raw)
        manifest = self.f.llama_v1_manifest
        m = json.loads(manifest.read_text(encoding="utf-8"))
        m["tensors"][0]["sha256"] = digest(path)
        manifest.write_text(json.dumps(m), encoding="utf-8")
        self.assertBlocked(self.f.llama, "nonfinite")

    def test_duplicate_json_fields_rejected(self):
        p = self.f.llama_v1_manifest
        p.write_text('{"schema":"one","schema":"two"}')
        self.assertBlocked(self.f.llama, "duplicate JSON")


if __name__ == "__main__":
    unittest.main(verbosity=2)
