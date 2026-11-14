#!/usr/bin/env python3
"""Deterministic source/manifest admission regressions for post-FFN scratch.

No real model execution, no CUDA APIs and no fake llama numerical oracle.
All artificial tensor bytes are unit-test fixture data, not GPU evidence.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "experiments" / "pp2_layer0_postffn_scratch_patch.py"
SOURCE = ROOT / "distributed" / "pp2_fwd.cu"
OLD = ROOT / "experiments" / "pp2_layer0_trace_scratch_patch.py"
spec = importlib.util.spec_from_file_location("h27_postffn_scratch_source", MODULE)
assert spec and spec.loader
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class PublishedSourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_bytes()
        cls.base_patch = p.load_patcher(OLD)
        cls.v2 = cls.base_patch.patched_source(cls.source.decode("utf-8"))
        cls.post = p.compose(cls.source, OLD).decode("utf-8")

    def test_current_source_frozen_hash_and_exact_pointer_geometry(self):
        self.assertEqual(p.SOURCE_SHA, hashlib.sha256(self.source).hexdigest())
        self.assertEqual(p.BASE_PATCHER_SHA, p.sha256_file(OLD))
        self.assertEqual(3, self.v2.count(p.BASE_TAG))
        self.assertEqual(0, self.v2.count(p.POST_TAG))
        self.assertEqual(3, self.post.count(p.BASE_TAG))
        self.assertEqual(3, self.post.count(p.POST_TAG))
        gdn_start = self.post.index("static bool run_gdn_decode_layer(")
        full_start = self.post.index("static bool run_full_decode_layer(")
        gdn = self.post[gdn_start:full_start]
        # Source-authentic scratch arena: d_h postnorm, d_down FFN projection,
        # d_x final additive residual. No device data is present in this test.
        for stage, pointer in (
            ("attn_post_norm", "d_h"),
            ("ffn_out", "d_down"),
            ("post_ffn", "d_x"),
        ):
            self.assertIn('{"' + stage + '",' + pointer + ',HID}', gdn)
        self.assertIn("float*    d_h    = (float*)(ws+0)", gdn)
        self.assertIn("float*    d_down = (float*)(ws+490240)", gdn)
        self.assertIn("const int T=1;", gdn)
        self.assertIn("size_t nH=(size_t)T*HID", gdn)
        self.assertEqual(set(p.POST_STAGES.values()), {5120})

    def test_capture_happens_after_ffn_before_existing_end_sync(self):
        gdn = self.post[
            self.post.index("static bool run_gdn_decode_layer("):
            self.post.index("static bool run_full_decode_layer(")]
        self.assertLess(gdn.index("rms_f32_k<<<T,512,(size_t)512*4,stream>>>(d_h,d_x,(float*)wPost->d,T,HID)"),
                        gdn.index("gemv_batch(d_hidb,(uint8_t*)wFd->d,d_down,FFN,HID,T,stream)"))
        self.assertLess(gdn.index("gemv_batch(d_hidb,(uint8_t*)wFd->d,d_down,FFN,HID,T,stream)"),
                        gdn.index("add_k<<<(int)((nH+255)/256),256,0,stream>>>(d_x,d_down,(int)nH)"))
        self.assertLess(gdn.index("add_k<<<(int)((nH+255)/256),256,0,stream>>>(d_x,d_down,(int)nH)"),
                        gdn.index('if(li==0 && std::getenv("HERMES_SCRATCH_LAYER0_POSTFFN_DIR"))'))
        self.assertLess(gdn.index('if(li==0 && std::getenv("HERMES_SCRATCH_LAYER0_POSTFFN_DIR"))'),
                        gdn.index("if(stream==0 && !decode_skip_layer_sync()) cudaDeviceSynchronize()"))
        self.assertEqual(1, gdn.count("l0_postffn_copy("))
        self.assertNotIn("l0_postffn_copy(", self.post[self.post.index(
            "static bool run_full_decode_layer("):])

    def test_sidecars_use_real_v2_invocation_same_state_and_sequence(self):
        self.assertIn("invocation.state != g_l0_trace_state.load()", self.post)
        self.assertIn("invocation.token != expected_tokens[seq]", self.post)
        self.assertIn("invocation.position != (long)seq", self.post)
        self.assertIn("g_l0_trace_sequence.load() != seq + 1", self.post)
        self.assertIn("g_l0_postffn_sequence=sequence", self.post)
        self.assertIn("std::memcmp(ctx_buffer,ctx_expected,(size_t)ne)!=0", self.post)
        self.assertIn("g_l0_trace_state.compare_exchange_strong", self.post)
        self.assertIn("L0TraceScope actual_decode_token(tok,pos,&ds);", self.post)
        self.assertEqual(b"token_id=6205\nabsolute_position=0\nsequence=0\n",
                         p.context_bytes(0))
        self.assertEqual(b"token_id=161421\nabsolute_position=1\nsequence=1\n",
                         p.context_bytes(1))

    def test_root_exact_pair_canonical_and_exclusive_output(self):
        for marker in ("std::filesystem::canonical(base_dir,ec)",
                       "std::filesystem::canonical(post_dir,ec)",
                       'std::snprintf(root_expected,sizeof(root_expected),',
                       'std::fopen(filename,"wbx")',
                       'std::fopen(sidecar,"wbx")',
                       'std::filesystem::directory_iterator(post_dir)',
                       "if(seq==1 && prior!=4)"):
            self.assertIn(marker, self.post)
        self.assertEqual(3, self.post.count(p.POST_TAG))
        self.assertIn('if(li==0 && std::getenv("HERMES_SCRATCH_LAYER0_POSTFFN_DIR"))', self.post)
        self.assertIn("stream != 0 || count != 3", self.post)
        self.assertIn("cudaStreamSynchronize(stream)!=cudaSuccess", self.post)
        self.assertIn("std::array<float,5120> host{}", self.post)
        self.assertNotIn("cudaDeviceReset(", p.POST_HELPER)

    def test_composition_deterministic_and_base_source_untouched(self):
        self.assertEqual(self.post, p.compose(self.source, OLD).decode("utf-8"))
        self.assertEqual(self.source, SOURCE.read_bytes())
        self.assertTrue(self.post.count("return true;") > 0)

    def test_changed_current_pp2_source_requires_new_review(self):
        mutated = self.source.replace(b"const int HID=5120", b"const int HID=5121", 1)
        self.assertNotEqual(mutated, self.source)
        with self.assertRaisesRegex(p.Blocked, "published frozen source"):
            p.compose(mutated, OLD)

    def test_frozen_patcher_sha_change_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            altered = Path(tmp) / OLD.name
            altered.write_bytes(OLD.read_bytes() + b"\n# tampered\n")
            with self.assertRaisesRegex(p.Blocked, "patcher SHA"):
                p.load_patcher(altered)

    def test_new_patcher_rejects_hook_marker_reapply(self):
        with self.assertRaisesRegex(p.Blocked, "re-patched"):
            p.extend_v2_source(self.post)
        with self.assertRaisesRegex(p.Blocked, "re-patched"):
            p.extend_v2_source(self.v2 + p.BASE_TAG)

    def test_mutants_missing_or_duplicate_ffn_producer_refused(self):
        self.assertEqual(2, self.v2.count(p.FFN_ANCHOR))
        gdn_start = self.v2.index("static bool run_gdn_decode_layer(")
        full_start = self.v2.index("static bool run_full_decode_layer(")
        gdn = self.v2[gdn_start:full_start]
        for mutant in (
            gdn.replace("gemv_batch(d_hidb,(uint8_t*)wFd->d,d_down,FFN,HID,T,stream)",
                        "gemv_batch(d_hidb,(uint8_t*)wFd->d,d_down,FFN,HID,T,0)", 1),
            gdn.replace("add_k<<<(int)((nH+255)/256),256,0,stream>>>(d_x,d_down,(int)nH)",
                        "add_k<<<1,1>>>(d_x,d_down,(int)nH)", 1),
            gdn.replace(p.FFN_ANCHOR, p.FFN_ANCHOR + p.FFN_ANCHOR, 1),
        ):
            with self.subTest(mutant_sha=hashlib.sha256(mutant.encode()).hexdigest()):
                source = self.v2[:gdn_start] + mutant + self.v2[full_start:]
                with self.assertRaisesRegex(p.Blocked, "postFFN hook"):
                    p.extend_v2_source(source)

    def test_mutants_missing_context_or_same_request_fence_refused(self):
        for original, replacement in (
            (p.HELPER_ANCHOR, "static bool l0_trace_scratch("),
            (p.CONTEXT_ANCHOR, p.CONTEXT_ANCHOR.replace("  return true;\n}\n", "")),
            ("L0TraceScope actual_decode_token(tok,pos,&ds);",
             "L0TraceScope actual_decode_token(tok,0,&ds);"),
            ("g_l0_trace_state.compare_exchange_strong",
             "g_l0_trace_state.store"),
            ("scratch layer0 trace refuses asynchronous/sliding dispatch",
             "disabled preflight"),
        ):
            with self.subTest(changed=original[:40]):
                self.assertIn(original, self.v2)
                with self.assertRaises(p.Blocked):
                    p.extend_v2_source(self.v2.replace(original, replacement, 1))

    def test_output_temp_only_and_never_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="h27-postffn-source-cpu-") as tmp:
            root = Path(tmp)
            source = root / "original" / "pp2_fwd.cu"
            target = root / "scratch" / "pp2_fwd.cu"
            source.parent.mkdir()
            target.parent.mkdir()
            source.write_bytes(self.source)
            with mock.patch.object(sys, "argv", [
                "postpatch", "--source", str(source), "--output", str(target)]):
                with redirect_stdout(io.StringIO()) as captured:
                    self.assertEqual(0, p.main())
                self.assertIn("PASS_CPU_SOURCE_ONLY_POSTFFN_NOT_ORACLE",
                              captured.getvalue())
                self.assertEqual(self.post.encode(), target.read_bytes())
                with self.assertRaisesRegex(p.Blocked, "must be NEW"):
                    p.main()
            self.assertEqual(self.source, source.read_bytes())


class SyntheticPostFFNMetadataContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h27-postffn-unit-")
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.base = root / "h27-unit"
        self.post = root / "h27-unit-postffn"
        self.base.mkdir()
        self.post.mkdir()
        for pos in (0, 1):
            for directory in (self.base, self.post):
                (directory / f"p{pos}_context.txt").write_bytes(p.context_bytes(pos))
            for stage, width in p.PRE_STAGES.items():
                (self.base / f"p{pos}_{stage}.f32").write_bytes(
                    struct.pack("<f", 0.25) * width)
            for stage, width in p.POST_STAGES.items():
                (self.post / f"p{pos}_{stage}.f32").write_bytes(
                    struct.pack("<f", 0.125) * width)

    def witness(self):
        return p.validate_post_sidecars(self.base, self.post, synthetic_test=True)

    def test_artificial_metadata_never_claims_real_capture(self):
        report = self.witness()
        self.assertEqual("h27.postffn-synthetic-cpu-fixture.v1",
                         report["schema"])
        self.assertEqual("SYNTHETIC_CPU_FIXTURE_NOT_RUNTIME", report["status"])
        self.assertIsNone(report["model_sha256"])
        self.assertFalse(report["independent_llama_numeric_oracle_verified"])
        self.assertTrue(report["not_release_gate"])
        self.assertEqual(6, len(report["postffn_tensors"]))
        self.assertEqual([0, 0, 0, 1, 1, 1],
                         [v["runtime_reported_position"] for v in report["postffn_tensors"]])

    def test_mock_never_admitted_as_genuine_physical_gguf(self):
        with self.assertRaisesRegex(p.Blocked, "full pinned physical GGUF"):
            p.validate_post_sidecars(self.base, self.post)
        fake = self.base.parent / "fake.gguf"
        fake.write_bytes(b"GGUF not a real model")
        with self.assertRaisesRegex(p.Blocked, "full pinned physical GGUF"):
            p.validate_post_sidecars(self.base, self.post, model=fake)

    def test_swapped_token_or_position_sidecar_rejected(self):
        file = self.post / "p1_context.txt"
        original = file.read_bytes()
        for mutant in (
            original.replace(b"161421", b"6205"),
            original.replace(b"position=1", b"position=0"),
            original.replace(b"sequence=1", b"sequence=0"),
            original + b"extra=1\n",
        ):
            with self.subTest(mutant=mutant):
                file.write_bytes(mutant)
                with self.assertRaisesRegex(p.Blocked, "token/absolute-position"):
                    self.witness()

    def test_missing_post_tensor_context_or_unknown_file(self):
        file = self.post / "p0_ffn_out.f32"
        raw = file.read_bytes()
        file.unlink()
        with self.assertRaisesRegex(p.Blocked, "missing/extra"):
            self.witness()
        file.write_bytes(raw)
        extra = self.post / "p0_unknown.f32"
        extra.write_bytes(b"untrusted")
        with self.assertRaisesRegex(p.Blocked, "missing/extra"):
            self.witness()
        extra.unlink()
        context = self.base / "p0_context.txt"
        context.unlink()
        with self.assertRaisesRegex(p.Blocked, "missing/extra"):
            self.witness()

    def test_hidden_partial_symlink_nested_fails_closed(self):
        hidden = self.post / ".partial"
        hidden.write_bytes(b"stale")
        with self.assertRaisesRegex(p.Blocked, "missing/extra"):
            self.witness()
        hidden.unlink()
        nested = self.post / "nested"
        nested.mkdir()
        with self.assertRaisesRegex(p.Blocked, "missing/extra"):
            self.witness()
        nested.rmdir()
        victim = self.post / "p0_attn_post_norm.f32"
        victim.unlink()
        try:
            victim.symlink_to(self.base / "p0_attn_norm.f32")
        except OSError as error:
            # Unprivileged Windows without Developer Mode cannot create a
            # symlink (WinError 1314). This case is source-audited, never
            # misreported as a dynamic symlink rejection on that runner.
            if getattr(error, "winerror", None) != 1314:
                raise
            self.assertIn("p.is_symlink()", MODULE.read_text(encoding="utf-8"))
            return
        with self.assertRaisesRegex(p.Blocked, "symlink/nonregular"):
            self.witness()

    def test_truncated_nonfinite_post_and_corrupted_base_rejected(self):
        post = self.post / "p0_ffn_out.f32"
        original = post.read_bytes()
        for bad in (original[:-4], struct.pack("<f", float("nan")) + original[4:],
                    struct.pack("<f", float("inf")) + original[4:]):
            with self.subTest(length=len(bad)):
                post.write_bytes(bad)
                with self.assertRaisesRegex(p.Blocked, "5120 finite"):
                    self.witness()
        post.write_bytes(original)
        pre = self.base / "p0_attn_norm.f32"
        pre.write_bytes(b"invalid pretrace")
        with self.assertRaisesRegex(p.Blocked, "pre-FFN v2 raw tensor"):
            self.witness()

    def test_wrong_base_post_directory_pair_is_rejected(self):
        with self.assertRaisesRegex(p.Blocked, "distinct and paired"):
            p.validate_post_sidecars(self.post, self.base, synthetic_test=True)
        with self.assertRaisesRegex(p.Blocked, "distinct and paired"):
            p.validate_post_sidecars(self.base, self.base, synthetic_test=True)


if __name__ == "__main__":
    unittest.main()
