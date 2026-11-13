#!/usr/bin/env python3
"""Offline source contracts for one-off last-head seeded PP2 scratch patch."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "prefill_last_head", ROOT / "experiments" / "prefill_last_head_scratch_patch.py")
patcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patcher)


class LastHeadContracts(unittest.TestCase):
    def sources(self):
        return {rel: (ROOT / rel).read_text(encoding="utf-8")
                for rel in patcher.FILES}

    def test_exact_existing_real_source_contract(self):
        original = self.sources()
        patched = patcher.transform(original)
        self.assertEqual(set(patched), set(original))
        for name in patcher.FILES:
            self.assertNotEqual(patched[name], original[name], name)
            self.assertEqual(
                patched[name].count("HERMES_PP2_SEEDED_LAST_HEAD_ONLY"),
                1, name)
        self.assertIn("last_head_only?(size_t)VOC:T*(size_t)VOC)",
                      patched["distributed/pp2_fwd.cu"])
        self.assertIn("last_head_only?1:T",
                      patched["distributed/pp2_fwd.cu"])
        self.assertIn("std::vector<float> logits=prefill_impl("
                      "ids,st,err,h_last_out,&ds,true,0,last_head_only);",
                      patched["distributed/pp2_fwd.cu"])
        self.assertIn("logits.data() + logits.size()",
                      patched["serving/server.cpp"])

    def test_tampered_real_callsite_rejected(self):
        original = self.sources()
        original["distributed/pp2_batch.cu"] = original[
            "distributed/pp2_batch.cu"].replace(
                "full.size() != prompt.size() * VOC",
                "full.size() >= prompt.size() * VOC")
        with self.assertRaisesRegex(ValueError, "source contract"):
            patcher.transform(original)

    def test_already_patched_source_rejected(self):
        source = self.sources()
        changed = patcher.transform(source)
        with self.assertRaisesRegex(ValueError, "source contract"):
            patcher.transform(changed)


if __name__ == "__main__":
    unittest.main()
