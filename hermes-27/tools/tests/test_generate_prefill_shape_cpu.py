#!/usr/bin/env python3
"""CPU-only fake PP2 shape/cardinality contract; no model or GPU inference."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


class FullPrefillShape(unittest.TestCase):
    binary: Path

    def invoke(self, shape: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["HERMES_TEST_PREFILL_SHAPE"] = shape
        return subprocess.run([
            str(self.binary), "--model", "/no-real-model-loaded.gguf",
            "--prompt-ids", "6205,161421", "--max-new-tokens", "1",
            "--eos-token", "2", "--verify-prefill",
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env=env, timeout=8, check=False)

    def test_short_nonempty_prefill_fails_before_decode(self):
        result = self.invoke("short")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("unexpected full-prefill logits count", result.stderr)
        self.assertNotIn("UNEXPECTED_STATEFUL_SEED", result.stderr)
        self.assertNotIn("UNEXPECTED_DECODE_STEP", result.stderr)

    def test_extra_prefill_fails_before_decode(self):
        result = self.invoke("extra")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("unexpected full-prefill logits count", result.stderr)

    def test_empty_prefill_fails_before_decode(self):
        result = self.invoke("empty")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("full prefill failed", result.stderr)

    def test_exact_prefill_reaches_existing_state_guard(self):
        result = self.invoke("exact")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("decode init failed", result.stderr)
        self.assertNotIn("unexpected full-prefill logits count", result.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_generate_prefill_shape_cpu.py MOCK_HERMES_GENERATE")
    FullPrefillShape.binary = Path(sys.argv[1]).resolve(strict=True)
    sys.argv = [sys.argv[0]]
    unittest.main(verbosity=2)
