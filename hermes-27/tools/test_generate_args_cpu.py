#!/usr/bin/env python3
"""CPU-only fail-closed CLI argument regression; never starts a GGUF upload."""

from __future__ import annotations

import subprocess
import sys
import unittest


class GenerateArgsCPU(unittest.TestCase):
    binary = ""

    def invoke(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.binary, *args], check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=8,
        )

    def test_help_documents_explicit_opt_in(self) -> None:
        result = self.invoke("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--batched-prefill", result.stdout)
        self.assertIn("T<=64", result.stdout)

    def test_batched_prefill_rejects_65_before_loading(self) -> None:
        result = self.invoke(
            "--model", "/definitely/not/a/model.gguf",
            "--prompt-ids", ",".join(["0"] * 65),
            "--batched-prefill", "--max-new-tokens", "1",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("batched state prefill limit 64", result.stderr)
        self.assertNotIn("upload failed", result.stderr)

    def test_seed_context_capacity_rejects_overflow(self) -> None:
        result = self.invoke(
            "--model", "/definitely/not/a/model.gguf",
            "--prompt-ids", "0,1",
            "--batched-prefill", "--max-new-tokens", "8192",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("context limit 8192", result.stderr)

    def test_invalid_token_and_generation_count(self) -> None:
        invalid = self.invoke(
            "--model", "/definitely/not/a/model.gguf",
            "--prompt-ids", "248320", "--batched-prefill",
        )
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("invalid token IDs", invalid.stderr)
        zero = self.invoke(
            "--model", "/definitely/not/a/model.gguf",
            "--prompt-ids", "1", "--max-new-tokens", "0",
        )
        self.assertEqual(zero.returncode, 2)
        self.assertIn("must be 1..8192", zero.stderr)

    def test_no_implicit_model_or_tokenizer(self) -> None:
        result = self.invoke("--batched-prefill", "--prompt-ids", "1")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--model and at least one --prompt-ids", result.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_generate_args_cpu.py /path/to/hermes_generate")
    GenerateArgsCPU.binary = sys.argv[1]
    sys.argv = [sys.argv[0]]
    unittest.main(verbosity=2)
