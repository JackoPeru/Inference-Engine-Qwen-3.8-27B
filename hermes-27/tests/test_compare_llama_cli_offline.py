#!/usr/bin/env python3
"""CPU-only fail-closed CLI contract for the independent llama token gate.

This test never starts a model, opens an HTTP connection, or uses a GPU.
"""

from __future__ import annotations

import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("compare_llama.py")


class IndependentGateCliOffline(unittest.TestCase):
    def run_gate(self, *argv: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *argv],
            text=True, capture_output=True, check=False, timeout=5,
        )

    def test_fixture_labels_match_actual_token_lengths(self) -> None:
        fixtures = runpy.run_path(str(SCRIPT), run_name="h27_fixture_inspection")["CASES"]
        self.assertEqual(len(fixtures), 4)
        self.assertEqual(len({case.name for case in fixtures}), len(fixtures))
        for case in fixtures:
            self.assertEqual(int(case.name.rsplit("-", 1)[-1]), len(case.ids))
            self.assertTrue(all(type(token) is int and token >= 0 for token in case.ids))

    def test_missing_gguf_returns_explicit_blocked_exit_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "no-model.gguf")
            result = self.run_gate("--gguf", missing, "--engine", missing)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("INCONCLUSIVE / BLOCKED:", result.stderr)
        self.assertIn("required GGUF or executable is missing", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_missing_engine_fails_before_llama_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gguf = Path(tmp) / "small-placeholder.gguf"
            gguf.write_bytes(b"not-a-real-model")
            result = self.run_gate(
                "--gguf", str(gguf), "--engine", str(Path(tmp) / "absent"),
                "--llama-url", "http://127.0.0.1:1",
            )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("INCONCLUSIVE / BLOCKED:", result.stderr)

    def test_malformed_url_is_rejected_before_any_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gguf = Path(tmp) / "model.gguf"
            engine = Path(tmp) / "engine"
            gguf.write_bytes(b"fixture")
            engine.write_bytes(b"fixture")
            result = self.run_gate(
                "--gguf", str(gguf), "--engine", str(engine),
                "--llama-url", "not-a-url",
            )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("--llama-url must start", result.stderr)

    def test_invalid_generation_arguments_fail_before_loading(self) -> None:
        for args in (("--tokens", "0"), ("--tokens", "33"),
                     ("--timeout", "0")):
            with self.subTest(args=args):
                result = self.run_gate("--gguf", "unused", "--engine", "unused", *args)
                self.assertEqual(result.returncode, 2)
                self.assertIn("--tokens must be 1..32", result.stderr)


if __name__ == "__main__":
    unittest.main()
