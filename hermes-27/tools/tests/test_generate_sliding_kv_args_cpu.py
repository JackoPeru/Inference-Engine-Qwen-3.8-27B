#!/usr/bin/env python3
"""Black-box CPU contract for the opt-in PP2 sliding CLI; never uses GPUs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class GenerateSlidingArgsCPU(unittest.TestCase):
    binary: Path

    def invoke(self, *args: str, env_overrides: dict[str, str] | None = None,
               timeout: int = 90) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        for key in (
            "HERMES_PP2_SLIDING_KV", "HERMES_NVFP4_GEMV",
            "HERMES_PP2_PREFILL_BF16", "HERMES_L26",
            "HERMES_TEST_FAIL_STEP", "HERMES_TEST_BAD_LOGITS",
        ):
            env.pop(key, None)
        env.update(env_overrides or {})
        return subprocess.run(
            [str(self.binary), *args], env=env, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
        )

    def base(self, *args: str) -> tuple[str, ...]:
        return ("--model", "/mock/model.gguf", "--prompt-ids", "42",
                "--eos-token", "1", *args)

    def assert_preupload_rejection(self, result: subprocess.CompletedProcess[str],
                                   message: str) -> None:
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(message, result.stderr)
        self.assertNotIn("MOCK_UPLOAD", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_help_documents_mode_and_limit(self) -> None:
        result = self.invoke("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--sliding-kv", result.stdout)
        self.assertIn("HERMES_PP2_SLIDING_KV=1", result.stdout)
        self.assertIn("8192", result.stdout)

    def test_default_capacity_and_context_limit_unchanged(self) -> None:
        ok = self.invoke(*self.base("--max-new-tokens", "2"))
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertIn("MOCK_INIT_CAPACITY=2 SLIDING=0", ok.stderr)
        self.assertEqual(json.loads(ok.stdout)["generated_token_ids"], [42, 42])
        bad = self.invoke(
            "--model", "/mock/model.gguf", "--prompt-ids", "42,42",
            "--eos-token", "1", "--max-new-tokens", "8192")
        self.assert_preupload_rejection(bad, "context limit 8192")

    def test_sliding_requires_exact_backend_optin(self) -> None:
        for env_value in (None, "0", "true", "01"):
            with self.subTest(value=env_value):
                kwargs = {} if env_value is None else {
                    "env_overrides": {"HERMES_PP2_SLIDING_KV": env_value}
                }
                result = self.invoke(*self.base("--sliding-kv"), **kwargs)
                self.assert_preupload_rejection(
                    result, "HERMES_PP2_SLIDING_KV=1")

    def test_backend_env_alone_cannot_implicitly_enable_sliding(self) -> None:
        result = self.invoke(
            *self.base("--max-new-tokens", "2"),
            env_overrides={"HERMES_PP2_SLIDING_KV": "1"})
        self.assert_preupload_rejection(result, "requires explicit --sliding-kv")

    def test_sliding_rejects_other_prefill_paths(self) -> None:
        env = {"HERMES_PP2_SLIDING_KV": "1"}
        for flag in ("--verify-prefill", "--batched-prefill"):
            with self.subTest(flag=flag):
                result = self.invoke(*self.base("--sliding-kv", flag),
                                     env_overrides=env)
                self.assert_preupload_rejection(result, "cannot combine")

    def test_sliding_rejects_experimental_environments(self) -> None:
        for name, value in (
            ("HERMES_NVFP4_GEMV", "packed4"),
            ("HERMES_PP2_PREFILL_BF16", "1"),
            ("HERMES_L26", "0"),
        ):
            with self.subTest(name=name):
                result = self.invoke(
                    *self.base("--sliding-kv"),
                    env_overrides={"HERMES_PP2_SLIDING_KV": "1", name: value})
                self.assert_preupload_rejection(result, "incompatible")

    def test_sliding_rejects_prompt_and_count_overflow(self) -> None:
        env = {"HERMES_PP2_SLIDING_KV": "1"}
        for count in ("0", "8193", "-1"):
            with self.subTest(max_new=count):
                result = self.invoke(
                    *self.base("--sliding-kv", "--max-new-tokens", count),
                    env_overrides=env)
                self.assert_preupload_rejection(result, "must be 1..8192")
        result = self.invoke(
            "--model", "/mock/model.gguf", "--prompt-ids",
            ",".join(["42"] * 8193), "--max-new-tokens", "2",
            "--sliding-kv", env_overrides=env)
        self.assert_preupload_rejection(result, "prompt 0 exceeds")

    def test_ring_boundary_is_real_absolute_decode_call(self) -> None:
        result = self.invoke(
            "--model", "/mock/model.gguf", "--prompt-ids",
            ",".join(["42"] * 8192), "--max-new-tokens", "2",
            "--eos-token", "1", "--sliding-kv",
            env_overrides={"HERMES_PP2_SLIDING_KV": "1"})
        self.assertEqual(result.returncode, 0, result.stderr[-1500:])
        parsed = json.loads(result.stdout)
        self.assertEqual(len(parsed["prompt_ids"]), 8192)
        self.assertEqual(parsed["generated_token_ids"], [42, 42])
        self.assertIn("MOCK_INIT_CAPACITY=8192 SLIDING=1", result.stderr)
        self.assertIn("MOCK_SEED_COMPLETE=8192", result.stderr)
        self.assertIn("MOCK_POS=8192", result.stderr)
        self.assertIn("MOCK_DECODE_FREE steps=8193", result.stderr)
        self.assertEqual(result.stderr.count("MOCK_DECODE_FREE"), 1)
        self.assertEqual(result.stderr.count("MOCK_FREE_ALL"), 1)
        self.assertNotIn("MOCK_UNEXPECTED_", result.stderr)

    def test_multiple_prompts_get_fresh_state(self) -> None:
        result = self.invoke(
            "--model", "/mock/model.gguf", "--prompt-ids", "42",
            "--prompt-ids", "42,42", "--max-new-tokens", "2",
            "--eos-token", "1", "--sliding-kv",
            env_overrides={"HERMES_PP2_SLIDING_KV": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(result.stdout.splitlines()), 2)
        self.assertEqual(result.stderr.count("MOCK_INIT_CAPACITY=8192 SLIDING=1"), 2)
        self.assertEqual(result.stderr.count("MOCK_DECODE_FREE"), 2)
        self.assertEqual(result.stderr.count("MOCK_FREE_ALL"), 1)
        self.assertNotIn("MOCK_BAD_POSITION", result.stderr)

    def test_decode_failure_closes_once_without_json_success(self) -> None:
        for env_error, marker in (
            ("HERMES_TEST_FAIL_STEP", "MOCK_INJECTED_STEP_FAILURE"),
            ("HERMES_TEST_BAD_LOGITS", "greedy selection failed"),
        ):
            with self.subTest(error=env_error):
                result = self.invoke(
                    "--model", "/mock/model.gguf", "--prompt-ids",
                    ",".join(["42"] * 8192), "--max-new-tokens", "2",
                    "--eos-token", "1", "--sliding-kv",
                    env_overrides={
                        "HERMES_PP2_SLIDING_KV": "1", env_error:
                            "8192" if env_error == "HERMES_TEST_FAIL_STEP" else "1",
                    })
                self.assertEqual(result.returncode, 1, result.stderr[-1500:])
                self.assertIn(marker, result.stderr)
                self.assertEqual(result.stderr.count("MOCK_DECODE_FREE"), 1)
                self.assertEqual(result.stderr.count("MOCK_FREE_ALL"), 1)
                self.assertNotIn("MOCK_DOUBLE_FREE", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_seed_failure_closes_once_without_partial_result(self) -> None:
        result = self.invoke(
            *self.base("--max-new-tokens", "2", "--sliding-kv"),
            env_overrides={"HERMES_PP2_SLIDING_KV": "1",
                           "HERMES_TEST_FAIL_STEP": "0"})
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("token-by-token seed failed at pos 0", result.stderr)
        self.assertEqual(result.stderr.count("MOCK_DECODE_FREE"), 1)
        self.assertEqual(result.stderr.count("MOCK_FREE_ALL"), 1)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_generate_sliding_kv_args_cpu.py MOCK_GENERATE")
    GenerateSlidingArgsCPU.binary = Path(sys.argv[1]).resolve(strict=True)
    sys.argv = [sys.argv[0]]
    unittest.main(verbosity=2)
