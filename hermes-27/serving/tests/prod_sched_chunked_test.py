#!/usr/bin/env python3
"""CPU mock HTTP acceptance: legacy-only opt-in T65..8192 chunked seed."""
import json
import os
import subprocess
import unittest
import urllib.request

from prod_sched_state_test import (
    ARGS, events, payload, post, running_server, trace_lines,
)


class LegacyChunkedServingCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        cls.tokenizer = Tokenizer.from_file(ARGS.tokenizer)
        cls.output_id = cls.tokenizer.encode(
            "hello", add_special_tokens=False).ids[0]
        cls.output_text = cls.tokenizer.decode([cls.output_id])

    def prompt(self, size):
        text = " x" * size
        self.assertEqual(
            len(self.tokenizer.encode(text, add_special_tokens=False).ids),
            size)
        return text

    def request(self, size, max_tokens=4, *, stream=False):
        body = payload(max_tokens, stream=stream, include_usage=stream)
        body["messages"][0]["content"] = self.prompt(size)
        return body

    def test_T65_T129_one_seed_same_state_no_replay(self):
        for size in (65, 129):
            with self.subTest(T=size):
                with running_server(
                    ids=[self.output_id], continuous=False,
                    flags=("--chunked-prefill", "--active", "1"),
                    env_extra={"HERMES_PP2_CHUNKED_PREFILL": "1",
                               "FAKE_STEP_MS": "0"},
                ) as (url, _, trace):
                    status, raw = post(url, self.request(size))
                    self.assertEqual(status, 200, raw)
                    response = json.loads(raw)
                    self.assertEqual(response["usage"]["prompt_tokens"], size)
                    self.assertEqual(response["usage"]["completion_tokens"], 4)
                    self.assertEqual(
                        response["choices"][0]["message"]["content"],
                        self.tokenizer.decode([self.output_id] * 4))
                    lines = trace_lines(trace)
                    self.assertEqual(
                        [line for line in lines if line.startswith("chunked ")],
                        [f"chunked {size}"], lines)
                    self.assertEqual(
                        [line for line in lines if line.startswith("step ")],
                        [f"step {size+i}" for i in range(3)], lines)
                    self.assertEqual(sum(x.startswith("open ") for x in lines), 1)
                    self.assertEqual(lines.count("close"), 1)
                    self.assertFalse(any(x.startswith("prefill ") for x in lines))
                    with urllib.request.urlopen(url + "/metrics", timeout=10) as r:
                        metrics = r.read().decode()
                    self.assertIn("hermes_prefill_attempts_total 1\n", metrics)
                    self.assertIn("hermes_prefill_seeded_total 1\n", metrics)
                    self.assertIn(f"hermes_prefill_prompt_tokens_total {size}\n", metrics)

    def test_T64_uses_existing_one_shot_seed(self):
        with running_server(
            ids=[self.output_id], continuous=False,
            flags=("--chunked-prefill", "--active", "1"),
            env_extra={"HERMES_PP2_CHUNKED_PREFILL": "1",
                       "FAKE_STEP_MS": "0"},
        ) as (url, _, trace):
            status, raw = post(url, self.request(64, max_tokens=2))
            self.assertEqual(status, 200, raw)
            lines = trace_lines(trace)
            self.assertIn("prefill 64", lines)
            self.assertIn("step 64", lines)
            self.assertFalse(any(x.startswith("chunked ") for x in lines))
            self.assertEqual(lines.count("close"), 1)

    def test_T129_sse_usage_and_state_owner(self):
        with running_server(
            ids=[self.output_id], continuous=False,
            flags=("--chunked-prefill", "--active", "1"),
            env_extra={"HERMES_PP2_CHUNKED_PREFILL": "1",
                       "FAKE_STEP_MS": "0"},
        ) as (url, _, trace):
            status, raw = post(url, self.request(129, stream=True))
            self.assertEqual(status, 200, raw)
            text, reasons, usages, done = events(raw)
            self.assertEqual(text, self.tokenizer.decode([self.output_id] * 4))
            self.assertEqual(reasons, ["length"])
            self.assertEqual(len(usages), 1)
            self.assertEqual(usages[0]["prompt_tokens"], 129)
            self.assertEqual(usages[0]["completion_tokens"], 4)
            self.assertEqual(usages[0]["total_tokens"], 133)
            self.assertEqual(done, 1)
            lines = trace_lines(trace)
            self.assertEqual(lines.count("chunked 129"), 1)
            self.assertEqual(lines.count("close"), 1)
            self.assertEqual([x for x in lines if x.startswith("step ")],
                             ["step 129", "step 130", "step 131"])

    def test_T129_without_flag_remains_stepwise_even_if_env_enabled(self):
        with running_server(
            ids=[self.output_id], continuous=False,
            flags=("--active", "1"),
            env_extra={"HERMES_PP2_CHUNKED_PREFILL": "1",
                       "FAKE_STEP_MS": "0"},
        ) as (url, _, trace):
            status, raw = post(url, self.request(129, max_tokens=2))
            self.assertEqual(status, 200, raw)
            lines = trace_lines(trace)
            self.assertFalse(any(x.startswith("chunked ") for x in lines))
            self.assertFalse(any(x.startswith("prefill ") for x in lines))
            self.assertEqual([x for x in lines if x.startswith("step ")],
                             [f"step {i}" for i in range(130)])
            self.assertEqual(lines.count("close"), 1)

    def test_T8192_one_output_no_invalid_continuation(self):
        with running_server(
            ids=[self.output_id], continuous=False,
            flags=("--chunked-prefill", "--active", "1",
                   "--context", "8192"),
            env_extra={"HERMES_PP2_CHUNKED_PREFILL": "1",
                       "FAKE_STEP_MS": "0"},
        ) as (url, _, trace):
            status, raw = post(url, self.request(8192, max_tokens=1))
            self.assertEqual(status, 200, raw)
            obj = json.loads(raw)
            self.assertEqual(obj["usage"]["prompt_tokens"], 8192)
            self.assertEqual(obj["usage"]["completion_tokens"], 1)
            lines = trace_lines(trace)
            self.assertIn("chunked 8192", lines)
            self.assertFalse(any(x.startswith("step ") for x in lines), lines)
            self.assertEqual(lines.count("close"), 1)

    def test_partial_chunk_error_and_nonfinite_never_replay(self):
        for fail_flag in ("FAKE_CHUNKED_FAIL", "FAKE_CHUNKED_NONFINITE"):
            with self.subTest(failure=fail_flag):
                with running_server(
                    ids=[self.output_id], continuous=False,
                    flags=("--chunked-prefill", "--active", "1"),
                    env_extra={"HERMES_PP2_CHUNKED_PREFILL": "1",
                               fail_flag: "1", "FAKE_STEP_MS": "0"},
                ) as (url, _, trace):
                    status, raw = post(url, self.request(65))
                    self.assertIn(status, (500, 503), raw)
                    lines = trace_lines(trace)
                    self.assertEqual(lines.count("chunked 65"), 1)
                    self.assertEqual(lines.count("close"), 1)
                    self.assertFalse(any(x.startswith("step ") for x in lines), lines)

    def test_invalid_startup_fails_before_model_upload(self):
        scenarios = [
            (("--chunked-prefill", "--active", "1"), {}),
            (("--chunked-prefill", "--active", "2"),
             {"HERMES_PP2_CHUNKED_PREFILL": "1"}),
            (("--chunked-prefill", "--active", "1",
              "--continuous-scheduler"),
             {"HERMES_PP2_CHUNKED_PREFILL": "1"}),
            (("--chunked-prefill", "--active", "1"),
             {"HERMES_PP2_CHUNKED_PREFILL": "1",
              "HERMES_NVFP4_GEMV": "packed4"}),
            (("--chunked-prefill", "--active", "1"),
             {"HERMES_PP2_CHUNKED_PREFILL": "1",
              "HERMES_L26": "1"}),
            (("--chunked-prefill", "--active", "1"),
             {"HERMES_PP2_CHUNKED_PREFILL": "1",
              "HERMES_PP2_ASYNC_STREAMS": "1"}),
            (("--chunked-prefill", "--active", "1"),
             {"HERMES_PP2_CHUNKED_PREFILL": "1",
              "HERMES_PP2_PREFILL_BF16": "1"}),
        ]
        for flags, env_extra in scenarios:
            with self.subTest(flags=flags, env=env_extra):
                env = os.environ.copy()
                for var in ("HERMES_PP2_CHUNKED_PREFILL", "HERMES_L26",
                            "HERMES_NVFP4_GEMV", "HERMES_PP2_ASYNC_STREAMS",
                            "HERMES_PP2_PREFILL_BF16"):
                    env.pop(var, None)
                env.update(env_extra)
                run = subprocess.run(
                    [ARGS.server, *flags], env=env,
                    capture_output=True, text=True, timeout=8)
                self.assertEqual(run.returncode, 2, run.stderr)
                self.assertTrue(
                    "chunked" in run.stderr.lower() or
                    "bf16" in run.stderr.lower(), run.stderr)


if __name__ == "__main__":
    # prod_sched_state_test consumes all application --server/--model args.
    unittest.main(argv=[__file__], verbosity=2)
