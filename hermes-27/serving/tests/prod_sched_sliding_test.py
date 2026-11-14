#!/usr/bin/env python3
"""CPU/mock-only opt-in rolling-KV HTTP boundary and ownership tests.

The fake backend records positions and allocation sizes but does not compute
sliding CUDA attention. No inference-quality or performance claim follows.
"""
import json
import os
import subprocess
import sys
import unittest

from prod_sched_state_test import ARGS, events, payload, post, running_server, trace_lines


class SlidingServingCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        cls.tokenizer = Tokenizer.from_file(ARGS.tokenizer)
        cls.output_id = cls.tokenizer.encode("hello", add_special_tokens=False).ids[0]

    def request(self, size, count=3, stream=False):
        text = " x" * size
        self.assertEqual(len(self.tokenizer.encode(text, add_special_tokens=False).ids), size)
        body = payload(count, stream=stream, include_usage=stream)
        body["messages"][0]["content"] = text
        return body

    def flags(self):
        return ("--sliding-kv", "--chunked-prefill", "--active", "1",
                "--context", "8192")

    def env(self):
        return {"HERMES_PP2_SLIDING_KV": "1",
                "HERMES_PP2_CHUNKED_PREFILL": "1", "FAKE_STEP_MS": "0"}

    def test_exact_T8192_plus_three_continues_on_one_state(self):
        with running_server(ids=[self.output_id], continuous=False,
                            flags=self.flags(), env_extra=self.env()) as (url, _, trace):
            status, raw = post(url, self.request(8192, 3))
            self.assertEqual(status, 200, raw)
            result = json.loads(raw)
            self.assertEqual(result["usage"], {"prompt_tokens": 8192,
                "completion_tokens": 3, "total_tokens": 8195})
            self.assertEqual(result["choices"][0]["message"]["content"],
                             self.tokenizer.decode([self.output_id] * 3))
            lines = trace_lines(trace)
            self.assertEqual([x for x in lines if x.startswith("open ")], ["open 8192"])
            self.assertEqual([x for x in lines if x.startswith("chunked ")], ["chunked 8192"])
            self.assertEqual([x for x in lines if x.startswith("step ")],
                             ["step 8192", "step 8193"])
            self.assertEqual(lines.count("close"), 1)

    def test_sse_wrap_preserves_usage_and_state(self):
        with running_server(ids=[self.output_id], continuous=False,
                            flags=self.flags(), env_extra=self.env()) as (url, _, trace):
            status, raw = post(url, self.request(8192, 2, stream=True))
            self.assertEqual(status, 200, raw)
            text, reasons, usages, done = events(raw)
            self.assertEqual(text, self.tokenizer.decode([self.output_id] * 2))
            self.assertEqual(reasons, ["length"])
            self.assertEqual(usages, [{"prompt_tokens": 8192,
                "completion_tokens": 2, "total_tokens": 8194}])
            self.assertEqual(done, 1)
            lines = trace_lines(trace)
            self.assertEqual(lines.count("chunked 8192"), 1)
            self.assertEqual([x for x in lines if x.startswith("step ")], ["step 8192"])
            self.assertEqual(lines.count("close"), 1)

    def test_short_prompt_under_flag_still_reserves_physical_8192(self):
        with running_server(ids=[self.output_id], continuous=False,
                            flags=self.flags(), env_extra=self.env()) as (url, _, trace):
            status, raw = post(url, self.request(129, 2))
            self.assertEqual(status, 200, raw)
            lines = trace_lines(trace)
            self.assertIn("open 8192", lines)
            self.assertEqual(lines.count("chunked 129"), 1)
            self.assertEqual([x for x in lines if x.startswith("step ")], ["step 129"])
            self.assertEqual(lines.count("close"), 1)

    def test_no_sliding_flag_still_rejects_T8192_continuation(self):
        with running_server(ids=[self.output_id], continuous=False,
                            flags=("--chunked-prefill", "--active", "1",
                                   "--context", "8192"),
                            env_extra={"HERMES_PP2_CHUNKED_PREFILL": "1"}) as (url, _, trace):
            status, raw = post(url, self.request(8192, 2))
            self.assertEqual(status, 400, raw)
            self.assertEqual(json.loads(raw)["error"]["code"], "context_length_exceeded")
            self.assertEqual(trace_lines(trace), [])

    def test_startup_rejects_incomplete_or_unsafe_combination(self):
        scenarios = (
            (("--sliding-kv",), {"HERMES_PP2_SLIDING_KV": "1"}),
            (("--sliding-kv", "--chunked-prefill", "--active", "1",
              "--context", "8192"), {"HERMES_PP2_CHUNKED_PREFILL": "1"}),
            (("--chunked-prefill", "--active", "1", "--context", "8192"),
             {"HERMES_PP2_CHUNKED_PREFILL": "1", "HERMES_PP2_SLIDING_KV": "1"}),
            (("--sliding-kv", "--chunked-prefill", "--active", "2",
              "--context", "8192"),
             {"HERMES_PP2_CHUNKED_PREFILL": "1", "HERMES_PP2_SLIDING_KV": "1"}),
            (("--sliding-kv", "--chunked-prefill", "--active", "1",
              "--context", "128"),
             {"HERMES_PP2_CHUNKED_PREFILL": "1", "HERMES_PP2_SLIDING_KV": "1"}),
            (("--sliding-kv", "--chunked-prefill", "--active", "1",
              "--context", "8192", "--continuous-scheduler"),
             {"HERMES_PP2_CHUNKED_PREFILL": "1", "HERMES_PP2_SLIDING_KV": "1"}),
        )
        for flags, updates in scenarios:
            with self.subTest(flags=flags):
                env = os.environ.copy()
                for name in ("HERMES_PP2_SLIDING_KV", "HERMES_PP2_CHUNKED_PREFILL",
                             "HERMES_PP2_ASYNC_STREAMS", "HERMES_NVFP4_GEMV"):
                    env.pop(name, None)
                env.update(updates)
                proc = subprocess.run([ARGS.server, *flags], env=env,
                                      capture_output=True, text=True, timeout=5)
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertTrue("sliding" in proc.stderr.lower() or
                                "chunked" in proc.stderr.lower(), proc.stderr)
                self.assertNotIn("pp2 upload", proc.stderr)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
