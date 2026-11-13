#!/usr/bin/env python3
"""CPU-only HTTP acceptance of explicit --parallel-steps scheduler admission.

Uses the real server.cpp and pp2_batch.cu with the deterministic CPU PP2
mock; no CUDA, model upload, live services or network beyond localhost.
GPU throughput and backend concurrency are NOT certified by this test.
"""
import concurrent.futures
import json
import os
import subprocess
import sys
import unittest
import urllib.request

from prod_sched_state_test import (
    ARGS, events, payload, post, running_server, trace_lines,
    assert_single_owner,
)


class ParallelServingCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        cls.tok = Tokenizer.from_file(ARGS.tokenizer)
        cls.out_id = cls.tok.encode("hello", add_special_tokens=False).ids[0]
        cls.content = cls.tok.decode([cls.out_id] * 4)

    def test_parallel_rejected_before_any_model_upload_without_contract(self):
        scenarios = (
            (("--parallel-steps",), {"HERMES_PP2_ASYNC_STREAMS": "1"}),
            (("--continuous-scheduler", "--parallel-steps"), {}),
            (("--continuous-scheduler", "--parallel-steps", "--batched-prefill"),
             {"HERMES_PP2_ASYNC_STREAMS": "1"}),
            (("--continuous-scheduler", "--parallel-steps"),
             {"HERMES_PP2_ASYNC_STREAMS": "0"}),
        )
        for flags, overrides in scenarios:
            with self.subTest(flags=flags, env=overrides):
                environment = os.environ.copy()
                environment.pop("HERMES_PP2_ASYNC_STREAMS", None)
                environment.update(overrides)
                result = subprocess.run(
                    [ARGS.server, *flags], env=environment,
                    capture_output=True, text=True, timeout=5,
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("--parallel-steps requires", result.stderr)
                self.assertNotIn("pp2 upload", result.stderr)

    def metric(self, url, name):
        with urllib.request.urlopen(url + "/metrics", timeout=10) as res:
            body = res.read().decode("utf-8")
        rows = [line for line in body.splitlines()
                if line.startswith(name + " ")]
        self.assertEqual(len(rows), 1, body)
        return int(rows[0].split()[1])

    def request_round(self, flags):
        with running_server(
            ids=[self.out_id], flags=flags, continuous=True,
            env_extra={"HERMES_PP2_ASYNC_STREAMS": "1", "FAKE_STEP_MS": "15"},
        ) as (url, _, trace):
            req = payload(4)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                responses = list(pool.map(lambda _: post(url, req), range(4)))
            parsed = []
            for status, raw in responses:
                self.assertEqual(status, 200, raw)
                body = json.loads(raw)
                self.assertEqual(body["choices"][0]["finish_reason"], "length")
                self.assertEqual(body["choices"][0]["message"]["content"], self.content)
                self.assertEqual(body["usage"]["completion_tokens"], 4)
                parsed.append(body)
            ids = [record["id"] for record in parsed]
            self.assertEqual(len(set(ids)), 4, ids)
            expected_tokens = parsed[0]["usage"]
            self.assertTrue(all(rec["usage"] == expected_tokens for rec in parsed))

            # Both SSE and nonstream must use the same decoder state,
            # usage, finish reason and Unicode handling.
            status, raw = post(url, payload(4, stream=True, include_usage=True))
            self.assertEqual(status, 200, raw)
            text, reasons, usage, done = events(raw)
            self.assertEqual(text, self.content)
            self.assertEqual(reasons, ["length"])
            self.assertEqual(done, 1)
            self.assertEqual(usage, [expected_tokens])

            # The server mock writes open, step and close exactly once per
            # request; execution order between different states may vary.
            lines = trace_lines(trace)
            assert_single_owner(self, lines, requests=5)
            self.assertEqual(sum(line.startswith("step ") for line in lines),
                             5 * (expected_tokens["prompt_tokens"] + 4 - 1))
            waves = self.metric(url, "hermes_parallel_waves_total")
            dispatched = self.metric(url, "hermes_parallel_dispatched_steps_total")
            if flags:
                self.assertGreater(waves, 0)
                self.assertGreaterEqual(dispatched, 5)
            else:
                self.assertEqual(waves, 0)
                self.assertEqual(dispatched, 0)
            return [rec["choices"][0]["message"]["content"] for rec in parsed], expected_tokens

    def test_parallel_optin_preserves_http_state_and_sse(self):
        serial_responses, serial_usage = self.request_round(())
        parallel_responses, parallel_usage = self.request_round(("--parallel-steps",))
        self.assertEqual(serial_usage, parallel_usage)
        self.assertEqual(serial_responses, parallel_responses)


if __name__ == "__main__":
    # The shared state-test helper intentionally owns the custom
    # --server/--model/--gguf-dump/--tokenizer/--python arguments.
    # Do not let unittest parse those application-specific switches.
    unittest.main(argv=[sys.argv[0]])
