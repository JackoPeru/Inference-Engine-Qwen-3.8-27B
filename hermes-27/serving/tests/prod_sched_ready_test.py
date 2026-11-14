#!/usr/bin/env python3
"""CPU-only HTTP contract for strictly opt-in ready-completion dispatch.

Actual server.cpp and ContinuousScheduler, fake PP2 backend. No CUDA model,
live network endpoint, GPU resource or actual throughput measurement.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
import unittest

from prod_sched_state_test import (
    ARGS, assert_single_owner, events, payload, post, running_server,
    trace_lines,
)


class ReadyCompletionServingCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(ARGS.tokenizer)
        cls.out_id = tok.encode("hello", add_special_tokens=False).ids[0]
        cls.expected = tok.decode([cls.out_id] * 4)

    def test_flag_requires_parallel_and_valid_async_preflight(self):
        cases = (
            (("--ready-completion",), {}),
            (("--continuous-scheduler", "--ready-completion"),
             {"HERMES_PP2_ASYNC_STREAMS": "1"}),
            (("--parallel-steps", "--ready-completion"),
             {"HERMES_PP2_ASYNC_STREAMS": "1"}),
            (("--continuous-scheduler", "--parallel-steps",
              "--ready-completion"), {}),
            (("--continuous-scheduler", "--parallel-steps",
              "--ready-completion", "--batched-prefill"),
             {"HERMES_PP2_ASYNC_STREAMS": "1"}),
            (("--continuous-scheduler", "--parallel-steps",
              "--ready-completion", "--chunked-prefill"),
             {"HERMES_PP2_ASYNC_STREAMS": "1",
              "HERMES_PP2_CHUNKED_PREFILL": "1"}),
        )
        for flags, updates in cases:
            with self.subTest(flags=flags):
                env = os.environ.copy()
                for name in ("HERMES_PP2_ASYNC_STREAMS",
                             "HERMES_PP2_CHUNKED_PREFILL"):
                    env.pop(name, None)
                env.update(updates)
                run = subprocess.run(
                    [ARGS.server, *flags], env=env, capture_output=True,
                    text=True, check=False, timeout=5,
                )
                self.assertEqual(run.returncode, 2, run.stderr)
                self.assertTrue("--ready-completion requires" in run.stderr or
                                "--parallel-steps requires" in run.stderr,
                                run.stderr)
                self.assertNotIn("pp2 upload", run.stderr)

    def request_round(self, ready):
        flags = ("--parallel-steps", "--ready-completion") if ready else (
            "--parallel-steps",)
        with running_server(
            ids=[self.out_id], continuous=True, flags=flags,
            env_extra={"HERMES_PP2_ASYNC_STREAMS": "1",
                       "FAKE_STEP_MS": "12"},
        ) as (url, _process, trace):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                responses = list(pool.map(lambda _:
                    post(url, payload(4)), range(4)))
            bodies = []
            for status, raw in responses:
                self.assertEqual(status, 200, raw)
                response = json.loads(raw)
                self.assertEqual(response["choices"][0]["finish_reason"], "length")
                self.assertEqual(response["choices"][0]["message"]["content"],
                                 self.expected)
                self.assertEqual(response["usage"]["completion_tokens"], 4)
                bodies.append(response)
            self.assertEqual(len({item["id"] for item in bodies}), 4)
            status, raw = post(url, payload(4, stream=True, include_usage=True))
            self.assertEqual(status, 200, raw)
            visible, reasons, usage, done = events(raw)
            self.assertEqual(visible, self.expected)
            self.assertEqual(reasons, ["length"])
            self.assertEqual(usage, [bodies[0]["usage"]])
            self.assertEqual(done, 1)
            lines = trace_lines(trace)
            assert_single_owner(self, lines, requests=5)
            self.assertEqual(sum(line.startswith("step ") for line in lines),
                             5 * (bodies[0]["usage"]["prompt_tokens"] + 4 - 1))
            return [item["choices"][0]["message"]["content"]
                    for item in bodies], bodies[0]["usage"]

    def test_ready_http_nonstream_and_sse_preserve_existing_contract(self):
        text_default, usage_default = self.request_round(False)
        text_ready, usage_ready = self.request_round(True)
        self.assertEqual(text_ready, text_default)
        self.assertEqual(usage_ready, usage_default)

    def test_terminal_oom_mock_nonstream_unblocks_and_next_job_survives(self):
        with running_server(
            ids=[self.out_id], continuous=True,
            flags=("--parallel-steps", "--ready-completion"),
            env_extra={"HERMES_PP2_ASYNC_STREAMS": "1",
                       "HERMES_TEST_FAIL_COMPLETE_ONCE": "1"},
        ) as (url, _, trace):
            status, raw = post(url, payload(2))
            self.assertEqual(status, 500, raw)
            self.assertIn("error", json.loads(raw))
            status, raw = post(url, payload(2))
            self.assertEqual(status, 200, raw)
            self.assertEqual(json.loads(raw)["usage"]["completion_tokens"], 2)
            assert_single_owner(self, trace_lines(trace), requests=2)

    def test_terminal_oom_mock_sse_unblocks_client(self):
        with running_server(
            ids=[self.out_id], continuous=True,
            flags=("--parallel-steps", "--ready-completion"),
            env_extra={"HERMES_PP2_ASYNC_STREAMS": "1",
                       "HERMES_TEST_FAIL_COMPLETE_ONCE": "1"},
        ) as (url, _, trace):
            # Once SSE headers are sent a terminal failure cannot retroactively
            # return an HTTP 500. It must still wake the streaming consumer.
            status, raw = post(url, payload(2, stream=True, include_usage=True))
            self.assertEqual(status, 200, raw)
            self.assertEqual(raw.count(b"data: [DONE]"), 1)
            status, raw = post(url, payload(2))
            self.assertEqual(status, 200, raw)
            self.assertEqual(json.loads(raw)["usage"]["completion_tokens"], 2)
            assert_single_owner(self, trace_lines(trace), requests=2)

    def test_terminal_callback_exception_nonstream_wakes_without_replay(self):
        with running_server(
            ids=[self.out_id], continuous=True,
            flags=("--parallel-steps", "--ready-completion"),
            env_extra={"HERMES_PP2_ASYNC_STREAMS": "1",
                       "HERMES_TEST_FAIL_TERMINAL_EVENT_ONCE": "1"},
        ) as (url, _, trace):
            status, raw = post(url, payload(2))
            self.assertEqual(status, 500, raw)
            self.assertIn("error", json.loads(raw))
            status, raw = post(url, payload(2))
            self.assertEqual(status, 200, raw)
            self.assertEqual(json.loads(raw)["usage"]["completion_tokens"], 2)
            assert_single_owner(self, trace_lines(trace), requests=2)

    def test_terminal_callback_exception_sse_wakes_without_replay(self):
        with running_server(
            ids=[self.out_id], continuous=True,
            flags=("--parallel-steps", "--ready-completion"),
            env_extra={"HERMES_PP2_ASYNC_STREAMS": "1",
                       "HERMES_TEST_FAIL_TERMINAL_EVENT_ONCE": "1"},
        ) as (url, _, trace):
            status, raw = post(url, payload(2, stream=True, include_usage=True))
            self.assertEqual(status, 200, raw)
            self.assertEqual(raw.count(b"data: [DONE]"), 1)
            status, raw = post(url, payload(2))
            self.assertEqual(status, 200, raw)
            self.assertEqual(json.loads(raw)["usage"]["completion_tokens"], 2)
            assert_single_owner(self, trace_lines(trace), requests=2)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
