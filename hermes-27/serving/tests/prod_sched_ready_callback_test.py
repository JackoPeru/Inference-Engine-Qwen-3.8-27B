#!/usr/bin/env python3
"""CPU-only real-server/real-scheduler HTTP contract for ready callback workers.

Uses the existing fake PP2 backend and tokenizer subprocess. No CUDA model,
GPU inference, live HERMES endpoint or performance claim.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import sys
import unittest
import urllib.request

from prod_sched_state_test import (
    ARGS, assert_single_owner, events, payload, post, running_server,
    trace_lines,
)


FLAGS = ("--parallel-steps", "--ready-completion", "--ready-callback-delivery")
ASYNC_ENV = {"HERMES_PP2_ASYNC_STREAMS": "1", "FAKE_STEP_MS": "9"}


class ReadyCallbackServingCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(ARGS.tokenizer)
        cls.tokenizer = tokenizer
        cls.token = tokenizer.encode("hello", add_special_tokens=False).ids[0]
        cls.expected = tokenizer.decode([cls.token] * 3)

    def test_flag_is_explicit_and_refuses_unsafe_scheduling(self):
        invalid = (
            (("--ready-callback-delivery",), {"HERMES_PP2_ASYNC_STREAMS": "1"}),
            (("--continuous-scheduler", "--ready-callback-delivery"),
             {"HERMES_PP2_ASYNC_STREAMS": "1"}),
            (("--continuous-scheduler", "--parallel-steps",
              "--ready-callback-delivery"), {"HERMES_PP2_ASYNC_STREAMS": "1"}),
            (("--continuous-scheduler", *FLAGS), {}),
            (("--continuous-scheduler", *FLAGS, "--active", "1"),
             {"HERMES_PP2_ASYNC_STREAMS": "1"}),
            (("--continuous-scheduler", *FLAGS, "--batched-prefill"),
             {"HERMES_PP2_ASYNC_STREAMS": "1"}),
            (("--continuous-scheduler", *FLAGS, "--chunked-prefill"),
             {"HERMES_PP2_ASYNC_STREAMS": "1",
              "HERMES_PP2_CHUNKED_PREFILL": "1"}),
            (("--continuous-scheduler", *FLAGS, "--sliding-kv", "--context", "8192"),
             {"HERMES_PP2_ASYNC_STREAMS": "1",
              "HERMES_PP2_SLIDING_KV": "1"}),
            (("--continuous-scheduler", *FLAGS),
             {"HERMES_PP2_ASYNC_STREAMS": "1", "HERMES_L26": "1"}),
        )
        for flags, updates in invalid:
            with self.subTest(flags=flags):
                env = os.environ.copy()
                for name in ("HERMES_PP2_ASYNC_STREAMS",
                             "HERMES_PP2_CHUNKED_PREFILL",
                             "HERMES_PP2_SLIDING_KV", "HERMES_L26"):
                    env.pop(name, None)
                env.update(updates)
                run = subprocess.run(
                    [ARGS.server, *flags], env=env, capture_output=True,
                    text=True, check=False, timeout=6,
                )
                self.assertEqual(run.returncode, 2, run.stderr)
                self.assertTrue(
                    "--ready-callback-delivery requires" in run.stderr or
                    "--parallel-steps requires" in run.stderr or
                    "--chunked-prefill requires" in run.stderr or
                    "--sliding-kv requires" in run.stderr,
                    run.stderr,
                )
                self.assertNotIn("pp2 upload", run.stderr)

    def test_concurrent_nonstream_sse_unicode_usage_and_single_owner(self):
        with running_server(
            ids=[self.token], continuous=True, flags=FLAGS,
            env_extra=ASYNC_ENV,
        ) as (url, _, trace):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda _:
                    post(url, payload(3)), range(4)))
            ids = set()
            usage = None
            for status, raw in results:
                self.assertEqual(status, 200, raw)
                response = json.loads(raw)
                self.assertEqual(response["choices"][0]["finish_reason"], "length")
                self.assertEqual(
                    response["choices"][0]["message"]["content"], self.expected)
                self.assertEqual(response["usage"]["completion_tokens"], 3)
                ids.add(response["id"])
                usage = response["usage"]
            self.assertEqual(len(ids), 4)
            status, raw = post(url, payload(3, stream=True, include_usage=True))
            self.assertEqual(status, 200, raw)
            visible, reasons, usages, done = events(raw)
            self.assertEqual(visible, self.expected)
            self.assertEqual(reasons, ["length"])
            self.assertEqual(usages, [usage])
            self.assertEqual(done, 1)
            assert_single_owner(self, trace_lines(trace), requests=5)

    def test_unicode_cross_token_sse_nonstream_and_usage_parity(self):
        tokens = self.tokenizer.encode(
            "Cafe\u0301, na\u00efve, fa\u00e7ade \U0001f680",
            add_special_tokens=False,
        ).ids
        expected = self.tokenizer.decode(tokens, skip_special_tokens=True)
        self.assertGreater(len(tokens), 3)
        with running_server(
            ids=tokens, continuous=True, flags=FLAGS, env_extra=ASYNC_ENV,
        ) as (url, _, trace):
            status, raw = post(url, payload(len(tokens)))
            self.assertEqual(status, 200, raw)
            response = json.loads(raw)
            self.assertEqual(
                response["choices"][0]["message"]["content"], expected)
            self.assertEqual(response["usage"]["completion_tokens"], len(tokens))
            status, raw = post(url, payload(
                len(tokens), stream=True, include_usage=True))
            self.assertEqual(status, 200, raw)
            text, reasons, usages, done = events(raw)
            self.assertEqual(text, expected)
            self.assertEqual(reasons, ["length"])
            self.assertEqual(done, 1)
            self.assertEqual(usages, [response["usage"]])
            assert_single_owner(self, trace_lines(trace), requests=2)

    def test_stop_crosses_tokens_and_terminal_sse_is_unique(self):
        tokens = self.tokenizer.encode(
            "hello good night world", add_special_tokens=False,
        ).ids
        expected = self.tokenizer.decode(tokens, skip_special_tokens=True)
        stop = "good night"
        self.assertIn(stop, expected)
        with running_server(
            ids=tokens, continuous=True, flags=FLAGS, env_extra=ASYNC_ENV,
        ) as (url, _, trace):
            status, raw = post(url, payload(
                len(tokens), stream=True, stops=stop, include_usage=True))
            self.assertEqual(status, 200, raw)
            visible, reasons, usages, done = events(raw)
            self.assertEqual(visible, expected.split(stop, 1)[0])
            self.assertEqual(reasons, ["stop"])
            self.assertEqual(done, 1)
            self.assertEqual(len(usages), 1)
            generated = usages[0]["completion_tokens"]
            self.assertGreater(generated, 0)
            self.assertLessEqual(generated, len(tokens))
            assert_single_owner(self, trace_lines(trace), requests=1)

    def test_explicit_cancel_during_sse_has_one_terminal_and_one_state(self):
        with running_server(
            ids=[self.token], continuous=True, flags=FLAGS,
            env_extra={**ASYNC_ENV, "FAKE_STEP_MS": "20"},
        ) as (url, _, trace):
            req = urllib.request.Request(
                url + "/v1/chat/completions",
                data=json.dumps(payload(
                    200, stream=True, include_usage=True)).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                first = response.readline()
                self.assertTrue(first.startswith(b"data: "), first)
                ident = json.loads(first[6:])["id"]
                status, raw = post(
                    url, {"id": ident}, endpoint="/v1/cancel")
                self.assertEqual(status, 202, raw)
                visible, reasons, usages, done = events(
                    first + response.read())
                self.assertEqual(reasons, ["cancelled"])
                self.assertEqual(done, 1)
                self.assertEqual(len(usages), 1)
                assert_single_owner(self, trace_lines(trace), requests=1)

    def test_callback_exception_is_fail_closed_then_next_request_survives(self):
        with running_server(
            ids=[self.token], continuous=True, flags=FLAGS,
            env_extra={**ASYNC_ENV, "HERMES_TEST_FAIL_TERMINAL_EVENT_ONCE": "1"},
        ) as (url, _, trace):
            status, raw = post(url, payload(2))
            self.assertEqual(status, 500, raw)
            self.assertIn("error", json.loads(raw))
            status, raw = post(url, payload(2, stream=True, include_usage=True))
            self.assertEqual(status, 200, raw)
            self.assertEqual(raw.count(b"data: [DONE]"), 1)
            assert_single_owner(self, trace_lines(trace), requests=2)

    def test_sse_terminal_allocation_fault_has_one_done_and_recovers(self):
        with running_server(
            ids=[self.token], continuous=True, flags=FLAGS,
            env_extra={**ASYNC_ENV, "HERMES_TEST_FAIL_COMPLETE_ONCE": "1"},
        ) as (url, _, trace):
            status, raw = post(url, payload(2, stream=True, include_usage=True))
            self.assertEqual(status, 200, raw)
            self.assertEqual(raw.count(b"data: [DONE]"), 1)
            self.assertNotIn(b'"finish_reason":"length"', raw)
            status, raw = post(url, payload(2))
            self.assertEqual(status, 200, raw)
            self.assertEqual(json.loads(raw)["usage"]["completion_tokens"], 2)
            assert_single_owner(self, trace_lines(trace), requests=2)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
