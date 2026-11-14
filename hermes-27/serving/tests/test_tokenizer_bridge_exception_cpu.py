#!/usr/bin/env python3
"""Targeted CPU-mock HTTP tests: fail-closed terminal callbacks and SSE.

Requires the existing HERMES_ADMISSION_CPU_ONLY hermes_server build, fake
PP2 backend, tokenizer fixtures and arguments accepted by
prod_sched_state_test.py. Never loads a CUDA model or connects to a remote
host. Uses only the test server's private loopback interface.

Example (on the Linux CPU test host, NOT production):
  python3 serving/tests/test_tokenizer_bridge_exception_cpu.py \
    --server /tmp/cpu-build/hermes_server --model /path/fixture.gguf \
    --gguf-dump /tmp/cpu-build/gguf_dump --tokenizer /path/tokenizer.json \
    --python /path/python
"""
from __future__ import annotations

import json
import os
import unittest
import urllib.error
import urllib.request

from prod_sched_state_test import (
    ARGS,
    assert_single_owner,
    events,
    payload,
    running_server,
    trace_lines,
)


ASYNC_READY = ("--parallel-steps", "--ready-completion")
READY_ENV = {"HERMES_PP2_ASYNC_STREAMS": "1"}
FAILURES = (
    "HERMES_TEST_FAIL_COMPLETE_ONCE",
    "HERMES_TEST_FAIL_TERMINAL_USAGE_ONCE",
    "HERMES_TEST_FAIL_TERMINAL_QUEUE_ONCE",
    "HERMES_TEST_FAIL_TERMINAL_EVENT_ONCE",
)


def bounded_post(url: str, data: dict, timeout: int = 8,
                 bearer: str | None = None) -> tuple[int, bytes]:
    headers = {"Content-Type": "application/json"}
    if bearer is not None:
        headers["Authorization"] = "Bearer " + bearer
    request = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(data).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class TokenizerBridgeTerminalExceptionCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        cls.tokenizer = Tokenizer.from_file(ARGS.tokenizer)
        cls.out_id = cls.tokenizer.encode(
            "hello", add_special_tokens=False).ids[0]

    def start_env(self, failpoint: str) -> dict[str, str]:
        self.assertIn(failpoint, FAILURES)
        self.assertFalse(
            any(name in os.environ for name in FAILURES),
            "unset inherited HERMES_TEST_* failpoints before CPU-mock tests",
        )
        env = dict(READY_ENV)
        env[failpoint] = "1"
        return env

    def test_existing_bearer_auth_gate_preserved(self):
        with running_server(
            ids=[self.out_id], continuous=True,
            flags=ASYNC_READY + ("--api-key", "test-only-secret"),
            env_extra=READY_ENV,
        ) as (url, _proc, trace):
            status, raw = bounded_post(url, payload(1))
            self.assertEqual(status, 401, raw)
            status, raw = bounded_post(
                url, payload(1), bearer="incorrect-test-key")
            self.assertEqual(status, 401, raw)
            status, raw = bounded_post(
                url, payload(1), bearer="test-only-secret")
            self.assertEqual(status, 200, raw)
            self.assertEqual(
                json.loads(raw)["choices"][0]["finish_reason"], "length")
            assert_single_owner(self, trace_lines(trace), requests=1)

    def test_nonstream_decode_response_fault_wakes_and_recovers(self):
        with running_server(
            ids=[self.out_id], continuous=True, flags=ASYNC_READY,
            env_extra=self.start_env("HERMES_TEST_FAIL_COMPLETE_ONCE"),
        ) as (url, _proc, trace):
            status, raw = bounded_post(url, payload(1))
            self.assertEqual(status, 500, raw)
            self.assertEqual(
                json.loads(raw)["error"]["message"],
                "request finalization failed",
            )
            status, raw = bounded_post(url, payload(1))
            self.assertEqual(status, 200, raw)
            result = json.loads(raw)
            self.assertEqual(result["choices"][0]["finish_reason"], "length")
            self.assertEqual(result["usage"]["completion_tokens"], 1)
            assert_single_owner(self, trace_lines(trace), requests=2)

    def test_early_terminal_callback_exception_uses_emergency_wakeup(self):
        with running_server(
            ids=[self.out_id], continuous=True, flags=ASYNC_READY,
            env_extra=self.start_env("HERMES_TEST_FAIL_TERMINAL_EVENT_ONCE"),
        ) as (url, _proc, trace):
            status, raw = bounded_post(url, payload(1))
            self.assertEqual(status, 500, raw)
            error = json.loads(raw)["error"]
            self.assertEqual(error["message"], "request finalization failed")
            status, raw = bounded_post(url, payload(1))
            self.assertEqual(status, 200, raw)
            self.assertEqual(json.loads(raw)["usage"]["completion_tokens"], 1)
            assert_single_owner(self, trace_lines(trace), requests=2)

    def assert_sse_failure_then_success(self, failpoint: str):
        with running_server(
            ids=[self.out_id], continuous=True, flags=ASYNC_READY,
            env_extra=self.start_env(failpoint),
        ) as (url, _proc, trace):
            request = payload(1, stream=True, include_usage=True)
            status, raw = bounded_post(url, request)
            # Once admitted, SSE necessarily has HTTP 200. A terminal
            # serialization fault must be signalled INSIDE the SSE stream,
            # never as a spurious successful finish_reason/usage frame.
            self.assertEqual(status, 200, raw)
            blocks = [line[6:] for line in raw.decode("utf-8").split("\n\n")
                      if line.startswith("data: ")]
            terminal_errors = [
                json.loads(line) for line in blocks
                if line != "[DONE]" and
                isinstance(json.loads(line), dict) and
                "error" in json.loads(line)
            ]
            self.assertEqual(len(terminal_errors), 1, raw)
            self.assertEqual(
                terminal_errors[0]["error"]["message"],
                "request finalization failed",
            )
            self.assertEqual(blocks.count("[DONE]"), 1, raw)
            self.assertFalse(any(
                line != "[DONE]" and
                (event.get("usage") is not None or any(
                    choice.get("finish_reason") is not None
                    for choice in event.get("choices", [])
                ))
                for line in blocks if line != "[DONE]"
                for event in (json.loads(line),)
            ), f"failed SSE must never expose success usage/finish: {blocks}")

            status, raw = bounded_post(url, request)
            self.assertEqual(status, 200, raw)
            _text, reasons, usages, done = events(raw)
            self.assertEqual(reasons, ["length"])
            self.assertEqual(len(usages), 1)
            self.assertEqual(usages[0]["completion_tokens"], 1)
            self.assertEqual(done, 1)
            assert_single_owner(self, trace_lines(trace), requests=2)

    def test_sse_complete_failure_does_not_hang(self):
        self.assert_sse_failure_then_success(
            "HERMES_TEST_FAIL_COMPLETE_ONCE")

    def test_sse_partial_success_staged_before_usage_fails(self):
        self.assert_sse_failure_then_success(
            "HERMES_TEST_FAIL_TERMINAL_USAGE_ONCE")

    def test_sse_queue_commit_failure_is_error_not_success(self):
        self.assert_sse_failure_then_success(
            "HERMES_TEST_FAIL_TERMINAL_QUEUE_ONCE")

    def test_sse_early_terminal_callback_exception_fails_closed(self):
        self.assert_sse_failure_then_success(
            "HERMES_TEST_FAIL_TERMINAL_EVENT_ONCE")


if __name__ == "__main__":
    import sys
    unittest.main(argv=[sys.argv[0]], verbosity=2)
