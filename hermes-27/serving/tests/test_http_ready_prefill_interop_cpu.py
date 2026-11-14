#!/usr/bin/env python3
"""Fresh integration gate: ready-completion versus RAW/legacy chunked seed.

Runs *only* on the isolated HERMES_ADMISSION_CPU_ONLY fake PP2 HTTP executable.
Uses local loopback of that freshly launched fake binary; no real GGUF, CUDA,
GPU, installed Hermes service or live inference requests. Existing 90+7
regressions are deliberately NOT rerun.

Command (in the CPU scratch, with matching published-source server build):
  python3 serving/tests/test_http_ready_prefill_interop_cpu.py \
    --server build/hermes_server --model fake.gguf \
    --gguf-dump gguf_dump_fake.py --tokenizer tokenizer.json --python python3
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
import unittest
import urllib.error
import urllib.request

from prod_sched_state_test import ARGS, payload, running_server, trace_lines


READY = ("--parallel-steps", "--ready-completion")
ASYNC = {"HERMES_PP2_ASYNC_STREAMS": "1", "FAKE_STEP_MS": "0"}
CHUNKED = {"HERMES_PP2_CHUNKED_PREFILL": "1", "FAKE_STEP_MS": "0"}
FAILPOINTS = (
    "HERMES_TEST_FAIL_COMPLETE_ONCE",
    "HERMES_TEST_FAIL_TERMINAL_EVENT_ONCE",
    "HERMES_TEST_FAIL_TERMINAL_USAGE_ONCE",
    "HERMES_TEST_FAIL_TERMINAL_QUEUE_ONCE",
)
SANITIZED_ENV = (
    "HERMES_PP2_ASYNC_STREAMS", "HERMES_PP2_CHUNKED_PREFILL",
    "HERMES_PP2_PREFILL_BF16", "HERMES_PP2_SLIDING_KV",
    "HERMES_L26", "HERMES_NVFP4_GEMV",
    *FAILPOINTS,
)


def bounded_post(url: str, data: dict, *, endpoint: str = "/v1/chat/completions",
                 timeout: int = 12) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url + endpoint, data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def parse_sse(raw: bytes) -> tuple[list[dict], int]:
    events, done = [], 0
    for block in raw.decode("utf-8").split("\n\n"):
        if not block:
            continue
        if not block.startswith("data: "):
            raise AssertionError(f"invalid SSE block: {block[:150]!r}")
        data = block[6:]
        if data == "[DONE]":
            done += 1
        else:
            events.append(json.loads(data))
    return events, done


def assert_terminal_once(test: unittest.TestCase, raw: bytes, *,
                         failure: bool = False, finish: str = "length",
                         prompt_tokens: int | None = None,
                         completion_tokens: int | None = None):
    items, done = parse_sse(raw)
    test.assertEqual(done, 1, raw)
    errors = [x for x in items if "error" in x]
    finish_events = [
        x for x in items
        if any(choice.get("finish_reason") is not None
               for choice in x.get("choices", []))
    ]
    usage = [x["usage"] for x in items if x.get("usage") is not None]
    if failure:
        test.assertEqual(len(errors), 1, raw)
        test.assertEqual(errors[0]["error"]["type"], "server_error")
        test.assertEqual(len(finish_events), 0, raw)
        test.assertEqual(len(usage), 0, raw)
    else:
        test.assertEqual(errors, [], raw)
        test.assertEqual(len(finish_events), 1, raw)
        test.assertEqual(finish_events[0]["choices"][0]["finish_reason"],
                         finish, raw)
        test.assertEqual(len(usage), 1, raw)
        if prompt_tokens is not None:
            test.assertEqual(usage[0]["prompt_tokens"], prompt_tokens, raw)
        if completion_tokens is not None:
            test.assertEqual(usage[0]["completion_tokens"], completion_tokens,
                             raw)
        test.assertEqual(usage[0]["total_tokens"],
                         usage[0]["prompt_tokens"] + usage[0]["completion_tokens"],
                         raw)
    return items


def first_content_then_cancel(test: unittest.TestCase, url: str,
                              request_payload: dict) -> bytes:
    request = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(request_payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=12) as response:
        test.assertEqual(response.status, 200)
        raw_chunks = []
        request_id = None
        got_content = False
        # Read only until a REAL visible generated token, rather than canceling
        # an unrelated pre-admission request. The first SSE event is role-only.
        for _ in range(100):
            line = response.readline()
            test.assertTrue(line, b"stream closed before first visible token")
            raw_chunks.append(line)
            if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                event = json.loads(line[6:])
                request_id = event.get("id", request_id)
                got_content = got_content or any(
                    bool(c.get("delta", {}).get("content"))
                    for c in event.get("choices", []))
                if got_content:
                    break
        test.assertTrue(got_content and request_id, "no visible token to cancel")
        status, raw = bounded_post(
            url, {"id": request_id}, endpoint="/v1/cancel", timeout=5)
        test.assertEqual(status, 202, raw)
        # One in-flight non-interruptible backend step may still finish; it
        # cannot cause a second SSE terminal or a second state owner.
        return b"".join(raw_chunks) + response.read()


class ReadyPrefillInteropCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        cls.tokenizer = Tokenizer.from_file(ARGS.tokenizer)
        cls.output_id = cls.tokenizer.encode(
            "hello", add_special_tokens=False).ids[0]
        cls.output_text = cls.tokenizer.decode([cls.output_id])

    def request(self, length: int, maximum: int, *, stream: bool = False):
        content = " x" * length
        ids = self.tokenizer.encode(content, add_special_tokens=False).ids
        self.assertEqual(len(ids), length,
                         f"tokenizer fixture must have exactly T={length} tokens")
        body = payload(maximum, stream=stream, include_usage=stream)
        body["messages"][0]["content"] = content
        return body

    def assert_trace(self, trace, *, count: int, prefill=(), chunked=(),
                     steps=None):
        lines = trace_lines(trace)
        self.assertEqual(len([x for x in lines if x.startswith("open ")]),
                         count, lines)
        self.assertEqual(lines.count("close"), count, lines)
        self.assertEqual(
            sorted(x for x in lines if x.startswith("prefill ")),
            sorted(f"prefill {n}" for n in prefill), lines)
        self.assertEqual(
            sorted(x for x in lines if x.startswith("chunked ")),
            sorted(f"chunked {n}" for n in chunked), lines)
        actual_steps = [
            int(x.split()[1]) for x in lines if x.startswith("step ")]
        if steps is not None:
            self.assertEqual(Counter(actual_steps), Counter(steps), lines)
        self.assertEqual(
            len(lines), 2 * count + len(prefill) + len(chunked) +
            len(actual_steps), lines)
        return lines

    def test_invalid_ready_plus_seed_flags_fail_before_model_upload(self):
        # --raw-prefill and --legacy-chunked are NOT CLI aliases. Exact
        # public spellings are --batched-prefill (RAW) and --chunked-prefill.
        cases = (
            ("ready alone", ("--continuous-scheduler", "--ready-completion"),
             {"HERMES_PP2_ASYNC_STREAMS": "1"}, "ready-completion requires"),
            ("missing continuous", READY,
             {"HERMES_PP2_ASYNC_STREAMS": "1"}, "parallel-steps requires"),
            ("missing async", ("--continuous-scheduler", *READY),
             {}, "parallel-steps requires"),
            ("raw seed plus ready", ("--continuous-scheduler", *READY,
                                     "--batched-prefill"),
             {"HERMES_PP2_ASYNC_STREAMS": "1"}, "parallel-steps requires"),
            ("legacy chunk plus ready", ("--continuous-scheduler", *READY,
                                         "--chunked-prefill", "--active", "1"),
             {"HERMES_PP2_ASYNC_STREAMS": "1",
              "HERMES_PP2_CHUNKED_PREFILL": "1"}, "parallel-steps requires"),
            ("legacy chunk plus continuous", ("--continuous-scheduler",
                                              "--chunked-prefill", "--active", "1"),
             {"HERMES_PP2_CHUNKED_PREFILL": "1"}, "chunked-prefill requires"),
            ("ready plus ring", ("--continuous-scheduler", *READY,
                                 "--sliding-kv", "--context", "8192"),
             {"HERMES_PP2_ASYNC_STREAMS": "1",
              "HERMES_PP2_SLIDING_KV": "1"}, "sliding-kv requires"),
            ("BF16 plus RAW seed", ("--batched-prefill",),
             {"HERMES_PP2_PREFILL_BF16": "1"}, "raw-only prefill"),
            ("unsupported raw alias", ("--continuous-scheduler", *READY,
                                       "--raw-prefill", "1"),
             {"HERMES_PP2_ASYNC_STREAMS": "1"}, "unknown option"),
            ("unsupported legacy alias", ("--legacy-chunked", "1"),
             {"HERMES_PP2_CHUNKED_PREFILL": "1"}, "unknown option"),
        )
        for name, flags, overrides, expected in cases:
            with self.subTest(case=name):
                env = os.environ.copy()
                for item in SANITIZED_ENV:
                    env.pop(item, None)
                env.update(overrides)
                result = subprocess.run(
                    [ARGS.server, *flags], env=env, capture_output=True,
                    text=True, timeout=6, check=False)
                self.assertEqual(result.returncode, 2, (name, result.stderr))
                self.assertIn(expected, result.stderr, name)
                self.assertNotIn("MODEL ready", result.stderr)
                self.assertNotIn("pp2 upload", result.stderr)
                self.assertNotIn("LISTEN", result.stderr)

    def test_ready_env_does_not_seed_without_flag_mixed_T64_T65(self):
        # Deliberately set legacy chunked env alone: no implicit second owner.
        env = dict(ASYNC)
        env["HERMES_PP2_CHUNKED_PREFILL"] = "1"
        with running_server(
            ids=[self.output_id], continuous=True,
            flags=(*READY, "--active", "2", "--queue", "4"),
            env_extra=env,
        ) as (url, _proc, trace):
            def one(pair):
                length, maximum = pair
                status, raw = bounded_post(
                    url, self.request(length, maximum))
                self.assertEqual(status, 200, raw)
                data = json.loads(raw)
                self.assertEqual(data["usage"]["prompt_tokens"], length)
                self.assertEqual(data["usage"]["completion_tokens"], maximum)
                self.assertEqual(
                    data["choices"][0]["message"]["content"],
                    self.tokenizer.decode([self.output_id] * maximum))
                return data["id"]
            with ThreadPoolExecutor(max_workers=2) as pool:
                ids = list(pool.map(one, [(64, 2), (65, 2)]))
            self.assertEqual(len(set(ids)), 2)
            self.assert_trace(trace, count=2,
                              steps=[*range(65), *range(66)])
            with urllib.request.urlopen(url + "/metrics", timeout=6) as r:
                metrics = r.read().decode()
            self.assertIn("hermes_prefill_attempts_total 0\n", metrics)
            self.assertIn("hermes_prefill_seeded_total 0\n", metrics)

    def test_raw_seed_allowed_separately_legacy_and_continuous_without_parallel(self):
        for continuous in (False, True):
            with self.subTest(continuous=continuous):
                with running_server(
                    ids=[self.output_id], continuous=continuous,
                    flags=("--batched-prefill",),
                    env_extra={"FAKE_STEP_MS": "0"},
                ) as (url, _proc, trace):
                    body = self.request(11, 3, stream=True)
                    status, raw = bounded_post(url, body)
                    self.assertEqual(status, 200, raw)
                    assert_terminal_once(
                        self, raw, prompt_tokens=11, completion_tokens=3)
                    self.assert_trace(trace, count=1, prefill=(11,),
                                      steps=(11, 12))

    def test_legacy_chunk_seed_one_state_without_parallel_owner(self):
        with running_server(
            ids=[self.output_id], continuous=False,
            flags=("--chunked-prefill", "--active", "1"),
            env_extra=CHUNKED,
        ) as (url, _proc, trace):
            status, raw = bounded_post(url, self.request(65, 3, stream=True))
            self.assertEqual(status, 200, raw)
            assert_terminal_once(
                self, raw, prompt_tokens=65, completion_tokens=3)
            self.assert_trace(trace, count=1, chunked=(65,),
                              steps=(65, 66))

    def test_ready_late_cancel_after_visible_token_one_owner(self):
        with running_server(
            ids=[self.output_id], continuous=True,
            flags=(*READY, "--active", "1"),
            env_extra={"HERMES_PP2_ASYNC_STREAMS": "1",
                       "FAKE_STEP_MS": "150"},
        ) as (url, _proc, trace):
            raw = first_content_then_cancel(
                self, url, self.request(1, 28, stream=True))
            items = assert_terminal_once(self, raw, finish="cancelled",
                                         prompt_tokens=1)
            usage = [x["usage"] for x in items if x.get("usage") is not None][0]
            self.assertGreaterEqual(usage["completion_tokens"], 1)
            self.assertLess(usage["completion_tokens"], 28)
            lines = self.assert_trace(trace, count=1)
            positions = [int(x.split()[1]) for x in lines
                         if x.startswith("step ")]
            self.assertEqual(len(positions), len(set(positions)), lines)
            self.assertTrue(all(p >= 0 for p in positions), lines)

    def test_chunked_late_cancel_after_seed_no_replay_or_double_close(self):
        with running_server(
            ids=[self.output_id], continuous=False,
            flags=("--chunked-prefill", "--active", "1"),
            env_extra={"HERMES_PP2_CHUNKED_PREFILL": "1",
                       "FAKE_STEP_MS": "150"},
        ) as (url, _proc, trace):
            raw = first_content_then_cancel(
                self, url, self.request(65, 28, stream=True))
            items = assert_terminal_once(
                self, raw, finish="cancelled", prompt_tokens=65)
            usage = [x["usage"] for x in items if x.get("usage") is not None][0]
            self.assertGreaterEqual(usage["completion_tokens"], 1)
            self.assertLess(usage["completion_tokens"], 28)
            lines = self.assert_trace(trace, count=1, chunked=(65,))
            positions = [int(x.split()[1]) for x in lines
                         if x.startswith("step ")]
            self.assertEqual(len(positions), len(set(positions)), lines)
            self.assertTrue(all(p >= 65 for p in positions), lines)

    def test_ready_sse_oom_single_terminal_then_normal_completion(self):
        with running_server(
            ids=[self.output_id], continuous=True,
            flags=READY,
            env_extra={"HERMES_PP2_ASYNC_STREAMS": "1",
                       "FAKE_STEP_MS": "0",
                       "HERMES_TEST_FAIL_TERMINAL_USAGE_ONCE": "1"},
        ) as (url, _proc, trace):
            status, raw = bounded_post(url, self.request(1, 2, stream=True))
            self.assertEqual(status, 200, raw)
            assert_terminal_once(self, raw, failure=True)
            status, raw = bounded_post(url, self.request(1, 2, stream=True))
            self.assertEqual(status, 200, raw)
            assert_terminal_once(self, raw, prompt_tokens=1,
                                 completion_tokens=2)
            self.assert_trace(trace, count=2, steps=(0, 1, 0, 1))

    def test_chunked_sse_oom_does_not_reseed_or_fake_success(self):
        with running_server(
            ids=[self.output_id], continuous=False,
            flags=("--chunked-prefill", "--active", "1"),
            env_extra={"HERMES_PP2_CHUNKED_PREFILL": "1",
                       "FAKE_STEP_MS": "0",
                       "HERMES_TEST_FAIL_TERMINAL_QUEUE_ONCE": "1"},
        ) as (url, _proc, trace):
            status, raw = bounded_post(url, self.request(65, 2, stream=True))
            self.assertEqual(status, 200, raw)
            assert_terminal_once(self, raw, failure=True)
            status, raw = bounded_post(url, self.request(65, 2, stream=True))
            self.assertEqual(status, 200, raw)
            assert_terminal_once(self, raw, prompt_tokens=65,
                                 completion_tokens=2)
            self.assert_trace(trace, count=2, chunked=(65, 65),
                              steps=(65, 65))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
