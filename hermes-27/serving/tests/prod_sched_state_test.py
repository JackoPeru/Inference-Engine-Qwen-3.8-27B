#!/usr/bin/env python3
"""CPU-only end-to-end state ownership tests for --continuous-scheduler.

Build serving/server.cpp with worker-6's pp2_batch.cu in
HERMES_ADMISSION_CPU_ONLY mode and prod_sched_mock_pp2.cpp. Never use a CUDA
binary for this test.
"""
import argparse
import concurrent.futures
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request

PARSER = argparse.ArgumentParser()
for name in ("server", "model", "gguf-dump", "tokenizer", "python"):
    PARSER.add_argument("--" + name, required=True)
ARGS, OTHER = PARSER.parse_known_args()


@contextmanager
def running_server(ids=(), flags=(), env_extra=None):
    with tempfile.TemporaryDirectory(prefix="hermes-prod-sched-") as temp:
        trace = Path(temp) / "states.log"
        env = os.environ.copy()
        env["FAKE_IDS"] = ",".join(str(token) for token in ids) if ids else "100"
        env["FAKE_TRACE_FILE"] = str(trace)
        env.update(env_extra or {})
        cmd = [
            ARGS.server, "--continuous-scheduler",
            "--model", ARGS.model, "--gguf-dump", ARGS.gguf_dump,
            "--tokenizer", ARGS.tokenizer, "--python", ARGS.python,
            "--context", "256", "--port", "0", *flags,
        ]
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        try:
            while True:
                line = proc.stderr.readline()
                if line.startswith("LISTEN "):
                    yield line.split()[1], proc, trace
                    break
                if proc.poll() is not None:
                    raise AssertionError(f"CPU mock server startup failed: {line}")
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=7)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=4)
            proc.stderr.close()


def payload(max_tokens, *, stream=False, stops=None, include_usage=False):
    request = {
        "model": "hermes-27", "messages": [{"role": "user", "content": "x"}],
        "max_tokens": max_tokens, "temperature": 0, "stream": stream,
    }
    if stops is not None:
        request["stop"] = stops
    if include_usage:
        request["stream_options"] = {"include_usage": True}
    return request


def post(url, data, endpoint="/v1/chat/completions"):
    request = urllib.request.Request(
        url + endpoint, data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=40) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def events(body):
    fragments, reasons, usages, done = [], [], [], 0
    for block in body.decode("utf-8").split("\n\n"):
        if not block:
            continue
        assert block.startswith("data: "), block
        value = block[6:]
        if value == "[DONE]":
            done += 1
            continue
        event = json.loads(value)
        if event.get("error"):
            raise AssertionError(event)
        if event.get("usage") is not None:
            usages.append(event["usage"])
        for choice in event["choices"]:
            fragments.append(choice["delta"].get("content", ""))
            if choice["finish_reason"] is not None:
                reasons.append(choice["finish_reason"])
    return "".join(fragments), reasons, usages, done


def trace_lines(path):
    return path.read_text().splitlines() if path.exists() else []


def assert_single_owner(test, lines, requests, expected_steps=None):
    opens = [line for line in lines if line.startswith("open ")]
    closes = [line for line in lines if line == "close"]
    steps = [line for line in lines if line.startswith("step ")]
    test.assertEqual(len(opens), requests, lines)
    test.assertEqual(len(closes), requests, lines)
    if expected_steps is not None:
        test.assertEqual(len(steps), expected_steps, lines)
    test.assertEqual(len(lines), len(opens) + len(closes) + len(steps), lines)
    return opens, closes, steps


class ProductionSchedulerOwnership(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        cls.tokenizer = Tokenizer.from_file(ARGS.tokenizer)

    def ids(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def test_exactly_one_state_per_completed_request(self):
        token_ids = self.ids("Caf\u00e9 \U0001f680 world!")
        self.assertGreater(len(token_ids), 1)
        expected = self.tokenizer.decode(token_ids)
        with running_server(token_ids) as (url, _, trace):
            status, raw = post(url, payload(len(token_ids)))
            self.assertEqual(status, 200, raw)
            result = json.loads(raw)
            self.assertEqual(result["choices"][0]["message"]["content"], expected)
            self.assertEqual(result["usage"]["completion_tokens"], len(token_ids))
            status, raw = post(
                url, payload(len(token_ids), stream=True, include_usage=True)
            )
            self.assertEqual(status, 200, raw)
            text, reasons, usages, done = events(raw)
            self.assertEqual(text, expected)
            self.assertEqual(reasons, ["length"])
            self.assertEqual(done, 1)
            self.assertEqual(len(usages), 1)
            self.assertEqual(usages[0]["completion_tokens"], len(token_ids))
            lines = trace_lines(trace)
            _, _, steps = assert_single_owner(
                self, lines, requests=2, expected_steps=2 * len(token_ids)
            )
            self.assertEqual(
                [int(step.split()[-1]) for step in steps],
                list(range(len(token_ids))) * 2,
            )

    def test_stop_uses_one_owner_and_trims_sse(self):
        token_ids = self.ids("hello good night world")
        expected = self.tokenizer.decode(token_ids)
        stop = "good night"
        with running_server(token_ids) as (url, _, trace):
            status, raw = post(url, payload(len(token_ids), stream=True, stops=stop,
                                            include_usage=True))
            self.assertEqual(status, 200, raw)
            text, reasons, usages, done = events(raw)
            self.assertEqual(text, expected.split(stop, 1)[0])
            self.assertEqual(reasons, ["stop"])
            self.assertEqual(done, 1)
            self.assertEqual(len(usages), 1)
            generated = usages[0]["completion_tokens"]
            self.assertGreater(generated, 0)
            self.assertLessEqual(generated, len(token_ids))
            assert_single_owner(self, trace_lines(trace), 1, generated)

    def test_expired_queue_does_not_open_second_state(self):
        with running_server(
            self.ids("z"), flags=("--active", "1", "--queue", "1",
                                   "--queue-timeout", "1"),
            env_extra={"FAKE_STEP_MS": "12"},
        ) as (url, _, trace):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(post, url, payload(200))
                for _ in range(100):
                    with urllib.request.urlopen(url + "/metrics", timeout=3) as response:
                        metrics = response.read().decode()
                    if "hermes_requests_active 1" in metrics:
                        break
                    time.sleep(.02)
                else:
                    self.fail("first request never acquired PP2 state")
                status, raw = post(url, payload(1, stream=True))
                self.assertEqual(status, 429, raw)
                self.assertEqual(json.loads(raw)["error"]["code"], "overloaded")
                status, raw = first.result(timeout=30)
                self.assertEqual(status, 200, raw)
                assert_single_owner(self, trace_lines(trace), 1, 200)

    def test_concurrent_requests_keep_independent_states(self):
        token_id = self.ids("a")[0]
        counts = (2, 3, 4, 5, 6, 7)
        with running_server([token_id], flags=("--active", "4", "--queue", "16")) as (url, _, trace):
            def run(n):
                status, raw = post(url, payload(n, stream=n % 2 == 0))
                self.assertEqual(status, 200, raw)
                if n % 2:
                    result = json.loads(raw)
                    self.assertEqual(result["usage"]["completion_tokens"], n)
                    return result["choices"][0]["message"]["content"]
                text, reasons, _, done = events(raw)
                self.assertEqual(done, 1)
                self.assertEqual(reasons, ["length"])
                return text
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                results = list(executor.map(run, counts))
            for actual, n in zip(results, counts):
                self.assertEqual(actual, self.tokenizer.decode([token_id] * n))
            assert_single_owner(self, trace_lines(trace), len(counts), sum(counts))

    def test_disconnect_cancellation_closes_state(self):
        with running_server(self.ids("hello")) as (url, _, trace):
            data = payload(200, stream=True, include_usage=True)
            req = urllib.request.Request(
                url + "/v1/chat/completions", data=json.dumps(data).encode(),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                first = response.readline()
                self.assertTrue(first.startswith(b"data: "), first)
                ident = json.loads(first[6:])["id"]
                status, raw = post(url, {"id": ident}, endpoint="/v1/cancel")
                self.assertEqual(status, 202, raw)
                whole = first + response.read()
                text, reasons, usages, done = events(whole)
                self.assertEqual(reasons, ["cancelled"])
                self.assertEqual(done, 1)
                self.assertEqual(len(usages), 1)
                assert_single_owner(self, trace_lines(trace), 1)

    def test_failed_open_allocates_no_state(self):
        with running_server(env_extra={"FAKE_INIT_FAIL": "1"}) as (url, _, trace):
            status, raw = post(url, payload(1, stream=True))
            self.assertEqual(status, 503, raw)
            self.assertEqual(json.loads(raw)["error"]["code"], "inference_error")
            self.assertEqual(trace_lines(trace), [])

    def test_shutdown_closes_state_once(self):
        with running_server(env_extra={"FAKE_STEP_MS": "20"}) as (url, process, trace):
            req = urllib.request.Request(
                url + "/v1/chat/completions",
                data=json.dumps(payload(200, stream=True)).encode(),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                first = response.readline()
                self.assertTrue(first.startswith(b"data: "), first)
                os.kill(process.pid, signal.SIGTERM)
                _, reasons, _, done = events(first + response.read())
                self.assertEqual(reasons, ["cancelled"])
                self.assertEqual(done, 1)
            self.assertEqual(process.wait(timeout=8), 0)
            assert_single_owner(self, trace_lines(trace), 1)


if __name__ == "__main__":
    unittest.main(argv=[__file__, *OTHER], verbosity=2)
