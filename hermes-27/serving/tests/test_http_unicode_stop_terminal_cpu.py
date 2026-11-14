#!/usr/bin/env python3
"""Focused real-HTTP/real-tokenizer regressions on the CUDA-free PP2 fake.

Build the real server.cpp with HERMES_ADMISSION_CPU_ONLY and
prod_sched_mock_pp2.cpp in isolated Linux scratch before running this test.
The server must bind its own ephemeral 127.0.0.1 port. It never opens a real
model, a CUDA context, a production listener, or any GPU service.

For the separate source-anchored allocation-exception witness, set
HERMES_STREAM_CHUNK_FAULT_BINARY to a scratch-only CPU fake binary compiled
from this same server.cpp with one std::bad_alloc injection immediately after
the `job->text.substr(job->sent_bytes)` allocation in stream_text(). The
additional test then requires error+[DONE] and post-fault reuse. The standard
tests never require that modified scratch binary.

For the independent legacy output-vector growth exception, set
HERMES_OUTPUT_PUSH_FAULT_BINARY to a scratch-only CPU fake binary compiled
with a single bad_alloc immediately before legacy_token()'s output.push_back.

Example:
  python3 serving/tests/test_http_unicode_stop_terminal_cpu.py \
    --server /tmp/cpu/build/hermes_server --model /tmp/cpu/fake.gguf \
    --gguf-dump /tmp/cpu/gguf_dump_fake.py \
    --tokenizer /opt/hermes/models/.../tokenizer.json \
    --python /opt/hermes/runtimes/exllamav3-venv/bin/python
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import unittest
import urllib.error
import urllib.request

from prod_sched_state_test import (
    ARGS, assert_single_owner, payload, running_server, trace_lines,
)


MODES = (
    ("legacy", False, (), {"FAKE_STEP_MS": "0"}),
    ("continuous", True, (), {"FAKE_STEP_MS": "0"}),
    ("ready", True, ("--parallel-steps", "--ready-completion"),
     {"HERMES_PP2_ASYNC_STREAMS": "1", "FAKE_STEP_MS": "0"}),
)


def post(url: str, body: dict, *, endpoint="/v1/chat/completions",
         timeout=10) -> tuple[int, bytes]:
    assert url.startswith("http://127.0.0.1:"), url
    request = urllib.request.Request(
        url + endpoint, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def parse_events(raw: bytes) -> tuple[list[dict], int]:
    blocks = [block for block in raw.decode("utf-8", "strict").split("\n\n")
              if block]
    assert blocks and all(block.startswith("data: ") for block in blocks), raw
    assert blocks[-1] == "data: [DONE]", blocks
    done = sum(block == "data: [DONE]" for block in blocks)
    events = [json.loads(block[6:]) for block in blocks
              if block != "data: [DONE]"]
    return events, done


def analyze(events: list[dict]) -> tuple[str, list[str], list[dict], list[dict]]:
    content, reasons, usage, errors = [], [], [], []
    for event in events:
        if "error" in event:
            errors.append(event)
            continue
        if event.get("usage") is not None:
            usage.append(event["usage"])
        for choice in event.get("choices", []):
            content.append(choice["delta"].get("content", ""))
            if choice["finish_reason"] is not None:
                reasons.append(choice["finish_reason"])
    return "".join(content), reasons, usage, errors


def fault_python(path: Path, mode: str) -> Path:
    """Wrap the original bridge at runtime; no shipped source is patched.

    server.cpp constructs tokenizer_bridge.py from its own scratch location
    and executes the --python path. This wrapper imports that exact bridge,
    injects one Python exception after stream_start, then calls its real main.
    """
    assert mode in ("push", "finish")
    wrapper = path / f"fault_{mode}_python"
    # argv[1] is always the real bridge path supplied by TokenizerBridge.
    wrapper.write_text(
        f"#!{ARGS.python}\n"
        "import importlib.util\n"
        "import sys\n"
        "bridge_path = sys.argv[1]\n"
        "spec = importlib.util.spec_from_file_location('real_hermes_bridge', bridge_path)\n"
        "bridge = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(bridge)\n"
        f"original = bridge.IncrementalByteLevel.{mode}\n"
        "fired = False\n"
        "def once(self, *args):\n"
        "    global fired\n"
        "    if not fired:\n"
        "        fired = True\n"
        f"        raise RuntimeError('injected stream_{mode} exception')\n"
        "    return original(self, *args)\n"
        f"bridge.IncrementalByteLevel.{mode} = once\n"
        "sys.argv = sys.argv[1:]\n"
        "raise SystemExit(bridge.main())\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o700)
    return wrapper


class UnicodeStopTerminalCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        cls.tokenizer = Tokenizer.from_file(ARGS.tokenizer)
        # Validate that this is the actual ByteLevel tokenizer: every split
        # UTF-8 byte below must have a distinct real single-byte token ID.
        import importlib.util
        helper = Path(ARGS.server).resolve().parent.parent / "serving/tokenizer_bridge.py"
        spec = importlib.util.spec_from_file_location("cpu_http_bytelevel", helper)
        bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bridge)
        cls.alphabet = {byte: char for char, byte in
                        bridge.bytelevel_alphabet().items()}
        if bridge.prepare_incremental(cls.tokenizer, ARGS.tokenizer) is None:
            raise unittest.SkipTest("fixture must use the verified real ByteLevel decoder")

    def raw_bytes(self, data: bytes) -> list[int]:
        ids = [self.tokenizer.token_to_id(self.alphabet[value]) for value in data]
        self.assertTrue(all(token is not None for token in ids), ids)
        return ids

    def ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def assert_sse_success(self, raw: bytes, *, text: str, reason: str,
                           completion_tokens: int):
        events, done = parse_events(raw)
        self.assertEqual(done, 1)
        output, reasons, usage, errors = analyze(events)
        self.assertEqual(errors, [], raw)
        self.assertEqual((output, reasons), (text, [reason]), raw)
        self.assertEqual(len(usage), 1, raw)
        self.assertEqual(usage[0]["completion_tokens"], completion_tokens, raw)
        self.assertEqual(usage[0]["total_tokens"],
                         usage[0]["prompt_tokens"] + completion_tokens)

    def assert_sse_error(self, raw: bytes):
        events, done = parse_events(raw)
        self.assertEqual(done, 1)
        _, reasons, usage, errors = analyze(events)
        self.assertEqual(reasons, [], raw)
        self.assertEqual(usage, [], raw)
        self.assertEqual(len(errors), 1, raw)
        self.assertEqual(errors[0]["error"]["type"], "server_error", raw)

    def test_emoji_split_into_four_raw_byte_tokens_stream_and_nonstream(self):
        generated = self.raw_bytes("🚀".encode("utf-8")) + self.ids(" end")
        self.assertEqual(self.tokenizer.decode(generated), "🚀 end")
        for label, continuous, flags, env in MODES:
            with self.subTest(mode=label):
                with running_server(generated, flags=flags, continuous=continuous,
                                    env_extra=env) as (url, _proc, trace):
                    for stream in (False, True):
                        status, raw = post(url, payload(
                            len(generated), stream=stream, include_usage=stream))
                        self.assertEqual(status, 200, raw)
                        if stream:
                            self.assert_sse_success(raw, text="🚀 end", reason="length",
                                                    completion_tokens=len(generated))
                        else:
                            body = json.loads(raw)
                            self.assertEqual(body["choices"][0]["message"]["content"],
                                             "🚀 end")
                    assert_single_owner(self, trace_lines(trace), requests=2)

    def test_stop_crosses_multibyte_and_token_boundaries_no_leak(self):
        prefix = self.ids("before ")
        suffix = self.raw_bytes("🚀".encode("utf-8")) + self.ids("END")
        generated = prefix + suffix + self.ids(" after")
        expected = self.tokenizer.decode(generated)
        self.assertIn("🚀END", expected)
        expected_before_stop = expected.split("🚀END", 1)[0]
        for label, continuous, flags, env in MODES:
            with self.subTest(mode=label):
                with running_server(generated, flags=flags, continuous=continuous,
                                    env_extra=env) as (url, _proc, trace):
                    for stream in (True, False):
                        status, raw = post(url, payload(
                            len(generated), stream=stream,
                            stops=["🚀END", "not present"], include_usage=stream))
                        self.assertEqual(status, 200, raw)
                        if stream:
                            self.assert_sse_success(raw, text=expected_before_stop,
                                                    reason="stop",
                                                    completion_tokens=len(prefix + suffix))
                        else:
                            result = json.loads(raw)
                            self.assertEqual(result["choices"][0]["message"]["content"],
                                             expected_before_stop)
                            self.assertEqual(result["choices"][0]["finish_reason"], "stop")
                    assert_single_owner(self, trace_lines(trace), requests=2)

    def test_partial_stop_prefix_flush_and_incomplete_utf8_at_finish(self):
        complete_emoji = self.raw_bytes("🚀".encode("utf-8"))
        incomplete = complete_emoji[:-1]
        self.assertEqual(self.tokenizer.decode(incomplete), "�")
        for label, continuous, flags, env in MODES:
            with self.subTest(mode=label):
                with running_server(complete_emoji, flags=flags, continuous=continuous,
                                    env_extra=env) as (url, _proc, trace):
                    status, raw = post(url, payload(
                        len(complete_emoji), stream=True, stops="🚀missing",
                        include_usage=True))
                    self.assertEqual(status, 200, raw)
                    self.assert_sse_success(raw, text="🚀", reason="length",
                                            completion_tokens=len(complete_emoji))
                    assert_single_owner(self, trace_lines(trace), requests=1)
                with running_server(incomplete, flags=flags, continuous=continuous,
                                    env_extra=env) as (url, _proc, trace):
                    status, raw = post(url, payload(
                        len(incomplete), stream=True, stops="�",
                        include_usage=True))
                    self.assertEqual(status, 200, raw)
                    self.assert_sse_success(raw, text="", reason="stop",
                                            completion_tokens=len(incomplete))
                    assert_single_owner(self, trace_lines(trace), requests=1)

    def test_actual_tokenizer_exception_after_stream_start_closes_state(self):
        generated = [self.ids("a")[0], self.ids("b")[0]]
        for kind in ("push", "finish"):
            for label, continuous, flags, env in MODES:
                with self.subTest(kind=kind, mode=label):
                    with tempfile.TemporaryDirectory(prefix="hermes-tokenizer-fault-") as tmp:
                        wrapper = fault_python(Path(tmp), kind)
                        with running_server(
                            generated, flags=(*flags, "--python", str(wrapper)),
                            continuous=continuous, env_extra=env,
                        ) as (url, _proc, trace):
                            request = payload(len(generated), stream=True,
                                              include_usage=True)
                            status, raw = post(url, request)
                            self.assertEqual(status, 200, raw)
                            self.assert_sse_error(raw)
                            status, raw = post(url, request)
                            self.assertEqual(status, 200, raw)
                            self.assert_sse_success(
                                raw, text=self.tokenizer.decode(generated),
                                reason="length", completion_tokens=len(generated))
                            assert_single_owner(self, trace_lines(trace), requests=2)

    def test_concurrent_one_terminal_allocation_fault_other_stays_valid(self):
        token = self.ids("a")[0]
        ready_flags = ("--parallel-steps", "--ready-completion", "--active", "2")
        env = {"HERMES_PP2_ASYNC_STREAMS": "1", "FAKE_STEP_MS": "0",
               "HERMES_TEST_FAIL_TERMINAL_USAGE_ONCE": "1"}
        with running_server([token], continuous=True, flags=ready_flags,
                            env_extra=env) as (url, _proc, trace):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(post, url, payload(
                    2, stream=True, include_usage=True)) for _ in range(2)]
                outcomes = [future.result(timeout=12) for future in futures]
            for status, _ in outcomes:
                self.assertEqual(status, 200)
            values = [analyze(parse_events(raw)[0]) for _, raw in outcomes]
            failed = [i for i, item in enumerate(values) if item[3]]
            self.assertEqual(len(failed), 1, outcomes)
            self.assert_sse_error(outcomes[failed[0]][1])
            self.assert_sse_success(outcomes[1 - failed[0]][1],
                                    text=self.tokenizer.decode([token] * 2),
                                    reason="length", completion_tokens=2)
            assert_single_owner(self, trace_lines(trace), requests=2)

    @unittest.skipUnless(os.environ.get("HERMES_STREAM_CHUNK_FAULT_BINARY"),
                         "requires a separately compiled CPU-only scratch fault witness")
    def test_stream_text_bad_alloc_must_fail_one_request_not_server(self):
        # stream_text() performs a fallible JSON/string allocation. An
        # exception from its call site must produce one failed SSE terminal
        # without escaping the native scheduler thread and killing the server.
        # The injection is in an isolated copy of the real source; this file
        # does not modify/monkeypatch the production server or fake PP2.
        fault_binary = os.environ["HERMES_STREAM_CHUNK_FAULT_BINARY"]
        self.assertTrue(Path(fault_binary).is_file(), fault_binary)
        token = self.ids("a")[0]
        saved_binary = ARGS.server
        try:
            ARGS.server = fault_binary
            for label, continuous, flags, env in MODES[::2]:
                with self.subTest(mode=label):
                    with running_server([token], flags=flags,
                                        continuous=continuous, env_extra={
                                            **env, "HERMES_TEST_FAIL_STREAM_TEXT_ONCE": "1",
                                        }) as (url, _proc, trace):
                        request = payload(2, stream=True, include_usage=True)
                        try:
                            status, raw = post(url, request)
                        except Exception as exc:
                            self.fail(f"{label} abandoned SSE transport after "
                                      f"stream_text allocation failure: {exc!r}")
                        self.assertEqual(status, 200, raw)
                        self.assert_sse_error(raw)
                        status, raw = post(url, request)
                        self.assertEqual(status, 200, raw)
                        self.assert_sse_success(
                            raw, text=self.tokenizer.decode([token] * 2),
                            reason="length", completion_tokens=2)
                        assert_single_owner(self, trace_lines(trace), requests=2)
        finally:
            ARGS.server = saved_binary

    @unittest.skipUnless(os.environ.get("HERMES_OUTPUT_PUSH_FAULT_BINARY"),
                         "requires a CPU-only scratch output-growth fault witness")
    def test_legacy_output_growth_bad_alloc_terminalizes_and_recovers(self):
        # std::vector::push_back may itself allocate before the tokenizer is
        # called; a short-lived simulated allocation failure must never
        # escape the legacy PP2 scheduler thread into std::terminate.
        fault_binary = os.environ["HERMES_OUTPUT_PUSH_FAULT_BINARY"]
        self.assertTrue(Path(fault_binary).is_file(), fault_binary)
        token = self.ids("a")[0]
        saved_binary = ARGS.server
        try:
            ARGS.server = fault_binary
            with running_server(
                [token], flags=("--active", "1"), continuous=False,
                env_extra={"FAKE_STEP_MS": "0",
                           "HERMES_TEST_FAIL_OUTPUT_PUSH_ONCE": "1"},
            ) as (url, _proc, trace):
                request = payload(2, stream=True, include_usage=True)
                try:
                    status, raw = post(url, request)
                except Exception as exc:
                    self.fail("legacy SSE transport broke after output-vector "
                              f"allocation failure: {exc!r}")
                self.assertEqual(status, 200, raw)
                self.assert_sse_error(raw)
                status, raw = post(url, request)
                self.assertEqual(status, 200, raw)
                self.assert_sse_success(
                    raw, text=self.tokenizer.decode([token] * 2),
                    reason="length", completion_tokens=2)
                assert_single_owner(self, trace_lines(trace), requests=2)
        finally:
            ARGS.server = saved_binary

    def test_concurrent_cancel_then_other_client_completes_once(self):
        token = self.ids("x")[0]
        for label, continuous, flags, env in MODES[::2]:
            with self.subTest(mode=label):
                with running_server([token], continuous=continuous,
                                    flags=(*flags, "--active", "2"),
                                    env_extra={**env, "FAKE_STEP_MS": "45"},
                                    ) as (url, _proc, trace):
                    initial = payload(45, stream=True, include_usage=True)
                    req = urllib.request.Request(
                        url + "/v1/chat/completions",
                        data=json.dumps(initial).encode(),
                        headers={"Content-Type": "application/json"}, method="POST")
                    with urllib.request.urlopen(req, timeout=10) as response:
                        seen = []
                        request_id = None
                        for _ in range(100):
                            line = response.readline()
                            self.assertTrue(line, "stream closed before first content")
                            seen.append(line)
                            if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                                event = json.loads(line[6:])
                                request_id = event.get("id", request_id)
                                if any(choice.get("delta", {}).get("content")
                                       for choice in event.get("choices", [])):
                                    break
                        else:
                            self.fail("no generated token received before cancellation")
                        with ThreadPoolExecutor(max_workers=1) as pool:
                            other = pool.submit(post, url, payload(
                                2, stream=True, include_usage=True))
                            status, raw = post(url, {"id": request_id},
                                               endpoint="/v1/cancel")
                            self.assertEqual(status, 202, raw)
                            cancelled = b"".join(seen) + response.read()
                            other_status, other_raw = other.result(timeout=10)
                    output, reasons, usage, errors = analyze(parse_events(cancelled)[0])
                    self.assertEqual(errors, [], cancelled)
                    self.assertTrue(output, "cancel must follow a visible token")
                    self.assertEqual(reasons, ["cancelled"])
                    self.assertEqual(len(usage), 1)
                    count = usage[0]["completion_tokens"]
                    self.assertGreaterEqual(count, 1)
                    self.assertLess(count, 45)
                    self.assertIn(output,
                                  [self.tokenizer.decode([token] * n)
                                   for n in range(count + 1)])
                    self.assert_sse_success(
                        cancelled, text=output, reason="cancelled",
                        completion_tokens=count)
                    self.assertEqual(other_status, 200)
                    self.assert_sse_success(other_raw,
                                            text=self.tokenizer.decode([token] * 2),
                                            reason="length", completion_tokens=2)
                    assert_single_owner(self, trace_lines(trace), requests=2)


if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)
