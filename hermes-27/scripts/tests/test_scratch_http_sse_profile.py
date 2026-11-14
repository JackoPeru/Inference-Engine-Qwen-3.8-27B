"""Deterministic, purely offline synthetic HTTP/SSE client tests.

Importing this file and running tests NEVER creates a real HTTPConnection,
GPU allocation, subprocess, service invocation or external request.
"""

from collections import deque
from dataclasses import replace
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import socket
import sys
import threading
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "scratch_http_sse_profile.py"
SPEC = importlib.util.spec_from_file_location("h27_scratch_http_sse_profile_test", SOURCE)
p = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = p
SPEC.loader.exec_module(p)


class Clock:
    def __init__(self, initial=100.0):
        self.now = initial
        self.lock = threading.RLock()

    def __call__(self):
        with self.lock:
            return self.now

    def advance(self, seconds):
        with self.lock:
            self.now += seconds


def frame(event, *, prefix=b"data: ", crlf=False):
    if isinstance(event, str):
        payload = event.encode("utf-8")
    else:
        payload = json.dumps(event, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ending = b"\r\n\r\n" if crlf else b"\n\n"
    return prefix + payload + ending


def role():
    return {"id": "chatcmpl-offline-123", "choices": [
        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None},
    ]}


def content(text):
    return {"id": "chatcmpl-offline-123", "choices": [
        {"index": 0, "delta": {"content": text}, "finish_reason": None},
    ]}


def reasoning(text):
    return {"id": "chatcmpl-offline-123", "choices": [
        {"index": 0, "delta": {"reasoning_content": text}, "finish_reason": None},
    ]}


def finish(reason="stop"):
    return {"id": "chatcmpl-offline-123", "choices": [
        {"index": 0, "delta": {}, "finish_reason": reason},
    ]}


def usage(count=4, prompt=17):
    return {"id": "chatcmpl-offline-123", "choices": [],
            "usage": {"prompt_tokens": prompt, "completion_tokens": count,
                      "total_tokens": count + prompt}}


def events(*, words=("Hello", " world"), count=4, reasoning_text=None):
    pieces = [role()]
    if reasoning_text is not None:
        pieces.append(reasoning(reasoning_text))
    pieces.extend(content(word) for word in words)
    pieces.extend((finish(), usage(count), "[DONE]"))
    return [frame(piece) for piece in pieces]


class FakeResponse:
    def __init__(self, segments, *, clock, status=200,
                 content_type="text/event-stream"):
        self.status = status
        self.content_type = content_type
        self.clock = clock
        self.segments = deque(
            (chunk, float(delay)) for chunk, delay in segments
        )
        self.read_calls = 0

    def getheader(self, key, fallback=None):
        if key.lower() == "content-type":
            return self.content_type
        return fallback

    def read1(self, amount):
        self.read_calls += 1
        if not self.segments:
            return b""
        data, delta = self.segments.popleft()
        self.clock.advance(delta)
        if len(data) > amount:
            self.segments.appendleft((data[amount:], 0.0))
            data = data[:amount]
        return data


class FakeConnection:
    def __init__(self, response, *, host, port, timeout, on_request=None):
        self.response = response
        self.host = host
        self.port = port
        self.timeout = timeout
        self.on_request = on_request
        self.requests = []
        self.closed = False
        self.sock = None

    def request(self, method, path, body=None, headers=None):
        data = (method, path, body, dict(headers or {}))
        self.requests.append(data)
        if self.on_request is not None:
            self.on_request(data)

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


class FakeFactory:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.connections = []
        self.lock = threading.RLock()

    def __call__(self, host, port, *, timeout):
        with self.lock:
            if not self.responses:
                raise AssertionError("no injected fake HTTP response left")
            connection = FakeConnection(
                self.responses.popleft(), host=host, port=port, timeout=timeout,
            )
            self.connections.append(connection)
            return connection


def fake_stream(clock, *, count=4, words=("Hello", " world"),
                reasoning_text=None, status=200,
                content_type="text/event-stream", coalesced=False):
    parts = events(words=words, count=count, reasoning_text=reasoning_text)
    if coalesced:
        return FakeResponse([(b"".join(parts), 0.4)], clock=clock,
                            status=status, content_type=content_type)
    timings = (.1, .3, .2, .1, .1, .1, .05)
    return FakeResponse([
        (part, timings[min(index, len(timings) - 1)])
        for index, part in enumerate(parts)
    ], clock=clock, status=status, content_type=content_type)


class ProtocolAndMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.config = p.Config(
            concurrency=(1,), rounds=1, max_tokens=12, timeout_s=10,
            run_label="unit",
        )

    def measure(self, response, **changes):
        config = replace(self.config, **changes)
        factory = FakeFactory([response])
        value = p.profile_request(
            config, clients=1, round_index=1, case=1,
            clock=self.clock, connection_factory=factory,
        )
        self.assertEqual(len(factory.connections), 1)
        self.assertTrue(factory.connections[0].closed)
        self.assertEqual(factory.connections[0].host, p.HOST)
        self.assertEqual(factory.connections[0].port, p.PORT)
        return value, factory.connections[0]

    def test_semantic_ttfc_excludes_role_and_headers(self):
        result, conn = self.measure(fake_stream(self.clock))
        self.assertEqual(result["status"], "OK")
        self.assertAlmostEqual(result["ttfc_ms"], 400.0, places=5)
        self.assertAlmostEqual(result["semantic_fragment_gaps_ms"][0], 200.0, places=5)
        self.assertEqual(result["semantic_fragments"], 2)
        self.assertEqual(result["role_only_events"], 1)
        self.assertEqual(result["completion_tokens"], 4)
        self.assertEqual(result["prompt_tokens"], 17)
        self.assertEqual(result["total_tokens"], 21)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertTrue(result["usage_final"])
        self.assertTrue(result["sse_done"])
        self.assertEqual(result["sse_completion_id"], "chatcmpl-offline-123")
        self.assertIsNone(result["ttft_ms"])
        self.assertIsNone(result["tpot_ms"])
        self.assertGreater(result["completion_tokens_per_wall_second"], 0)
        self.assertEqual(conn.requests[0][0], "POST")
        self.assertEqual(conn.requests[0][1], "/v1/chat/completions")
        self.assertEqual(
            conn.requests[0][3]["X-Profile-Request-ID"], result["request_id"],
        )
        body = json.loads(conn.requests[0][2])
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["max_tokens"], self.config.max_tokens)

    def test_no_true_tpot_when_two_sse_fragments_coalesce_in_one_read(self):
        result, _ = self.measure(fake_stream(self.clock, coalesced=True))
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["semantic_fragments"], 2)
        self.assertEqual(result["coalesced_adjacent_semantic_events"], 1)
        self.assertEqual(result["semantic_fragment_gaps_ms"], [0.0])
        self.assertIsNone(result["tpot_ms"])
        self.assertIsNone(result["ttft_ms"])

    def test_reasoning_semantic_delta_can_be_first(self):
        result, _ = self.measure(fake_stream(
            self.clock, words=("final",), count=5, reasoning_text="thinking",
        ))
        self.assertEqual(result["status"], "OK")
        self.assertAlmostEqual(result["ttfc_ms"], 400.0, places=5)
        self.assertEqual(result["semantic_fragments"], 2)
        self.assertGreater(result["generated_reasoning_bytes"], 0)

    def test_utf8_split_between_arbitrary_socket_reads(self):
        text = "L'été à Napoli: 中文 😀"
        segments = b"".join(events(words=(text,), count=7))
        # Simulates HTTP transport that divides a multi-byte character.
        pos = segments.index("😀".encode("utf-8")) + 1
        response = FakeResponse(
            [(segments[:pos], .25), (segments[pos:], .4)],
            clock=self.clock,
        )
        result, _ = self.measure(response)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["generated_text_bytes"], len(text.encode("utf-8")))
        self.assertEqual(result["semantic_fragments"], 1)

    def test_crlf_and_data_space_are_accepted(self):
        chunks = [frame(event, crlf=True) for event in (
            role(), content("yes"), finish(), usage(1), "[DONE]",
        )]
        result, _ = self.measure(FakeResponse(
            [(b"".join(chunks), .3)], clock=self.clock,
        ))
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["completion_tokens"], 1)

    def test_multiline_data_event_accepts_json_whitespace(self):
        item = json.dumps(content("hello"), separators=(",", ":"))
        split = item.index('"choices"')
        multi = (
            b"data: " + item[:split].encode() + b"\n"
            + b"data: " + item[split:].encode() + b"\n\n"
        )
        stream = b"".join([
            frame(role()), multi, frame(finish()), frame(usage(2)),
            frame("[DONE]"),
        ])
        result, _ = self.measure(FakeResponse(
            [(stream, .4)], clock=self.clock,
        ))
        self.assertEqual(result["status"], "OK")

    def test_sse_comment_is_not_first_semantic_event(self):
        stream = b": heartbeat\n\n" + b"".join(events(words=("hi",), count=1))
        result, _ = self.measure(FakeResponse(
            [(stream, .4)], clock=self.clock,
        ))
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["semantic_fragments"], 1)

    def test_single_final_content_event_with_finish_is_accepted(self):
        terminal = content("end")
        terminal["choices"][0]["finish_reason"] = "stop"
        stream = b"".join([
            frame(role()), frame(terminal), frame(usage(2)), frame("[DONE]"),
        ])
        result, _ = self.measure(FakeResponse(
            [(stream, .1)], clock=self.clock,
        ))
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["finish_reason"], "stop")

    def test_true_usage_not_equated_with_fragment_count(self):
        result, _ = self.measure(fake_stream(self.clock, count=11))
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["semantic_fragments"], 2)
        self.assertEqual(result["completion_tokens"], 11)
        self.assertIsNone(result["tpot_ms"])

    def test_bounded_request_identifier_is_stable_across_repeat_runs(self):
        response1 = fake_stream(self.clock)
        first, _ = self.measure(response1)
        response2 = fake_stream(self.clock)
        second, _ = self.measure(response2)
        self.assertEqual(first["request_id"], "h27-unit-n1-r01-q01")
        self.assertEqual(first["request_id"], second["request_id"])

    def test_no_redirect_302_is_hard_error_not_followed(self):
        result, conn = self.measure(FakeResponse(
            [], clock=self.clock, status=302,
            content_type="text/event-stream",
        ))
        self.assertEqual(result["status"], "ERROR")
        self.assertEqual(result["error"], "http_status_not_200")
        self.assertEqual(result["http_status"], 302)
        self.assertEqual(len(conn.requests), 1)

    def test_non_sse_content_type_rejected(self):
        result, _ = self.measure(fake_stream(
            self.clock, content_type="application/json",
        ))
        self.assertEqual(result["error"], "not_sse_content_type")

    def test_unapproved_sse_charset_rejected(self):
        result, _ = self.measure(fake_stream(
            self.clock, content_type="text/event-stream; charset=iso-8859-1",
        ))
        self.assertEqual(result["error"], "unsupported_sse_charset")

    def test_missing_done_rejected(self):
        pieces = events()[:-1]
        result, _ = self.measure(FakeResponse(
            [(b"".join(pieces), .3)], clock=self.clock,
        ))
        self.assertEqual(result["error"], "missing_done")
        self.assertIsNone(result["completion_tokens_per_wall_second"])

    def test_missing_usage_rejected_without_guessed_token_count(self):
        pieces = events()
        pieces.pop(-2)
        result, _ = self.measure(FakeResponse(
            [(b"".join(pieces), .3)], clock=self.clock,
        ))
        self.assertEqual(result["error"], "missing_final_usage")
        self.assertIsNone(result["completion_tokens"])
        self.assertIsNone(result["completion_tokens_per_wall_second"])

    def test_missing_finish_is_error(self):
        pieces = [frame(role()), frame(content("x")), frame(usage(2)),
                  frame("[DONE]")]
        result, _ = self.measure(FakeResponse(
            [(b"".join(pieces), .3)], clock=self.clock,
        ))
        self.assertEqual(result["error"], "missing_finish_reason")

    def test_duplicate_usage_is_error(self):
        pieces = events()
        pieces.insert(-1, frame(usage()))
        result, _ = self.measure(FakeResponse(
            [(b"".join(pieces), .3)], clock=self.clock,
        ))
        self.assertEqual(result["error"], "invalid_or_duplicate_usage")

    def test_duplicate_done_is_error(self):
        parts = b"".join(events() + [frame("[DONE]")])
        result, _ = self.measure(FakeResponse(
            [(parts, .3)], clock=self.clock,
        ))
        self.assertEqual(result["error"], "data_after_done")

    def test_sse_data_after_done_is_error(self):
        parts = b"".join(events()) + frame(content("late"))
        result, _ = self.measure(FakeResponse(
            [(parts, .3)], clock=self.clock,
        ))
        self.assertEqual(result["error"], "data_after_done")

    def test_generated_data_after_prior_finish_is_error(self):
        pieces = [frame(role()), frame(content("first")), frame(finish()),
                  frame(content("late")), frame(usage()), frame("[DONE]")]
        result, _ = self.measure(FakeResponse(
            [(b"".join(pieces), .3)], clock=self.clock,
        ))
        self.assertEqual(result["error"], "semantic_after_finish_or_usage")

    def test_nontext_content_is_rejected(self):
        malformed = content("x")
        malformed["choices"][0]["delta"]["content"] = [{"text": "x"}]
        parts = [frame(role()), frame(malformed), frame(finish()),
                 frame(usage()), frame("[DONE]")]
        result, _ = self.measure(FakeResponse(
            [(b"".join(parts), .3)], clock=self.clock,
        ))
        self.assertEqual(result["error"], "nontext_semantic_delta")

    def test_tool_calls_are_not_silently_counted_as_text(self):
        malformed = content("x")
        malformed["choices"][0]["delta"]["tool_calls"] = []
        parts = [frame(role()), frame(malformed), frame(finish()),
                 frame(usage()), frame("[DONE]")]
        result, _ = self.measure(FakeResponse(
            [(b"".join(parts), .3)], clock=self.clock,
        ))
        self.assertEqual(result["error"], "non_text_generation_not_supported")

    def test_changed_completion_id_fails_closed(self):
        bad = content("next")
        bad["id"] = "chatcmpl-different"
        parts = [frame(role()), frame(bad), frame(finish()),
                 frame(usage()), frame("[DONE]")]
        result, _ = self.measure(FakeResponse(
            [(b"".join(parts), .3)], clock=self.clock,
        ))
        self.assertEqual(result["error"], "completion_id_changed")

    def test_server_sse_error_does_not_leak_message(self):
        secret = "raw-secret-prompt-should-not-be-logged"
        parts = b"".join([
            frame(role()), frame({"error": {"message": secret}}),
            frame("[DONE]"),
        ])
        result, _ = self.measure(FakeResponse(
            [(parts, .3)], clock=self.clock,
        ))
        self.assertEqual(result["error"], "server_sse_error")
        self.assertNotIn(secret, json.dumps(result))

    def test_bool_or_negative_usage_tokens_rejected(self):
        for value in (True, False, -1, 4.2):
            with self.subTest(value=value):
                invalid = usage()
                invalid["usage"]["completion_tokens"] = value
                parts = [frame(role()), frame(content("hi")),
                         frame(finish()), frame(invalid), frame("[DONE]")]
                result, _ = self.measure(FakeResponse(
                    [(b"".join(parts), .3)], clock=self.clock,
                ))
                self.assertEqual(result["error"], "invalid_usage_counts")

    def test_usage_above_max_tokens_rejected(self):
        result, _ = self.measure(fake_stream(self.clock, count=13))
        self.assertEqual(result["error"], "completion_usage_out_of_range")

    def test_usage_inconsistent_total_rejected(self):
        bad = usage()
        bad["usage"]["total_tokens"] = 999
        parts = [frame(role()), frame(content("hi")),
                 frame(finish()), frame(bad), frame("[DONE]")]
        result, _ = self.measure(FakeResponse(
            [(b"".join(parts), .3)], clock=self.clock,
        ))
        self.assertEqual(result["error"], "inconsistent_usage_total")

    def test_max_sse_bytes_enforced_across_multiple_reads(self):
        huge = b": " + b"A" * 1090 + b"\n\n"
        result, _ = self.measure(FakeResponse(
            [(huge[:600], .1), (huge[600:], .1)],
            clock=self.clock,
        ), max_sse_bytes=1024)
        self.assertEqual(result["error"], "sse_max_bytes")
        self.assertIsNone(result["completion_tokens_per_wall_second"])

    def test_max_line_bytes_enforced(self):
        huge = b": " + b"A" * (p.MAX_LINE_BYTES + 5)
        result, _ = self.measure(FakeResponse(
            [(huge, .1)], clock=self.clock,
        ), max_sse_bytes=2 * p.MAX_LINE_BYTES)
        self.assertEqual(result["error"], "sse_line_too_long")

    def test_invalid_utf8_event_fails(self):
        parts = b"data: \xff\xfe\n\n" + b"".join(events())
        result, _ = self.measure(FakeResponse(
            [(parts, .1)], clock=self.clock,
        ))
        self.assertEqual(result["error"], "invalid_utf8_sse")

    def test_unsupported_sse_event_type_fails(self):
        parts = b"event: custom\ndata: {\"choices\":[]}\n\n"
        result, _ = self.measure(FakeResponse(
            [(parts, .1)], clock=self.clock,
        ))
        self.assertEqual(result["error"], "unknown_sse_event_type")

    def test_broken_json_fails(self):
        parts = b"data: {bad-json}\n\n" + b"".join(events())
        result, _ = self.measure(FakeResponse(
            [(parts, .1)], clock=self.clock,
        ))
        self.assertEqual(result["error"], "malformed_sse_json")

    def test_cancelled_pre_connect_never_opens_connection(self):
        token = threading.Event()
        token.set()
        never = mock.Mock(side_effect=AssertionError("network forbidden"))
        outcome = p.profile_request(
            self.config, clients=1, round_index=1, case=1,
            clock=self.clock, connection_factory=never, cancelled=token,
        )
        self.assertEqual(outcome["error"], "cancelled")
        never.assert_not_called()

    def test_cancellation_during_stream_returns_error(self):
        token = threading.Event()
        class CancelResponse(FakeResponse):
            def read1(self, amount):
                data = super().read1(amount)
                token.set()
                return data
        response = CancelResponse(
            [(b"".join(events()), .1)], clock=self.clock,
        )
        factory = FakeFactory([response])
        result = p.profile_request(
            self.config, clients=1, round_index=1, case=1,
            clock=self.clock, connection_factory=factory, cancelled=token,
        )
        self.assertEqual(result["error"], "cancelled")
        self.assertTrue(factory.connections[0].closed)

    def test_absolute_wall_timeout_is_enforced_even_if_data_arrives(self):
        result, _ = self.measure(FakeResponse(
            [(b"".join(events()), 11.0)], clock=self.clock,
        ))
        self.assertEqual(result["error"], "timeout")

    def test_socket_timeout_is_not_sse_completion(self):
        class TimeoutResponse(FakeResponse):
            def read1(self, size):
                raise socket.timeout("fake transport stalls")
        response = TimeoutResponse([], clock=self.clock)
        result, _ = self.measure(response)
        self.assertEqual(result["error"], "timeout")

    def test_request_validation_bounds_reject_invalid_run_early(self):
        invalid = (
            {"concurrency": (3,)}, {"concurrency": (1, 1)},
            {"max_tokens": 0}, {"rounds": 17},
            {"timeout_s": float("nan")},
            {"run_label": "../unsafe"}, {"model": "http://external"},
            {"read_bytes": 0}, {"max_sse_bytes": 100},
            {"prompt": "X" * 8193},
        )
        for case in invalid:
            with self.subTest(case=case), self.assertRaises(ValueError):
                replace(self.config, **case).validate()

    def test_type7_percentile_and_no_fake_zero_for_empty(self):
        self.assertEqual(p.percentile([], .95), None)
        self.assertEqual(p.distribution([]), {"count": 0, "p50": None, "p95": None})
        self.assertEqual(p.percentile([0., 100.], .95), 95.)
        self.assertEqual(p.percentile([4.], .95), 4.)
        with self.assertRaises(ValueError):
            p.percentile([1.], 1.01)

    def test_output_hash_is_invariant_to_sse_fragment_boundary(self):
        first, _ = self.measure(fake_stream(
            self.clock, words=("hello ", "world"), count=2,
        ))
        second, _ = self.measure(fake_stream(
            self.clock, words=("hel", "lo world"), count=2,
        ))
        self.assertEqual(first["status"], "OK")
        self.assertEqual(second["status"], "OK")
        self.assertEqual(first["semantic_output_sha256"],
                         second["semantic_output_sha256"])
        self.assertIsNone(first["tpot_ms"])


class WorkloadAndMetricsTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()

    def test_all_n_levels_and_two_repeated_rounds_offline(self):
        levels = (1, 2, 4, 8)
        rounds = 2
        expected = rounds * sum(levels)
        injected = FakeFactory([
            fake_stream(self.clock, coalesced=(i % 3 == 0))
            for i in range(expected)
        ])
        report = p.run_profile(
            p.Config(
                concurrency=levels, rounds=rounds, timeout_s=10,
                run_label="offline",
            ),
            clock=self.clock,
            connection_factory=injected,
        )
        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(report["schema"], p.SCHEMA)
        self.assertEqual(report["overall"]["attempts"], 30)
        self.assertEqual(report["overall"]["ok"], 30)
        self.assertGreater(report["overall"][
            "aggregate_completion_tokens_per_round_wall_second"], 0)
        self.assertIsNone(report["tpot_ms"])
        self.assertIsNone(report["ttft_ms"])
        self.assertEqual(len(injected.connections), expected)
        ids = set()
        for group, n in zip(report["by_concurrency"], levels):
            self.assertEqual(group["clients"], n)
            self.assertEqual(group["summary"]["attempts"], n * rounds)
            self.assertTrue(group["summary"]["all_requests_complete"])
            self.assertGreater(group["summary"][
                "aggregate_completion_tokens_per_round_wall_second"], 0)
            self.assertEqual(group["summary"]["ttfc_ms"]["count"], n * rounds)
            self.assertEqual(group["summary"]["request_wall_ms"]["count"], n * rounds)
            self.assertEqual(len(group["rounds"]), 2)
            for rnd, record in enumerate(group["rounds"], 1):
                self.assertEqual(record["round"], rnd)
                self.assertEqual(record["stats"]["attempts"], n)
                self.assertGreater(record["stats"]["semantic_fragment_gap_ms"]["count"], 0)
                self.assertEqual(len(record["requests"]), n)
                for case, req in enumerate(record["requests"], 1):
                    self.assertEqual(req["request_id"],
                                     f"h27-offline-n{n}-r{rnd:02d}-q{case:02d}")
                    self.assertEqual(req["status"], "OK")
                    self.assertIsNone(req["tpot_ms"])
                    ids.add(req["request_id"])
        self.assertEqual(len(ids), 30)
        self.assertTrue(all(conn.closed for conn in injected.connections))
        self.assertTrue(all(conn.host == "127.0.0.1" and conn.port == 18127
                            for conn in injected.connections))
        self.assertTrue(all(
            conn.requests[0][0] == "POST" and
            conn.requests[0][1] == "/v1/chat/completions"
            for conn in injected.connections
        ))

    def test_one_http_failure_invalidates_aggregate_rate_not_count(self):
        injected = FakeFactory([
            fake_stream(self.clock),
            fake_stream(self.clock, status=500),
        ])
        report = p.run_profile(
            p.Config(concurrency=(2,), rounds=1,
                     timeout_s=10, run_label="partial"),
            clock=self.clock, connection_factory=injected,
        )
        self.assertEqual(report["status"], "INCOMPLETE")
        stats = report["by_concurrency"][0]["rounds"][0]["stats"]
        self.assertEqual(stats["attempts"], 2)
        self.assertEqual(stats["errors"], 1)
        self.assertEqual(stats["ok"], 1)
        self.assertIsNone(stats["aggregate_completion_tokens_per_group_wall_second"])
        self.assertEqual(stats["completion_tokens_success_only"], 4)
        self.assertEqual(report["overall"]["errors"], 1)
        self.assertIsNone(report["overall"][
            "aggregate_completion_tokens_per_round_wall_second"])

    def test_pre_cancelled_run_represents_all_planned_requests(self):
        token = threading.Event()
        token.set()
        never = mock.Mock(side_effect=AssertionError("must not connect"))
        config = p.Config(concurrency=(1, 2, 4, 8), rounds=2)
        result = p.run_profile(
            config, connection_factory=never, clock=self.clock,
            cancelled=token,
        )
        never.assert_not_called()
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["overall"]["attempts"], 30)
        self.assertEqual(result["overall"]["errors"], 30)
        self.assertIsNone(result["overall"][
            "aggregate_completion_tokens_per_round_wall_second"])
        self.assertEqual(result["overall"]["ttfc_ms"]["count"], 0)

    def test_metrics_optional_does_not_create_any_get_by_default(self):
        factory = FakeFactory([fake_stream(self.clock)])
        result = p.run_profile(
            p.Config(concurrency=(1,), rounds=1, timeout_s=10, metrics=False),
            clock=self.clock, connection_factory=factory,
        )
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(len(factory.connections), 1)
        self.assertEqual(factory.connections[0].requests[0][0], "POST")
        self.assertIsNone(result["by_concurrency"][0]["rounds"][0]["metrics_before"])

    def test_optional_metrics_vram_get_only_fixed_route_and_units(self):
        metrics = (
            b"# TYPE nvidia_vram_used_bytes gauge\n"
            b'nvidia_vram_used_bytes{gpu="0"} 1048576\n'
            b'nvidia_vram_used_bytes{gpu="1"} 2097152\n'
            b"gpu_memory_allocated 10.5\n"
            b"random_metric 999999\n"
        )
        factory = FakeFactory([
            FakeResponse([(metrics, .1)], clock=self.clock),
            fake_stream(self.clock),
            FakeResponse([(metrics, .1)], clock=self.clock),
        ])
        outcome = p.run_profile(
            p.Config(
                concurrency=(1,), rounds=1,
                timeout_s=10, metrics=True,
            ),
            clock=self.clock, connection_factory=factory,
        )
        self.assertEqual(outcome["status"], "COMPLETE")
        rec = outcome["by_concurrency"][0]["rounds"][0]
        self.assertEqual(rec["metrics_before"]["status"], "OK")
        self.assertEqual(rec["metrics_after"]["status"], "OK")
        self.assertEqual(len(rec["metrics_before"]["series"]), 3)
        self.assertEqual(rec["metrics_before"]["series"][0]["unit"],
                         "bytes_from_metric_name")
        self.assertEqual(rec["metrics_before"]["series"][2]["unit"],
                         "unknown_raw_metric_unit")
        self.assertEqual([c.requests[0][:2] for c in factory.connections],
                         [("GET", "/metrics"),
                          ("POST", "/v1/chat/completions"),
                          ("GET", "/metrics")])
        self.assertTrue(all(c.host == p.HOST and c.port == p.PORT
                            for c in factory.connections))

    def test_metrics_redirect_rejected_without_affecting_sse_count(self):
        factory = FakeFactory([
            FakeResponse([], clock=self.clock, status=302),
            fake_stream(self.clock),
            FakeResponse([], clock=self.clock, status=301),
        ])
        result = p.run_profile(
            p.Config(concurrency=(1,), rounds=1, timeout_s=10, metrics=True),
            clock=self.clock, connection_factory=factory,
        )
        self.assertEqual(len(factory.connections), 3)
        self.assertEqual(result["status"], "COMPLETE")
        rec = result["by_concurrency"][0]["rounds"][0]
        self.assertEqual(rec["metrics_before"]["error"], "metrics_status_not_200")
        self.assertEqual(rec["metrics_after"]["error"], "metrics_status_not_200")

    def test_metrics_max_bytes_returns_error_not_false_zero(self):
        factory = FakeFactory([
            FakeResponse([(b"A" * (p.MAX_METRICS_BYTES + 1), .1)],
                         clock=self.clock),
        ])
        outcome = p.sample_vram_metrics(
            p.Config(timeout_s=10), clock=self.clock,
            connection_factory=factory,
        )
        self.assertEqual(outcome["status"], "ERROR")
        self.assertEqual(outcome["error"], "metrics_max_bytes")
        self.assertEqual(outcome["series"], [])

    def test_metrics_unknown_field_is_not_interpreted_as_vram(self):
        factory = FakeFactory([
            FakeResponse(
                [(b"gpu_utilization_percent 98\ncpu_memory_total 32768\n", .1)],
                clock=self.clock,
            ),
        ])
        outcome = p.sample_vram_metrics(
            p.Config(timeout_s=10), clock=self.clock,
            connection_factory=factory,
        )
        self.assertEqual(outcome["status"], "OK")
        self.assertEqual(outcome["series"], [])

    def test_cli_defaults_to_dry_run_without_opening_any_socket(self):
        stdout = io.StringIO()
        with mock.patch.object(p, "run_profile",
                               side_effect=AssertionError("run forbidden")), \
                mock.patch.object(p.http.client, "HTTPConnection",
                                  side_effect=AssertionError("network forbidden")), \
                contextlib.redirect_stdout(stdout):
            self.assertEqual(p.main([]), 0)
        data = json.loads(stdout.getvalue())
        self.assertEqual(data["status"], "DRY_RUN_NO_NETWORK")
        self.assertEqual(data["target"], "http://127.0.0.1:18127")
        self.assertEqual(data["planned_requests"], 30)
        self.assertEqual(data["metrics_gets"], 0)

    def test_cli_preview_metrics_calculates_safe_get_count_no_network(self):
        output = io.StringIO()
        with mock.patch.object(p, "run_profile",
                               side_effect=AssertionError("run forbidden")), \
                contextlib.redirect_stdout(output):
            self.assertEqual(p.main([
                "--metrics-vram", "--clients", "2", "8", "--rounds", "3",
            ]), 0)
        info = json.loads(output.getvalue())
        self.assertEqual(info["planned_requests"], (2 + 8) * 3)
        self.assertEqual(info["metrics_gets"], 2 * 3 * 2)

    def test_external_base_url_is_not_a_supported_cli_parameter(self):
        with contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as error:
            p.main(["--base-url", "https://evil.example"])
        self.assertEqual(error.exception.code, 2)

    def test_default_factory_is_httpclient_not_urllib_proxy_opener(self):
        self.assertIs(p.profile_request.__kwdefaults__["connection_factory"],
                      p.http.client.HTTPConnection)
        self.assertIs(p.sample_vram_metrics.__kwdefaults__["connection_factory"],
                      p.http.client.HTTPConnection)

    def test_invalid_cli_concurrency_fails_without_network(self):
        with contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as error:
            p.main(["--run", "--clients", "16"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
