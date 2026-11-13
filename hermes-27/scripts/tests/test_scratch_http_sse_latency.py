#!/usr/bin/env python3
"""Offline mocked SSE contract tests; no server, GPU, live client or network."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import json

SCRIPT = Path(__file__).resolve().parents[1] / "scratch_http_sse_latency.py"
spec = importlib.util.spec_from_file_location("scratch_http_sse_latency", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Response:
    def __init__(self, events, *, content_type="text/event-stream",
                 status=200):
        self.events = events
        self.headers = {"Content-Type": content_type}
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __iter__(self):
        for data in self.events:
            if data == "[DONE]":
                yield b"data: [DONE]\n\n"
            else:
                yield b"data: " + json.dumps(data, ensure_ascii=False).encode() + b"\n\n"


def clean_events():
    return [
        {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "L’"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "éxemple"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}]},
        {"choices": [], "usage": {"prompt_tokens": 47, "completion_tokens": 2}},
        "[DONE]",
    ]


class SSEContract(unittest.TestCase):
    def run_response(self, response):
        with patch.object(module.urllib.request, "urlopen", return_value=response):
            return module.consume_sse(
                "http://127.0.0.1:18127", 0, 1, 16, 10)

    def test_success_unicode_exact_sha_and_usage(self):
        import hashlib
        record = self.run_response(Response(clean_events()))
        self.assertEqual(record["completion_tokens"], 2)
        self.assertEqual(record["prompt_tokens"], 47)
        self.assertEqual(record["finish"], "length")
        self.assertEqual(record["visible_deltas"], 2)
        self.assertEqual(
            record["content_sha256"],
            hashlib.sha256("L’éxemple".encode("utf-8")).hexdigest())
        self.assertGreaterEqual(record["elapsed_sec"], record["ttfc_sec"])

    def test_missing_done_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "terminal"):
            self.run_response(Response(clean_events()[:-1]))

    def test_duplicate_usage_rejected(self):
        events = clean_events()
        events.insert(-1, events[-2])
        with self.assertRaisesRegex(RuntimeError, "duplicate SSE usage"):
            self.run_response(Response(events))

    def test_error_event_rejected(self):
        events = clean_events()
        events.insert(1, {"error": {"message": "GPU failed"}})
        with self.assertRaisesRegex(RuntimeError, "server SSE error"):
            self.run_response(Response(events))

    def test_no_visible_content_rejected(self):
        events = [event for event in clean_events() if not
                  isinstance(event, dict) or not
                  any(choice.get("delta", {}).get("content")
                      for choice in event.get("choices", []))]
        with self.assertRaisesRegex(RuntimeError, "invalid SSE"):
            self.run_response(Response(events))

    def test_wrong_content_type_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "SSE Content-Type"):
            self.run_response(Response(clean_events(), content_type="application/json"))

    def test_unexpected_status_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "unexpected HTTP"):
            self.run_response(Response(clean_events(), status=503))

    def test_percentile_small_and_interpolated(self):
        self.assertEqual(module.percentile([1.0], .95), 1.0)
        self.assertAlmostEqual(module.percentile([0.0, 10.0], .95), 9.5)
        with self.assertRaisesRegex(ValueError, "requires samples"):
            module.percentile([], .5)


if __name__ == "__main__":
    unittest.main()
