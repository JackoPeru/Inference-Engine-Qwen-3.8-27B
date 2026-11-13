#!/usr/bin/env python3
"""Pure-CPU end-to-end tests of the release QA harness; no real HERMES or GPU."""
from __future__ import annotations

import hashlib
import http.server
import json
import sys
import tempfile
import threading
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
from reliability_core import (GateFailure, RunReport, metric_values, request, require,
                              sse_request, validate_chat, validate_sse)
from verify_release_online import OnlineGate
import verify_release_logits as logits


class MockHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def emit(self, code, obj, content_type="application/json"):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def error(self, code, reason):
        self.emit(code, {"error": {"code": reason, "message": reason, "type": "invalid_request_error"}})

    def authenticate(self):
        if self.headers.get("Authorization") == "Bearer test-secret":
            return True
        self.error(401, "invalid_api_key")
        return False

    def do_GET(self):
        if not self.authenticate():
            return
        if self.path == "/health":
            return self.emit(200, {"ready": True, "status": "ok"})
        if self.path == "/v1/models":
            return self.emit(200, {"data": [{"id": "hermes-27"}]})
        if self.path == "/metrics":
            body = ("hermes_server_ready 1\nhermes_requests_active 0\n"
                    "hermes_requests_queued 0\nhermes_inference_errors_total 0\n"
                    "hermes_requests_cancelled_total " +
                    ("1" if self.server.cancelled.is_set() else "0") + "\n"
                    'hermes_vram_used_mib{gpu="0"} 1000\n'
                    'hermes_vram_used_mib{gpu="1"} 1100\n')
            raw = body.encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(raw)
            return
        self.error(404, "not_found")

    def do_POST(self):
        if not self.authenticate():
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            obj = json.loads(raw)
        except ValueError:
            return self.error(400, "invalid_json")
        if self.path == "/v1/cancel":
            if (isinstance(obj, dict) and obj.get("id") == "fake-id" and
                    self.server.stream_waiting):
                self.server.cancelled.set()
                return self.emit(202, {"id": "fake-id", "cancelled": True})
            return self.error(404, "not_found")
        if self.path != "/v1/chat/completions":
            return self.error(404, "not_found")
        if not isinstance(obj, dict):
            return self.error(400, "invalid_type")
        supported = {"model", "messages", "max_tokens", "max_completion_tokens",
                     "reasoning", "stream", "temperature", "top_p"}
        if set(obj) - supported:
            return self.error(400, "unsupported_parameter")
        if obj.get("model") != "hermes-27":
            return self.error(404, "model_not_found")
        if obj.get("temperature", 0) != 0 or obj.get("top_p", 1) != 1:
            return self.error(400, "unsupported_value")
        if ("max_completion_tokens" in obj and "max_tokens" in obj):
            return self.error(400, "invalid_value")
        n = obj.get("max_tokens", 3)
        if type(n) is not int or n < 1 or n > 8192:
            return self.error(400, "invalid_value")
        messages = obj.get("messages")
        if not isinstance(messages, list) or not messages:
            return self.error(400, "invalid_value")
        if any(m.get("role") not in ("system", "user", "assistant") or
               not isinstance(m.get("content"), str) for m in messages):
            return self.error(400, "unsupported_value")
        prompt = messages[0]["content"]
        text = "QA:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12] + " 🚀 中文"
        first = prompt.startswith("fixture:")
        prompt_len = int(prompt.split(":")[-1]) if first else 11
        if obj.get("stream"):
            if prompt.startswith("Write a very long"):
                initial = {"id": "fake-id", "model": "hermes-27",
                           "object": "chat.completion.chunk",
                           "choices": [{"delta": {"role": "assistant"},
                                        "finish_reason": None}]}
                terminal = {"id": "fake-id", "model": "hermes-27",
                            "object": "chat.completion.chunk",
                            "choices": [{"delta": {}, "finish_reason": "cancelled"}]}
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.server.stream_waiting = True
                self.wfile.write(("data: " + json.dumps(initial) + "\n\n").encode())
                self.wfile.flush()
                self.server.cancelled.wait(timeout=6)
                self.wfile.write(("data: " + json.dumps(terminal) +
                                  "\n\ndata: [DONE]\n\n").encode())
                self.wfile.flush()
                self.server.stream_waiting = False
                self.close_connection = True
                return
            chunks = ["QA:", text[3:10], text[10:16], text[16:]]
            stream = []
            def chunk(delta, finish=None):
                return {"id": "fake-id", "model": "hermes-27",
                        "object": "chat.completion.chunk",
                        "choices": [{"delta": delta, "finish_reason": finish}]}
            stream.append(chunk({"role": "assistant"}))
            for value in chunks:
                stream.append(chunk({"content": value}))
            stream.append(chunk({}, "length"))
            payload = "".join("data: " + json.dumps(row, ensure_ascii=False) +
                              "\n\n" for row in stream) + "data: [DONE]\n\n"
            binary = payload.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(binary)))
            self.end_headers()
            self.wfile.write(binary)
            return
        self.emit(200, {"object": "chat.completion", "id": "fake-id",
                        "model": "hermes-27",
                        "choices": [{"message": {"role": "assistant", "content": text},
                                     "finish_reason": "length"}],
                        "usage": {"prompt_tokens": prompt_len, "completion_tokens": n,
                                  "total_tokens": prompt_len + n}})


class TestReliability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        cls.server.daemon_threads = True
        cls.server.cancelled = threading.Event()
        cls.server.stream_waiting = False
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        cls.args = SimpleNamespace(base_url=cls.base, api_key="test-secret",
                                   model="hermes-27", timeout=8, concurrency=[1, 2, 4, 8])
        cls.gate = OnlineGate(cls.args)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)

    def test_http_validation_and_auth(self):
        self.assertTrue(self.gate.health())
        self.assertEqual(self.gate.errors()["cases"], 13)
        self.assertEqual(self.gate.auth()["unauthenticated_health_and_chat"], 401)

    def test_backpressure_mock_cannot_claim_pass(self):
        # A fast fake service with no queue capacity bound must NOT satisfy
        # the live 429/Retry-After release requirement.
        with self.assertRaises(GateFailure):
            self.gate.backpressure()

    def test_active_stream_cancel_and_metrics(self):
        self.server.cancelled.clear()
        self.assertEqual(self.gate.cancellation()["finish_reason"], "cancelled")

    def test_unicode_sse_and_concurrency(self):
        result = self.gate.unicode()
        self.assertEqual(len(result["unicode_cases"]), 8)
        for n in (1, 2, 4, 8):
            with self.subTest(n=n):
                self.assertEqual(self.gate.concurrency(n, 1)["requests"], n)

    def test_memory_accounting_and_exact_context(self):
        with patch("verify_release_online.nvidia_vram", return_value=(1000, 1200)):
            result = self.gate.memory(2, 0)
        self.assertEqual(len(result["checkpoints"]), 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            path.write_text(json.dumps({"schema": "h27.context-fixtures.v1",
                "cases": [{"tokens": n, "messages": [{"role": "user",
                 "content": f"fixture:{n}"}], "thinking": False}
                          for n in (129, 1024, 8192)]}), encoding="utf-8")
            self.assertEqual(self.gate.contexts(path)["validated_contexts"], [129, 1024, 8192])

    def test_malformed_sse_is_fatal(self):
        with self.assertRaises(GateFailure):
            validate_sse([{"error": {"message": "oops"}}, "[DONE]"], "hermes-27")
        with self.assertRaises(GateFailure):
            validate_sse([{"object": "chat.completion.chunk",
                           "model": "hermes-27", "id": "x",
                           "choices": [{"delta": {"content": "ok"},
                                        "finish_reason": "length"}]},
                          "[DONE]", "[DONE]"], "hermes-27")

    def test_release_outcome_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            r = RunReport("cpu-test")
            r.record("positive", lambda: {"fine": True})
            r.skip("missing-gpu", "no permission")
            self.assertEqual(r.save(Path(d) / "report.json")[0], 2)
            r.record("broken", lambda: require(False, "mismatch"))
            self.assertEqual(r.save(Path(d) / "report.json")[0], 1)
            report = json.loads((Path(d) / "report.json").read_text())
            self.assertEqual(report["status"], "NO_GO")

    def test_logits_manifest_exact_checkpoint_and_no_stale(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            model = root / "actual.gguf"
            model.write_bytes(b"GGUF" * 1024)
            binary = root / "exporter.bin"
            binary.write_bytes(b"independent-trace-exporter")
            record = {"prompt_index": 0, "prompt_ids": [1, 2, 3],
                      "generated_token_ids": [42], "stop_reason": "length"}
            jsonl = root / "tokens.jsonl"
            jsonl.write_text(json.dumps(record) + "\n")
            manifests = []
            for backend in ("hermes-27", "llama.cpp"):
                folder = root / backend
                folder.mkdir()
                values = array("f", [0.0]) * logits.VOCAB
                values[42] = 2.0
                (folder / "prompt_0_step_0.f32").write_bytes(values.tobytes())
                a = SimpleNamespace(backend=backend, run_id="qarelease-123456789",
                                    exporter_version="selftest (mock)", gguf=model,
                                    exporter_binary=binary,
                                    jsonl=jsonl, dump_dir=folder, max_age_seconds=3600)
                result = logits.make_manifest(a)
                path = root / (backend + ".json")
                path.write_text(json.dumps(result))
                manifests.append(path)
            c = SimpleNamespace(hermes_manifest=manifests[0],
                                llama_manifest=manifests[1], max_age_seconds=3600,
                                max_abs=.002, max_rmse=.0005)
            self.assertEqual(logits.compare(c)["matched_steps"], 1)
            # Modifying reference after capture must fail even when sizes match.
            with (root / "llama.cpp/prompt_0_step_0.f32").open("r+b") as f:
                f.seek(100)
                f.write(b"XXXX")
            with self.assertRaises(GateFailure):
                logits.compare(c)


if __name__ == "__main__":
    unittest.main(verbosity=2)
