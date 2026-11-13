#!/usr/bin/env python3
"""CPU-only production contract tests using the real serving C++ and tokenizer.

The server binary MUST be linked against a fake PP2 backend. It must not
upload weights or touch GPUs. Run this against isolated scratch only.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from contextlib import contextmanager


PARSER = argparse.ArgumentParser()
for name in ("server", "model", "gguf-dump", "tokenizer", "python"):
    PARSER.add_argument("--" + name, required=True)
PARSER.add_argument("--slow-reader", action="store_true",
                    help="requires a CPU fake backend that honors FAKE_STEP_MS=0")
ARGS, REMAINING = PARSER.parse_known_args()


@contextmanager
def server(fake_ids=None, extra=(), env_extra=None, with_process=False):
    env = os.environ.copy()
    if fake_ids is not None:
        env["FAKE_IDS"] = ",".join(map(str, fake_ids))
    if env_extra:
        env.update(env_extra)
    command = [
        ARGS.server, "--model", ARGS.model, "--gguf-dump", ARGS.gguf_dump,
        "--tokenizer", ARGS.tokenizer, "--python", ARGS.python, "--port", "0",
        "--context", "256", *extra,
    ]
    proc = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        env=env, text=True, bufsize=1,
    )
    try:
        while True:
            line = proc.stderr.readline()
            if line.startswith("LISTEN "):
                url = line.split()[1]
                yield (url, proc) if with_process else url
                break
            if proc.poll() is not None:
                raise AssertionError(f"server startup failed: {line}")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=7)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=4)
        proc.stderr.close()


def request(url, payload=None, *, endpoint="/v1/chat/completions", headers=None):
    if payload is None:
        data = None
        method = "GET"
    else:
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
    req = urllib.request.Request(
        url + endpoint, data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as res:
            return res.status, dict(res.headers), res.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def base_payload(**kwargs):
    return {
        "model": "hermes-27", "messages": [{"role": "user", "content": "x"}],
        "max_tokens": 8, "temperature": 0, **kwargs,
    }


def collect(body):
    fragments = []
    reasons = []
    usage = []
    done = 0
    for block in body.decode("utf-8").split("\n\n"):
        if not block:
            continue
        assert block.startswith("data: "), block
        entry = block[6:]
        if entry == "[DONE]":
            done += 1
            continue
        obj = json.loads(entry)
        if obj.get("usage") is not None:
            usage.append(obj["usage"])
        for choice in obj["choices"]:
            fragments.append(choice["delta"].get("content", ""))
            if choice["finish_reason"] is not None:
                reasons.append(choice["finish_reason"])
    return "".join(fragments), reasons, usage, done


class ProductionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        cls.tokenizer = Tokenizer.from_file(ARGS.tokenizer)

    def ids(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def assert_error(self, url, payload, status, code, **kwargs):
        received, _, body = request(url, payload, **kwargs)
        self.assertEqual(received, status, body)
        self.assertEqual(json.loads(body)["error"]["code"], code, body)

    def test_strict_numeric_and_invalid_requests(self):
        with server(extra=("--context", "256")) as url:
            for kwargs in (
                {"n": 4294967297},
                {"n": -4294967295},
            ):
                self.assert_error(url, base_payload(**kwargs), 400, "unsupported_value")
            for kwargs in (
                {"max_tokens": 4294967297},
                {"max_tokens": -1},
            ):
                self.assert_error(url, base_payload(**kwargs), 400, "invalid_value")
            self.assert_error(url, base_payload(model="missing"), 404, "model_not_found")
            self.assert_error(url, base_payload(top_p=0.8), 400, "unsupported_value")
            self.assert_error(url, base_payload(temperature=1), 400, "unsupported_value")

    def test_stop_sequence_crosses_tokens_and_streaming_parity(self):
        generated = self.ids("hello good night world")
        expected = self.tokenizer.decode(generated)
        stop = "good night"
        index = expected.find(stop)
        self.assertGreater(index, 0)
        with server(fake_ids=generated) as url:
            data = base_payload(stop=stop, max_tokens=len(generated))
            status, _, body = request(url, data)
            self.assertEqual(status, 200, body)
            completion = json.loads(body)
            self.assertEqual(completion["choices"][0]["message"]["content"], expected[:index])
            self.assertEqual(completion["choices"][0]["finish_reason"], "stop")
            status, _, body = request(url, {**data, "stream": True})
            self.assertEqual(status, 200, body)
            text, reasons, _, done = collect(body)
            self.assertEqual(text, expected[:index])
            self.assertEqual(reasons, ["stop"])
            self.assertEqual(done, 1)

    def test_stop_unicode_and_partial_prefix(self):
        ids = self.ids("Here is 🍎 a blue apple")
        reference = self.tokenizer.decode(ids)
        stop = "🍎 a"
        with server(fake_ids=ids) as url:
            options = base_payload(stop=[stop, "not-here"], max_tokens=len(ids))
            for streaming in (False, True):
                status, _, output = request(url, {**options, "stream": streaming})
                self.assertEqual(status, 200, output)
                if streaming:
                    text, reason, _, done = collect(output)
                    self.assertEqual((reason, done), (["stop"], 1))
                else:
                    result = json.loads(output)
                    text = result["choices"][0]["message"]["content"]
                self.assertEqual(text, reference[:reference.find(stop)])

    def test_stop_prefix_flushes_at_max_tokens(self):
        ids = self.ids("abc")
        with server(fake_ids=ids) as url:
            data = base_payload(max_tokens=len(ids), stop=["abcd"])
            for streaming in (False, True):
                status, _, output = request(url, {**data, "stream": streaming})
                self.assertEqual(status, 200, output)
                if streaming:
                    text, reason, _, done = collect(output)
                    self.assertEqual((reason, done), (["length"], 1))
                else:
                    result = json.loads(output)
                    text = result["choices"][0]["message"]["content"]
                self.assertEqual(text, self.tokenizer.decode(ids))

    def test_stop_parameter_validation(self):
        with server() as url:
            invalid = ([], [""], ["a"] * 5, 2, {"x": 1}, [123], "")
            for stop in invalid:
                self.assert_error(url, base_payload(stop=stop), 400, "invalid_value")

    def test_stream_usage_contract(self):
        ids = self.ids("hello there")
        with server(fake_ids=ids) as url:
            status, _, output = request(
                url, base_payload(stream=True, stream_options={"include_usage": True},
                                  max_completion_tokens=len(ids), max_tokens=None)
            )
            self.assertEqual(status, 200, output)
            text, reason, usage, done = collect(output)
            self.assertEqual(text, self.tokenizer.decode(ids))
            self.assertEqual(reason, ["length"])
            self.assertEqual(done, 1)
            self.assertEqual(len(usage), 1)
            self.assertEqual(usage[0]["completion_tokens"], len(ids))
            self.assertEqual(
                usage[0]["total_tokens"],
                usage[0]["prompt_tokens"] + usage[0]["completion_tokens"],
            )

    def test_nonloopback_requires_auth(self):
        # CLI argument validation occurs before any GPU activity or model upload.
        cmd = [
            ARGS.server, "--model", ARGS.model, "--gguf-dump", ARGS.gguf_dump,
            "--tokenizer", ARGS.tokenizer, "--python", ARGS.python,
            "--host", "0.0.0.0", "--port", "0",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("api-key", result.stderr)
        with server(extra=("--api-key", "example-secret")) as url:
            status, _, _ = request(url, endpoint="/health")
            self.assertEqual(status, 401)
            status, _, body = request(
                url, endpoint="/v1/models",
                headers={"Authorization": "Bearer example-secret"},
            )
            self.assertEqual(status, 200, body)

    def test_private_api_key_file(self):
        with tempfile.TemporaryDirectory() as temp:
            key_path = Path(temp) / "api-key"
            key_path.write_text("long-random-test-secret\n", encoding="utf-8")
            key_path.chmod(0o600)
            with server(extra=("--api-key-file", str(key_path))) as url:
                status, _, body = request(url, endpoint="/health")
                self.assertEqual(status, 401, body)
                status, _, body = request(
                    url, endpoint="/health",
                    headers={"Authorization": "Bearer long-random-test-secret"},
                )
                self.assertEqual(status, 200, body)
            key_path.chmod(0o644)
            cmd = [
                ARGS.server, "--model", ARGS.model, "--gguf-dump", ARGS.gguf_dump,
                "--tokenizer", ARGS.tokenizer, "--python", ARGS.python,
                "--api-key-file", str(key_path), "--port", "0",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
            self.assertEqual(proc.returncode, 2, proc.stderr)
            self.assertIn("api-key-file", proc.stderr)

    def test_strict_content_type_and_usage_model_id(self):
        with server(extra=("--model-id", "hermes-27")) as url:
            payload = base_payload(max_tokens=1)
            status, _, raw = request(
                url, payload, headers={"Content-Type": "text/plain"}
            )
            self.assertEqual(status, 415, raw)
            self.assertEqual(json.loads(raw)["error"]["code"], "unsupported_media_type")
            status, _, raw = request(
                url, payload, headers={"Content-Type": "Application/JSON; charset=UTF-8"}
            )
            self.assertEqual(status, 200, raw)
            body = json.loads(raw)
            self.assertEqual(body["model"], "hermes-27")
            self.assertEqual(
                body["usage"]["total_tokens"],
                body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"],
            )

    def test_text_content_parts_developer_and_named_messages(self):
        with server() as url:
            payload = base_payload(max_tokens=1)
            payload["messages"] = [
                {"role": "developer", "content": "System instructions", "name": "policy"},
                {"role": "user", "content": [
                    {"type": "text", "text": "x"},
                    {"type": "text", "text": "y"},
                ]},
            ]
            status, _, body = request(url, payload)
            self.assertEqual(status, 200, body)
            self.assertEqual(json.loads(body)["usage"]["completion_tokens"], 1)
            payload["messages"][1]["content"].append({"type": "image_url", "image_url": "x"})
            self.assert_error(url, payload, 400, "unsupported_value")

    def test_mixed_concurrent_streams_and_usage(self):
        ids = self.ids("Café rocket 🚀")
        expected = self.tokenizer.decode(ids)
        with server(fake_ids=ids, extra=("--active", "4", "--queue", "16")) as url:
            def one_request(index):
                streaming = (index % 2 == 0)
                stop = ["rocket"] if index % 3 == 0 else None
                payload = base_payload(
                    max_tokens=len(ids), stream=streaming, stop=stop
                )
                if streaming:
                    payload["stream_options"] = {"include_usage": True}
                status, _, raw = request(url, payload)
                self.assertEqual(status, 200, raw)
                if streaming:
                    text, reasons, usage, done = collect(raw)
                    self.assertEqual(done, 1)
                    self.assertEqual(len(usage), 1)
                    self.assertEqual(usage[0]["total_tokens"],
                                     usage[0]["prompt_tokens"] + usage[0]["completion_tokens"])
                    event = next(
                        json.loads(chunk[6:]) for chunk in raw.decode().split("\n\n")
                        if chunk.startswith("data: {")
                    )
                    identity = event["id"]
                else:
                    response = json.loads(raw)
                    text = response["choices"][0]["message"]["content"]
                    reasons = [response["choices"][0]["finish_reason"]]
                    identity = response["id"]
                if stop:
                    self.assertEqual(text, expected.split("rocket", 1)[0])
                    self.assertEqual(reasons, ["stop"])
                else:
                    self.assertEqual(text, expected)
                    self.assertEqual(reasons, ["length"])
                return identity

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                ids_seen = list(pool.map(one_request, range(8)))
            self.assertEqual(len(ids_seen), len(set(ids_seen)))

    def test_custom_model_id_and_null_optional_fields(self):
        with server(extra=("--model-id", "my-qwen")) as url:
            status, _, raw = request(url, endpoint="/v1/models")
            self.assertEqual(status, 200, raw)
            self.assertEqual(json.loads(raw)["data"][0]["id"], "my-qwen")
            self.assert_error(url, base_payload(max_tokens=1), 404, "model_not_found")
            payload = base_payload(model="my-qwen", max_tokens=None)
            payload["max_completion_tokens"] = 1
            payload["temperature"] = None
            payload["top_p"] = None
            payload["reasoning"] = None
            payload["stream"] = None
            status, _, raw = request(url, payload)
            self.assertEqual(status, 200, raw)

    def test_bridge_recovers_after_child_crash(self):
        with server(with_process=True) as (url, process):
            status, _, raw = request(url, base_payload(max_tokens=1))
            self.assertEqual(status, 200, raw)
            children_file = Path(f"/proc/{process.pid}/task/{process.pid}/children")
            children = [int(pid) for pid in children_file.read_text().split()]
            self.assertEqual(len(children), 1, children)
            os.kill(children[0], signal.SIGKILL)
            # The next request must transparently restart the bridge. The
            # in-flight tokenizer state is deliberately lost, but subsequent
            # encode/decode operations remain usable.
            for _ in range(2):
                status, _, raw = request(url, base_payload(max_tokens=1))
                self.assertEqual(status, 200, raw)

    def test_queued_stream_times_out_before_http_sse_headers(self):
        with server(extra=("--active", "1", "--queue", "2", "--queue-timeout", "1")) as url:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(request, url, base_payload(max_tokens=200))
                for _ in range(100):
                    _, _, raw = request(url, endpoint="/metrics")
                    if "hermes_requests_active 1" in raw.decode():
                        break
                    time.sleep(0.02)
                else:
                    self.fail("first request did not reach scheduler")
                status, _, raw = request(
                    url, base_payload(max_tokens=1, stream=True)
                )
                self.assertEqual(status, 429, raw)
                self.assertEqual(json.loads(raw)["error"]["code"], "overloaded")
                first_status, _, first_body = first.result(timeout=25)
                self.assertEqual(first_status, 200, first_body)

    def test_vram_admission_rejects_before_stream(self):
        with server(
            extra=("--gpu0-used-limit-mib", "16000", "--gpu1-used-limit-mib", "16000"),
            env_extra={"FAKE_GPU_USED_MIB": "15300"},
        ) as url:
            status, _, raw = request(url, base_payload(max_tokens=1, stream=True))
            self.assertEqual(status, 429, raw)
            self.assertEqual(json.loads(raw)["error"]["code"], "overloaded")

    def test_init_error_returns_http503_before_sse(self):
        with server(env_extra={"FAKE_INIT_FAIL": "1"}) as url:
            status, _, raw = request(url, base_payload(stream=True, max_tokens=1))
            self.assertEqual(status, 503, raw)
            self.assertEqual(json.loads(raw)["error"]["code"], "inference_error")

    def test_decode_error_after_sse_headers_then_next_request_recovers(self):
        with server(env_extra={"FAKE_DECODE_FAIL_ONCE": "1"}) as url:
            status, _, raw = request(url, base_payload(stream=True, max_tokens=1))
            self.assertEqual(status, 200, raw)
            self.assertIn(b"data: [DONE]", raw)
            self.assertIn(b"server_error", raw)
            status, _, raw = request(url, base_payload(stream=False, max_tokens=1))
            self.assertEqual(status, 200, raw)
            self.assertEqual(json.loads(raw)["usage"]["completion_tokens"], 1)

    def test_slow_sse_reader_triggers_bounded_backpressure(self):
        if not ARGS.slow_reader:
            self.skipTest("use --slow-reader with an instant CPU fake PP2 backend")
        with server(
            extra=("--context", "8192", "--active", "1", "--queue", "2"),
            env_extra={"FAKE_STEP_MS": "0"},
        ) as url:
            port = int(url.rsplit(":", 1)[1])
            sock = socket.socket()
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
                sock.settimeout(3)
                sock.connect(("127.0.0.1", port))
                payload = json.dumps(
                    base_payload(max_tokens=8192, stream=True)
                ).encode()
                request_bytes = (
                    b"POST /v1/chat/completions HTTP/1.1\r\n"
                    b"Host: 127.0.0.1\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Content-Length: " + str(len(payload)).encode() +
                    b"\r\nConnection: keep-alive\r\n\r\n" + payload
                )
                sock.sendall(request_bytes)
                for _ in range(125):
                    status, _, metrics = request(url, endpoint="/metrics")
                    self.assertEqual(status, 200)
                    lines = metrics.decode().splitlines()
                    count = next(
                        int(line.split()[-1]) for line in lines
                        if line.startswith("hermes_stream_backpressure_total ")
                    )
                    if count:
                        self.assertEqual(count, 1)
                        self.assertIn("hermes_requests_active 0", metrics.decode())
                        break
                    time.sleep(0.08)
                else:
                    self.fail("slow reader did not saturate SSE queue")
            finally:
                sock.close()

    def test_graceful_signal_shutdown_ends_active_sse(self):
        with server(with_process=True) as (url, process):
            payload = base_payload(max_tokens=200, stream=True)
            req = urllib.request.Request(
                url + "/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=25) as response:
                # The first role chunk is available once the GPU state was
                # admitted; the 200-token decode should still be active.
                first_line = response.readline()
                self.assertTrue(first_line.startswith(b"data: "), first_line)
                os.kill(process.pid, signal.SIGTERM)
                rest = first_line + response.read()
                self.assertIn(b"data: [DONE]", rest)
                text, reasons, _, done = collect(rest)
                self.assertEqual(done, 1)
                self.assertEqual(reasons, ["cancelled"])
            process.wait(timeout=8)
            self.assertEqual(process.returncode, 0)


if __name__ == "__main__":
    unittest.main(argv=[__file__, *REMAINING], verbosity=2)
