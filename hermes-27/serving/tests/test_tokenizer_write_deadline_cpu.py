#!/usr/bin/env python3
"""Linux CPU-only HTTP regression for a tokenizer child stuck on stdin.

The actual server.cpp and pp2_batch.cu must be linked with
HERMES_ADMISSION_CPU_ONLY and prod_sched_mock_pp2.cpp. The supplied executable
is checked for fake-backend symbols before it is ever launched. A private
executable child passed via --python writes the normal tokenizer ready JSON,
then deliberately never reads stdin. Every HTTP connection is loopback.

Usage:
  python3 serving/tests/test_tokenizer_write_deadline_cpu.py \
      --server /tmp/cpu-scratch/build/hermes_server

Optional pre-fix negative witness:
  ... --baseline-server /tmp/old-cpu-scratch/build/hermes_server

The baseline binary must actually predate the nonblocking write deadline.
This script does not generate an old binary or mutate production sources.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request


WRITE_TIMEOUT_MS = "150"  # CPU-only override; never an accepted CUDA setting.
LOCAL_HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
STUB = '''#!/usr/bin/env python3
"""Test-only tokenizer fake: valid hello, then NEVER drain stdin."""
import json
import os
import time

print(json.dumps({"ready": True, "incremental": False}), flush=True)
with open(os.environ["HERMES_TEST_TOKENIZER_CHILD_LOG"], "a", encoding="utf-8") as f:
    f.write("ready\\n")
# Do not read sys.stdin. Parent writes fill the kernel pipe.
while True:
    time.sleep(3600)
'''


def assert_cpu_binary(path: Path, *, old: bool) -> None:
    """Never accidentally launch a production CUDA executable in this test."""
    if not path.is_file() or not os.access(path, os.X_OK):
        raise AssertionError(f"missing executable CPU mock: {path}")
    data = path.read_bytes()
    if not data.startswith(b"\x7fELF"):
        raise AssertionError("CPU mock must be an ELF executable, not a launcher")
    for marker in (b"FAKE_TRACE_FILE", b"FAKE_IDS", b"FAKE_INIT_FAIL"):
        if marker not in data:
            raise AssertionError(f"not the isolated fake PP2 CPU backend: {marker!r}")
    fixed = b"HERMES_TEST_TOKENIZER_WRITE_TIMEOUT_MS" in data
    if old and fixed:
        raise AssertionError("baseline CPU mock already contains the write deadline")
    if not old and not fixed:
        raise AssertionError("candidate CPU mock lacks the write deadline")


def saturation_size() -> int:
    """Keep JSON under kMaxBody=1MiB while exceeding a Linux pipe capacity."""
    read_fd, write_fd = os.pipe()
    try:
        pipe_capacity = fcntl.fcntl(write_fd, fcntl.F_GETPIPE_SZ)
    finally:
        os.close(read_fd)
        os.close(write_fd)
    size = min(900_000, max(320_000, 3 * pipe_capacity))
    if size <= pipe_capacity:
        raise unittest.SkipTest(
            f"kernel pipe capacity {pipe_capacity} exceeds the safe HTTP-body bound")
    return size


def large_request(content_bytes: int) -> bytes:
    request = {
        "model": "hermes-27",
        "messages": [{"role": "user", "content": "z" * content_bytes}],
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
    }
    body = json.dumps(request, separators=(",", ":")).encode("utf-8")
    if not 200_000 < len(body) < 1024 * 1024:
        raise AssertionError("HTTP fixture must exceed 200KB and stay under 1MiB")
    return body


def bounded_post(url: str, body: bytes, timeout: float = 6) -> tuple[int, bytes]:
    req = urllib.request.Request(
        url + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with LOCAL_HTTP.open(req, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def metric_requests(url: str) -> int:
    with LOCAL_HTTP.open(url + "/metrics", timeout=0.5) as response:
        data = response.read().decode("utf-8")
    for line in data.splitlines():
        if line.startswith("hermes_requests_total "):
            return int(line.rsplit(" ", 1)[1])
    raise AssertionError("mock server omitted hermes_requests_total")


def wait_for_requests(url: str, minimum: int, seconds: float) -> None:
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        try:
            if metric_requests(url) >= minimum:
                return
        except (OSError, TimeoutError, ValueError):
            pass
        time.sleep(0.02)
    raise AssertionError(f"{minimum} requests did not enter mock HTTP handlers")


class MockServer:
    def __init__(self, server: Path):
        self.binary = server
        self.proc: subprocess.Popen | None = None
        self.url = ""
        self.scratch = tempfile.TemporaryDirectory(prefix="h27-tok-write-cpu-")
        self.root = Path(self.scratch.name)
        self.stub = self.root / "no-stdin-tokenizer"
        self.log = self.root / "tokenizer-ready.log"
        self.stderr = self.root / "server.log"
        self.stub.write_text(STUB, encoding="utf-8")
        self.stub.chmod(0o700)
        for name in ("mock.gguf", "mock-tokenizer.json", "mock-gguf-dump"):
            (self.root / name).touch()

    def start(self) -> "MockServer":
        env = os.environ.copy()
        # Deliberate opt-in test-mode environment is active only in a CPU mock.
        env.update({
            "HERMES_TEST_TOKENIZER_CHILD_LOG": str(self.log),
            "HERMES_TEST_TOKENIZER_WRITE_TIMEOUT_MS": WRITE_TIMEOUT_MS,
            "FAKE_IDS": "100",
        })
        args = [
            str(self.binary), "--model", str(self.root / "mock.gguf"),
            "--gguf-dump", str(self.root / "mock-gguf-dump"),
            "--tokenizer", str(self.root / "mock-tokenizer.json"),
            "--python", str(self.stub),
            "--host", "127.0.0.1", "--port", "0", "--context", "256",
        ]
        with self.stderr.open("wb") as f:
            self.proc = subprocess.Popen(
                args, env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=f,
                start_new_session=True,
            )
        until = time.monotonic() + 7
        while time.monotonic() < until:
            output = self.stderr.read_text(encoding="utf-8", errors="replace")
            for line in output.splitlines():
                if line.startswith("LISTEN "):
                    self.url = line.split()[1]
                    parsed = urllib.parse.urlsplit(self.url)
                    if parsed.hostname != "127.0.0.1":
                        raise AssertionError("server is not bound to loopback")
                    return self
            if self.proc.poll() is not None:
                raise AssertionError(
                    f"CPU mock exited before LISTEN: {self.proc.returncode}: {output}")
            time.sleep(0.02)
        raise AssertionError("CPU mock did not report LISTEN within 7 seconds")

    def terminate(self) -> None:
        if self.proc is None:
            return
        # Stop only this isolated process group. Try graceful SIGTERM first;
        # use SIGKILL only after a provable timeout in the intentional
        # pre-fix hang case, never for routine cleanup.
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait(timeout=3)

    def close(self) -> None:
        self.terminate()
        self.scratch.cleanup()


@contextmanager
def isolated_server(binary: Path):
    instance = MockServer(binary)
    try:
        yield instance.start()
    finally:
        instance.close()


class TokenizerWriteDeadlineCPU(unittest.TestCase):
    fixed_server: Path
    baseline_server: Path | None
    request_body: bytes

    @classmethod
    def setUpClass(cls):
        cls.request_body = large_request(saturation_size())
        assert_cpu_binary(cls.fixed_server, old=False)
        if cls.baseline_server:
            assert_cpu_binary(cls.baseline_server, old=True)

    def test_fixed_write_deadline_returns_503_and_retries_only_once(self):
        with isolated_server(self.fixed_server) as server:
            started = time.monotonic()
            status, payload = bounded_post(server.url, self.request_body)
            elapsed = time.monotonic() - started
            self.assertEqual(status, 503, (status, payload[:400]))
            self.assertLess(elapsed, 4, (elapsed, payload[:400]))
            result = json.loads(payload)
            self.assertIn("error", result)
            self.assertEqual(result["error"]["code"], "tokenizer_unavailable")
            self.assertEqual(result["error"]["type"], "server_error")
            self.assertIn("tokenizer", result["error"]["message"].lower())
            self.assertEqual(
                server.log.read_text(encoding="utf-8").count("ready"), 2,
                "encode retry must restart the stuck child exactly once")

    def test_two_blocked_encodes_and_concurrent_sigterm_drain(self):
        with isolated_server(self.fixed_server) as server:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(bounded_post, server.url, self.request_body)
                second = pool.submit(bounded_post, server.url, self.request_body)
                wait_for_requests(server.url, 2, seconds=2)
                self.assertIsNone(server.proc.poll(), "server exited before shutdown probe")
                server.proc.send_signal(signal.SIGTERM)
                # The HTTP handlers may still be inside tokenizer_.call() when
                # SIGTERM arrives. Both must finish so listener.join is bounded.
                self.assertEqual(server.proc.wait(timeout=5), 0)
                for future in (first, second):
                    try:
                        response = future.result(timeout=6)
                    except (OSError, TimeoutError, urllib.error.URLError):
                        # Shutdown may cut the HTTP socket after a bounded
                        # tokenizer failure. Closing within deadline is key.
                        continue
                    self.assertGreaterEqual(response[0], 500, response)

    def test_pre_fix_blocking_writer_hangs_until_isolated_process_killed(self):
        if self.baseline_server is None:
            self.skipTest("provide --baseline-server with a pre-fix CPU mock ELF")
        with isolated_server(self.baseline_server) as server:
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                result = pool.submit(bounded_post, server.url, self.request_body)
                wait_for_requests(server.url, 1, seconds=2)
                time.sleep(0.85)
                self.assertFalse(result.done(), "baseline unexpectedly returned")
                self.assertIsNone(server.proc.poll())
                server.proc.send_signal(signal.SIGTERM)
                time.sleep(0.45)
                self.assertIsNone(
                    server.proc.poll(),
                    "baseline shutdown unexpectedly drained a blocked tokenizer write")
                server.terminate()  # Release the blocked HTTP caller.
                try:
                    result.result(timeout=5)
                except (OSError, TimeoutError, urllib.error.URLError):
                    pass
            finally:
                # Crucial for an expected-hang witness: do not let the
                # ThreadPoolExecutor context manager wait on the wedged call.
                server.terminate()
                pool.shutdown(wait=False, cancel_futures=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True,
                        help="Actual Linux HERMES_ADMISSION_CPU_ONLY mock ELF")
    parser.add_argument("--baseline-server", type=Path,
                        help="Optional pre-deadline CPU mock ELF for original hang")
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("requires isolated Linux CPU/loopback; no Windows/GPU")
    TokenizerWriteDeadlineCPU.fixed_server = args.server.resolve()
    TokenizerWriteDeadlineCPU.baseline_server = (
        args.baseline_server.resolve() if args.baseline_server else None)
    unittest.main(argv=[sys.argv[0]], verbosity=2)


if __name__ == "__main__":
    main()
