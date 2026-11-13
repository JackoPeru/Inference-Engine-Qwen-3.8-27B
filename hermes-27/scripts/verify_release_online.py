#!/usr/bin/env python3
"""Strict HERMES-27 HTTP reliability gates. NEVER starts or stops the server."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from reliability_core import (GateBlocked, GateFailure, RunReport, chat_payload,
                              metric_values, nvidia_vram, request, require,
                              sse_request, validate_chat, validate_sse, vram_from_metrics)

UNICODE = (
    "Italiano: caffè, perché, più, città.",
    "Decomposed: cafe\u0301 and \u212b.",
    "中文 日本語 한국어",
    "العربية עברית हिन्दी",
    "Emoji: 🧪🚀👩‍💻👨‍👩‍👧‍👦",
    "Math ∀x∈ℝ: x² ≥ 0; ∑ π",
    "Multiline:\n\tindented\nthird line",
    "Quotes: \"double\" 'single' backslash \\",
)


class OnlineGate:
    def __init__(self, args):
        self.a = args
        self.url = args.base_url.rstrip("/")
        self.key = args.api_key
        self.model = args.model
        self.timeout = args.timeout

    def api(self, path, *, payload=None, raw=None, method=None, key_override=None):
        return request(self.url, path, payload=payload, raw=raw, method=method,
                       api_key=self.key if key_override is None else key_override,
                       timeout=self.timeout)

    def metrics(self) -> dict[str, float]:
        reply = self.api("/metrics")
        require(reply.status == 200, f"metrics endpoint status={reply.status}")
        return metric_values(reply.body)

    def health(self):
        obj = self.api("/health")
        require(obj.status == 200 and obj.json().get("ready") is True,
                "health is not ready")
        models = self.api("/v1/models")
        require(models.status == 200, "models endpoint not available")
        items = models.json().get("data", [])
        require(isinstance(items, list) and any(isinstance(m, dict) and
                m.get("id") == self.model for m in items), "configured model missing")
        metrics = self.metrics()
        for name in ("hermes_server_ready", "hermes_requests_active",
                     "hermes_requests_queued", "hermes_inference_errors_total"):
            require(name in metrics, f"metric missing: {name}")
        require(metrics["hermes_server_ready"] == 1, "not ready in Prometheus metrics")
        return {"model": self.model, "active": metrics["hermes_requests_active"],
                "queued": metrics["hermes_requests_queued"]}

    def errors(self):
        base = chat_payload(self.model, "error validation")
        cases = [
            ("unknown-route", "/unknown/reliability", None, None, "GET", 404, "not_found"),
            ("model-404", "/v1/chat/completions",
             {**base, "model": "__nonexistent__"}, None, "POST", 404, "model_not_found"),
            ("tools-reject", "/v1/chat/completions",
             {**base, "tools": []}, None, "POST", 400, "unsupported_parameter"),
            ("sample-reject", "/v1/chat/completions",
             {**base, "temperature": .7}, None, "POST", 400, "unsupported_value"),
            ("top-p-reject", "/v1/chat/completions",
             {**base, "top_p": .9}, None, "POST", 400, "unsupported_value"),
            ("token-limit-overflow", "/v1/chat/completions",
             {**base, "max_tokens": 4294967297}, None, "POST", 400, "invalid_value"),
            ("token-limit-zero", "/v1/chat/completions",
             {**base, "max_tokens": 0}, None, "POST", 400, "invalid_value"),
            ("double-token-limit", "/v1/chat/completions",
             {**base, "max_completion_tokens": 1}, None, "POST", 400, "invalid_value"),
            ("bad-role", "/v1/chat/completions",
             {**base, "messages": [{"role": "tool", "content": "x"}]},
             None, "POST", 400, "unsupported_value"),
            ("bad-content", "/v1/chat/completions",
             {**base, "messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}]},
             None, "POST", 400, "unsupported_value"),
            ("empty-messages", "/v1/chat/completions",
             {**base, "messages": []}, None, "POST", 400, "invalid_value"),
            ("malformed-json", "/v1/chat/completions",
             None, b'{"messages":', "POST", 400, "invalid_json"),
            ("cancel-nonexistent", "/v1/cancel",
             {"id": "__missing_reliability__"}, None, "POST", 404, "not_found"),
        ]
        for name, path, body, raw, method, code, error_code in cases:
            res = self.api(path, payload=body, raw=raw, method=method)
            require(res.status == code, f"{name}: expected {code} got {res.status}")
            if name == "unknown-route":
                # cpp-httplib may return HTML for an unregistered route; only
                # explicitly implemented OpenAI routes promise JSON errors.
                continue
            obj = res.json()
            require(isinstance(obj.get("error"), dict) and
                    obj["error"].get("code") == error_code,
                    f"{name}: unexpected error shape/code: {obj}")
        return {"cases": len(cases), "all_explicit_http_errors": True}

    def auth(self):
        if not self.key:
            raise GateBlocked("no API key configured: security test requires a secured server")
        for path, payload in (("/health", None),
                              ("/v1/chat/completions", chat_payload(self.model, "no auth"))):
            reply = request(self.url, path, payload=payload, timeout=self.timeout)
            require(reply.status == 401 and reply.json().get("error", {}).get("code") ==
                    "invalid_api_key", f"missing bearer token accepted at {path}")
        return {"unauthenticated_health_and_chat": 401}

    def query(self, prompt: str, max_tokens: int = 3) -> tuple[str, str, int]:
        payload = chat_payload(self.model, prompt, max_tokens=max_tokens)
        reply = self.api("/v1/chat/completions", payload=payload)
        return validate_chat(reply, self.model, max_tokens)

    def unicode(self):
        cases = []
        for prompt in UNICODE:
            payload = chat_payload(self.model, prompt, max_tokens=4)
            plain = self.api("/v1/chat/completions", payload=payload)
            text, reason, _ = validate_chat(plain, self.model, 4)
            code, _, events = sse_request(self.url, "/v1/chat/completions",
                                          {**payload, "stream": True}, api_key=self.key,
                                          timeout=self.timeout)
            require(code == 200, f"Unicode SSE status={code}")
            streamed, streamed_reason, _ = validate_sse(events, self.model)
            require((streamed, streamed_reason) == (text, reason),
                    f"SSE vs nonstream mismatch for {prompt[:35]!r}")
            cases.append({"prompt_sha256_prefix": __import__("hashlib").sha256(
                prompt.encode("utf-8")).hexdigest()[:16], "length": len(text),
                          "finish_reason": reason})
        return {"unicode_cases": cases, "all_text_exact": True}

    def cancellation(self):
        before = self.metrics().get("hermes_requests_cancelled_total", 0)
        cancelled: list[str] = []
        def cancel_first(event):
            require(isinstance(event.get("id"), str) and bool(event["id"]),
                    "SSE cancel request ID missing")
            res = self.api("/v1/cancel", payload={"id": event["id"]})
            require(res.status == 202 and res.json().get("cancelled") is True,
                    f"active SSE cancellation refused: {res.status} {res.body[:200]!r}")
            cancelled.append(event["id"])
        payload = chat_payload(self.model, "Write a very long explanation of "
                               "binary search with examples and proofs.",
                               max_tokens=32, stream=True)
        code, _, events = sse_request(self.url, "/v1/chat/completions", payload,
                                     api_key=self.key, timeout=self.timeout,
                                     on_first=cancel_first)
        require(code == 200 and len(cancelled) == 1, "cancelled SSE did not return HTTP 200")
        require(events and events[-1] == "[DONE]" and events.count("[DONE]") == 1,
                "cancelled SSE did not terminate exactly once")
        endings = [e["choices"][0]["finish_reason"] for e in events
                   if isinstance(e, dict) and "choices" in e and
                   e["choices"][0].get("finish_reason") is not None]
        require(endings == ["cancelled"],
                f"cancelled SSE must finish once with reason cancelled: {endings}")
        self.assert_drained()
        after = self.metrics().get("hermes_requests_cancelled_total", 0)
        require(after >= before + 1, "server cancelled counter did not increase")
        return {"cancelled_request": cancelled[0], "finish_reason": "cancelled",
                "active_jobs_drained": True}

    def backpressure(self):
        # Run separately on a server started with --active 1 --queue 1.
        # Using a production --active 8 --queue 32 server should NOT be
        # claimed as an exercised bounded-backpressure gate.
        begin = threading.Event()
        def invoke(i):
            begin.wait(timeout=10)
            payload = chat_payload(self.model, f"Backpressure {i}: write about algorithms",
                                   max_tokens=16)
            return self.api("/v1/chat/completions", payload=payload)
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            futures = [pool.submit(invoke, i) for i in range(12)]
            begin.set()
            replies = [future.result(timeout=self.timeout * 3) for future in futures]
        statuses = [response.status for response in replies]
        require(429 in statuses and 200 in statuses,
                f"did not exercise bounded queue; expected 429 and 200, got {statuses}. "
                "Start a separate test server with --active 1 --queue 1")
        for reply in replies:
            if reply.status == 429:
                code = reply.json().get("error", {}).get("code")
                require(code in ("queue_full", "overloaded") and
                        reply.headers.get("Retry-After", "").isdigit(),
                        f"429 missing queue error/retry-after: {reply.body[:200]!r}")
            else:
                validate_chat(reply, self.model, 16)
        self.assert_drained()
        return {"statuses": statuses, "retry_after_verified": True}

    def concurrency(self, n: int, repeats: int) -> dict:
        prompts = [
            f"QA isolation key={i} group={n}. Reply with a short factual statement. "
            f"Unicode marker={UNICODE[i % len(UNICODE)]}" for i in range(n)
        ]
        # Deterministic serial baseline; model internals and unknown text are
        # never used as a synthetic expected answer.
        baseline = [self.query(p, max_tokens=3) for p in prompts]
        latencies: list[float] = []
        for rep in range(repeats):
            start = threading.Event()
            def invoke(i):
                start.wait(timeout=10)
                ts = time.monotonic()
                answer = self.query(prompts[i], max_tokens=3)
                return i, answer, time.monotonic() - ts
            with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
                futures = [pool.submit(invoke, i) for i in range(n)]
                start.set()
                # Avoid swallowing worker exceptions as an apparent pass.
                completed = [f.result(timeout=self.timeout * (n + 1)) for f in futures]
            for i, response, duration in completed:
                require(response == baseline[i],
                        f"N={n} repeat={rep} request={i}: different text/reason/tokens "
                        f"serial={baseline[i]!r} concurrent={response!r}")
                latencies.append(duration)
            self.assert_drained()
        return {"N": n, "repeats": repeats, "requests": n * repeats,
                "p50_latency_seconds": round(statistics.median(latencies), 3),
                "p95_latency_seconds": round(sorted(latencies)[
                    max(0, math.ceil(.95 * len(latencies)) - 1)], 3),
                "deterministic_text_isolation": True}

    def assert_drained(self):
        for attempt in range(15):
            current = self.metrics()
            if current.get("hermes_requests_active") == 0 and current.get("hermes_requests_queued") == 0:
                return
            time.sleep(.2)
        raise GateFailure(f"jobs did not drain: {current}")

    def memory(self, cycles: int, mib: int):
        self.query("VRAM warmup: answer one word.", max_tokens=1)
        self.assert_drained()
        first = self.metrics()
        v0 = vram_from_metrics(first)
        external0 = nvidia_vram()
        checkpoints = []
        for rep in range(cycles):
            for n in (1, 2, 4, 8):
                if n not in self.a.concurrency:
                    continue
                # Use already-validated workload path, and all iterations check
                # quiescent persistent state (not transient peak VRAM).
                self.concurrency(n, 1)
            v = vram_from_metrics(self.metrics())
            external = nvidia_vram()
            checkpoints.append({"cycle": rep + 1, "metrics_mib": v, "smi_mib": external})
            for i in range(2):
                require(v[i] - v0[i] <= mib,
                        f"GPU{i} metrics leak/growth {v[i] - v0[i]}MiB > {mib}MiB")
                require(external[i] - external0[i] <= mib,
                        f"GPU{i} nvidia-smi growth {external[i] - external0[i]}MiB > {mib}MiB")
        return {"warm_baseline_metrics_mib": v0, "warm_baseline_smi_mib": external0,
                "checkpoints": checkpoints, "growth_tolerance_mib": mib,
                "note": "steady-state only; does not bound transient peaks"}

    def contexts(self, filename: Path):
        if not filename.is_file():
            raise GateBlocked(f"context fixture missing: {filename}")
        data = json.loads(filename.read_text(encoding="utf-8"))
        require(data.get("schema") == "h27.context-fixtures.v1", "unrecognized context fixture")
        cases = data.get("cases", [])
        require(sorted(case.get("tokens") for case in cases) == [129, 1024, 8192],
                "requires exact 129/1024/8192-token fixtures")
        executed = []
        for case in cases:
            t = case["tokens"]
            payload = {"model": self.model, "messages": case["messages"],
                       "reasoning": {"enabled": case["thinking"]},
                       "temperature": 0, "max_tokens": 1}
            resp = self.api("/v1/chat/completions", payload=payload)
            if resp.status == 400 and resp.json().get("error", {}).get("code") == "context_length_exceeded":
                raise GateBlocked(f"server --context < {t}; restart in isolated GPU window with --context 8192")
            _, _, completion = validate_chat(resp, self.model, 1)
            observed = resp.json()["usage"]["prompt_tokens"]
            require(observed == t, f"context fixture expected {t} tokens, server got {observed}")
            require(completion == 1, f"context {t}: did not generate 1 token")
            executed.append(t)
        return {"validated_contexts": executed, "decode_state_capacity_exact": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="hermes-27")
    p.add_argument("--api-key", help="secure server bearer key (never written to report)")
    p.add_argument("--timeout", type=float, default=240, help="seconds per HTTP request")
    p.add_argument("--concurrency", default="1,2,4,8")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--memory-cycles", type=int, default=2)
    p.add_argument("--memory-tolerance-mib", type=int, default=128)
    p.add_argument("--context-fixtures", type=Path)
    p.add_argument("--run-long-contexts", action="store_true",
                   help="EXTRA HIGH COST: test exact 129/1024/8192-token live inference")
    p.add_argument("--skip-memory", action="store_true")
    p.add_argument("--skip-security", action="store_true")
    p.add_argument("--backpressure-only", action="store_true",
                   help="separate test instance must use --active 1 --queue 1")
    p.add_argument("--gpu-exclusive", action="store_true",
                   help="mandatory explicit consent from coordinator; no server lifecycle managed")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    try:
        a.concurrency = sorted(set(int(v) for v in a.concurrency.split(",")))
    except ValueError:
        p.error("invalid --concurrency")
    if (a.concurrency != [1, 2, 4, 8] or a.repeats < 1 or
            a.memory_cycles < 1 or a.timeout <= 0 or a.memory_tolerance_mib < 0):
        p.error("release gate requires N=1,2,4,8; positive repeats/cycles/timeout")
    report = RunReport("hermes27-online-v1", {
        "url": a.base_url, "model": a.model, "concurrency": a.concurrency,
        "repeats": a.repeats, "gpu_exclusive_approved": a.gpu_exclusive
    })
    if not a.gpu_exclusive:
        for name in ("health", "error-paths", "security", "unicode-and-sse",
                     "stream-cancellation", "queue-backpressure",
                     "concurrency-N1", "concurrency-N2", "concurrency-N4",
                     "concurrency-N8", "steady-vram", "contexts-129-1024-8192"):
            report.skip(name, "GPU permission not granted; no HTTP inference attempted")
        status, label = report.save(a.out)
        print(f"RELEASE ONLINE {label}: {a.out}")
        return status
    gate = OnlineGate(a)
    if a.backpressure_only:
        report.record("queue-backpressure", gate.backpressure)
        code, label = report.save(a.out)
        print(f"RELEASE BACKPRESSURE {label}: {a.out}")
        return code
    report.record("health", gate.health)
    report.record("error-paths", gate.errors)
    if a.skip_security:
        report.skip("security", "operator explicitly skipped protected-endpoint validation")
    else:
        report.record("security", gate.auth)
    report.record("unicode-and-sse", gate.unicode)
    report.record("stream-cancellation", gate.cancellation)
    for n in a.concurrency:
        report.record(f"concurrency-N{n}", lambda n=n: gate.concurrency(n, a.repeats))
    if a.skip_memory:
        report.skip("steady-vram", "operator explicitly skipped memory-leak gate")
    else:
        report.record("steady-vram", lambda: gate.memory(a.memory_cycles,
                                                        a.memory_tolerance_mib))
    if not a.run_long_contexts:
        report.skip("contexts-129-1024-8192",
                    "EXTRA HIGH COST: opt in --run-long-contexts and supply --context-fixtures")
    else:
        report.record("contexts-129-1024-8192", lambda: gate.contexts(
            a.context_fixtures or Path("/nonexistent/context-fixtures.json")))
    code, label = report.save(a.out)
    print(f"RELEASE ONLINE {label}: {a.out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
