#!/usr/bin/env python3
"""Measured streaming HTTP concurrency, TTFC and per-visible-delta timing.

Strict isolated localhost-only HERMES-27 scratch acceptance probe. The
first *nonempty visible content delta* can lag the first generated model
token (UTF-8 buffering, invisible/special tokens). Therefore TTFC is NOT
automatically TTFT; SSE delta gaps are NOT per-token TPOT. No GPU or
process management: caller exclusively owns scratch-server lifecycle.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires samples")
    ordered = sorted(values)
    x = (len(ordered) - 1) * quantile
    lo, hi = math.floor(x), math.ceil(x)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (x - lo)


def consume_sse(url: str, case: int, rnd: int, max_tokens: int,
                timeout: float) -> dict:
    prompt = (
        "Explain in several complete English sentences why GPU memory bandwidth, "
        "occupancy, and attention caching determine LLM inference performance. "
        f"Give concrete technical details. Case {case} round {rnd}."
    )
    payload = {
        "model": "hermes-27",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "reasoning": {"effort": "none"},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.monotonic()
    chunks = []
    content_timestamps = []
    usage = None
    finish = []
    done = 0
    role_seen = False
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"unexpected HTTP {response.status}")
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            raise RuntimeError("not SSE Content-Type")
        for raw in response:
            line = raw.strip()
            if not line:
                continue
            if not line.startswith(b"data: "):
                raise RuntimeError(f"unexpected SSE line: {raw[:100]!r}")
            now = time.monotonic()
            data = line[6:]
            if data == b"[DONE]":
                done += 1
                continue
            event = json.loads(data.decode("utf-8", errors="strict"))
            if "error" in event:
                raise RuntimeError(f"server SSE error: {event['error']!r}")
            if event.get("usage") is not None:
                if usage is not None:
                    raise RuntimeError("duplicate SSE usage event")
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("finish_reason") is not None:
                    finish.append(choice["finish_reason"])
                delta = choice.get("delta") or {}
                if delta.get("role") == "assistant":
                    role_seen = True
                part = delta.get("content", "")
                if not isinstance(part, str):
                    raise RuntimeError("content delta must be string")
                if part:
                    content_timestamps.append(now)
                    chunks.append(part)
    completed = time.monotonic()
    if done != 1 or len(finish) != 1 or usage is None or not role_seen:
        raise RuntimeError("missing/duplicate SSE terminal, usage, role or DONE")
    if finish[0] not in ("length", "stop"):
        raise RuntimeError(f"unexpected finish reason {finish!r}")
    tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    if (not isinstance(tokens, int) or isinstance(tokens, bool) or
            not 1 <= tokens <= max_tokens or
            not isinstance(prompt_tokens, int) or prompt_tokens < 1 or
            not content_timestamps):
        raise RuntimeError(f"invalid SSE token usage/content: {usage!r}")
    text = "".join(chunks)
    gaps = [(b - a) for a, b in
            zip(content_timestamps, content_timestamps[1:])]
    return {
        "case": case, "round": rnd,
        "elapsed_sec": completed - started,
        "ttfc_sec": content_timestamps[0] - started,
        "last_visible_sec": content_timestamps[-1] - started,
        "visible_deltas": len(content_timestamps),
        "delta_gaps_sec": gaps,
        "completion_tokens": tokens,
        "prompt_tokens": prompt_tokens,
        "finish": finish[0],
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "visible_text_bytes": len(text.encode("utf-8")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18127")
    parser.add_argument("--clients", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--variant", choices=("parallel", "legacy", "serial"),
                        required=True)
    args = parser.parse_args()
    parsed = urllib.parse.urlparse(args.base_url)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or
            parsed.port != 18127 or parsed.path not in ("", "/") or
            parsed.username or parsed.password or parsed.query or parsed.fragment):
        parser.error("ONLY isolated http://127.0.0.1:18127 is allowed")
    if (args.rounds not in range(1, 4) or
            args.max_tokens not in range(2, 65) or
            not args.clients or any(n not in (1, 2, 4, 8) for n in args.clients) or
            args.timeout < 5 or args.timeout > 240):
        parser.error("rounds 1..3, tokens 2..64, clients 1/2/4/8, timeout 5..240")
    all_results = []
    for n in args.clients:
        for rnd in range(args.rounds):
            started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
                jobs = [
                    pool.submit(consume_sse, args.base_url, case, rnd,
                                args.max_tokens, args.timeout)
                    for case in range(n)
                ]
                results = [job.result() for job in jobs]
            wall = time.monotonic() - started
            if len(results) != n:
                raise RuntimeError("request count mismatch")
            all_results.extend(results)
            tokens = sum(x["completion_tokens"] for x in results)
            gaps = [gap for x in results for gap in x["delta_gaps_sec"]]
            report = {
                "schema": "h27.sse-timing.v1",
                "variant": args.variant, "clients": n, "round": rnd,
                "max_tokens": args.max_tokens,
                "requests": n,
                "completion_tokens": tokens,
                "prompt_tokens": sum(x["prompt_tokens"] for x in results),
                "wall_sec": round(wall, 6),
                "aggregate_completion_tok_per_wall_sec": round(tokens / wall, 6),
                "request_latency_p50_sec": round(statistics.median(x["elapsed_sec"] for x in results), 6),
                "request_latency_p95_sec": round(percentile([x["elapsed_sec"] for x in results], .95), 6),
                "ttfc_p50_sec": round(statistics.median(x["ttfc_sec"] for x in results), 6),
                "ttfc_p95_sec": round(percentile([x["ttfc_sec"] for x in results], .95), 6),
                "visible_delta_gap_p50_sec": round(statistics.median(gaps), 6) if gaps else None,
                "visible_delta_gap_p95_sec": round(percentile(gaps, .95), 6) if gaps else None,
                "outputs": [
                    {"case": x["case"], "sha256": x["content_sha256"],
                     "finish": x["finish"], "completion_tokens": x["completion_tokens"],
                     "ttfc_sec": round(x["ttfc_sec"], 6),
                     "visible_deltas": x["visible_deltas"]}
                    for x in sorted(results, key=lambda x: x["case"])
                ],
                "note": "TTFC is first visible content, NOT exact GPU first-token TTFT; delta gaps are NOT token TPOT",
            }
            print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
        raise SystemExit(f"SSE_LATENCY_GATE_FAIL: {error}")
