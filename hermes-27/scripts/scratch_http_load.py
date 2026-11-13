#!/usr/bin/env python3
"""Isolated HTTP concurrency check; reports measured throughput, not engine decode TPS.

Requires an already running 127.0.0.1 scratch HERMES-27 server. Fails closed
on any HTTP status/error, unusable usage count, or missing final content.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request


def request(url: str, prompt: str, count: int, timeout: int) -> dict:
    body = {
        "model": "hermes-27",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": count,
        "temperature": 0,
        "reasoning": {"effort": "none"},
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url + "/v1/chat/completions", data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            code = response.status
            result = json.loads(response.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from e
    elapsed = time.monotonic() - started
    if code != 200 or "error" in result:
        raise RuntimeError(f"unexpected HTTP response: {code} {result}")
    choices = result.get("choices", [])
    usage = result.get("usage", {})
    generated = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    if (not isinstance(generated, int) or isinstance(generated, bool) or
            generated < 1 or generated > count or
            not isinstance(prompt_tokens, int) or prompt_tokens < 1 or
            len(choices) != 1 or not isinstance(choices[0].get("message", {}).get("content"), str)):
        raise RuntimeError(f"malformed counts or response: {result}")
    return {"elapsed_sec": elapsed, "completion_tokens": generated,
            "prompt_tokens": prompt_tokens, "finish_reason": choices[0].get("finish_reason")}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    low = math.floor(idx)
    high = math.ceil(idx)
    return ordered[low] + (ordered[high] - ordered[low]) * (idx - low)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18127")
    parser.add_argument("--clients", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--prompt-style", choices=("arithmetic", "explain"),
                        default="arithmetic",
                        help="arithmetic often stops early; explain asks for a longer answer")
    args = parser.parse_args()
    url = urllib.parse.urlparse(args.base_url)
    if url.scheme != "http" or url.hostname != "127.0.0.1" or url.port != 18127:
        parser.error("only isolated http://127.0.0.1:18127 is allowed")
    if (args.rounds < 1 or args.rounds > 10 or
            args.max_tokens < 1 or args.max_tokens > 64 or
            any(n not in (1, 2, 4, 8) for n in args.clients)):
        parser.error("rounds 1..10, tokens 1..64, clients 1/2/4/8")
    for n in args.clients:
        for rnd in range(args.rounds):
            if args.prompt_style == "arithmetic":
                prompts = [f"Answer briefly in English: what is 2 + 2? Request {i}, round {rnd}."
                           for i in range(n)]
            else:
                prompts = [
                    "Explain in several complete English sentences why GPU memory bandwidth, "
                    "occupancy, and attention caching determine LLM inference performance. "
                    f"Give concrete technical details. Case {i} round {rnd}."
                    for i in range(n)
                ]
            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=n) as pool:
                futures = [pool.submit(request, args.base_url, p, args.max_tokens, args.timeout)
                           for p in prompts]
                results = [f.result() for f in futures]
            elapsed = time.monotonic() - started
            tokens = sum(r["completion_tokens"] for r in results)
            values = [r["elapsed_sec"] for r in results]
            print(json.dumps({
                "kind": "HTTP_END_TO_END_CONCURRENCY",
                "clients": n, "round": rnd, "requests": len(results),
                "prompt_style": args.prompt_style,
                "wall_sec": round(elapsed, 4),
                "completion_tokens": tokens,
                "prompt_tokens": sum(r["prompt_tokens"] for r in results),
                "completion_tokens_per_wall_sec_including_prefill": round(tokens / elapsed, 3),
                "request_latency_p50_sec": round(statistics.median(values), 4),
                "request_latency_p95_sec": round(percentile(values, .95), 4),
            }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
