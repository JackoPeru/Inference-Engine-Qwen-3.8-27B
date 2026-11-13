#!/usr/bin/env python3
"""Fail-closed, localhost-only HERMES-27 HTTP N4/N8 stability harness.

The caller must start the isolated scratch server separately. This module
never uploads a model, creates a GPU context, or manages a process/service.
It has NO network side effects until main()/run_soak() is explicitly invoked.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import http.client
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable


BASE_URL = "http://127.0.0.1:18127"
MODEL = "hermes-27"
REQUIRED_COUNTERS = (
    "hermes_requests_total",
    "hermes_requests_accepted_total",
    "hermes_requests_completed_total",
    "hermes_requests_rejected_total",
    "hermes_requests_cancelled_total",
    "hermes_stream_backpressure_total",
    "hermes_inference_errors_total",
    "hermes_prefill_failed_total",
    "hermes_prompt_tokens_total",
    "hermes_completion_tokens_total",
)
REQUIRED_GAUGES = ("hermes_server_ready", "hermes_requests_active", "hermes_requests_queued")
METRIC_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^{}]*\})?\s+(\S+)(?:\s+.*)?$")


class SoakFailure(RuntimeError):
    """Unsafe/invalid test precondition or reliability regression; stop now."""


def validate_url(raw: str) -> str:
    """Deny alternate hosts, redirect pivots, credentials, paths and DNS names."""
    if not isinstance(raw, str) or raw not in (BASE_URL, BASE_URL + "/"):
        raise SoakFailure("only exact http://127.0.0.1:18127 is permitted")
    parsed = urllib.parse.urlsplit(raw)
    if (parsed.scheme, parsed.hostname, parsed.port) != ("http", "127.0.0.1", 18127):
        raise SoakFailure("only scratch loopback endpoint is permitted")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise SoakFailure("URL credentials, queries and fragments are forbidden")
    return BASE_URL


@dataclass(frozen=True)
class Config:
    base_url: str = BASE_URL
    clients: tuple[int, ...] = (4, 8)
    rounds: int = 5
    max_tokens: int = 8
    timeout: int = 90
    drain_timeout: float = 15.0
    poll_interval: float = 0.25
    prompt_style: str = "arithmetic"
    vram_allowance_mib: int = 64
    vram_warmup_limit_mib: int = 256
    vram_step_mib: int = 8
    rising_steps: int = 3
    model: str = MODEL
    api_key: str | None = None

    def checked(self) -> "Config":
        validate_url(self.base_url)
        if (not self.clients or len(set(self.clients)) != len(self.clients) or
                any(n not in (4, 8) for n in self.clients)):
            raise SoakFailure("client counts must be unique, selected from N4/N8")
        if not 2 <= self.rounds <= 12:
            raise SoakFailure("rounds must be 2..12 for repeatability")
        if not 1 <= self.max_tokens <= 64:
            raise SoakFailure("max_tokens must be 1..64")
        if not 1 <= self.timeout <= 120:
            raise SoakFailure("request timeout must be 1..120 seconds")
        if not 0.1 <= self.drain_timeout <= 60:
            raise SoakFailure("drain timeout must be 0.1..60 seconds")
        if not 0.01 <= self.poll_interval <= 2:
            raise SoakFailure("poll interval must be 0.01..2 seconds")
        if self.prompt_style not in ("arithmetic", "explain"):
            raise SoakFailure("unsupported prompt style")
        if (not 0 <= self.vram_allowance_mib <= 512 or
                not 0 <= self.vram_warmup_limit_mib <= 1024 or
                not 1 <= self.vram_step_mib <= 128):
            raise SoakFailure("invalid GPU VRAM trend thresholds")
        if not 2 <= self.rising_steps <= 6:
            raise SoakFailure("rising_steps must be 2..6")
        if self.model != MODEL:
            raise SoakFailure("model must match the scratch hermes-27 endpoint")
        if self.api_key is not None and (not self.api_key or "\r" in self.api_key or
                                          "\n" in self.api_key):
            raise SoakFailure("invalid API key value")
        return self


class DenyRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise SoakFailure("redirect refused: no requests may leave localhost scratch endpoint")


class HttpClient:
    """Injectable urllib opener; production opener refuses ALL redirects."""

    def __init__(self, config: Config, opener=None):
        self.cfg = config.checked()
        # Explicitly ignore HTTP_PROXY/HTTPS_PROXY/ALL_PROXY from the shell.
        # A proxy could otherwise forward even a loopback URL to an external
        # process. The harness makes only direct 127.0.0.1 connections.
        self.opener = (opener if opener is not None else
                       urllib.request.build_opener(urllib.request.ProxyHandler({}),
                                                   DenyRedirects()))

    def fetch(self, path: str, payload=None) -> tuple[int, bytes, str]:
        if path not in ("/health", "/v1/models", "/metrics", "/v1/chat/completions"):
            raise SoakFailure("unexpected HTTP path")
        if (payload is None) == (path == "/v1/chat/completions"):
            raise SoakFailure("invalid HTTP method/payload")
        url = validate_url(self.cfg.base_url) + path
        headers = {"Accept": "application/json" if path != "/metrics" else "text/plain"}
        if self.cfg.api_key is not None:
            headers["Authorization"] = "Bearer " + self.cfg.api_key
        if payload is not None:
            headers["Content-Type"] = "application/json"
        data = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                if payload is not None else None)
        req = urllib.request.Request(url, data=data,
                                     method="POST" if data is not None else "GET",
                                     headers=headers)
        try:
            with self.opener.open(req, timeout=self.cfg.timeout) as response:
                status = response.status
                # Reject mocked/custom opener redirects as well.
                actual = response.geturl() if hasattr(response, "geturl") else url
                if actual != url:
                    raise SoakFailure("response URL changed (redirect blocked)")
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                # Limit all responses including metrics and error bodies.
                content = response.read(2 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as exc:
            raise SoakFailure(f"{path}: HTTP {exc.code}; aborting soak") from exc
        except (urllib.error.URLError, TimeoutError, OSError,
                http.client.HTTPException) as exc:
            raise SoakFailure(f"{path}: HTTP transport failed: {type(exc).__name__}") from exc
        if status != 200:
            raise SoakFailure(f"{path}: expected HTTP 200, got {status}")
        if len(content) > 2 * 1024 * 1024:
            raise SoakFailure(f"{path}: response too large")
        expected = "text/plain" if path == "/metrics" else "application/json"
        if content_type != expected:
            raise SoakFailure(f"{path}: wrong Content-Type {content_type!r}")
        if not content:
            raise SoakFailure(f"{path}: empty HTTP response")
        return status, content, content_type

    def json(self, path: str, payload=None) -> dict:
        _, raw, _ = self.fetch(path, payload)
        try:
            result = json.loads(raw)
        except (UnicodeError, ValueError) as exc:
            raise SoakFailure(f"{path}: malformed JSON response") from exc
        if not isinstance(result, dict) or "error" in result:
            raise SoakFailure(f"{path}: error payload or non-object response")
        return result


def _count(value: object, label: str, *, allow_zero: bool = True) -> int:
    if (type(value) is not int or value < (0 if allow_zero else 1)):
        raise SoakFailure(f"{label}: invalid nonnegative integer")
    return value


@dataclass(frozen=True)
class Metrics:
    counters: dict[str, int]
    active: int
    queued: int
    ready: int
    vram: tuple[int, int]

    def counter(self, name: str) -> int:
        return self.counters[name]


def parse_metrics(content: bytes) -> Metrics:
    try:
        body = content.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise SoakFailure("metrics are not valid UTF-8") from exc
    scalars: dict[str, int] = {}
    vram: dict[int, int] = {}
    wanted = set(REQUIRED_COUNTERS) | set(REQUIRED_GAUGES)
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = METRIC_LINE.fullmatch(line)
        if not match:
            raise SoakFailure("malformed Prometheus metrics line")
        name, labels, raw = match.groups()
        if name not in wanted and name != "hermes_vram_used_mib":
            continue
        if not re.fullmatch(r"[0-9]+", raw):
            raise SoakFailure(f"{name}: noninteger/negative/nonfinite gauge or counter")
        value = int(raw)
        if name == "hermes_vram_used_mib":
            if labels not in ('{gpu="0"}', '{gpu="1"}'):
                raise SoakFailure("unexpected or missing GPU VRAM label")
            gpu = 0 if labels == '{gpu="0"}' else 1
            if gpu in vram or value > 16384:
                raise SoakFailure("duplicate or impossible GPU VRAM metric")
            vram[gpu] = value
        else:
            if labels or name in scalars:
                raise SoakFailure(f"duplicate/labeled required metric {name}")
            scalars[name] = value
    if wanted - set(scalars) or set(vram) != {0, 1}:
        raise SoakFailure("missing required active/queue/error/usage/VRAM metrics")
    if scalars["hermes_server_ready"] != 1:
        raise SoakFailure("scratch server readiness metric is not 1")
    if scalars["hermes_requests_active"] > 16 or scalars["hermes_requests_queued"] > 32:
        raise SoakFailure("invalid request occupancy gauge")
    return Metrics(
        counters={k: scalars[k] for k in REQUIRED_COUNTERS},
        active=scalars["hermes_requests_active"],
        queued=scalars["hermes_requests_queued"],
        ready=scalars["hermes_server_ready"],
        vram=(vram[0], vram[1]),
    )


def get_metrics(http: HttpClient) -> Metrics:
    return parse_metrics(http.fetch("/metrics")[1])


def wait_for_drain(http: HttpClient, timeout: float, poll_interval: float,
                   *, now: Callable[[], float] = time.monotonic,
                   sleep: Callable[[float], None] = time.sleep) -> Metrics:
    deadline = now() + timeout
    last_idle_vram: tuple[int, int] | None = None
    while True:
        snapshot = get_metrics(http)
        if snapshot.active == 0 and snapshot.queued == 0:
            # 'active == 0' can precede the final async decode_free's CUDA
            # stream drain. Require two successive zero/zero readings with
            # equal GPU-used values; never sample a transient post-request
            # CUDA deallocation as the settled memory high-water.
            if snapshot.vram == last_idle_vram:
                return snapshot
            last_idle_vram = snapshot.vram
        else:
            last_idle_vram = None
        if now() >= deadline:
            raise SoakFailure(f"drain timeout: active={snapshot.active} queued={snapshot.queued}")
        sleep(min(poll_interval, max(0.0, deadline - now())))


def make_prompt(style: str, clients: int, slot: int) -> str:
    # Crucially NO round number: the same case must yield the same output
    # and usage in repeated deterministic rounds.
    if style == "arithmetic":
        return (f"N{clients} case-{slot}. Answer only with the integer: "
                "what is two plus two? Do not explain.")
    return (f"N{clients} case-{slot}. In three short English sentences, explain "
            "how GPU memory bandwidth, occupancy and KV caching affect LLM inference.")


def validate_chat(data: dict, max_tokens: int, expected_model: str) -> dict:
    if (data.get("object") != "chat.completion" or data.get("model") != expected_model or
            not isinstance(data.get("id"), str) or not data["id"]):
        raise SoakFailure("wrong chat response object/model/id")
    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise SoakFailure("chat response must contain exactly one choice")
    first = choices[0]
    if first.get("index") != 0 or first.get("finish_reason") not in ("stop", "length"):
        raise SoakFailure("unexpected choice index, cancellation, or finish reason")
    message = first.get("message")
    if (not isinstance(message, dict) or message.get("role") != "assistant" or
            not isinstance(message.get("content"), str) or not message["content"].strip()):
        raise SoakFailure("missing or empty assistant response content")
    usage = data.get("usage")
    if not isinstance(usage, dict):
        raise SoakFailure("missing usage counters")
    prompt = _count(usage.get("prompt_tokens"), "prompt_tokens", allow_zero=False)
    completion = _count(usage.get("completion_tokens"), "completion_tokens", allow_zero=False)
    total = _count(usage.get("total_tokens"), "total_tokens", allow_zero=False)
    if completion > max_tokens or prompt + completion != total:
        raise SoakFailure("invalid total or exceeding max completion tokens")
    text = message["content"]
    fingerprint = {
        "text": text,
        "finish_reason": first["finish_reason"],
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }
    canonical = json.dumps(fingerprint, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    return {
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "case_sha256": hashlib.sha256(canonical).hexdigest(),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "finish_reason": first["finish_reason"],
    }


def check_deltas(before: Metrics, after: Metrics, count: int,
                 sum_prompt: int, sum_completion: int) -> None:
    expected = {
        "hermes_requests_total": count,
        "hermes_requests_accepted_total": count,
        "hermes_requests_completed_total": count,
        "hermes_prompt_tokens_total": sum_prompt,
        "hermes_completion_tokens_total": sum_completion,
    }
    for name in REQUIRED_COUNTERS:
        change = after.counter(name) - before.counter(name)
        if change != expected.get(name, 0):
            raise SoakFailure(
                f"{name}: expected delta {expected.get(name, 0)}, got {change}; "
                "unexpected inference error, rejection, cancellation, external traffic or reset")


class VramGuard:
    """Thresholded trend checks on *post-drain* server GPU-used metrics.

    First round may establish lazy cache/allocator high-water. Later plateaus
    are cache-LIKE, never proof that retained memory is not leaked. Persistent
    upward drift is a suspected leak and causes a fail-closed exit.
    """

    def __init__(self, config: Config, baseline: tuple[int, int]):
        self.cfg = config
        self.baseline = baseline
        self.settled: tuple[int, int] | None = None
        self.previous: tuple[int, int] | None = None
        self.consecutive = [0, 0]

    def check(self, sample: tuple[int, int]) -> str:
        if self.settled is None:
            for gpu in (0, 1):
                if sample[gpu] - self.baseline[gpu] > self.cfg.vram_warmup_limit_mib:
                    raise SoakFailure(
                        f"GPU{gpu} warmup VRAM rose >{self.cfg.vram_warmup_limit_mib} MiB "
                        "relative to pre-cohort baseline: unexplained allocation")
            self.settled = sample  # first successful round is a warmup
            self.previous = sample
            return "warmup_cache_unclassified"
        assert self.previous is not None
        for gpu in (0, 1):
            above_warm = sample[gpu] - self.settled[gpu]
            step = sample[gpu] - self.previous[gpu]
            self.consecutive[gpu] = (self.consecutive[gpu] + 1
                                     if step >= self.cfg.vram_step_mib else 0)
            if above_warm > self.cfg.vram_allowance_mib:
                raise SoakFailure(
                    f"GPU{gpu} VRAM grew {above_warm} MiB after warmup "
                    f"(max {self.cfg.vram_allowance_mib} MiB): possible leak")
            if self.consecutive[gpu] >= self.cfg.rising_steps:
                raise SoakFailure(
                    f"GPU{gpu} VRAM rose at least {self.cfg.vram_step_mib} MiB "
                    f"for {self.consecutive[gpu]} successive post-warmup rounds: possible leak")
        self.previous = sample
        return ("cache_or_allocator_plateau_not_proven_leak_free"
                if sample != self.baseline or self.settled != self.baseline else "stable")


def _one_chat(http: HttpClient, clients: int, slot: int,
              *, now: Callable[[], float]) -> dict:
    body = {
        "model": http.cfg.model,
        "messages": [{"role": "user", "content": make_prompt(http.cfg.prompt_style, clients, slot)}],
        "max_tokens": http.cfg.max_tokens,
        "temperature": 0,
        "reasoning": {"effort": "none"},
        "stream": False,
    }
    start = now()
    answer = validate_chat(http.json("/v1/chat/completions", body),
                           http.cfg.max_tokens, http.cfg.model)
    elapsed = now() - start
    if not math.isfinite(elapsed) or elapsed < 0:
        raise SoakFailure("invalid request monotonic latency")
    answer["latency_sec"] = elapsed
    answer["slot"] = slot
    return answer


def run_soak(config: Config, *, opener=None,
             now: Callable[[], float] = time.monotonic,
             sleep: Callable[[float], None] = time.sleep,
             emit: Callable[[dict], None] | None = None) -> list[dict]:
    """Run against a previously started scratch server (tests inject opener)."""
    config.checked()  # no HTTP access until even the entire config is valid
    http = HttpClient(config, opener=opener)
    health = http.json("/health")
    if health.get("ready") is not True or health.get("status") != "ok":
        raise SoakFailure("/health does not confirm scratch server readiness")
    models = http.json("/v1/models").get("data")
    if (not isinstance(models, list) or not any(
            isinstance(model, dict) and model.get("id") == config.model for model in models)):
        raise SoakFailure("scratch model ID absent from /v1/models")
    previous = get_metrics(http)
    if previous.active != 0 or previous.queued != 0:
        raise SoakFailure("preflight refused: scratch server already has active/queued requests")
    fingerprints: dict[tuple[int, int], str] = {}
    reports: list[dict] = []
    for clients in config.clients:
        # Workload shape changes from N4 to N8 may legitimately create new
        # allocator high-water/caches. Give EACH cohort one warmup sample,
        # and measure sustained rises only within the same fixed-N cohort.
        guard = VramGuard(config, previous.vram)
        for rnd in range(config.rounds):
            started = now()
            # ThreadPoolExecutor is bounded by the requested N and the network
            # timeouts. On any exception no following round is scheduled.
            with ThreadPoolExecutor(max_workers=clients) as pool:
                futures = [pool.submit(_one_chat, http, clients, slot, now=now)
                           for slot in range(clients)]
                results = [future.result() for future in futures]
            elapsed = now() - started
            if elapsed <= 0 or not math.isfinite(elapsed):
                raise SoakFailure("invalid round elapsed monotonic duration")
            # Crucial: check every case against the first successful round;
            # no round number is part of the prompt or checksum.
            for result in results:
                key = (clients, result["slot"])
                previous_sha = fingerprints.setdefault(key, result["case_sha256"])
                if previous_sha != result["case_sha256"]:
                    raise SoakFailure(
                        f"deterministic output or usage drift: N{clients} "
                        f"slot={result['slot']} round={rnd}")
            settled = wait_for_drain(http, config.drain_timeout, config.poll_interval,
                                     now=now, sleep=sleep)
            sum_prompt = sum(result["prompt_tokens"] for result in results)
            sum_completion = sum(result["completion_tokens"] for result in results)
            check_deltas(previous, settled, clients, sum_prompt, sum_completion)
            memory_status = guard.check(settled.vram)
            record = {
                "kind": "HERMES_HTTP_SOAK_ROUND",
                "clients": clients,
                "round": rnd,
                "wall_sec": round(elapsed, 5),
                "requests": clients,
                "prompt_tokens": sum_prompt,
                "completion_tokens": sum_completion,
                "completion_tokens_per_wall_sec_including_prefill": round(sum_completion / elapsed, 4),
                "vram_used_mib": list(settled.vram),
                "vram_baseline_mib": list(guard.baseline),
                "vram_post_warmup_reference_mib": list(guard.settled or settled.vram),
                "vram_assessment": memory_status,
                "active": settled.active,
                "queued": settled.queued,
                "inference_errors_delta": 0,
                "cases": [
                    {k: value for k, value in result.items() if k != "latency_sec"}
                    | {"latency_sec": round(result["latency_sec"], 5)}
                    for result in results
                ],
            }
            reports.append(record)
            if emit is not None:
                emit(record)
            previous = settled
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--clients", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--drain-timeout", type=float, default=15.0)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--prompt-style", choices=("arithmetic", "explain"), default="arithmetic")
    parser.add_argument("--vram-allowance-mib", type=int, default=64)
    parser.add_argument("--vram-warmup-limit-mib", type=int, default=256)
    parser.add_argument("--vram-step-mib", type=int, default=8)
    parser.add_argument("--rising-steps", type=int, default=3)
    parser.add_argument("--api-key-env", default=None,
                        help="Optional existing environment variable with scratch server bearer token; never printed")
    args = parser.parse_args(argv)
    try:
        # No accidental network calls if a secret env var does not exist.
        api_key = None
        if args.api_key_env is not None:
            import os
            if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,79}", args.api_key_env):
                raise SoakFailure("invalid API-key environment variable name")
            api_key = os.environ.get(args.api_key_env)
            if not api_key:
                raise SoakFailure("API-key environment variable is not populated")
        cfg = Config(base_url=args.base_url, clients=tuple(args.clients),
                     rounds=args.rounds, max_tokens=args.max_tokens,
                     timeout=args.timeout, drain_timeout=args.drain_timeout,
                     poll_interval=args.poll_interval, prompt_style=args.prompt_style,
                     vram_allowance_mib=args.vram_allowance_mib,
                     vram_warmup_limit_mib=args.vram_warmup_limit_mib,
                     vram_step_mib=args.vram_step_mib, rising_steps=args.rising_steps,
                     api_key=api_key).checked()
        reports = run_soak(cfg, emit=lambda item: print(json.dumps(item, sort_keys=True), flush=True))
        print(json.dumps({"kind": "HERMES_HTTP_SOAK_PASS", "rounds": len(reports),
                          "cases_per_round": list(cfg.clients),
                          "note": "cache-like plateau is not proof of leak freedom"},
                         sort_keys=True), flush=True)
        return 0
    except SoakFailure as exc:
        print(json.dumps({"kind": "HERMES_HTTP_SOAK_FAIL_CLOSED",
                          "reason": str(exc)}, sort_keys=True), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
