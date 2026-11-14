#!/usr/bin/env python3
"""Opt-in, localhost-only OpenAI chat SSE timing profiler for HERMES-27.

No GPU/service control. Import and default CLI do not perform network I/O.
An explicit --run is required to contact 127.0.0.1:18127. Standard-library
http.client sends direct HTTP, so it does not obey HTTP_PROXY or redirects.

TTFC = first nonempty generated text/reasoning SSE delta observed by client;
TTFC is NOT verified GPU TTFT. SSE fragments need not be model tokens, so
true TPOT is never inferred from their spacing. Use final usage for the
completion token denominator of wall-clock throughput, never chunk count.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
import hashlib
import http.client
import json
import math
from pathlib import Path
import re
import socket
import statistics
import threading
import time
from typing import Callable


HOST = "127.0.0.1"
PORT = 18127
CHAT_PATH = "/v1/chat/completions"
METRICS_PATH = "/metrics"
ALLOWED_CONCURRENCY = (1, 2, 4, 8)
SCHEMA = "h27.scratch-http-sse-profile.v1"
MAX_EVENT_BYTES = 131072
MAX_LINE_BYTES = 65536
MAX_METRICS_BYTES = 262144
DEFAULT_PROMPT = (
    "Explain how GPU memory bandwidth, kernel scheduling and KV caching "
    "affect language model inference performance. Give a concise technical "
    "answer with several specific examples."
)


class ProfileFailure(RuntimeError):
    """A bounded, sanitized failure code (never response or prompt text)."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class Config:
    model: str = "hermes-27"
    prompt: str = DEFAULT_PROMPT
    max_tokens: int = 32
    rounds: int = 2
    concurrency: tuple[int, ...] = ALLOWED_CONCURRENCY
    timeout_s: float = 120.0
    max_sse_bytes: int = 4 * 1024 * 1024
    read_bytes: int = 4096
    metrics: bool = False
    run_label: str = "baseline"

    def validate(self) -> None:
        if not isinstance(self.model, str) or not re.fullmatch(r"[A-Za-z0-9._/-]{1,128}", self.model):
            raise ValueError("invalid model identifier")
        if (not isinstance(self.prompt, str) or not 1 <= len(self.prompt.encode("utf-8")) <= 8192
                or "\x00" in self.prompt):
            raise ValueError("prompt length/format invalid")
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 4096:
            raise ValueError("max_tokens outside 1..4096")
        if type(self.rounds) is not int or not 1 <= self.rounds <= 16:
            raise ValueError("rounds outside 1..16")
        if (not isinstance(self.concurrency, tuple) or not self.concurrency
                or len(self.concurrency) != len(set(self.concurrency))
                or any(type(n) is not int or n not in ALLOWED_CONCURRENCY for n in self.concurrency)):
            raise ValueError("concurrency must be a unique subset of N1,N2,N4,N8")
        if (type(self.timeout_s) not in (float, int) or not math.isfinite(self.timeout_s)
                or not 1 <= self.timeout_s <= 1200):
            raise ValueError("timeout_s outside 1..1200")
        if (type(self.max_sse_bytes) is not int or
                not 1024 <= self.max_sse_bytes <= 64 * 1024 * 1024):
            raise ValueError("max_sse_bytes outside 1 KiB..64 MiB")
        if type(self.read_bytes) is not int or not 1 <= self.read_bytes <= 65536:
            raise ValueError("read_bytes outside 1..65536")
        if type(self.metrics) is not bool:
            raise ValueError("metrics must be boolean")
        if (not isinstance(self.run_label, str) or
                not re.fullmatch(r"[a-zA-Z0-9_-]{1,48}", self.run_label)):
            raise ValueError("run_label must be 1..48 safe ASCII characters")


def percentile(values: list[float], quantile: float) -> float | None:
    """Linear interpolation between order statistics (type-7), not nearest-rank."""
    if not values:
        return None
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0,1]")
    if any(not math.isfinite(x) for x in values):
        raise ValueError("nonfinite statistic")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values: list[float]) -> dict:
    """Both p50/p95 are null if no samples; no artificial zero timings."""
    return {"count": len(values), "p50": percentile(values, 0.5),
            "p95": percentile(values, 0.95)}


def _strict_nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


class SSEDecoder:
    """Bounded byte-oriented event parser: UTF-8 can span arbitrary reads."""

    def __init__(self, *, max_bytes: int):
        self.max_bytes = max_bytes
        self.total_bytes = 0
        self._partial_line = bytearray()
        self._event_data: list[bytes] = []
        self._event_length = 0
        self._event_type = "message"
        self.done = False

    def feed(self, data: bytes) -> list[str]:
        if not isinstance(data, bytes):
            raise ProfileFailure("sse_nonbytes")
        self.total_bytes += len(data)
        if self.total_bytes > self.max_bytes:
            raise ProfileFailure("sse_max_bytes")
        events: list[str] = []
        for char in data:
            if self.done:
                # Allow only whitespace CR/LF after DONE, never another event.
                if char not in (10, 13):
                    raise ProfileFailure("data_after_done")
                continue
            if char != 10:
                self._partial_line.append(char)
                if len(self._partial_line) > MAX_LINE_BYTES:
                    raise ProfileFailure("sse_line_too_long")
                continue
            line = bytes(self._partial_line)
            self._partial_line.clear()
            if line.endswith(b"\r"):
                line = line[:-1]
            if not line:
                if self._event_data:
                    if self._event_type != "message":
                        raise ProfileFailure("unknown_sse_event_type")
                    raw = b"\n".join(self._event_data)
                    try:
                        text = raw.decode("utf-8", "strict")
                    except UnicodeDecodeError:
                        raise ProfileFailure("invalid_utf8_sse") from None
                    if text == "[DONE]":
                        self.done = True
                    events.append(text)
                self._event_data = []
                self._event_length = 0
                self._event_type = "message"
                continue
            if line.startswith(b":"):
                continue  # SSE comment/keepalive, not semantic output.
            key, sep, value = line.partition(b":")
            if not sep:
                # MIME permits valueless fields, but they cannot prove
                # an OpenAI chat event; disallow ambiguity.
                raise ProfileFailure("unsupported_sse_field")
            if value.startswith(b" "):
                value = value[1:]
            if key == b"data":
                self._event_length += len(value) + 1
                if self._event_length > MAX_EVENT_BYTES:
                    raise ProfileFailure("sse_event_too_large")
                self._event_data.append(value)
            elif key == b"event":
                try:
                    self._event_type = value.decode("ascii", "strict")
                except UnicodeDecodeError:
                    raise ProfileFailure("invalid_sse_event_type") from None
            elif key in (b"id", b"retry"):
                # OpenAI chunk identity is JSON "id", not optional SSE
                # transport 'id'. Do not use SSE id for request correlation.
                continue
            else:
                raise ProfileFailure("unsupported_sse_field")
        return events

    def finish(self) -> None:
        if not self.done:
            raise ProfileFailure("missing_done")
        if self._partial_line or self._event_data:
            raise ProfileFailure("unterminated_sse_event")


class ChatEvents:
    """Track OpenAI semantic deltas without asserting token/chunk parity."""

    def __init__(self):
        self.first_semantic_s: float | None = None
        self.last_semantic_s: float | None = None
        self.semantic_event_times: list[float] = []
        self.semantic_fragments = 0
        self.coalesced_adjacent_events = 0
        self.generated_text_bytes = 0
        self.generated_reasoning_bytes = 0
        self.text_sha256 = hashlib.sha256()
        self.reasoning_sha256 = hashlib.sha256()
        self.sse_completion_id: str | None = None
        self.finish_reason: str | None = None
        self.usage: dict | None = None
        self.role_only_events = 0

    def observe(self, data: str, now_s: float) -> None:
        if data == "[DONE]":
            return
        try:
            item = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            raise ProfileFailure("malformed_sse_json") from None
        if not isinstance(item, dict):
            raise ProfileFailure("invalid_sse_object")
        if item.get("error") is not None:
            raise ProfileFailure("server_sse_error")
        completion_id = item.get("id")
        if completion_id is not None:
            if (not isinstance(completion_id, str) or
                    not re.fullmatch(r"[A-Za-z0-9._-]{1,256}", completion_id)):
                raise ProfileFailure("invalid_completion_id")
            if self.sse_completion_id is None:
                self.sse_completion_id = completion_id
            elif self.sse_completion_id != completion_id:
                raise ProfileFailure("completion_id_changed")
        usage = item.get("usage")
        if usage is not None:
            if self.usage is not None or not isinstance(usage, dict):
                raise ProfileFailure("invalid_or_duplicate_usage")
            if (not _strict_nonnegative_int(usage.get("prompt_tokens")) or
                    not _strict_nonnegative_int(usage.get("completion_tokens"))):
                raise ProfileFailure("invalid_usage_counts")
            total = usage.get("total_tokens")
            if (total is not None and
                    (not _strict_nonnegative_int(total) or
                     total != usage["prompt_tokens"] + usage["completion_tokens"])):
                raise ProfileFailure("inconsistent_usage_total")
            self.usage = dict(usage)
        choices = item.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise ProfileFailure("invalid_choices")
        if not choices:
            if usage is None:
                raise ProfileFailure("empty_choices_without_usage")
            return
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("index", 0) != 0:
            raise ProfileFailure("invalid_choice_index")
        prior_finished = self.finish_reason is not None
        finish = choice.get("finish_reason")
        if finish is not None:
            if self.finish_reason is not None or finish not in ("stop", "length"):
                raise ProfileFailure("invalid_or_duplicate_finish")
            self.finish_reason = finish
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            raise ProfileFailure("invalid_delta")
        if any(k in delta for k in ("tool_calls", "function_call", "audio", "images")):
            raise ProfileFailure("non_text_generation_not_supported")
        parts = []
        for key in ("reasoning_content", "content"):
            chunk = delta.get(key)
            if chunk is None or chunk == "":
                continue
            if not isinstance(chunk, str):
                raise ProfileFailure("nontext_semantic_delta")
            parts.append((key, chunk))
        # A final chunk may legally carry content AND finish_reason.
        # Content after an earlier finish chunk is not allowed.
        if parts and (prior_finished or self.usage is not None):
            raise ProfileFailure("semantic_after_finish_or_usage")
        if not parts:
            if delta.get("role") is not None:
                self.role_only_events += 1
            return
        # Timing is assigned when bytes were made available in a socket
        # read. Two SSE events in one read share a timestamp, not model time.
        if self.first_semantic_s is None:
            self.first_semantic_s = now_s
        if self.last_semantic_s == now_s:
            self.coalesced_adjacent_events += 1
        self.last_semantic_s = now_s
        self.semantic_event_times.append(now_s)
        self.semantic_fragments += 1
        for key, chunk in parts:
            raw = chunk.encode("utf-8")
            if key == "reasoning_content":
                self.generated_reasoning_bytes += len(raw)
                self.reasoning_sha256.update(raw)
            else:
                self.generated_text_bytes += len(raw)
                self.text_sha256.update(raw)

    def finish(self, *, decoder: SSEDecoder, max_tokens: int) -> None:
        decoder.finish()
        if self.first_semantic_s is None:
            raise ProfileFailure("no_generated_semantic_delta")
        if self.finish_reason is None:
            raise ProfileFailure("missing_finish_reason")
        if self.usage is None:
            raise ProfileFailure("missing_final_usage")
        count = self.usage["completion_tokens"]
        if count < 1 or count > max_tokens:
            raise ProfileFailure("completion_usage_out_of_range")
        if self.usage["prompt_tokens"] < 1:
            raise ProfileFailure("prompt_usage_missing")


def _connection(factory: Callable, timeout_s: float):
    """The only connection target; no user URL, DNS, proxy or redirect path."""
    return factory(HOST, PORT, timeout=timeout_s)


def _reader(response):
    # HTTPResponse.read1 returns currently available bytes without waiting
    # for an entire buffer of HTTP chunked SSE to accumulate (unlike read).
    return response.read1 if callable(getattr(response, "read1", None)) else response.read


def _read_with_deadline(
    conn, response, size: int, *, start_s: float, timeout_s: float,
    clock: Callable[[], float], cancelled: threading.Event,
) -> bytes:
    if cancelled.is_set():
        raise ProfileFailure("cancelled")
    remaining = timeout_s - (clock() - start_s)
    if remaining <= 0:
        raise ProfileFailure("timeout")
    sock = getattr(conn, "sock", None)
    if sock is not None and callable(getattr(sock, "settimeout", None)):
        sock.settimeout(min(remaining, timeout_s))
    data = _reader(response)(size)
    if cancelled.is_set():
        raise ProfileFailure("cancelled")
    if clock() - start_s > timeout_s:
        raise ProfileFailure("timeout")
    if not isinstance(data, bytes):
        raise ProfileFailure("response_nonbytes")
    return data


def _result_template(request_id: str, clients: int, round_index: int, case: int) -> dict:
    return {
        "request_id": request_id,
        "clients": clients,
        "round": round_index,
        "case": case,
        "status": "ERROR",
        "error": None,
        "http_status": None,
        "sse_completion_id": None,
        "sse_done": False,
        "finish_reason": None,
        "usage_final": False,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "semantic_fragments": 0,
        "coalesced_adjacent_semantic_events": 0,
        "role_only_events": 0,
        "ttfc_ms": None,
        "ttft_ms": None,
        "tpot_ms": None,
        "semantic_fragment_gaps_ms": [],
        "semantic_fragment_gap_ms": distribution([]),
        "wall_completion_ms": None,
        "completion_tokens_per_wall_second": None,
        "generated_text_bytes": 0,
        "generated_reasoning_bytes": 0,
        "semantic_output_sha256": None,
        "sse_bytes": 0,
        "start_monotonic_s": None,
        "end_monotonic_s": None,
        "timing_provenance": (
            "TTFC from first nonempty SSE content/reasoning delta; "
            "TTFT and true TPOT unavailable without token boundary proof"
        ),
    }


def _request_payload(config: Config) -> bytes:
    # Same deterministic model prompt/decoding settings for every request.
    # No hidden/real-time request ID injected into prompt text.
    return json.dumps({
        "model": config.model,
        "messages": [{"role": "user", "content": config.prompt}],
        "temperature": 0,
        "max_tokens": config.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def profile_request(
    config: Config, *, clients: int, round_index: int, case: int,
    clock: Callable[[], float] = time.monotonic,
    connection_factory: Callable = http.client.HTTPConnection,
    cancelled: threading.Event | None = None,
    ready: threading.Barrier | None = None,
) -> dict:
    """One HTTP chat request; all network I/O is direct localhost and opt-in.

    For tests inject a fake connection factory and monotonic clock. A
    thread.Barrier can synchronize N workers. Failure results contain only
    bounded codes, never raw response/prompt, credentials or SSE text.
    """
    config.validate()
    if clients not in config.concurrency or not 1 <= case <= clients:
        raise ValueError("case/client outside configured bounded profile")
    if not 1 <= round_index <= config.rounds:
        raise ValueError("round outside configured bounded profile")
    identifier = (
        f"h27-{config.run_label}-n{clients}-r{round_index:02d}-q{case:02d}"
    )
    outcome = _result_template(identifier, clients, round_index, case)
    token = cancelled if cancelled is not None else threading.Event()
    conn = None
    events = ChatEvents()
    decoder = SSEDecoder(max_bytes=config.max_sse_bytes)
    start_s: float | None = None
    try:
        if ready is not None:
            try:
                ready.wait(timeout=config.timeout_s)
            except threading.BrokenBarrierError:
                raise ProfileFailure("launch_barrier_broken") from None
        if token.is_set():
            raise ProfileFailure("cancelled")
        start_s = clock()
        outcome["start_monotonic_s"] = start_s
        conn = _connection(connection_factory, config.timeout_s)
        conn.request(
            "POST", CHAT_PATH, body=_request_payload(config), headers={
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "text/event-stream",
                "Connection": "close",
                "X-Profile-Request-ID": identifier,
            },
        )
        response = conn.getresponse()
        status = response.status
        outcome["http_status"] = status
        if type(status) is not int or status != 200:
            # No redirect handling whatsoever: 3xx is a hard error.
            raise ProfileFailure("http_status_not_200")
        content_type = response.getheader("Content-Type", "")
        if not isinstance(content_type, str):
            raise ProfileFailure("invalid_content_type")
        pieces = [piece.strip().lower() for piece in content_type.split(";")]
        if not pieces or pieces[0] != "text/event-stream":
            raise ProfileFailure("not_sse_content_type")
        if any(piece.startswith("charset=") and
               piece not in ("charset=utf-8", 'charset="utf-8"')
               for piece in pieces[1:]):
            raise ProfileFailure("unsupported_sse_charset")

        while True:
            block = _read_with_deadline(
                conn, response, config.read_bytes, start_s=start_s,
                timeout_s=config.timeout_s, clock=clock, cancelled=token,
            )
            observed_s = clock()
            if not block:
                break
            for data in decoder.feed(block):
                events.observe(data, observed_s)
        # An HTTP 200 that disconnects early is an ERROR, not a success
        # with a guessed token count or a zero duration.
        events.finish(decoder=decoder, max_tokens=config.max_tokens)
        ended_s = clock()
        if ended_s - start_s > config.timeout_s:
            raise ProfileFailure("timeout")
        elapsed_s = ended_s - start_s
        if elapsed_s <= 0:
            raise ProfileFailure("nonpositive_elapsed_time")
        event_gaps_ms = [
            (later - earlier) * 1000
            for earlier, later in zip(
                events.semantic_event_times, events.semantic_event_times[1:],
            )
        ]
        if (not math.isfinite(elapsed_s) or
                not math.isfinite(events.first_semantic_s - start_s) or
                events.first_semantic_s < start_s or
                any(not math.isfinite(gap) or gap < 0 for gap in event_gaps_ms)):
            raise ProfileFailure("unstable_monotonic_clock")
        count = events.usage["completion_tokens"]
        outcome.update({
            "status": "OK",
            "sse_completion_id": events.sse_completion_id,
            "sse_done": decoder.done,
            "finish_reason": events.finish_reason,
            "usage_final": True,
            "prompt_tokens": events.usage["prompt_tokens"],
            "completion_tokens": count,
            "total_tokens": events.usage.get(
                "total_tokens",
                count + events.usage["prompt_tokens"],
            ),
            "semantic_fragments": events.semantic_fragments,
            "coalesced_adjacent_semantic_events": events.coalesced_adjacent_events,
            "role_only_events": events.role_only_events,
            "ttfc_ms": (events.first_semantic_s - start_s) * 1000,
            # SSE chunks may be coalesced/split and contain arbitrary
            # numbers of tokens, so never pretend these are TTFT/TPOT.
            "ttft_ms": None,
            "tpot_ms": None,
            "semantic_fragment_gaps_ms": event_gaps_ms,
            "semantic_fragment_gap_ms": distribution(event_gaps_ms),
            "wall_completion_ms": elapsed_s * 1000,
            "completion_tokens_per_wall_second": count / elapsed_s,
            "generated_text_bytes": events.generated_text_bytes,
            "generated_reasoning_bytes": events.generated_reasoning_bytes,
            # SHA over concatenated text and reasoning channel digests:
            # invariant to SSE fragment boundaries, unlike per-event hashes.
            "semantic_output_sha256": hashlib.sha256(
                b"text:" + events.text_sha256.digest()
                + b":reasoning:" + events.reasoning_sha256.digest()
            ).hexdigest(),
        })
    except ProfileFailure as exc:
        outcome["error"] = exc.code
    except (socket.timeout, TimeoutError):
        outcome["error"] = "timeout"
    except (OSError, http.client.HTTPException, UnicodeError, ValueError):
        outcome["error"] = "transport_or_protocol_error"
    except Exception:
        # Do not leak uncaught raw server messages or sensitive prompt
        # content into logs. A bug is a failure, never silent success.
        outcome["error"] = "unexpected_client_failure"
    finally:
        if start_s is not None:
            outcome["end_monotonic_s"] = clock()
            outcome["sse_bytes"] = decoder.total_bytes
            if outcome["status"] != "OK":
                outcome["sse_done"] = decoder.done
                outcome["sse_completion_id"] = events.sse_completion_id
                outcome["finish_reason"] = events.finish_reason
                outcome["usage_final"] = events.usage is not None
                outcome["semantic_fragments"] = events.semantic_fragments
                outcome["role_only_events"] = events.role_only_events
                if events.first_semantic_s is not None:
                    outcome["ttfc_ms"] = (
                        events.first_semantic_s - start_s
                    ) * 1000
        if conn is not None:
            try:
                conn.close()
            except Exception:
                # A failed close cannot legitimize incomplete SSE data;
                # if completion succeeded, still report cleanup failure.
                if outcome["status"] == "OK":
                    outcome["status"] = "ERROR"
                    outcome["error"] = "connection_close_failed"
                    outcome["completion_tokens_per_wall_second"] = None
    return outcome


def sample_vram_metrics(
    config: Config, *, clock: Callable[[], float] = time.monotonic,
    connection_factory: Callable = http.client.HTTPConnection,
    cancelled: threading.Event | None = None,
) -> dict:
    """Optional read-only GET /metrics on the same exact local endpoint.

    Return raw VRAM-related metric series with original unit in the name;
    these are NOT independent hardware-owner/VRAM attestations. Missing
    endpoint, redirection, auth prompts and parser faults are diagnostics.
    """
    result = {"status": "ERROR", "error": None, "series": [],
              "metrics_bytes": 0}
    token = cancelled if cancelled is not None else threading.Event()
    conn = None
    try:
        start_s = clock()
        conn = _connection(connection_factory, config.timeout_s)
        conn.request("GET", METRICS_PATH, headers={
            "Accept": "text/plain", "Connection": "close",
        })
        response = conn.getresponse()
        if type(response.status) is not int or response.status != 200:
            raise ProfileFailure("metrics_status_not_200")
        incoming = bytearray()
        while True:
            block = _read_with_deadline(
                conn, response, 8192, start_s=start_s,
                timeout_s=config.timeout_s, clock=clock, cancelled=token,
            )
            if not block:
                break
            if len(incoming) + len(block) > MAX_METRICS_BYTES:
                raise ProfileFailure("metrics_max_bytes")
            incoming.extend(block)
        result["metrics_bytes"] = len(incoming)
        try:
            source = incoming.decode("utf-8", "strict")
        except UnicodeDecodeError:
            raise ProfileFailure("invalid_utf8_metrics") from None
        series = []
        for line in source.splitlines():
            if not line or line.startswith("#"):
                continue
            # Prometheus text exposition: only numeric gauge-like values.
            # NEVER parse arbitrary label values into the response.
            match = re.fullmatch(
                r"([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]{0,1024}\})?"
                r"\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
                r"(?:\s+[0-9]+)?", line,
            )
            if not match:
                continue
            name = match.group(1)
            if not re.search(r"vram|gpu.*(?:memory|mem).*(?:used|alloc|reserv)", name, re.I):
                continue
            try:
                number = float(match.group(2))
            except ValueError:
                continue
            if not math.isfinite(number) or number < 0:
                continue
            if len(series) >= 64:
                raise ProfileFailure("metrics_series_limit")
            series.append({"name": name, "value": number,
                           "unit": ("bytes_from_metric_name" if name.endswith("_bytes")
                                    else "unknown_raw_metric_unit")})
        result.update(status="OK", series=series)
    except ProfileFailure as exc:
        result["error"] = exc.code
    except (socket.timeout, TimeoutError):
        result["error"] = "metrics_timeout"
    except (OSError, http.client.HTTPException, UnicodeError, ValueError):
        result["error"] = "metrics_transport_error"
    except Exception:
        result["error"] = "metrics_unexpected_client_failure"
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                if result["status"] == "OK":
                    result["status"] = "ERROR"
                    result["error"] = "metrics_connection_close_failed"
                    result["series"] = []
    return result


def _request_stats(results: list[dict], *, group_wall_s: float) -> dict:
    completed = [item for item in results if item["status"] == "OK"]
    all_complete = len(completed) == len(results) and bool(results)
    tokens = sum(item["completion_tokens"] for item in completed)
    # A partial-failure group must never report a deceptively high
    # success-only aggregate as measured N-user throughput.
    token_rate = tokens / group_wall_s if all_complete and group_wall_s > 0 else None
    semantic_gaps = [
        gap for item in completed
        for gap in item["semantic_fragment_gaps_ms"]
    ]
    # On wire event coalescence: the spacing can be 0, still NOT TPOT.
    return {
        "attempts": len(results),
        "ok": len(completed),
        "errors": len(results) - len(completed),
        "error_counts": {
            reason: sum(item["error"] == reason for item in results)
            for reason in sorted({item["error"] for item in results if item["error"] is not None})
        },
        "completion_tokens_success_only": tokens,
        "group_wall_s": group_wall_s,
        "aggregate_completion_tokens_per_group_wall_second": token_rate,
        "ttfc_ms": distribution([
            item["ttfc_ms"] for item in completed if item["ttfc_ms"] is not None
        ]),
        "request_wall_ms": distribution([
            item["wall_completion_ms"] for item in completed
            if item["wall_completion_ms"] is not None
        ]),
        "per_request_completion_tokens_per_wall_second": distribution([
            item["completion_tokens_per_wall_second"] for item in completed
            if item["completion_tokens_per_wall_second"] is not None
        ]),
        "semantic_fragment_gap_ms": distribution(semantic_gaps),
        "ttft_ms": None,
        "tpot_ms": None,
        "all_requests_complete": all_complete,
    }


def _summarize_rounds(rounds: list[dict]) -> dict:
    requests = [request for run in rounds for request in run["requests"]]
    complete = [request for request in requests if request["status"] == "OK"]
    all_complete = bool(rounds) and all(run["stats"]["all_requests_complete"] for run in rounds)
    total_wall_s = sum(run["stats"]["group_wall_s"] for run in rounds)
    # Each round's launch-to-complete span is added; no summing the
    # individual overlapping request latencies at N>1.
    total_tokens = sum(item["completion_tokens"] for item in complete)
    return {
        "attempts": len(requests),
        "ok": len(complete),
        "errors": len(requests) - len(complete),
        "total_completion_tokens_success_only": total_tokens,
        "sum_round_wall_s": total_wall_s,
        "aggregate_completion_tokens_per_round_wall_second": (
            total_tokens / total_wall_s if all_complete and total_wall_s > 0 else None
        ),
        "ttfc_ms": distribution([
            item["ttfc_ms"] for item in complete if item["ttfc_ms"] is not None
        ]),
        "request_wall_ms": distribution([
            item["wall_completion_ms"] for item in complete
            if item["wall_completion_ms"] is not None
        ]),
        "per_request_completion_tokens_per_wall_second": distribution([
            item["completion_tokens_per_wall_second"] for item in complete
            if item["completion_tokens_per_wall_second"] is not None
        ]),
        "semantic_fragment_gap_ms": distribution([
            gap for item in complete
            for gap in item["semantic_fragment_gaps_ms"]
        ]),
        "all_requests_complete": all_complete,
        "ttft_ms": None,
        "tpot_ms": None,
    }


def run_profile(
    config: Config, *,
    clock: Callable[[], float] = time.monotonic,
    connection_factory: Callable = http.client.HTTPConnection,
    cancelled: threading.Event | None = None,
) -> dict:
    """Run N1/N2/N4/N8 rounds with bounded ThreadPoolExecutor concurrency.

    Does network I/O ONLY when called explicitly. CLI calls it only after
    --run; tests pass fake connection factories (no sockets created).
    """
    config.validate()
    token = cancelled if cancelled is not None else threading.Event()
    output = {
        "schema": SCHEMA,
        "status": "INCOMPLETE",
        "target": f"http://{HOST}:{PORT}",
        "request_path": CHAT_PATH,
        "model": config.model,
        "run_label": config.run_label,
        "rounds_per_concurrency": config.rounds,
        "concurrency": list(config.concurrency),
        "max_tokens": config.max_tokens,
        "timeout_s": config.timeout_s,
        "max_sse_bytes": config.max_sse_bytes,
        "token_accounting": "SSE final usage.completion_tokens only",
        "ttfc_definition": (
            "request start to first nonempty SSE delta.content or "
            "delta.reasoning_content observed by client"
        ),
        "ttft_ms": None,
        "tpot_ms": None,
        "tpot_reason": (
            "OpenAI SSE semantic fragments are not provably one token each; "
            "true TTFT/TPOT require server-side model-token timestamps"
        ),
        "by_concurrency": [],
    }
    for n in config.concurrency:
        runs = []
        for rnd in range(1, config.rounds + 1):
            if token.is_set():
                # Keep all planned rounds represented; never silently
                # skip a round and report an optimistic aggregate.
                requests = [
                    _result_template(
                        f"h27-{config.run_label}-n{n}-r{rnd:02d}-q{index:02d}",
                        n, rnd, index,
                    ) for index in range(1, n + 1)
                ]
                for request in requests:
                    request["error"] = "cancelled"
                runs.append({
                    "round": rnd, "requests": requests,
                    "metrics_before": None, "metrics_after": None,
                    "stats": _request_stats(requests, group_wall_s=0.0),
                })
                continue
            before = (sample_vram_metrics(
                config, clock=clock, connection_factory=connection_factory,
                cancelled=token,
            ) if config.metrics else None)
            gate = threading.Barrier(n + 1)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n, thread_name_prefix=f"h27-sse-n{n}"
            ) as executor:
                tasks = [
                    executor.submit(
                        profile_request, config,
                        clients=n, round_index=rnd, case=index, clock=clock,
                        connection_factory=connection_factory,
                        cancelled=token, ready=gate,
                    )
                    for index in range(1, n + 1)
                ]
                started = clock()
                try:
                    gate.wait(timeout=config.timeout_s)
                except threading.BrokenBarrierError:
                    token.set()
                requests = []
                for index, task in enumerate(tasks, 1):
                    try:
                        request = task.result()
                    except Exception:
                        request = _result_template(
                            f"h27-{config.run_label}-n{n}-r{rnd:02d}-q{index:02d}",
                            n, rnd, index,
                        )
                        request["error"] = "worker_internal_failure"
                    requests.append(request)
                ended = clock()
            after = (sample_vram_metrics(
                config, clock=clock, connection_factory=connection_factory,
                cancelled=token,
            ) if config.metrics else None)
            runs.append({
                "round": rnd,
                "requests": requests,
                "metrics_before": before,
                "metrics_after": after,
                "stats": _request_stats(
                    requests, group_wall_s=max(0.0, ended - started)
                ),
            })
        output["by_concurrency"].append({
            "clients": n,
            "rounds": runs,
            "summary": _summarize_rounds(runs),
        })
    output["status"] = (
        "COMPLETE" if all(
            item["summary"]["all_requests_complete"]
            for item in output["by_concurrency"]
        ) else "INCOMPLETE"
    )
    output["overall"] = _summarize_rounds([
        rnd for group in output["by_concurrency"]
        for rnd in group["rounds"]
    ])
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="Explicitly permit requests to ONLY 127.0.0.1:18127")
    parser.add_argument("--clients", nargs="+", type=int, default=list(ALLOWED_CONCURRENCY))
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-sse-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--read-bytes", type=int, default=4096)
    parser.add_argument("--model", default="hermes-27")
    parser.add_argument("--run-label", default="baseline")
    parser.add_argument("--metrics-vram", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = Config(
            model=args.model, concurrency=tuple(args.clients),
            rounds=args.rounds, max_tokens=args.max_tokens,
            timeout_s=args.timeout, max_sse_bytes=args.max_sse_bytes,
            read_bytes=args.read_bytes, metrics=args.metrics_vram,
            run_label=args.run_label,
        )
        config.validate()
    except ValueError as exc:
        parser.error(str(exc))
    if not args.run:
        print(json.dumps({
            "status": "DRY_RUN_NO_NETWORK",
            "target": f"http://{HOST}:{PORT}",
            "clients": list(config.concurrency),
            "rounds": config.rounds,
            "max_tokens": config.max_tokens,
            "planned_requests": sum(config.concurrency) * config.rounds,
            "metrics_gets": (
                len(config.concurrency) * config.rounds * 2 if config.metrics else 0
            ),
            "run_required": "--run",
            "note": "TTFC not exact token TTFT; no true token TPOT from SSE",
        }, sort_keys=True))
        return 0
    try:
        output = run_profile(config)
    except KeyboardInterrupt:
        print(json.dumps({"schema": SCHEMA, "status": "CANCELLED",
                          "target": f"http://{HOST}:{PORT}"}))
        return 130
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0 if output["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
