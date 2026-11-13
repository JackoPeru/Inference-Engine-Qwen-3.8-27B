#!/usr/bin/env python3
"""Strict, stdlib-only HTTP/SSE and release-report primitives; no GPU imports."""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class GateFailure(AssertionError):
    """A tested invariant is false."""


class GateBlocked(RuntimeError):
    """A mandatory gate cannot be tested or its prerequisites are absent."""


@dataclass
class Response:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> dict[str, Any]:
        try:
            obj = json.loads(self.body.decode("utf-8", errors="strict"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise GateFailure(f"HTTP {self.status}: invalid UTF-8 JSON response") from exc
        require(isinstance(obj, dict), "response JSON must be an object")
        return obj


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateFailure(message)


def request(base: str, path: str, *, payload: Any = None, raw: bytes | None = None,
            api_key: str | None = None, timeout: float = 180,
            method: str | None = None) -> Response:
    if payload is not None and raw is not None:
        raise ValueError("payload and raw are mutually exclusive")
    body = (json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else raw)
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(base.rstrip("/") + path, data=body, headers=headers,
                                 method=method or ("POST" if body is not None else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return Response(resp.status, dict(resp.headers), resp.read())
    except urllib.error.HTTPError as exc:
        return Response(exc.code, dict(exc.headers), exc.read())
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise GateBlocked(f"{path}: request unavailable: {exc}") from exc


def sse_request(base: str, path: str, payload: dict[str, Any], *,
                api_key: str | None = None, timeout: float = 180, on_first=None
                ) -> tuple[int, dict[str, str], list[dict[str, Any] | str]]:
    headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(base.rstrip("/") + path,
                                 data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                 headers=headers, method="POST")
    try:
        handle = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        # A rejection must use a real non-200 HTTP status, not a successful
        # SSE stream containing an 'error' object.
        reject = Response(exc.code, dict(exc.headers), exc.read())
        return reject.status, reject.headers, [reject.json()]
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise GateBlocked(f"SSE connection unavailable: {exc}") from exc
    try:
        with handle as resp:
            content_type = resp.headers.get("Content-Type", "")
            require(resp.status == 200, f"SSE unexpected status {resp.status}")
            require(content_type.lower().startswith("text/event-stream"),
                    f"SSE unexpected content-type: {content_type!r}")
            events: list[dict[str, Any] | str] = []
            lines: list[str] = []

            def flush() -> None:
                nonlocal lines
                fields = [line[5:].lstrip(" ") for line in lines if line.startswith("data:")]
                lines = []
                if not fields:
                    return
                raw_data = "\n".join(fields)
                if raw_data == "[DONE]":
                    events.append("[DONE]")
                else:
                    try:
                        obj = json.loads(raw_data)
                    except ValueError as exc:
                        raise GateFailure("SSE event is malformed JSON") from exc
                    require(isinstance(obj, dict), "SSE data must be a JSON object")
                    events.append(obj)
                    if on_first is not None and len(events) == 1:
                        on_first(obj)

            for line in resp:
                try:
                    item = line.decode("utf-8", errors="strict").rstrip("\r\n")
                except UnicodeDecodeError as exc:
                    raise GateFailure("SSE contains invalid UTF-8") from exc
                if item == "":
                    flush()
                    if events and events[-1] == "[DONE]":
                        break
                elif not item.startswith(":"):
                    lines.append(item)
            if lines:
                flush()
            return resp.status, dict(resp.headers), events
    except (OSError, TimeoutError) as exc:
        raise GateBlocked(f"SSE aborted: {exc}") from exc


def chat_payload(model: str, prompt: str, *, max_tokens: int = 3,
                 stream: bool = False, thinking: bool = False) -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0,
            "reasoning": {"enabled": thinking}, "stream": stream}


def validate_chat(resp: Response, model: str, max_tokens: int) -> tuple[str, str, int]:
    require(resp.status == 200, f"chat HTTP {resp.status}: {resp.body[:300]!r}")
    obj = resp.json()
    require(obj.get("object") == "chat.completion" and obj.get("model") == model,
            "chat completion object/model mismatch")
    require(isinstance(obj.get("id"), str) and bool(obj["id"]), "missing completion id")
    choices = obj.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, "choices must be length 1")
    choice = choices[0]
    require(isinstance(choice, dict), "choice is not an object")
    msg = choice.get("message")
    require(isinstance(msg, dict) and msg.get("role") == "assistant" and
            isinstance(msg.get("content"), str), "assistant message/content invalid")
    reason = choice.get("finish_reason")
    require(reason in ("stop", "length"), f"unexpected finish_reason {reason!r}")
    usage = obj.get("usage")
    require(isinstance(usage, dict), "missing token usage")
    n = usage.get("completion_tokens")
    p = usage.get("prompt_tokens")
    require(type(n) is int and 1 <= n <= max_tokens and type(p) is int and p >= 1,
            f"invalid token counts: {usage}")
    require(usage.get("total_tokens") == n + p, "invalid usage total")
    if reason == "length":
        require(n == max_tokens, f"length completion consumed only {n}/{max_tokens}")
    return msg["content"], reason, n


def validate_sse(events: list[dict[str, Any] | str], model: str
                 ) -> tuple[str, str, str]:
    require(len(events) >= 3 and events[-1] == "[DONE]" and
            events.count("[DONE]") == 1, "SSE must terminate with exactly one [DONE]")
    ids: set[str] = set()
    reasons: list[str] = []
    chunks: list[str] = []
    for event in events[:-1]:
        require(isinstance(event, dict) and event.get("object") == "chat.completion.chunk",
                "invalid SSE chunk or error embedded in 200 stream")
        require(event.get("model") == model and isinstance(event.get("id"), str),
                "SSE model/id mismatch")
        ids.add(event["id"])
        choices = event.get("choices")
        require(isinstance(choices, list) and len(choices) == 1,
                "SSE choice count must be 1")
        choice = choices[0]
        delta = choice.get("delta")
        require(isinstance(delta, dict), "SSE delta must be an object")
        if "content" in delta:
            require(isinstance(delta["content"], str), "SSE content must be str")
            chunks.append(delta["content"])
        if choice.get("finish_reason") is not None:
            reasons.append(choice["finish_reason"])
    require(len(ids) == 1 and len(reasons) == 1 and reasons[0] in ("stop", "length"),
            f"SSE must have exactly one id and terminal reason, got {ids}/{reasons}")
    return "".join(chunks), reasons[0], next(iter(ids))


def metric_values(payload: bytes) -> dict[str, float]:
    try:
        content = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise GateFailure("invalid metrics UTF-8") from exc
    values: dict[str, float] = {}
    for line in content.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*(?:\{[^}]+\})?)\s+(-?\d+(?:\.\d+)?)$', line)
        if match:
            values[match.group(1)] = float(match.group(2))
    return values


def vram_from_metrics(metrics: dict[str, float]) -> tuple[float, float]:
    labels = ('hermes_vram_used_mib{gpu="0"}', 'hermes_vram_used_mib{gpu="1"}')
    if any(key not in metrics for key in labels):
        raise GateBlocked("server metrics have no two per-GPU VRAM gauges")
    return metrics[labels[0]], metrics[labels[1]]


def nvidia_vram() -> tuple[int, int]:
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise GateBlocked(f"nvidia-smi unavailable for memory check: {exc}") from exc
    rows: dict[int, int] = {}
    for line in proc.stdout.splitlines():
        items = [part.strip() for part in line.split(",")]
        if len(items) == 2:
            try:
                rows[int(items[0])] = int(items[1])
            except ValueError:
                pass
    if 0 not in rows or 1 not in rows:
        raise GateBlocked(f"expected GPU0 and GPU1 VRAM readings, got {rows}")
    return rows[0], rows[1]


@dataclass
class RunReport:
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)

    def record(self, name: str, fn, *, mandatory: bool = True) -> Any:
        t0 = time.monotonic()
        try:
            detail = fn()
            self.checks.append({"name": name, "status": "PASS", "mandatory": mandatory,
                                "elapsed_sec": round(time.monotonic() - t0, 3),
                                "detail": detail})
            print(f"PASS {name}: {detail}", flush=True)
            return detail
        except GateBlocked as exc:
            self.checks.append({"name": name, "status": "BLOCKED", "mandatory": mandatory,
                                "elapsed_sec": round(time.monotonic() - t0, 3),
                                "detail": str(exc)})
            print(f"BLOCKED {name}: {exc}", flush=True)
        except (GateFailure, AssertionError) as exc:
            self.checks.append({"name": name, "status": "FAIL", "mandatory": mandatory,
                                "elapsed_sec": round(time.monotonic() - t0, 3),
                                "detail": str(exc)})
            print(f"FAIL {name}: {exc}", flush=True)
        except Exception as exc:
            self.checks.append({"name": name, "status": "BLOCKED", "mandatory": mandatory,
                                "elapsed_sec": round(time.monotonic() - t0, 3),
                                "detail": f"unhandled {type(exc).__name__}: {exc}"})
            print(f"BLOCKED {name}: unhandled {type(exc).__name__}: {exc}", flush=True)
        return None

    def skip(self, name: str, reason: str, *, mandatory: bool = True) -> None:
        self.checks.append({"name": name, "status": "NOT_RUN", "mandatory": mandatory,
                            "elapsed_sec": 0.0, "detail": reason})
        print(f"NOT_RUN {name}: {reason}", flush=True)

    def outcome(self) -> tuple[int, str]:
        must = [c for c in self.checks if c["mandatory"]]
        if any(c["status"] == "FAIL" for c in must):
            return 1, "NO_GO"
        if not must or any(c["status"] != "PASS" for c in must):
            return 2, "BLOCKED"
        return 0, "PASS_SCOPED_GATES_ONLY"

    def save(self, path: str | Path) -> tuple[int, str]:
        code, status = self.outcome()
        data = {"schema": "h27.reliability.v1", "label": self.label,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": status, "exit_code": code, "metadata": self.metadata,
                "checks": self.checks}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
        return code, status
