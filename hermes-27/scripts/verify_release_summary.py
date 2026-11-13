#!/usr/bin/env python3
"""Fail-closed HERMES-27 RELEASE_GATE aggregator.

No server or GPU is touched: validates structured evidence from independent
scripts. Exit 0 means the scoped canary gate has enough fresh evidence, not
that the engine is production-proven. Missing evidence always returns 2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


EXPECTED = {
    "static": {
        "label": "hermes27-static-v1",
        "checks": ("source-invariants", "real-gguf-semantic-preflight",
                   "CLI-invalid-arguments"),
    },
    "deploy": {
        "label": "hermes27-deploy-cpu-v1",
        "checks": ("scratch-build-current-GGUF-parser", "source-gguf-chat-template",
                   "source-tokenizer-Jinja-Unicode", "deployed-gguf-dump-parity"),
    },
    "online": {
        "label": "hermes27-online-v1",
        "checks": ("health", "error-paths", "security", "unicode-and-sse",
                   "stream-cancellation", "concurrency-N1", "concurrency-N2",
                   "concurrency-N4", "concurrency-N8", "steady-vram",
                   "contexts-129-1024-8192"),
    },
    "queue": {
        "label": "hermes27-online-v1",
        "checks": ("queue-backpressure",),
    },
    "llama_tokens": {
        "label": "hermes27-independent-llama-token-v1",
        "checks": ("same-checkpoint-llama-greedy",),
    },
    "eos": {
        "label": "hermes27-eos-v1",
        "checks": ("EOS-and-early-stop",),
    },
    "prefill": {
        "label": "hermes27-prefill-decode-v1",
        "checks": ("prefill-vs-decode-four-prompts",),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def evidence(path: Path, expected: dict[str, Any], max_age_sec: int
             ) -> tuple[str, str, dict[str, Any] | None]:
    if not path.is_file():
        return "BLOCKED", f"missing evidence: {path}", None
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(item, dict) or item.get("schema") != "h27.reliability.v1":
            return "FAIL", "unknown/invalid reliability evidence schema", None
        if item.get("label") != expected["label"]:
            return "FAIL", f"wrong test family: {item.get('label')!r}", None
        stamp = item.get("created_utc")
        if not isinstance(stamp, str) or not stamp.endswith("Z"):
            return "FAIL", "missing UTC creation timestamp", None
        # time.mktime uses local time; the report timestamp is always UTC.
        import calendar
        age = time.time() - calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))
        if not math.isfinite(age) or age < -60 or age > max_age_sec:
            return "BLOCKED", f"stale/future report age={age:.0f}s", None
        checks = item.get("checks")
        if not isinstance(checks, list) or not checks:
            return "FAIL", "no checks in report", None
        names = [check.get("name") for check in checks if isinstance(check, dict)]
        if len(set(names)) != len(names):
            return "FAIL", "duplicate check names", None
        by_name = {check["name"]: check for check in checks if isinstance(check, dict)}
        for name in expected["checks"]:
            if name not in by_name:
                return "BLOCKED", f"mandatory check absent: {name}", None
            check = by_name[name]
            if check.get("status") != "PASS" or check.get("mandatory") is not True:
                status = check.get("status")
                return ("FAIL" if status == "FAIL" else "BLOCKED",
                        f"{name} {status}: {check.get('detail')}", None)
        if item.get("status") != "PASS_SCOPED_GATES_ONLY" or item.get("exit_code") != 0:
            return "BLOCKED", f"original report did not PASS: {item.get('status')}", None
        return "PASS", f"{len(expected['checks'])} required checks verified", item
    except (OSError, ValueError, TypeError, OverflowError, KeyError) as exc:
        return "FAIL", f"malformed report: {exc}", None


def full_fp32(path: Path, gguf_hash: str, max_age_sec: int
              ) -> tuple[str, str]:
    if not path.is_file():
        return "BLOCKED", f"missing independent llama full-logit comparison: {path}"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("schema") != "h27.independent-fp32-gate.v1":
            return "FAIL", "independent numerical evidence has unknown schema"
        if report.get("passed") is False:
            return ("BLOCKED" if report.get("status") == "BLOCKED" else "FAIL",
                    "independent full-logit gate " +
                    str(report.get("status", "FAIL")) + ": " + str(report.get("error", "")))
        if (report.get("passed") is not True or
                report.get("reference") != "llama.cpp" or
                report.get("candidate") != "hermes-27"):
            return "FAIL", "full-vocab evidence is not independently labeled/validated"
        age = time.time() - report["created_unix"]
        if age < -60 or age > max_age_sec:
            return "BLOCKED", "full-logit comparison stale"
        if report.get("sha256_gguf") != gguf_hash:
            return "FAIL", "full-logit comparison used a different GGUF"
        if report.get("matched_steps", 0) < 12:
            return "BLOCKED", "require >=12 independently matched steps (4 prompts x 3 steps)"
        if report.get("max_abs_gate", 1e9) > 0.2 or report.get("max_rmse_gate", 1e9) > .02:
            return "FAIL", "numerical thresholds relaxed above release policy"
        return "PASS", f"{report['matched_steps']} full-vocabulary steps compared"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return "FAIL", f"invalid independent full-logit report: {exc}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", required=True, type=Path)
    for label in EXPECTED:
        p.add_argument("--" + label.replace("_", "-"), required=True, type=Path)
    p.add_argument("--llama-fp32", required=True, type=Path)
    p.add_argument("--max-age-hours", type=int, default=24)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.max_age_hours <= 0 or args.max_age_hours > 168:
        p.error("--max-age-hours must be 1..168")
    criteria = []
    code = 0
    reports: dict[str, dict[str, Any]] = {}
    ttl = args.max_age_hours * 3600
    for label, expected in EXPECTED.items():
        source = getattr(args, label)
        state, message, result = evidence(source, expected, ttl)
        criteria.append({"gate": label, "status": state, "detail": message,
                         "evidence_path": str(source)})
        if result is not None:
            reports[label] = result
        if state == "FAIL":
            code = 1
        elif state != "PASS" and code != 1:
            code = 2
    gguf_digest = None
    if not args.gguf.is_file():
        state, message = "BLOCKED", "exact checkpoint missing"
    else:
        gguf_digest = sha256(args.gguf)
        state, message = full_fp32(args.llama_fp32, gguf_digest, ttl)
    criteria.append({"gate": "llama-fp32", "status": state, "detail": message,
                     "evidence_path": str(args.llama_fp32)})
    if state == "FAIL":
        code = 1
    elif state != "PASS" and code != 1:
        code = 2
    # The independent token gate records its own SHA256. Demand same GGUF.
    if gguf_digest is not None and "llama_tokens" in reports:
        matching = any(
            check.get("name") == "same-checkpoint-llama-greedy" and
            isinstance(check.get("detail"), dict) and
            check["detail"].get("model_sha256") == gguf_digest
            for check in reports["llama_tokens"].get("checks", [])
        )
        criteria.append({"gate": "cross-check-checkpoint-SHA256",
                         "status": "PASS" if matching else "FAIL",
                         "detail": "real llama/HERMES token proof matches exact GGUF" if matching
                         else "independent token comparison missing/mismatched GGUF hash"})
        if not matching:
            code = 1
    else:
        criteria.append({"gate": "cross-check-checkpoint-SHA256", "status": "BLOCKED",
                         "detail": "no fresh GGUF or llama token evidence"})
        if code == 0:
            code = 2
    status = "GO_FOR_CONTROLLED_CANARY_ONLY" if code == 0 else (
        "NO_GO" if code == 1 else "NO_GO_EVIDENCE_INCOMPLETE")
    summary = {"schema": "h27.release-summary.v1",
               "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "status": status, "exit_code": code, "gguf_sha256": gguf_digest,
               "criteria": criteria,
               "scope": "controlled canary only; 24h soak, crash recovery and production security separate"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    for item in criteria:
        print(f"{item['status']:7s} {item['gate']}: {item['detail']}")
    print(f"RELEASE DECISION {status}: {args.out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
