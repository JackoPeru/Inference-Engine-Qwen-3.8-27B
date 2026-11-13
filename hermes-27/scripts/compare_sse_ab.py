#!/usr/bin/env python3
"""Fail-closed offline comparison of paired HERMES-27 SSE benchmark logs.

Verifies identical workload, full output hashes, finish reasons and token
usage for parallel vs legacy before computing any performance ratio.
This comparison is *not* independent llama.cpp full-logit validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


CLIENTS = (1, 2, 4, 8)
ROUNDS = (0, 1)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            sha.update(part)
    return sha.hexdigest()


def parse_log(path: Path, expected: str) -> dict:
    items = {}
    content = path.read_text(encoding="utf-8")
    for line in content.splitlines():
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("schema") != "h27.sse-timing.v1":
            continue
        if row.get("variant") != expected:
            raise ValueError(f"{expected} log contains {row.get('variant')!r} variant")
        key = row.get("clients"), row.get("round")
        if key in items:
            raise ValueError(f"duplicate round {key} in {expected}")
        if key not in {(n, rnd) for n in CLIENTS for rnd in ROUNDS}:
            raise ValueError(f"unexpected workload {key!r}")
        if (row.get("max_tokens") != 16 or row.get("requests") != key[0] or
                row.get("completion_tokens") != 16 * key[0] or
                row.get("prompt_tokens") != 47 * key[0]):
            raise ValueError(f"incomplete/wrong workload for {expected} {key}")
        outputs = row.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != key[0]:
            raise ValueError(f"wrong output vector for {expected} {key}")
        for i, output in enumerate(outputs):
            sha = output.get("sha256")
            if (output.get("case") != i or output.get("completion_tokens") != 16 or
                    output.get("finish") != "length" or
                    not isinstance(sha, str) or len(sha) != 64 or
                    any(c not in "0123456789abcdef" for c in sha)):
                raise ValueError(f"bad output/usage {expected} {key} case {i}")
        for metric in ("aggregate_completion_tok_per_wall_sec",
                       "ttfc_p50_sec", "ttfc_p95_sec",
                       "request_latency_p50_sec", "request_latency_p95_sec"):
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{expected} invalid {metric} for {key}")
        items[key] = row
    expected_keys = {(n, rnd) for n in CLIENTS for rnd in ROUNDS}
    if set(items) != expected_keys:
        raise ValueError(f"{expected} missing workloads {sorted(expected_keys-set(items))}")
    if f"SCRATCH SSE TIMING: PASS variant={expected}" not in content:
        raise ValueError(f"{expected} source log did not finish with PASS")
    if ("POST-RUN GPU MEMORY:" not in content or
            "0, 486 MiB, 0 %" not in content or
            "1, 1289 MiB, 0 %" not in content):
        raise ValueError(f"{expected} source log missing exact idle GPU restoration")
    return items


def compare(parallel: dict, legacy: dict) -> dict:
    summary = []
    for n in CLIENTS:
        pairs = [(parallel[(n, r)], legacy[(n, r)]) for r in ROUNDS]
        for rnd, (p, l) in enumerate(pairs):
            for ip, il in zip(p["outputs"], l["outputs"]):
                if ip["sha256"] != il["sha256"]:
                    raise ValueError(f"full generated text mismatch N{n} round{rnd} case{ip['case']}")
                if ip["finish"] != il["finish"] or ip["completion_tokens"] != il["completion_tokens"]:
                    raise ValueError(f"finish/usage mismatch N{n} round{rnd} case{ip['case']}")
        mp = statistics.mean(p["aggregate_completion_tok_per_wall_sec"] for p, _ in pairs)
        ml = statistics.mean(l["aggregate_completion_tok_per_wall_sec"] for _, l in pairs)
        summary.append({
            "clients": n,
            "parallel_tok_per_wall_sec": round(mp, 6),
            "legacy_tok_per_wall_sec": round(ml, 6),
            "relative_parallel_percent": round(100 * (mp / ml - 1), 3),
            "parallel_ttfc_p95_sec": round(statistics.mean(
                p["ttfc_p95_sec"] for p, _ in pairs), 6),
            "legacy_ttfc_p95_sec": round(statistics.mean(
                l["ttfc_p95_sec"] for _, l in pairs), 6),
            "parallel_request_p95_sec": round(statistics.mean(
                p["request_latency_p95_sec"] for p, _ in pairs), 6),
            "legacy_request_p95_sec": round(statistics.mean(
                l["request_latency_p95_sec"] for _, l in pairs), 6),
        })
    return {"schema": "h27.sse-ab-parity.v1", "status": "PASS_SCOPED_SSE_PARITY",
            "quality_scope": "same-PP2 SSE text hashes/usage only; independent full logits still FAIL",
            "request_count_per_variant": sum(n*len(ROUNDS) for n in CLIENTS),
            "output_hashes_identical": True, "data": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel", required=True, type=Path)
    parser.add_argument("--legacy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        p = parse_log(args.parallel, "parallel")
        l = parse_log(args.legacy, "legacy")
        result = compare(p, l)
        result["source_sha256"] = {
            "parallel": digest(args.parallel), "legacy": digest(args.legacy)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n",
                               encoding="utf-8")
        for row in result["data"]:
            print(f"N{row['clients']} parallel={row['parallel_tok_per_wall_sec']:.3f} "
                  f"legacy={row['legacy_tok_per_wall_sec']:.3f} "
                  f"delta={row['relative_parallel_percent']:+.2f}% "
                  f"p95-TTFC={row['parallel_ttfc_p95_sec']:.3f}/"
                  f"{row['legacy_ttfc_p95_sec']:.3f}s")
        print("SSE_A_B_PARITY=PASS 30/30 deterministic content hashes, "
              "completion usage/finish and GPU restoration; NOT llama parity")
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"SSE_A_B_PARITY=FAIL_CLOSED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
