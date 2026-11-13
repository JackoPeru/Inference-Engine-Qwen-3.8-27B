#!/usr/bin/env python3
"""Real-GGUF batched-prefill vs incremental-decode full-FP32 gate.

Runs the production CLI's --verify-prefill path on four prompt classes.
No GPU work unless --gpu-exclusive is explicit; no CMake/source edits.
This is an internal HERMES parity gate, NOT an independent llama oracle.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from reliability_core import GateBlocked, RunReport, require


PROMPTS = (
    (6205, 161421, 303, 220, 17, 1374, 9911, 1147, 27053, 6059, 13),
    (8917, 5437, 537, 279, 10814, 314, 279, 8323, 21440, 303, 2250, 22157, 13),
    (248045, 846, 198, 49, 158648, 13138, 384, 1147, 183687, 272,
     21817, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271),
    (6205, 161421, 303, 220),
)
MATCH = re.compile(
    r"VERIFY prompt=(\d+) full_prefill max_abs=([0-9eE+.\-]+) "
    r"mean_abs=([0-9eE+.\-]+) argmax=(\d+)/(\d+) (PASS|FAIL)"
)


def perform(a):
    for path in (a.engine, a.gguf):
        if not path.is_file():
            raise GateBlocked(f"real model or CLI unavailable: {path}")
    args = [str(a.engine), "--model", str(a.gguf), "--verify-prefill",
            "--max-new-tokens", "1"]
    for prompt in PROMPTS:
        args += ["--prompt-ids", ",".join(map(str, prompt))]
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=a.timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GateBlocked(f"real HERMES prefill inference unavailable: {exc}") from exc
    require(proc.returncode == 0, f"PP2 batched-prefill/full-decode numerical gate failed: "
            f"exit={proc.returncode}, stderr={proc.stderr[-3000:]}")
    found = MATCH.findall(proc.stderr)
    require(len(found) == len(PROMPTS),
            f"expected {len(PROMPTS)} full-vocab comparison lines; got {len(found)}")
    comparisons = []
    for i, (index, maxerr, meanerr, pretop, dectop, verdict) in enumerate(found):
        require(int(index) == i, "reordered/duplicate CLI prefill result")
        mx = float(maxerr)
        me = float(meanerr)
        require(mx >= 0 and me >= 0 and mx <= .002 and
                int(pretop) == int(dectop) and verdict == "PASS",
                f"prefill case {i} actual max_abs={mx} mean={me} "
                f"argmax {pretop}/{dectop} result={verdict}")
        comparisons.append({"prompt_index": i, "length": len(PROMPTS[i]),
                            "max_abs": mx, "mean_abs": me,
                            "prefill_argmax": int(pretop),
                            "decode_argmax": int(dectop)})
    records = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    require(len(records) == len(PROMPTS), "CLI JSONL generated wrong number of requests")
    for i, entry in enumerate(records):
        require(entry.get("prompt_index") == i and entry.get("prompt_ids") == list(PROMPTS[i]) and
                len(entry.get("generated_token_ids", [])) == 1,
                f"prefill case {i} changed prompt IDs or output shape")
    return {"real_gguf": str(a.gguf), "verified_cases": comparisons,
            "threshold_max_abs": .002,
            "note": "HERMES batched-prefill vs incremental decode only, not llama.cpp"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--engine", required=True, type=Path)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--gpu-exclusive", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.timeout < 1:
        p.error("timeout must be positive")
    report = RunReport("hermes27-prefill-decode-v1", {
        "model_path": str(a.gguf), "gpu_exclusive_approved": a.gpu_exclusive})
    if a.gpu_exclusive:
        report.record("prefill-vs-decode-four-prompts", lambda: perform(a))
    else:
        report.skip("prefill-vs-decode-four-prompts",
                    "GPU exclusive permission not granted; no real inference")
    code, status = report.save(a.out)
    print(f"RELEASE PREFILL {status}: {a.out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
