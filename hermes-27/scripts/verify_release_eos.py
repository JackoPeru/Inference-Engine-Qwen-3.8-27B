#!/usr/bin/env python3
"""Real-model EOS gate: derive first greedy token, then force it as EOS.

Runs HERMES twice with the same GGUF and prompt; this intentionally costs
two model uploads, so it requires explicit GPU-exclusive permission.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from reliability_core import GateBlocked, RunReport, require


def run_engine(exe: Path, model: Path, ids: list[int], max_tokens: int,
               timeout: int, eos: int | None = None) -> dict:
    args = [str(exe), "--model", str(model),
            "--prompt-ids", ",".join(map(str, ids)),
            "--max-new-tokens", str(max_tokens)]
    if eos is not None:
        args += ["--eos-token", str(eos)]
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GateBlocked(f"real CLI inference unavailable: {exc}") from exc
    require(p.returncode == 0, f"CLI failed: {p.returncode}: {p.stderr[-1500:]}")
    lines = [json.loads(line) for line in p.stdout.splitlines() if line.strip()]
    require(len(lines) == 1, "expected exactly one JSONL output")
    return lines[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", required=True, type=Path)
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--ids", default="6205,161421,303,220,17,1374,9911,1147,27053,6059,13")
    p.add_argument("--timeout", type=int, default=360)
    p.add_argument("--gpu-exclusive", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    ids = [int(v) for v in a.ids.split(",")]
    report = RunReport("hermes27-eos-v1", {"prompt_ids": ids,
                       "gpu_exclusive_approved": a.gpu_exclusive})
    def eos():
        if not a.gguf.is_file() or not a.engine.is_file():
            raise GateBlocked("engine or GGUF missing")
        first = run_engine(a.engine, a.gguf, ids, 1, a.timeout)
        t = first.get("generated_token_ids")
        require(type(t) is list and len(t) == 1 and type(t[0]) is int,
                "first-token oracle invalid")
        forced = run_engine(a.engine, a.gguf, ids, 4, a.timeout, eos=t[0])
        require(forced.get("generated_token_ids") == t and
                forced.get("stop_reason") == "eos",
                f"forced EOS failed: initial={first}, forced={forced}")
        return {"first_greedy_token": t[0], "forced_eos_stop": True}
    if a.gpu_exclusive:
        report.record("EOS-and-early-stop", eos)
    else:
        report.skip("EOS-and-early-stop", "needs two real model loads; no GPU permission")
    code, status = report.save(a.out)
    print(f"RELEASE EOS {status}: {a.out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
