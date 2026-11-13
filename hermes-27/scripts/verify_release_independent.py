#!/usr/bin/env python3
"""Independent llama.cpp greedy token gate on the exact same GGUF.

Uses tests/compare_llama.py (real llama-server /completion, no stored golden).
The separate full-vocabulary numerical gate is verify_release_logits.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from reliability_core import GateBlocked, GateFailure, RunReport, require

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def actual_comparison(a):
    for path in (a.gguf, a.engine):
        if not path.is_file():
            raise GateBlocked(f"required real file missing: {path}")
    # compare_llama.py verifies llama-server /props absolute model_path and
    # os.path.samefile with THIS GGUF, before requesting any predictions.
    cmd = [sys.executable, str(ROOT / "tests/compare_llama.py"),
           "--gguf", str(a.gguf), "--engine", str(a.engine),
           "--llama-url", a.llama_url, "--tokens", str(a.tokens),
           "--timeout", str(a.timeout)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout * 12)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GateBlocked(f"independent inference unavailable: {exc}") from exc
    if proc.returncode == 2:
        raise GateBlocked("llama/HERMES setup unavailable: " + proc.stderr[-2500:])
    require(proc.returncode == 0, "greedy comparison diverged: " + proc.stdout[-2500:] +
            proc.stderr[-2500:])
    return {"model_sha256": sha256(a.gguf), "llama_url": a.llama_url,
            "prompt_cases": 4, "max_greedy_tokens": a.tokens, "matched": True,
            "trace": proc.stdout[-4000:]}


def offline_manifest_comparison(a):
    """CPU-only validate independent C-API capture already obtained on GPUs.

    Does not make another model call. Requires BOTH manifests from
    verify_release_logits.py and recomputes the model SHA256; both exporters'
    binaries and every raw FP32 file must remain unchanged since capture.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from verify_release_logits import check_manifest
    if not a.gguf.is_file():
        raise GateBlocked("same-checkpoint GGUF missing")
    if not a.llama_manifest.is_file() or not a.hermes_manifest.is_file():
        raise GateBlocked("independent llama/HERMES manifests missing")
    try:
        llama = json.loads(a.llama_manifest.read_text(encoding="utf-8"))
        hermes = json.loads(a.hermes_manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GateBlocked(f"cannot load independent trace manifests: {exc}") from exc
    check_manifest(llama, a.max_age_seconds)
    check_manifest(hermes, a.max_age_seconds)
    require(llama.get("backend") == "llama.cpp" and hermes.get("backend") == "hermes-27",
            "need separate independent llama C-API and HERMES exporters")
    require(llama.get("run_id") == hermes.get("run_id") and bool(llama.get("run_id")),
            "independent captures belong to different release sessions")
    digest = sha256(a.gguf)
    require(llama.get("gguf_sha256") == hermes.get("gguf_sha256") == digest,
            "independent/backend GGUF SHA256 mismatch with current checkpoint")
    llama_records, hermes_records = llama.get("entries"), hermes.get("entries")
    require(isinstance(llama_records, list) and len(llama_records) >= 4 and
            isinstance(hermes_records, list), "require >=4 independent prompts")
    require(llama_records == hermes_records,
            "same GGUF but greedy token histories/finish reasons differ")
    steps = sum(len(row["generated_token_ids"]) for row in llama_records)
    require(steps >= 12, "require >=12 total independent greedy token steps")
    return {"model_sha256": digest, "prompt_cases": len(llama_records),
            "max_greedy_tokens": max(len(row["generated_token_ids"]) for row in llama_records),
            "matched_greedy_steps": steps, "matched": True,
            "method": "independent llama.cpp C-API full-FP32 traces, offline manifest audit",
            "note": "greedy match is a separate gate; full-FP32 numerical gate may still fail"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--engine", type=Path,
                   help="HERMES CLI for live llama-server comparison")
    p.add_argument("--llama-url", default="http://127.0.0.1:8080")
    p.add_argument("--tokens", default=4, type=int)
    p.add_argument("--timeout", default=300, type=float)
    p.add_argument("--gpu-exclusive", action="store_true")
    p.add_argument("--llama-manifest", type=Path,
                   help="already-generated independent C-API llama manifest, CPU-only audit")
    p.add_argument("--hermes-manifest", type=Path,
                   help="already-generated HERMES trace manifest, CPU-only audit")
    p.add_argument("--max-age-seconds", default=3600, type=int)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if not 1 <= a.tokens <= 32 or a.timeout <= 0:
        p.error("--tokens must be 1..32 and --timeout >0")
    if bool(a.llama_manifest) != bool(a.hermes_manifest):
        p.error("--llama-manifest and --hermes-manifest are required together")
    if a.max_age_seconds <= 0:
        p.error("--max-age-seconds must be positive")
    report = RunReport("hermes27-independent-llama-token-v1", {
        "gguf": str(a.gguf), "llama_url": a.llama_url,
        "engine": str(a.engine) if a.engine else None,
        "gpu_exclusive_approved": a.gpu_exclusive,
        "offline_manifest_only": bool(a.llama_manifest)})
    if a.llama_manifest and a.hermes_manifest:
        report.record("same-checkpoint-llama-greedy",
                      lambda: offline_manifest_comparison(a))
    elif not a.gpu_exclusive:
        report.skip("same-checkpoint-llama-greedy", "no GPU permission; neither server started")
    elif a.engine is None:
        report.skip("same-checkpoint-llama-greedy",
                    "live llama-server mode requires --engine")
    else:
        report.record("same-checkpoint-llama-greedy", lambda: actual_comparison(a))
    code, label = report.save(a.out)
    print(f"RELEASE INDEPENDENT {label}: {a.out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
