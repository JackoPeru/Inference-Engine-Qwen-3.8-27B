#!/usr/bin/env python3
"""Full-vocabulary FP32 gate with same-GGUF SHA256 and per-run trace provenance.

This tool NEVER invents a llama.cpp exporter. The reference must actually
export raw FP32 logits from llama.cpp at identical token histories. A token
trace alone or HERMES-vs-HERMES output can never satisfy the independent gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from reliability_core import GateBlocked, GateFailure, require

ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile(r"^prompt_(\d+)_step_(\d+)\.f32$")
VOCAB = 248320


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def records_from_jsonl(path: Path) -> list[dict]:
    entries: list[dict] = []
    for index, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except ValueError as exc:
            raise GateFailure(f"line {index + 1} of {path} not JSON") from exc
        require(type(obj) is dict and obj.get("prompt_index") == len(entries),
                f"noncontiguous prompt index at {index}")
        ids = obj.get("prompt_ids")
        out = obj.get("generated_token_ids")
        require(type(ids) is list and ids and all(type(v) is int and 0 <= v < VOCAB
                                                for v in ids), "invalid prompt IDs")
        require(type(out) is list and out and all(type(v) is int and 0 <= v < VOCAB
                                                 for v in out), "invalid generated IDs")
        require(obj.get("stop_reason") in ("eos", "length"),
                "unknown generated stop_reason")
        entries.append({"prompt_index": obj["prompt_index"], "prompt_ids": ids,
                        "generated_token_ids": out, "stop_reason": obj["stop_reason"]})
    require(bool(entries), "no valid JSONL inference entries")
    return entries


def make_manifest(a) -> dict:
    if a.backend not in ("hermes-27", "llama.cpp"):
        raise GateFailure("backend must be literally hermes-27 or llama.cpp")
    if not a.run_id or len(a.run_id) < 12:
        raise GateFailure("run-id must be a shared fresh identifier >=12 chars")
    if not a.exporter_version or not a.exporter_version.strip():
        raise GateFailure("exporter-version is required")
    if not a.gguf.is_file():
        raise GateBlocked("model checkpoint missing")
    if not a.exporter_binary.is_file():
        raise GateBlocked(f"backend exporter binary missing: {a.exporter_binary}")
    age_jsonl = time.time() - a.jsonl.stat().st_mtime
    require(-60 <= age_jsonl <= a.max_age_seconds,
            f"inference JSONL is stale/future dated ({age_jsonl:.0f}s)")
    entries = records_from_jsonl(a.jsonl)
    expected = {f"prompt_{i}_step_{j}.f32" for i, item in enumerate(entries)
                for j in range(len(item["generated_token_ids"]))}
    trace_dir = a.dump_dir.resolve(strict=True)
    files = {f.name: f for f in trace_dir.iterdir() if f.is_file() and NAME.fullmatch(f.name)}
    require(expected == set(files), f"FP32 dump mismatch: missing {sorted(expected - set(files))[:8]} "
            f"unexpected {sorted(set(files) - expected)[:8]}")
    now = time.time()
    hashes = {}
    # Reuse prime's strict FP32 loader for finite/shape checks; provenance
    # additionally binds traces to the model digest and greedy token sequence.
    sys.path.insert(0, str(ROOT / "tools"))
    from compare_fp32_logits import read_logits
    for name in sorted(files):
        f = files[name]
        age = now - f.stat().st_mtime
        require(-60 <= age <= a.max_age_seconds,
                f"{name} is stale/future dated (age {age:.0f}s); regenerate in this session")
        values = read_logits(f, VOCAB)
        i, j = map(int, NAME.fullmatch(name).groups())
        greedy = max(range(VOCAB), key=values.__getitem__)
        require(greedy == entries[i]["generated_token_ids"][j],
                f"{name}: stored greedy ID {entries[i]['generated_token_ids'][j]} != "
                f"dumped argmax {greedy}")
        hashes[name] = digest(f)
    return {"schema": "h27.fp32-traces.v1", "backend": a.backend,
            "exporter_version": a.exporter_version, "run_id": a.run_id,
            "created_unix": now, "gguf_path": str(a.gguf.resolve()),
            "gguf_size": a.gguf.stat().st_size, "gguf_sha256": digest(a.gguf),
            "exporter_binary": str(a.exporter_binary.resolve()),
            "exporter_binary_sha256": digest(a.exporter_binary),
            "dump_dir": str(trace_dir), "entries": entries, "files_sha256": hashes}


def check_manifest(manifest: dict, age_limit: int) -> None:
    require(manifest.get("schema") == "h27.fp32-traces.v1", "unexpected trace manifest schema")
    require(manifest.get("backend") in ("llama.cpp", "hermes-27"), "invalid backend label")
    age = time.time() - manifest.get("created_unix", 0)
    require(-60 <= age <= age_limit, f"manifest stale/future dated {age:.0f}s")
    require(len(manifest.get("gguf_sha256", "")) == 64, "GGUF SHA256 missing")
    exporter = Path(manifest.get("exporter_binary", "/nonexistent"))
    require(exporter.is_file() and
            digest(exporter) == manifest.get("exporter_binary_sha256"),
            "source exporter binary replaced or missing after capture")
    traces = manifest.get("files_sha256")
    require(type(traces) is dict and bool(traces), "empty file digest table")
    directory = Path(manifest["dump_dir"])
    for name, original in traces.items():
        require(NAME.fullmatch(name) is not None, f"unsafe trace file name {name!r}")
        f = directory / name
        require(f.is_file(), f"trace missing: {f}")
        require(digest(f) == original, f"trace modified after provenance capture: {f}")


def compare(a) -> dict:
    ha = json.loads(a.hermes_manifest.read_text(encoding="utf-8"))
    la = json.loads(a.llama_manifest.read_text(encoding="utf-8"))
    check_manifest(ha, a.max_age_seconds)
    check_manifest(la, a.max_age_seconds)
    require(ha["backend"] == "hermes-27" and la["backend"] == "llama.cpp",
            "need HERMES versus an independently instrumented llama.cpp, not same backend")
    require(ha["run_id"] == la["run_id"], "different release capture sessions")
    require(ha["gguf_sha256"] == la["gguf_sha256"] and
            ha["gguf_size"] == la["gguf_size"], "reference/candidate GGUF differs")
    # An old manifest must not pass after its underlying checkpoint changes.
    # Cache samefile digest to avoid reading a 19.7GB file more than once.
    verification_cache: dict[str, str] = {}
    for manifest in (ha, la):
        path = Path(manifest["gguf_path"])
        require(path.is_file(), f"original GGUF missing: {path}")
        key = str(path.resolve())
        if key not in verification_cache:
            verification_cache[key] = digest(path)
        require(verification_cache[key] == manifest["gguf_sha256"] and
                path.stat().st_size == manifest["gguf_size"],
                "GGUF changed after inference trace capture")
    require(ha["entries"] == la["entries"],
            "prompt IDs, greedy continuation or stop reasons differ; logits histories not aligned")
    require(set(ha["files_sha256"]) == set(la["files_sha256"]),
            "different FP32 step sets")
    argv = [sys.executable, str(ROOT / "tools/compare_fp32_logits.py"),
            ha["dump_dir"], la["dump_dir"], "--max-abs", str(a.max_abs),
            "--max-rmse", str(a.max_rmse)]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=240)
    require(proc.returncode == 0,
            f"independent full logits mismatch: {proc.stdout[-3000:]} {proc.stderr[-1000:]}")
    return {"matched_steps": len(ha["files_sha256"]), "sha256_gguf": ha["gguf_sha256"],
            "schema": "h27.independent-fp32-gate.v1", "passed": True,
            "created_unix": time.time(), "reference": "llama.cpp",
            "candidate": "hermes-27",
            "run_id": ha["run_id"], "max_abs_gate": a.max_abs,
            "max_rmse_gate": a.max_rmse, "output": proc.stdout[-4000:]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="action", required=True)
    manifest = commands.add_parser("manifest", help="capture fresh GGUF+trace provenance")
    manifest.add_argument("--backend", choices=("hermes-27", "llama.cpp"), required=True)
    manifest.add_argument("--exporter-version", required=True)
    manifest.add_argument("--exporter-binary", required=True, type=Path,
                          help="actual HERMES or independently built llama exporter executable")
    manifest.add_argument("--run-id", required=True)
    manifest.add_argument("--gguf", required=True, type=Path)
    manifest.add_argument("--jsonl", required=True, type=Path)
    manifest.add_argument("--dump-dir", required=True, type=Path)
    manifest.add_argument("--out", required=True, type=Path)
    manifest.add_argument("--max-age-seconds", type=int, default=3600)
    verification = commands.add_parser("compare", help="strict independent full-FP32 gate")
    verification.add_argument("--hermes-manifest", required=True, type=Path)
    verification.add_argument("--llama-manifest", required=True, type=Path)
    verification.add_argument("--max-abs", type=float, default=.2)
    verification.add_argument("--max-rmse", type=float, default=.02)
    verification.add_argument("--max-age-seconds", type=int, default=3600)
    verification.add_argument("--out", type=Path)
    a = p.parse_args()
    try:
        if a.max_age_seconds <= 0:
            p.error("--max-age-seconds must be positive")
        if a.action == "manifest":
            result = make_manifest(a)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(f"PASS fresh provenance backend={a.backend} model={result['gguf_sha256'][:16]} "
                  f"steps={len(result['files_sha256'])}: {a.out}")
        else:
            if a.max_abs < 0 or a.max_rmse < 0:
                p.error("tolerances must be nonnegative")
            result = compare(a)
            if a.out:
                a.out.parent.mkdir(parents=True, exist_ok=True)
                a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print("PASS independent llama.cpp full-FP32 logits: "
                  f"{result['matched_steps']} matching steps, GGUF sha256={result['sha256_gguf']}")
        return 0
    except (OSError, ValueError, GateFailure, GateBlocked, KeyError,
            subprocess.SubprocessError) as exc:
        code = 2 if isinstance(exc, (OSError, GateBlocked, KeyError)) else 1
        label = "BLOCKED" if code == 2 else "FAIL"
        print(f"{label} independent full-FP32 gate: {exc}", file=sys.stderr)
        if a.action == "compare" and a.out:
            # Do not turn a failed/missing comparison into an absent (ambiguous)
            # result. Explicit machine-readable failure remains fail-closed.
            failure = {"schema": "h27.independent-fp32-gate.v1", "passed": False,
                       "status": label, "created_unix": time.time(),
                       "reference": "llama.cpp", "candidate": "hermes-27",
                       "error": str(exc), "exit_code": code,
                       "max_abs_gate": a.max_abs, "max_rmse_gate": a.max_rmse}
            try:
                a.out.parent.mkdir(parents=True, exist_ok=True)
                a.out.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
            except OSError as write_exc:
                print(f"cannot save failure evidence: {write_exc}", file=sys.stderr)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
