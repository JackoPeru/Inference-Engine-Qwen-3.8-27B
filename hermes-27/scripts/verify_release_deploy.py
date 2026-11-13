#!/usr/bin/env python3
"""CPU-only deployment gate: compile CURRENT gguf_dump from source in scratch,
validate Jinja/Unicode against actual GGUF and tokenizer, then fail if the
deployed gguf_dump binary differs. No CMake, CUDA, daemon or production writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from reliability_core import GateBlocked, GateFailure, RunReport, require


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_source(root: Path, scratch: Path, compiler: str) -> Path:
    required = [root / "tools/gguf_dump.cpp", root / "model/src/gguf.cpp",
                root / "model/src/weights.cpp", root / "model/include/hermes/gguf.hpp",
                root / "model/include/hermes/weights.hpp"]
    for filename in required:
        if not filename.is_file():
            raise GateBlocked(f"current GGUF parser source missing: {filename}")
    output = scratch / "gguf_dump_from_current_source"
    argv = [compiler, "-std=c++20", "-O2", "-I" + str(root / "model/include"),
            str(root / "tools/gguf_dump.cpp"), str(root / "model/src/gguf.cpp"),
            str(root / "model/src/weights.cpp"), "-o", str(output)]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GateBlocked(f"scratch C++ compiler unavailable: {exc}") from exc
    require(proc.returncode == 0, "current GGUF parser scratch compile failed: " +
            proc.stderr[-2000:])
    require(output.is_file(), "GGUF source build created no binary")
    return output


def get_template(binary: Path, gguf: Path) -> bytes:
    try:
        proc = subprocess.run([str(binary), str(gguf), "--chat-template"],
                              capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GateBlocked(f"gguf_dump cannot extract chat template: {exc}") from exc
    require(proc.returncode == 0, f"gguf_dump --chat-template exited {proc.returncode}: "
            f"{proc.stderr[-300:]!r}")
    raw = proc.stdout
    require(raw and b"{%" in raw, "chat template missing Jinja syntax")
    require(not raw.startswith(b"version=") and b"\nKV-S" not in raw,
            "STALE gguf_dump: --chat-template returned metadata dump, not template")
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise GateFailure("chat template output has invalid UTF-8") from exc
    return raw


def bridge_check(python: Path, bridge: Path, tokenizer: Path,
                 gguf: Path, gguf_dump: Path) -> dict:
    for path in (python, bridge, tokenizer):
        if not path.is_file():
            raise GateBlocked(f"tokenizer prerequisite missing: {path}")
    cmd = [str(python), str(bridge), "--model", str(gguf),
           "--tokenizer", str(tokenizer), "--gguf-dump", str(gguf_dump)]
    unicode_text = "Unicode: caffè 🚀 中文 العربية 👩‍💻"
    request = {"op": "encode", "messages": [{"role": "user", "content": unicode_text}],
               "thinking": False}

    def call(req: dict) -> list[dict]:
        try:
            proc = subprocess.run(cmd, input=json.dumps(req, ensure_ascii=False) + "\n",
                                  capture_output=True, text=True, encoding="utf-8",
                                  timeout=90)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GateBlocked(f"tokenizer bridge failed to start: {exc}") from exc
        lines = proc.stdout.splitlines()
        if not lines:
            raise GateBlocked(f"tokenizer bridge emitted no ready reply: {proc.stderr[-400:]}")
        try:
            answers = [json.loads(line) for line in lines]
        except ValueError as exc:
            raise GateFailure(f"bridge emitted invalid JSONL: {lines[:2]!r}") from exc
        require(proc.returncode == 0 and len(answers) == 2 and
                answers[0].get("ready") is True and not answers[1].get("error"),
                f"bridge startup/encode invalid: {answers[:2]} {proc.stderr[-300:]}")
        return answers

    encoded = call(request)[1]
    ids = encoded.get("ids")
    require(isinstance(ids, list) and ids and
            all(type(t) is int and 0 <= t < 248320 for t in ids),
            "bridge returned no valid model token IDs")
    decoded = call({"op": "decode", "ids": ids})[1].get("text")
    require(isinstance(decoded, str) and unicode_text in decoded,
            f"tokenizer Unicode roundtrip mismatch: {decoded!r}")
    return {"token_count": len(ids), "unicode_roundtrip": True,
            "token_sha256": sha256(",".join(map(str, ids)).encode())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--tokenizer-json", required=True, type=Path)
    p.add_argument("--python-bin", required=True, type=Path,
                   help="existing Python with jinja2 and tokenizers installed")
    p.add_argument("--bridge", type=Path)
    p.add_argument("--deployed-gguf-dump", type=Path,
                   help="check actual deployment binary; required for full release PASS")
    p.add_argument("--compiler", default="g++")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    bridge = a.bridge or (root / "serving/tokenizer_bridge.py")
    report = RunReport("hermes27-deploy-cpu-v1", {
        "root": str(root), "gguf": str(a.gguf),
        "deployed_binary": str(a.deployed_gguf_dump) if a.deployed_gguf_dump else None,
        "gpu_workloads": False})
    with tempfile.TemporaryDirectory(prefix="h27-release-deploy-") as temporary:
        scratch = Path(temporary)
        built: Path | None = None
        raw: bytes | None = None
        def build():
            nonlocal built
            built = build_source(root, scratch, a.compiler)
            return {"scratch_executable": str(built),
                    "sha256_binary": sha256(built.read_bytes())}
        report.record("scratch-build-current-GGUF-parser", build)
        def template():
            nonlocal raw
            if built is None or not a.gguf.is_file():
                raise GateBlocked("source-built gguf_dump or GGUF not available")
            raw = get_template(built, a.gguf)
            return {"utf8_bytes": len(raw), "sha256_template": sha256(raw)}
        report.record("source-gguf-chat-template", template)
        def tokenizer_gate():
            if built is None or raw is None:
                raise GateBlocked("corrected source-built extractor unavailable")
            return bridge_check(a.python_bin, bridge, a.tokenizer_json, a.gguf, built)
        report.record("source-tokenizer-Jinja-Unicode", tokenizer_gate)
        if a.deployed_gguf_dump:
            def deployed():
                if raw is None or not a.deployed_gguf_dump.is_file():
                    raise GateBlocked("source model/template or deployed binary missing")
                candidate = get_template(a.deployed_gguf_dump, a.gguf)
                require(candidate == raw,
                        "DEPLOYMENT STALE: live gguf_dump template differs from current sources")
                return {"deployed_sha256_binary":
                        sha256(a.deployed_gguf_dump.read_bytes()), "matches_source": True}
            report.record("deployed-gguf-dump-parity", deployed)
        else:
            report.skip("deployed-gguf-dump-parity",
                        "pass --deployed-gguf-dump to reject stale deployment binaries")
    code, label = report.save(a.out)
    print(f"RELEASE DEPLOY {label} (CPU-only): {a.out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
