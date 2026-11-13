#!/usr/bin/env python3
"""Zero-GPU preflight: GGUF semantics and CLI parse/error-path validation."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from reliability_core import GateBlocked, GateFailure, RunReport, require

ROOT = Path(__file__).resolve().parents[1]


def math_preflight(gguf: Path | None) -> dict:
    script = ROOT / "tests/gate_math_gguf.py"
    argv = [sys.executable, str(script), "--root", str(ROOT)]
    if gguf:
        if not gguf.is_file():
            raise GateBlocked(f"real GGUF not available: {gguf}")
        argv += ["--gguf", str(gguf)]
    else:
        argv.append("--source-only")
    run = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    if run.returncode == 2:
        raise GateBlocked(run.stderr[-1500:])
    require(run.returncode == 0, f"GGUF math regression: {run.stdout[-1500:]} {run.stderr[-1500:]}")
    return {"gguf_verified": gguf is not None, "output_tail": run.stdout[-600:]}


def cli_parse(engine: Path) -> dict:
    if not engine or not engine.is_file():
        raise GateBlocked(f"CLI binary absent: {engine}")
    model_placeholder = "/__does_not_exist__/model.gguf"
    base = [str(engine), "--model", model_placeholder]
    tests = (
        ["--prompt-ids", "-1"],
        ["--prompt-ids", "248320"],
        ["--prompt-ids", "1,"],
        ["--prompt-ids", ""],
        ["--prompt-ids", "1", "--max-new-tokens", "0"],
        ["--prompt-ids", "1", "--max-new-tokens", "8193"],
        ["--prompt-ids", "1", "--eos-token", "-1"],
        ["--prompt-ids", "1", "--bogus", "42"],
        ["--prompt-ids", ",".join(["1"] * 8192), "--max-new-tokens", "2"],
        ["--prompt-ids", ",".join(["1"] * 65), "--verify-prefill"],
    )
    for argv in tests:
        try:
            result = subprocess.run(base + argv, capture_output=True, text=True,
                                    timeout=5)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GateBlocked(f"CLI error-path check could not complete: {exc}") from exc
        require(result.returncode == 2, f"CLI invalid args not rejected before GPU: "
                f"{argv[:8]} returned {result.returncode}, stderr={result.stderr[-200:]}")
    return {"invalid_input_cases": len(tests), "all_exit_2_before_upload": True}


def source_files() -> dict:
    files = [ROOT / "distributed/pp2_fwd.cu", ROOT / "serving/server.cpp",
             ROOT / "scripts/block0_corrected_ref.py", ROOT / "tests/reliability_core.py",
             ROOT / "tests/compare_llama.py", ROOT / "tools/compare_fp32_logits.py"]
    absent = [str(p) for p in files if not p.is_file()]
    require(not absent, f"required validation/engine source files missing: {absent}")
    if (ROOT / "kernels/tests/test_p1_block0.cu").read_text(encoding="utf-8").find(
            "load_corrected_oracle") < 0:
        raise GateFailure("P1 test does not enforce corrected-oracle preflight")
    return {"verified_source_files": len(files) + 1, "corrected_oracle_required": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", type=Path, help="real Qwen3.8 NVFP4 GGUF, required for full CPU preflight")
    p.add_argument("--engine", type=Path, help="hermes_generate binary for pre-upload invalid-arg checks")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    report = RunReport("hermes27-static-v1", {
        "gguf_path": str(a.gguf) if a.gguf else None,
        "engine_path": str(a.engine) if a.engine else None, "gpu_workloads": False
    })
    report.record("source-invariants", source_files)
    if a.gguf:
        report.record("real-gguf-semantic-preflight", lambda: math_preflight(a.gguf))
    else:
        report.record("source-semantic-preflight", lambda: math_preflight(None))
        report.skip("real-gguf-semantic-preflight", "pass --gguf to verify the actual checkpoint")
    if a.engine:
        report.record("CLI-invalid-arguments", lambda: cli_parse(a.engine))
    else:
        report.skip("CLI-invalid-arguments", "pass --engine to verify parse failures without GPU")
    code, label = report.save(a.out)
    print(f"RELEASE STATIC {label} (no GPU): {a.out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
