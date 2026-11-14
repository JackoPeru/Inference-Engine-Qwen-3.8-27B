#!/usr/bin/env python3
"""Fail-closed verification of the two-token HERMES scratch layer0 trace.

No GPU, network or model execution. This verifies trace bytes and the
scratch runtime's recorded decode_step(token, absolute_position) arguments;
it does not establish independent llama.cpp numerical fidelity.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import struct


EXPECTED_GGUF_SHA = (
    "db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1"
)
EXPECTED_GGUF_BYTES = 19694360544
STAGES = {
    "attn_norm": 5120,
    "linear_attn_qkv_mixed": 10240,
    "conv_output_silu": 10240,
    "beta_sigmoid": 48,
    "gate": 48,
    "q_norm_scaled": 6144,
    "k_norm": 6144,
    "attn_output": 6144,
    "final_output": 6144,
    "linear_attn_out": 5120,
    "attn_residual": 5120,
}


class Blocked(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(4 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def verify(
    *, capture_dir: Path, model: Path, source: Path, patched_source: Path,
    binary: Path, expected_sha: str = EXPECTED_GGUF_SHA,
    expected_bytes: int = EXPECTED_GGUF_BYTES,
) -> dict:
    files = [capture_dir, model, source, patched_source, binary]
    if not capture_dir.is_dir() or any(not p.is_file() for p in files[1:]):
        raise Blocked("missing capture directory/model/source/binary")
    if model.stat().st_size != expected_bytes or sha256_file(model) != expected_sha:
        raise Blocked("physical GGUF size/hash not verified")
    original = source.read_bytes()
    modified = patched_source.read_bytes()
    if original == modified or b"HERMES_SCRATCH_LAYER0_TRACE_DIR" in original:
        raise Blocked("unmodified original source identity invalid")
    if modified.count(b"HERMES_SCRATCH_LAYER0_TRACE_DIR") != 2:
        raise Blocked("not the expected opt-in scratch layer0 patch")
    if not (binary.stat().st_size and source.stat().st_size and
            patched_source.stat().st_size):
        raise Blocked("empty executable or source")

    expected_files = {f"p{pos}_{name}.f32"
                      for pos in (0, 1) for name in STAGES}
    actual_files = {path.name for path in capture_dir.glob("*.f32")}
    if actual_files != expected_files:
        raise Blocked(f"missing or unexpected trace tensor files: "
                      f"missing={sorted(expected_files-actual_files)} "
                      f"unexpected={sorted(actual_files-expected_files)}")
    expected_contexts = {f"p{pos}_context.txt" for pos in (0, 1)}
    actual_contexts = {path.name for path in capture_dir.glob("p*_context.txt")}
    if actual_contexts != expected_contexts:
        raise Blocked("missing/unexpected runtime token-position contexts")
    expected_tokens = (6205, 161421)
    context_records = []
    for pos, token in enumerate(expected_tokens):
        context_path = capture_dir / f"p{pos}_context.txt"
        if context_path.is_symlink():
            raise Blocked(f"runtime context symlink rejected: {context_path}")
        raw_context = context_path.read_bytes()
        expected_context = (f"token_id={token}\nabsolute_position={pos}\n"
                            f"sequence={pos}\n").encode("ascii")
        if raw_context != expected_context:
            raise Blocked(f"runtime decode token/position mismatch: {context_path}")
        context_records.append({
            "runtime_token_id": token,
            "runtime_absolute_position": pos,
            "decode_sequence": pos,
            "context_sha256": hashlib.sha256(raw_context).hexdigest(),
        })
    tensor_records = []
    for pos in (0, 1):
        for name, count in STAGES.items():
            path = capture_dir / f"p{pos}_{name}.f32"
            if path.is_symlink() or path.stat().st_size != count * 4:
                raise Blocked(f"trace size/symlink mismatch: {path}")
            raw = path.read_bytes()
            floats = array("f")
            if floats.itemsize != 4 or struct.pack("=I", 1) != struct.pack("<I", 1):
                raise Blocked("host must be little-endian FP32")
            floats.frombytes(raw)
            if any(not math.isfinite(value) for value in floats):
                raise Blocked(f"nonfinite trace values: {path}")
            tensor_records.append({
                "name": name, "runtime_reported_position": pos,
                "dtype": "little-endian-fp32", "shape": [count],
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            })
    return {
        "schema": "h27.hermes-scratch-layer0-capture.v2",
        "backend": "hermes-27-scratch",
        "status": "RUNTIME_DECODE_ARGUMENTS_RECORDED_NOT_LLAMA_COMPARED",
        "not_release_gate": True,
        "prompt_ids_runtime_reported": list(expected_tokens),
        "absolute_positions_runtime_reported": [0, 1],
        "runtime_decode_contexts": context_records,
        "model_sha256": expected_sha, "model_bytes": expected_bytes,
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "patched_source_sha256": hashlib.sha256(modified).hexdigest(),
        "binary_sha256": sha256_file(binary),
        "tensors": tensor_records,
        "limitations": [
            "Token IDs and positions were recorded inside the instrumented HERMES decode_step, not independently inside llama.cpp",
            "No independent llama stage numeric comparison",
            "Scratch-only trace files may contain hidden activations; never commit them",
        ],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-dir", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--patched-source", required=True, type=Path)
    p.add_argument("--binary", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.out.exists():
        raise Blocked("never overwrite capture manifest")
    data = verify(capture_dir=args.capture_dir, model=args.model,
                  source=args.source, patched_source=args.patched_source,
                  binary=args.binary)
    args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"PASS SCRATCH_LAYER0_BYTES_VERIFIED stages={len(data['tensors'])} "
          f"not_release_gate=true manifest={args.out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Blocked, OSError, ValueError) as exc:
        print("BLOCKED_LAYER0_TRACE_MANIFEST:", exc)
        raise SystemExit(2) from exc
