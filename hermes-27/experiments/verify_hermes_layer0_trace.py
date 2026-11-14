#!/usr/bin/env python3
"""Fail-closed verification of the two-token HERMES scratch layer0 trace.

No GPU, network or model execution. This verifies trace bytes and the
scratch runtime's recorded decode_step(token, absolute_position) arguments;
it does not establish independent llama.cpp numerical fidelity. Test-only
synthetic inputs use a DIFFERENT schema and can never become an oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct


EXPECTED_GGUF_SHA = (
    "db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1"
)
EXPECTED_GGUF_BYTES = 19694360544
SCRATCH_PATCHER_SHA = "80c0be0c1e29faf5db837da306af59ee729d4207c9bc21022163cfdef049c886"
RUNTIME_TOKENS = (6205, 161421)
SYNTHETIC_SCHEMA = "h27.hermes-scratch-layer0-synthetic-fixture.v1"
SYNTHETIC_STATUS = "SYNTHETIC_CPU_FIXTURE_NOT_PHYSICAL_RUNTIME"
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


def reconstructed_patch(original: bytes) -> bytes:
    """Require exact frozen patch transformation, not two marker strings."""
    patcher_path = Path(__file__).with_name("pp2_layer0_trace_scratch_patch.py")
    if not patcher_path.is_file() or sha256_file(patcher_path) != SCRATCH_PATCHER_SHA:
        raise Blocked("frozen PP2 scratch patcher physical SHA256 mismatch")
    spec = importlib.util.spec_from_file_location("h27_exact_l0_scratch_patcher", patcher_path)
    if spec is None or spec.loader is None:
        raise Blocked("cannot inspect the trusted PP2 scratch patcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.patched_source(original.decode("utf-8")).encode("utf-8")
    except (UnicodeError, ValueError) as error:
        raise Blocked("original PP2 source cannot produce exact scratch patch") from error


def runtime_sidecar(pos: int, token: int) -> bytes:
    return (f"token_id={token}\nabsolute_position={pos}\n"
            f"sequence={pos}\n").encode("ascii")


def verify(
    *, capture_dir: Path, model: Path, source: Path, patched_source: Path,
    binary: Path, expected_sha: str = EXPECTED_GGUF_SHA,
    expected_bytes: int = EXPECTED_GGUF_BYTES,
    test_fixture: bool = False,
) -> dict:
    if type(test_fixture) is not bool:
        raise Blocked("invalid test-only flag")
    if not test_fixture and (expected_sha != EXPECTED_GGUF_SHA or
                             expected_bytes != EXPECTED_GGUF_BYTES):
        raise Blocked("genuine HERMES trace requires fixed GGUF SHA/size")
    files = [capture_dir, model, source, patched_source, binary]
    if not capture_dir.is_dir() or any(not p.is_file() for p in files[1:]):
        raise Blocked("missing capture directory/model/source/binary")
    if capture_dir.is_symlink() or any(p.is_symlink() for p in files[2:]):
        raise Blocked("capture directory/source/patch/binary symlink rejected")
    if (source.resolve() == patched_source.resolve() or
            source.resolve() == binary.resolve() or
            patched_source.resolve() == binary.resolve()):
        raise Blocked("source, scratch patch, and executable must be different files")
    if source.name != "pp2_fwd.cu" or patched_source.name != "pp2_fwd.cu":
        raise Blocked("source and scratch C++ files must each be pp2_fwd.cu")
    if model.stat().st_size != expected_bytes or sha256_file(model) != expected_sha:
        raise Blocked("physical GGUF size/hash not verified")
    original = source.read_bytes()
    modified = patched_source.read_bytes()
    if original == modified or b"HERMES_SCRATCH_LAYER0_TRACE_DIR" in original:
        raise Blocked("unmodified original source identity invalid")
    if modified != reconstructed_patch(original):
        raise Blocked("scratch C++ patch differs from exact frozen source transformation")
    if sha256_file(source) != hashlib.sha256(original).hexdigest() or \
            sha256_file(patched_source) != hashlib.sha256(modified).hexdigest():
        raise Blocked("source or scratch patch changed during verification")
    if not (binary.stat().st_size and source.stat().st_size and
            patched_source.stat().st_size):
        raise Blocked("empty executable or source")

    expected_files = {f"p{pos}_{name}.f32"
                      for pos in (0, 1) for name in STAGES}
    expected_contexts = {f"p{pos}_context.txt" for pos in (0, 1)}
    actual_paths = list(capture_dir.iterdir())
    actual_files = {path.name for path in actual_paths}
    if any(path.is_symlink() or not path.is_file() for path in actual_paths):
        raise Blocked("scratch capture contains a symlink or nonregular file")
    if actual_files != expected_files | expected_contexts or len(actual_paths) != 24:
        raise Blocked(f"missing/unexpected runtime context/tensor files: "
                      f"missing={sorted((expected_files | expected_contexts)-actual_files)} "
                      f"unexpected={sorted(actual_files-(expected_files | expected_contexts))}")
    context_records = []
    for pos, token in enumerate(RUNTIME_TOKENS):
        context_path = capture_dir / f"p{pos}_context.txt"
        if context_path.is_symlink():
            raise Blocked(f"runtime context symlink rejected: {context_path}")
        raw_context = context_path.read_bytes()
        expected_context = runtime_sidecar(pos, token)
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
            if struct.pack("=I", 1) != struct.pack("<I", 1):
                raise Blocked("host must be little-endian FP32")
            if any(not math.isfinite(value) for (value,) in
                   struct.iter_unpack("<f", raw)):
                raise Blocked(f"nonfinite trace values: {path}")
            tensor_records.append({
                "name": name, "runtime_reported_position": pos,
                "dtype": "little-endian-fp32", "shape": [count],
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            })
    return {
        "schema": (SYNTHETIC_SCHEMA if test_fixture else
                   "h27.hermes-scratch-layer0-capture.v2"),
        "backend": "hermes-27-synthetic-fixture" if test_fixture else "hermes-27-scratch",
        "status": (SYNTHETIC_STATUS if test_fixture else
                   "RUNTIME_DECODE_ARGUMENTS_RECORDED_NOT_LLAMA_COMPARED"),
        "not_release_gate": True,
        "prompt_ids_runtime_reported": list(RUNTIME_TOKENS),
        "absolute_positions_runtime_reported": [0, 1],
        "runtime_decode_contexts": context_records,
        "model_sha256": expected_sha, "model_bytes": expected_bytes,
        "model_path": str(model.resolve()),
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "patched_source_sha256": hashlib.sha256(modified).hexdigest(),
        "binary_sha256": sha256_file(binary),
        "tensors": tensor_records,
        "runtime_identity_scope": (
            "SYNTHETIC_CPU_INPUTS_ONLY" if test_fixture else
            "SCRATCH_RUNTIME_DECODE_SELF_REPORT_ONLY"),
        "independent_llama_tensor_identity_verified": False,
        "cross_engine_comparison_status": "BLOCKED_NO_INDEPENDENT_LLAMA_CALLBACK",
        "tensor_axis_contract": "FLAT_F32_PER_STAGE_NO_INDEPENDENT_HEAD_AXIS_PROOF",
        "qk_cross_engine_mapping": {
            "hermes_stages": ["q_norm_scaled", "k_norm"],
            "llama_candidate_callbacks": ["q_conv_predelta-0", "k_conv_predelta-0"],
            "status": "BLOCKED_UNVERIFIED_HEAD_REPETITION_AND_Q_SCALING",
            "independent_semantic_equivalence_verified": False,
        },
        "limitations": [
            "Token IDs and positions were recorded inside the instrumented HERMES decode_step, not independently inside llama.cpp",
            "No independent llama stage numeric comparison",
            "HERMES Q/K flat width cannot certify llama head/position axes, repetition or Q scaling",
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
