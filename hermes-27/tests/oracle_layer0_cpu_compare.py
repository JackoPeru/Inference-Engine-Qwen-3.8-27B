#!/usr/bin/env python3
"""Independent GGUF block0 stage oracle vs llama.cpp cb_eval trace.

Computes both FP32-input and BF16-RNE-input projection predictions using real
GGUF unpacked weights on CPU. It does not model llama.cpp's internal Q8_1
activation quantization or HERMES GPU FP32 reduction order. This is a
first-divergence diagnostic, never a whole-model release PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as inp:
        for chunk in iter(lambda: inp.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bf16_round_nearest_even(values, np):
    """Finite FP32 -> BF16 RNE -> FP32, independently of torch/CUDA."""
    a = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(a)):
        raise ValueError("non-finite BF16-rounding input")
    bits = np.ascontiguousarray(a).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    words = (rounded >> 16).astype(np.uint32)
    return np.ascontiguousarray(words << 16).view(np.float32).reshape(a.shape)


def stage_metrics(a, b, name, np):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(f"invalid shape or nonfinite diagnostic {name}")
    delta = a - b
    idx = int(np.argmax(np.abs(delta)))
    max_abs = float(np.max(np.abs(delta)))
    rmse = float(np.sqrt(np.mean(delta * delta)))
    scale = float(np.sqrt(np.mean(a * a)))
    result = {
        "stage": name,
        "elements": int(a.size),
        "max_abs": max_abs,
        "rmse": rmse,
        "relative_rmse": rmse / max(1e-15, scale),
        "max_error_index": idx,
        "reference_at_max": float(a.flat[idx]),
        "candidate_at_max": float(b.flat[idx]),
    }
    print(f"DIAGNOSTIC {name} n={a.size} max_abs={max_abs:.9g} "
          f"rmse={rmse:.9g} relative_rmse={result['relative_rmse']:.9g} "
          f"worst_idx={idx}")
    return result


def compare(llama_dir: Path, model: Path, output: Path) -> dict:
    import numpy as np
    from scripts.block0_corrected_ref import (
        GGUF, bf16_to_f32, rms, silu,
    )

    manifest_path = llama_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "h27.reliability-layer-cb.v1"
            or manifest.get("backend") != "llama.cpp"
            or manifest.get("status") != "CAPTURED_NOT_INDEPENDENTLY_COMPARED"
            or manifest.get("prompt_ids") != [6205]
            or manifest.get("absolute_position") != 0
            or manifest.get("layer") != 0):
        raise ValueError("unsupported model/token/position/stage capture provenance")
    if Path(manifest.get("model_path", "")).resolve() != model.resolve():
        raise ValueError("capture and oracle GGUF path differ")
    expected = manifest.get("model_sha256", "")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("missing GGUF SHA256 identity")
    if file_sha256(model) != expected:
        raise ValueError("GGUF SHA256 differs from independent capture")

    stage_names = (
        ("attn_norm-0", 5120),
        ("linear_attn_qkv_mixed-0", 10240),
        ("conv_output_silu-0", 10240),
    )
    observed = {}
    for stage in manifest.get("tensors", []):
        observed[stage["name"]] = stage
    captures = {}
    for name, n in stage_names:
        entry = observed.get(name)
        if not entry or entry.get("shape") != [n, 1, 1, 1]:
            raise ValueError(f"missing or incorrectly shaped llama tensor {name}")
        filename = entry.get("file", "")
        if filename != f"{name}.f32":
            raise ValueError(f"unsafe or unexpected capture filename {filename!r}")
        path = llama_dir / filename
        if path.stat().st_size != n * 4 or file_sha256(path) != entry.get("sha256"):
            raise ValueError(f"invalid capture byte count/hash for {name}")
        arr = np.fromfile(path, dtype="<f4", count=n)
        if arr.size != n or not np.all(np.isfinite(arr)):
            raise ValueError(f"invalid FP32 logits at {name}")
        captures[name] = arr

    with model.open("rb") as inp:
        gguf = GGUF(inp, np)
        embeddings = gguf.tensor("token_embd.weight", "bf16")
        x = bf16_to_f32(embeddings[6205], np).astype(np.float32)
        h = np.asarray(
            rms(x.reshape(1, -1), gguf.tensor("blk.0.attn_norm.weight", "f32"), np),
            dtype=np.float32,
        ).reshape(-1)
        del embeddings
        h_bf16 = bf16_round_nearest_even(h, np)
        weight = gguf.tensor("blk.0.attn_qkv.weight", "nvfp4")
        qkv = np.asarray(h.reshape(1, -1) @ weight.T, dtype=np.float32).reshape(-1)
        qkv_bf16 = np.asarray(h_bf16.reshape(1, -1) @ weight.T,
                              dtype=np.float32).reshape(-1)
        del weight
        conv_weight = gguf.tensor("blk.0.ssm_conv1d.weight", "f32").reshape(10240, 4)
        # For token pos0 the three causal predecessor taps are zero.
        conv = np.asarray(silu(qkv * conv_weight[:, 3], np), dtype=np.float32)
        conv_bf16 = np.asarray(
            silu(qkv_bf16 * conv_weight[:, 3], np), dtype=np.float32
        )
        del conv_weight

    references = dict(zip((x[0] for x in stage_names), (h, qkv, conv)))
    results = []
    for name, n in stage_names:
        if len(references[name]) != n:
            raise ValueError(f"nonfinite or invalid CPU reference {name}")
        results.append(stage_metrics(references[name], captures[name],
                                     name + ":CPU_FP32_vs_LLAMA", np))
    results.extend([
        stage_metrics(h_bf16, h, "attn_norm-0:BF16_RNE_vs_FP32", np),
        stage_metrics(qkv_bf16, qkv, "qkv:CPU_BF16_RNE_vs_CPU_FP32", np),
        stage_metrics(qkv_bf16, captures["linear_attn_qkv_mixed-0"],
                      "qkv:CPU_BF16_RNE_vs_LLAMA", np),
        stage_metrics(conv_bf16, conv, "conv:CPU_BF16_RNE_vs_CPU_FP32", np),
        stage_metrics(conv_bf16, captures["conv_output_silu-0"],
                      "conv:CPU_BF16_RNE_vs_LLAMA", np),
    ])
    result = {
        "status": "DIAGNOSTIC_NOT_RELEASE_GATE",
        "model_sha256": expected,
        "llama_manifest": str(manifest_path),
        "model": str(model),
        "token_id": 6205,
        "absolute_position": 0,
        "reference_backend": "CPU_NUMPY_GGUF_UNPACKED_FLOAT32_AND_BF16_RNE_INPUT",
        "llama_backend": "llama.cpp C API cb_eval",
        "warning": "QKV BF16-RNE is a CPU approximation of HERMES input precision, "
                   "NOT a capture of HERMES CUDA kernel nor an exact llama MMVQ oracle.",
        "stages": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def selftest() -> None:
    import numpy as np
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        p = Path(folder) / "bytes.bin"
        p.write_bytes(b"abc")
        if file_sha256(p) != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad":
            raise AssertionError("SHA256 selftest failed")
    values = np.array([1.0, 1.00390625, 1.01171875, -1.00390625], dtype=np.float32)
    converted = bf16_round_nearest_even(values, np)
    np.testing.assert_array_equal(
        converted, np.array([1.0, 1.0, 1.015625, -1.0], dtype=np.float32)
    )
    try:
        bf16_round_nearest_even(np.array([np.inf], dtype=np.float32), np)
        raise AssertionError("non-finite BF16 input unexpectedly accepted")
    except ValueError:
        pass
    print("SELFTEST PASS layer0 CPU oracle SHA256 and input validation")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-dir", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return 0
    if args.llama_dir is None or args.model is None or args.out is None:
        parser.error("--llama-dir, --model and --out are required")
    compare(args.llama_dir, args.model, args.out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, AssertionError) as e:
        print(f"BLOCKED: {e}", file=sys.stderr)
        raise SystemExit(2)
