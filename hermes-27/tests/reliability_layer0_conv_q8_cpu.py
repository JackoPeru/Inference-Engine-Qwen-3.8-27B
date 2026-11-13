#!/usr/bin/env python3
"""Independent CPU-only exact-GGUF layer0 Q8_1 -> causal conv+SiLU comparison.

Reads immutable llama.cpp cb_eval captures and GGUF. Does not run GPU/llama/HERMES.
Requires the source-tree GGUF parser and numpy. Uses bounded OpenBLAS threads.
A CPU double matvec + f32 conv approximates (not bit-exact emulates) CUDA kernels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

EXPECTED_GGUF_SHA = "db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1"
EXPECTED_LIB_SHA = "fd9e3e0810c6eca2e37b57abf443bcf15963671cbd0ee8489a37461a521094d4"
TOKEN, POSITION, LAYER, DIM, CHANNELS, TAPS = 6205, 0, 0, 5120, 10240, 4
SCHEMA = "h27.reliability-layer0-q8-conv-cpu.v1"
STAGES = {"attn_norm-0": DIM, "linear_attn_qkv_mixed-0": CHANNELS,
          "conv_output_silu-0": CHANNELS}


class Blocked(RuntimeError):
    """Untrusted or incompatible input; no synthetic success."""


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise Blocked(msg)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_capture(manifest_path: Path, model: Path, np) -> tuple[dict, dict]:
    require(manifest_path.is_file(), "missing independent llama manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "h27.reliability-layer-cb.v1"
            and manifest.get("backend") == "llama.cpp"
            and manifest.get("status") == "CAPTURED_NOT_INDEPENDENTLY_COMPARED"
            and manifest.get("prompt_ids") == [TOKEN]
            and manifest.get("absolute_position") == POSITION
            and manifest.get("layer") == LAYER and manifest.get("n_gpu_layers") == 80,
            "invalid independent llama capture source/token/position/shape or execution mode")
    require(manifest.get("model_sha256") == EXPECTED_GGUF_SHA,
            "capture GGUF not authenticated expected checkpoint")
    require(model.resolve() == Path(manifest.get("model_path", "")).resolve(),
            "physical GGUF path not same as capture source")
    require(sha256_file(model) == EXPECTED_GGUF_SHA,
            "physical GGUF sha256 not same as independent capture")
    require(manifest.get("libllama_sha256") == EXPECTED_LIB_SHA,
            "unexpected captured libllama identity")
    require(sha256_file(Path(manifest.get("libllama_path", ""))) == EXPECTED_LIB_SHA,
            "installed runtime libllama no longer matches captured library")
    require(isinstance(manifest.get("exporter_sha256"), str)
            and len(manifest["exporter_sha256"]) == 64
            and manifest.get("tensor_data_format") == "little-endian-fp32-ne0-fastest",
            "missing exporter hash or wrong tensor format")
    captures = {}
    found = set()
    for record in manifest.get("tensors", []):
        name = record.get("name")
        require(name in STAGES and name not in found,
                "unexpected/duplicated stage in capture")
        found.add(name)
        n = STAGES[name]
        require(record.get("file") == name + ".f32"
                and record.get("source_type") == "f32"
                and record.get("shape") == [n, 1, 1, 1]
                and record.get("source_strides_bytes") == [4, n * 4, n * 4, n * 4]
                and record.get("elements") == n and record.get("bytes") == n * 4
                and record.get("buffer") == "CUDA0",
                name + ": unexpected strides, dtype, shape or device")
        path = manifest_path.parent / (name + ".f32")
        require(path.is_file() and path.stat().st_size == n * 4,
                name + ": absent or short capture")
        require(sha256_file(path) == record.get("sha256"),
                name + ": corrupted trace SHA256")
        arr = np.fromfile(path, dtype="<f4")
        require(arr.size == n and np.isfinite(arr).all(),
                name + ": invalid FP32 data")
        captures[name] = arr
    require(found == set(STAGES), "missing actual llama capture stages")
    return manifest, captures


def round_bf16_rne(x, np):
    a = np.ascontiguousarray(x, dtype=np.float32)
    require(bool(np.isfinite(a).all()), "nonfinite BF16 input")
    bits = a.view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1))
            & np.uint32(0xffff0000)).view(np.float32)


def quant_q8_1_per32(x, np):
    """CPU GGML-style per-32 Q8_1: FP32 amax/127, signed i8, FP16 scale.

    ggml-cuda/quantize.cu quantizes with roundf (ties away from zero); emulate
    ties-away explicitly. This still is NOT an arithmetic-exact CUDA/DP4A replay.
    """
    a = np.ascontiguousarray(x, dtype=np.float32)
    require(a.ndim == 1 and a.size % 32 == 0
            and bool(np.isfinite(a).all()), "invalid Q8_1 input")
    chunks = a.reshape(-1, 32)
    amax = np.max(np.abs(chunks), axis=1, keepdims=True)
    scale = amax / np.float32(127.)
    ratio = np.divide(chunks, scale, out=np.zeros_like(chunks),
                      where=scale != 0)
    q = np.copysign(np.floor(np.abs(ratio) + np.float32(0.5)), ratio)
    q = np.clip(q, -127, 127).astype(np.int8)
    scale_stored = scale.astype(np.float16).astype(np.float32)
    return np.ascontiguousarray((q.astype(np.float32) *
                                 scale_stored).reshape(-1))


def causal_conv_silu_pos0(qkv, conv_w, np):
    """Four-tap depthwise conv with x[-3],x[-2],x[-1] = 0, current x[0].

    Source contract: llama-mainline ggml/src/ggml-cuda/ssm-conv.cu:29-56;
    PP2 distributed/pp2_fwd.cu:98-105 and 1068-1078.
    """
    x = np.asarray(qkv, dtype=np.float32)
    w = np.asarray(conv_w, dtype=np.float32)
    require(x.shape == (CHANNELS,) and w.shape == (CHANNELS, TAPS)
            and bool(np.isfinite(x).all()) and bool(np.isfinite(w).all()),
            "bad actual convolution QKV/kernel shape or finite check")
    # Preserve target float32 rounding: 0*three weights + x[t]*w3.
    raw = (x * w[:, 3]).astype(np.float32)
    y = (raw / (np.float32(1) + np.exp(-raw))).astype(np.float32)
    require(bool(np.isfinite(y).all()), "nonfinite convolution output")
    return y


def comparison(reference, candidate, np):
    ref = np.asarray(reference, dtype=np.float64)
    got = np.asarray(candidate, dtype=np.float64)
    require(ref.shape == got.shape and ref.size == CHANNELS
            and bool(np.isfinite(ref).all()) and bool(np.isfinite(got).all()),
            "nonfinite or mismatched 10240-element stage comparison")
    delta = got - ref
    i = int(np.argmax(np.abs(delta)))
    return {"elements": CHANNELS, "rmse": float(np.sqrt(np.mean(delta * delta))),
            "max_abs": float(np.max(np.abs(delta))),
            "index_max_error": i,
            "llama_at_max_error": float(ref[i]),
            "candidate_at_max_error": float(got[i]),
            "mean_abs": float(np.mean(np.abs(delta))),
            "fraction_abs_gt_1e_3": float(np.mean(np.abs(delta) > 1.e-3)),
            "pearson": float(np.corrcoef(ref, got)[0, 1])}


def inspect_conv_source(path: Path) -> str:
    """Authenticate the relevant source expression, not just an assertion in doc."""
    require(path.is_file(), "reference llama CUDA ssm-conv.cu missing")
    code = path.read_text(encoding="utf-8")
    require("x[(i + j) % d_conv] * w[j]" in code
            and "i + d_conv - 1" in code and "ggml_cuda_op_silu_single" in code,
            "installed llama depthwise convolution semantics changed")
    return sha256_file(path)


def compute(manifest_path: Path, model_path: Path, source_root: Path,
            llama_conv_source: Path, output_path: Path) -> dict:
    import numpy as np
    sys.path.insert(0, str(source_root))
    from scripts.block0_corrected_ref import GGUF, bf16_to_f32, rms

    manifest, llama = validate_capture(manifest_path, model_path, np)
    ref_source_sha = inspect_conv_source(llama_conv_source)
    with model_path.open("rb") as source:
        gguf = GGUF(source, np)
        dims, dtype, _ = gguf.tensors.get("blk.0.ssm_conv1d.weight", ((), -1, -1))
        require(tuple(dims) == (4, CHANNELS) and dtype == 0,
                "GGUF conv must be real FP32 [4, 10240] with causal tap order")
        conv_w = gguf.tensor("blk.0.ssm_conv1d.weight", "f32")
        require(conv_w.shape == (CHANNELS, TAPS)
                and bool(np.isfinite(conv_w).all()), "invalid real GGUF conv")
        w_dims, w_type, _ = gguf.tensors.get("blk.0.attn_qkv.weight", ((), -1, -1))
        require(tuple(w_dims) == (DIM, CHANNELS) and w_type == 40,
                "GGUF blk0 NVFP4 QKV tensor must have shape 10240x5120")
        embeddings = gguf.tensor("token_embd.weight", "bf16")
        require(embeddings.shape[1] == DIM and embeddings.shape[0] > TOKEN,
                "GGUF real token embedding missing/wrong shape")
        embedding = bf16_to_f32(embeddings[TOKEN], np).copy()
        norm_w = gguf.tensor("blk.0.attn_norm.weight", "f32")
        check_norm = np.asarray(rms(embedding[None, :], norm_w, np),
                                dtype=np.float32).reshape(-1)
        norm_diff = np.abs(check_norm.astype(np.float64)
                           - llama["attn_norm-0"].astype(np.float64))
        require(np.max(norm_diff) < 1.e-5,
                "exact GGUF norm and llama-captured input do not match")
        input_x = llama["attn_norm-0"]
        modes = {"q8_1": quant_q8_1_per32(input_x, np),
                 "bf16_rne": round_bf16_rne(input_x, np),
                 "fp32": input_x.copy()}
        # The llama-QKV-only control separates conv math from projection errors.
        all_qkv = {"llama_qkv_passthrough": llama["linear_attn_qkv_mixed-0"]}
        # Full 10240 x 5120 GGUF weights; float32 200 MiB, projected in bounded
        # FP64 panels to avoid a second 400 MiB all-weights float64 materialization.
        weights = gguf.tensor("blk.0.attn_qkv.weight", "nvfp4")
        require(weights.shape == (CHANNELS, DIM)
                and bool(np.isfinite(weights).all()), "invalid decoded weights")
        out = {name: np.empty(CHANNELS, dtype=np.float32) for name in modes}
        x64 = {name: vec.astype(np.float64) for name, vec in modes.items()}
        for start in range(0, CHANNELS, 128):
            end = min(start + 128, CHANNELS)
            panel = weights[start:end, :].astype(np.float64)
            for name, input64 in x64.items():
                out[name][start:end] = (panel @ input64).astype(np.float32)
        del weights
        all_qkv.update(out)

    llama_qkv = llama["linear_attn_qkv_mixed-0"]
    llama_conv = llama["conv_output_silu-0"]
    projections = {name: comparison(llama_qkv, qkv, np)
                   for name, qkv in all_qkv.items()}
    convolved = {name: causal_conv_silu_pos0(qkv, conv_w, np)
                 for name, qkv in all_qkv.items()}
    conv_comparisons = {name: comparison(llama_conv, y, np)
                        for name, y in convolved.items()}
    qq = projections["q8_1"]["rmse"]
    cq = conv_comparisons["q8_1"]["rmse"]
    clf = conv_comparisons["llama_qkv_passthrough"]["rmse"]
    result = {
        "schema": SCHEMA, "status": "CPU_ONLY_AUTHENTIC_CAPTURE_COMPARISON_NOT_GPU_RELEASE_GATE",
        "model_sha256": EXPECTED_GGUF_SHA, "token_id": TOKEN,
        "absolute_position": POSITION, "layer": LAYER,
        "llama_manifest_path": str(manifest_path),
        "llama_manifest_sha256": sha256_file(manifest_path),
        "llama_lib_sha256": manifest["libllama_sha256"],
        "llama_exporter_sha256_manifest": manifest["exporter_sha256"],
        "real_conv_source_path": str(llama_conv_source),
        "real_conv_source_sha256": ref_source_sha,
        "real_gguf_conv_shape": [CHANNELS, TAPS],
        "full_qkv_projection_elements": CHANNELS,
        "real_model_norm_vs_llama_max_abs": float(np.max(norm_diff)),
        "tensor_source_sha256": {name: record["sha256"]
           for record in manifest["tensors"] for name in STAGES
           if record["name"] == name},
        "projection_vs_llama_qkv": projections,
        "causal_conv_silu_vs_llama": conv_comparisons,
        "q8_conv_rmse_divided_by_bf16_conv_rmse": (
            cq / conv_comparisons["bf16_rne"]["rmse"]),
        "q8_conv_rmse_divided_by_fp32_conv_rmse": (
            cq / conv_comparisons["fp32"]["rmse"]),
        "verified_llama_qkv_conv_control_rmse": clf,
        "causal_depthwise_conv_matched": clf < 1.e-5,
        "q8_qkv_matched": qq < 1.e-5,
        "q8_conv_matched": cq < 1.e-5,
        "limitations": [
            "One genuine llama capture at token6205 position0 only; no later-position state",
            "CPU FP64 projected GEMV and CPU FP32 NumPy SiLU not bitwise ggml CUDA",
            "Not a current HERMES PP2 GPU runtime trace; no whole-model or release pass",
            "GGUF SHA256 checked by streaming physical model; exporter SHA is manifest-bound",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print("GGUF sha256 verified; captured tensor hashes verified; source verified")
    for mode in all_qkv:
        q, c = projections[mode], conv_comparisons[mode]
        print(f"{mode:22} QKV rmse={q['rmse']:.10g} max={q['max_abs']:.10g}"
              f" | CONV rmse={c['rmse']:.10g} max={c['max_abs']:.10g}")
    print(f"CONV_SEMANTICS_MATCH={result['causal_depthwise_conv_matched']}"
          f" Q8_CONV_MATCH={result['q8_conv_matched']}"
          f" output={output_path}")
    return result


def selftest() -> None:
    import numpy as np
    check = np.array([1., 1.00390625, 1.01171875, -1.00390625], dtype=np.float32)
    np.testing.assert_array_equal(round_bf16_rne(check, np),
                                  np.array([1., 1., 1.015625, -1.]))
    blocks = np.ones(32, dtype=np.float32)
    # Stored Q8_1 scale is FP16: a unit-amplitude block does not decode to
    # exactly 1 because (1/127)_fp16 * 127 = 0.99993896484375.
    expected_unit = np.float32(np.float16(np.float32(1 / 127))) * np.float32(127)
    np.testing.assert_array_equal(quant_q8_1_per32(blocks, np),
                                  np.full(32, expected_unit, dtype=np.float32))
    np.testing.assert_array_equal(quant_q8_1_per32(np.zeros(32, dtype=np.float32), np),
                                  np.zeros(32, dtype=np.float32))
    # Convolution unit: zero causal predecessors, weight3-only contribution.
    x = np.zeros(CHANNELS, dtype=np.float32); x[0] = 2.; x[1] = -2.
    w = np.zeros((CHANNELS, TAPS), dtype=np.float32)
    w[:, :3] = 100.  # Must NOT contribute at pos0
    w[0, 3] = .5; w[1, 3] = 1.
    got = causal_conv_silu_pos0(x, w, np)
    assert np.isclose(got[0], 1 / (1 + math.exp(-1)), rtol=1e-6)
    assert np.isclose(got[1], -2 / (1 + math.exp(2)), rtol=1e-6)
    assert np.all(got[2:] == 0)
    assert sha256_file(Path(__file__)) and len(sha256_file(Path(__file__))) == 64
    for invalid in [np.zeros(31, dtype=np.float32),
                    np.array([np.nan] * 32, dtype=np.float32)]:
        try:
            quant_q8_1_per32(invalid, np)
            raise AssertionError("accepted invalid quant input")
        except Blocked:
            pass
    print("SELFTEST_PASS: BF16-RNE Q8_1 zero/constant/bounds, causal conv at pos0")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--gguf", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--llama-conv-source", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest(); return 0
    if any(x is None for x in
           (args.manifest, args.gguf, args.source_root, args.llama_conv_source, args.out)):
        parser.error("--manifest --gguf --source-root --llama-conv-source --out required")
    compute(args.manifest, args.gguf, args.source_root,
            args.llama_conv_source, args.out)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (Blocked, ValueError, KeyError, OSError, ImportError) as exc:
        print(f"BLOCKED_CPU_LAYER0_CONV: {exc}", file=sys.stderr)
        sys.exit(2)
