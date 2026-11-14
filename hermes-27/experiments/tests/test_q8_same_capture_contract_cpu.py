#!/usr/bin/env python3
"""Offline source-bound Q8_1 scalar/DP4A and BF16 warp comparison.

The --manifest mode refuses non-authentic input: it hashes the actual 19.7-GB
checkpoint, installed libllama and captured FP32 stage files before reading the
raw GGUF NVFP4 tensor. No CUDA or model inference is performed. Source-level
CPU approximations are deliberately NOT CUDA numerical or performance tests.

Usage: python -B experiments/tests/test_q8_same_capture_contract_cpu.py --selftest
       python -B experiments/tests/test_q8_same_capture_contract_cpu.py \
         --manifest /tmp/h27-layer0-llama-pos0-prime3/manifest.json \
         --gguf /home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf \
         --source-root /home/matteo/hermes-27-gh/hermes-27
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

CAPTURE_SHA = "49b0052027532e1e8572d9d865517b2beef92c0e8176a02adacd35efe46d540b"
SOURCES = {
    "experiments/q8_dp4a_pp2_scratch_patch.py":
        "27d9b8fe31f3bc0ca8c5fb120769e93e20d466bdc07878310b6d0cccfe0b598c",
    "kernels/src/nvfp4_gemv_q8_1_experimental.cu":
        "7cd1e3010d3d038caeb19d31cf2dcbd1f6e619477917f88ddea2958e37922d7c",
    "kernels/src/nvfp4_gemv_warp_variants.cu":
        "bb3ba6c14ebda27e17cec39d294ca4357ac3dae675423ff0804c321df5cbcfc9",
    "tests/reliability_layer0_conv_q8_cpu.py":
        "f41b46eeef2818cd8f15bff04e23b2e19629744bb28d696e5d07b835c7d062fc",
    "scripts/block0_corrected_ref.py":
        "b1ef0c53c0e3b426754f70f124a054e897a814ce17dbce2b5c4731677f61370b",
}
NVFP4_X2 = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)
VARIANTS = ("PackedU16x2", "PackedU32x4", "Unroll2", "WarpSplit2", "WarpSplit4", "Warp8Rows")


class Blocked(RuntimeError):
    """Missing proof, changed source, incompatible dimensions or data."""


def need(condition, reason):
    if not condition:
        raise Blocked(reason)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def check_sources(root):
    for rel, digest in SOURCES.items():
        path = root / rel
        need(path.is_file() and sha(path) == digest,
             "changed/unavailable audited numerical source: " + rel)


def as_f32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def scale_ue4m3(code):
    if code in (0, 127):
        return 0.0
    e, m = code >> 3, code & 7
    return math.ldexp(float(m), -9) if e == 0 else math.ldexp(1 + m / 8, e - 7)


def q8_quant(x, np):
    """Exact packed Q8 integers and stored FP16 scale per 32 input floats.

    The FP32 ratio is rounded away from zero; an FP16 scale is saved *after*
    rounding the integers, as in ggml-cuda and the experimental quant kernel.
    """
    a = np.asarray(x, dtype=np.float32)
    need(a.ndim == 1 and a.size % 256 == 0 and np.isfinite(a).all(),
         "Q8 input must be finite FP32 with K divisible by 256")
    groups = a.reshape(-1, 32)
    amax = np.max(np.abs(groups), axis=1)
    d32 = np.float32(amax / np.float32(127.))
    ratio = np.divide(groups, d32[:, None],
                      out=np.zeros_like(groups), where=d32[:, None] != 0)
    qfloat = np.copysign(np.floor(np.abs(ratio) + np.float32(.5)), ratio)
    need(np.isfinite(qfloat).all() and (np.abs(qfloat) <= 127).all(),
         "Q8 quantizer overflow/incompatible input")
    q = qfloat.astype(np.int8)
    d16 = d32.astype(np.float16).astype(np.float32)
    need(np.isfinite(d16).all(), "Q8 FP16 scale overflow")
    return q, d16


def bf16_rne(x, np):
    a = np.asarray(x, dtype=np.float32).copy()
    bits = a.view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1))
            & np.uint32(0xffff0000)).view(np.float32)


def scales_array(codes, np):
    exp = codes >> 3
    mant = codes & 7
    s = np.where(exp == 0, mant.astype(np.float32) * np.float32(2**-9),
                 (np.float32(1.) + mant.astype(np.float32) / np.float32(8.))
                 * np.exp2(exp.astype(np.float32) - np.float32(7.)))
    return np.where((codes == 0) | (codes == 127), np.float32(0.), s)


def tree_reduce(values, np):
    """CUDA __shfl_down over 32 active lanes; only lane0 is used."""
    a = np.array(values, dtype=np.float32, copy=True)
    need(a.shape[-1] == 32, "expected 32-lane warp")
    for step in (16, 8, 4, 2, 1):
        prev = a.copy()
        a[..., :32-step] = np.float32(prev[..., :32-step] + prev[..., step:])
    return a[..., 0]


def project_panel(raw, q, d, x_bf16, np):
    """R rows x B blocks of raw GGUF NVFP4; entire K dimension per row.

    Q8_scalar follows per-subscale eight-product int reduction and its 4-lane
    float tree; DP4A follows per-4-int chunk, four GGUF blocks/warp, float
    accumulation and 32-lane tree. BF16 FP64 is a mathematical warp oracle,
    distinct from a bitwise CUDA FMA replay.
    """
    rows, nb, size = raw.shape
    need(size == 36 and nb > 0 and nb % 4 == 0, "invalid NVFP4 DP4A geometry")
    need(q.shape == (nb * 2, 32) and d.shape == (nb * 2,)
         and x_bf16.shape == (nb * 64,), "Q8/BF16 stage shapes differ")
    nib = raw[:, :, 4:].reshape(rows, nb, 4, 8)
    lut2 = np.array(NVFP4_X2, dtype=np.int32)
    half0 = lut2[nib & 15]
    half1 = lut2[nib >> 4]
    q16 = q.reshape(nb, 4, 16).astype(np.int32)
    dot8 = (half0 * q16[None, :, :, :8] +
            half1 * q16[None, :, :, 8:])
    dot0 = np.sum(dot8[..., :4], axis=-1, dtype=np.int32)
    dot1 = np.sum(dot8[..., 4:], axis=-1, dtype=np.int32)
    sf = np.float32(scales_array(raw[:, :, :4], np) * np.float32(.5))
    d4 = np.repeat(d.reshape(nb, 2), 2, axis=1)

    # gemv_q8_nvfp4_kernel: piece = float(dot)*(sf*0.5f)*d;
    p_scalar = np.float32(np.float32(np.float32(dot0 + dot1) * sf)
                          * d4[None, :, :])
    acc4 = np.zeros((rows, 4), dtype=np.float32)
    for b in range(nb):
        acc4 = np.float32(acc4 + p_scalar[:, b, :])
    scalar = np.float32(np.float32(acc4[:, 0] + acc4[:, 2])
                        + np.float32(acc4[:, 1] + acc4[:, 3]))

    # gemv_q8_nvfp4_dp4a_batched_kernel: sf = ue4m3/2;
    # __dp4a(two signed 4x8 groups); accum += float(dot)*(sf*d).
    dot_chunks = np.stack((dot0, dot1), axis=-1)
    sf_d = np.float32(sf * d4[None, :, :])
    chunks = np.float32(np.float32(dot_chunks) * sf_d[..., None])
    lane = np.zeros((rows, 4, 4, 2), dtype=np.float32)
    for bbase in range(0, nb, 4):
        lane = np.float32(lane + chunks[:, bbase:bbase+4, :, :])
    optimized = tree_reduce(lane.reshape(rows, 32), np)

    # Independent mathematically decoded weights: same raw bytes and BF16
    # rounded input; FP64 result is NOT a CUDA warp implementation.
    lut = np.array(NVFP4_X2, dtype=np.float32) * np.float32(.5)
    weights = np.empty((rows, nb, 4, 16), dtype=np.float32)
    weights[..., :8] = lut[nib & 15]
    weights[..., 8:] = lut[nib >> 4]
    weights = np.float32(weights * scales_array(raw[:, :, :4], np)[..., None])
    mat = weights.reshape(rows, nb * 64).astype(np.float64)
    bf16_math = (mat @ x_bf16.astype(np.float64)).astype(np.float32)
    q8_x = (q.astype(np.float32) * d[:, None]).reshape(-1)
    q8_math = (mat @ q8_x.astype(np.float64)).astype(np.float32)
    return {"q8_scalar_cpu": scalar, "q8_dp4a_cpu": optimized,
            "q8_fp64_math": q8_math, "bf16_fp64_math": bf16_math}


def warp_one_row(raw_row, x_bf16, variant):
    """Source layout emulation for six BF16 warp variants on selected rows.

    Float FMA uses rounded FP32 software arithmetic; this is CPU approximate
    even if the indexed GGUF weights and BF16 input are authentic.
    """
    need(variant in VARIANTS, "unknown BF16 warp variant")
    nb = len(raw_row) // 36
    need(nb and nb * 64 == len(x_bf16), "invalid BF16 warp geometry")

    def term(b, sub, j):
        data = raw_row[b*36 + 4 + sub*8 + j]
        k = b*64 + sub*16 + j
        return as_f32(x_bf16[k] * (NVFP4_X2[data & 15] * .5) +
                      x_bf16[k+8] * (NVFP4_X2[data >> 4] * .5))

    def fma(b, sub, t, acc):
        return as_f32(scale_ue4m3(raw_row[b*36+sub]) * term(b, sub, t) + acc)

    def reduce32(values):
        a = values[:]
        for delta in (16, 8, 4, 2, 1):
            prev = a[:]
            for i in range(32-delta):
                a[i] = as_f32(prev[i] + prev[i+delta])
        return a[0]

    if variant in ("PackedU16x2", "PackedU32x4"):
        A = 2 if variant == "PackedU16x2" else 4
        lpb = 32 // A
        a = [0.0] * 32
        for bbase in range(0, nb, A):
            for lane in range(32):
                b = bbase + lane//lpb
                if b >= nb:
                    continue
                t = lane % lpb
                sub = (t*A)//8
                pos = (t*A)%8
                for j in range(A):
                    a[lane] = fma(b, sub, pos+j, a[lane])
        return reduce32(a)

    if variant in ("Unroll2", "Warp8Rows"):
        a, c = [0.0]*32, [0.0]*32
        for b in range(0, nb, 2):
            for lane in range(32):
                sub, j = lane >> 3, lane & 7
                a[lane] = fma(b, sub, j, a[lane])
                if b + 1 < nb:
                    c[lane] = fma(b+1, sub, j, c[lane])
        return reduce32([as_f32(a[i]+c[i]) for i in range(32)])

    WPR = 2 if variant == "WarpSplit2" else 4
    partial = []
    for wid in range(WPR):
        a = [0.0]*32
        for b in range(wid, nb, WPR):
            for lane in range(32):
                a[lane] = fma(b, lane >> 3, lane & 7, a[lane])
        partial.append(reduce32(a))
    value = partial[0]
    for p in partial[1:]:
        value = as_f32(value + p)
    return value


def stats(reference, got, np):
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(got, dtype=np.float64)
    need(a.shape == b.shape and a.size > 0 and np.isfinite(a).all()
         and np.isfinite(b).all(), "invalid result comparison")
    e = b-a
    return {"elements": int(a.size), "rmse": float(np.sqrt(np.mean(e*e))),
            "max_abs": float(np.max(np.abs(e))),
            "max_error_index": int(np.argmax(np.abs(e)))}


def run_authentic(manifest_path, model_path, root):
    import numpy as np
    check_sources(root)
    need(manifest_path.is_file() and sha(manifest_path) == CAPTURE_SHA,
         "manifest hash differs from first independently authenticated llama capture")
    sys.path.insert(0, str(root / "tests"))
    from reliability_layer0_conv_q8_cpu import validate_capture
    sys.path.insert(0, str(root))
    from scripts.block0_corrected_ref import GGUF
    manifest, stages = validate_capture(manifest_path, model_path, np)
    x = stages["attn_norm-0"]
    target = stages["linear_attn_qkv_mixed-0"]
    need(x.shape == (5120,) and target.shape == (10240,),
         "independent llama stage shape incompatible with QKV projection")
    q, d = q8_quant(x, np)
    x_bf16 = bf16_rne(x, np)
    with model_path.open("rb") as physical:
        gguf = GGUF(physical, np)
        dims, typ, offset = gguf.tensors["blk.0.attn_qkv.weight"]
        need(tuple(dims) == (5120, 10240) and typ == 40,
             "GGUF first QKV must be 10240x5120 NVFP4")
        raw = np.frombuffer(gguf.mm, dtype=np.uint8, count=10240*80*36,
                            offset=gguf.data_offset+offset).reshape(10240, 80, 36)
        need(len(raw.tobytes()) == 29491200, "raw GGUF QKV byte count wrong")
        got = {name: np.empty(10240, dtype=np.float32)
               for name in ("q8_scalar_cpu", "q8_dp4a_cpu",
                            "q8_fp64_math", "bf16_fp64_math")}
        selected = (0, 1, 31, 128, 4585, 10239)
        samples = {name: {} for name in VARIANTS}
        for start in range(0, 10240, 64):
            end = min(start+64, 10240)
            panel = project_panel(raw[start:end], q, d, x_bf16, np)
            for key, value in panel.items():
                got[key][start:end] = value
            for index in selected:
                if start <= index < end:
                    block = raw[index].tobytes()
                    for name in VARIANTS:
                        samples[name][str(index)] = float(warp_one_row(block, x_bf16, name))

    result = {"schema": "h27.q8-same-capture-cpu.v1",
              "status": "CPU_ONLY_AUTHENTIC_SOURCE_CONTRACT_NOT_CUDA_PARITY",
              "source_sha256": SOURCES,
              "llama_manifest_sha256": sha(manifest_path),
              "physical_gguf_sha256": manifest["model_sha256"],
              "token_id": 6205, "position": 0, "layer": 0,
              "weight_tensor": "blk.0.attn_qkv.weight",
              "weight_dtype": "GGUF NVFP4 raw", "weight_shape": [10240, 5120],
              "quantization": "Q8_1 per32 FP32 round-away/FP16 d; BF16 RNE",
              "q8_scalar_dp4a": stats(got["q8_scalar_cpu"], got["q8_dp4a_cpu"], np),
              "vs_independent_llama": {name: stats(target, value, np)
                                       for name, value in got.items()},
              "bf16_warp_variants_sampled_rows": samples,
              "bf16_warp_sample_max_abs_vs_bf16_fp64_math": {
                  name: max(abs(value[str(row)]-float(got["bf16_fp64_math"][row]))
                            for row in selected)
                  for name, value in samples.items()},
              "limitations": ["No CUDA kernel or measured throughput",
                              "FP32 FMA/FP64 matrix oracle is approximate",
                              "BF16 warp variants emulated for six actual GGUF output rows",
                              "Only llama token 6205 position0; no MTP, head or later GDN"],
              }
    need(all(result["vs_independent_llama"][name]["max_abs"] < 1.e-3
             for name in ("q8_scalar_cpu", "q8_dp4a_cpu")),
         "Q8 CPU candidate diverges from independent llama layer0 capture")
    return result


def selftest():
    import numpy as np
    import random
    import tempfile

    x = np.zeros(256, dtype=np.float32)
    q, d = q8_quant(x, np)
    need(np.count_nonzero(q) == 0 and np.count_nonzero(d) == 0,
         "zero Q8 quantization failed")
    x[:] = 1.
    q, d = q8_quant(x, np)
    need(np.all(q == 127) and
         float(d[0] * np.float32(127.)) == 0.99993896484375,
         "Q8 FP16 stored-scale semantics wrong")
    need(scale_ue4m3(0) == 0 and scale_ue4m3(127) == 0 and
         scale_ue4m3(1) == 2**-9 and scale_ue4m3(8) == 2**-6,
         "reserved/subnormal UE4M3 semantics wrong")
    need(NVFP4_X2[1] == 1 and NVFP4_X2[7] == 12 and
         NVFP4_X2[15] == -12, "E2M1 signed mapping wrong")
    rng = random.Random(0xD4A112)
    x = np.asarray([rng.uniform(-1., 1.) for _ in range(256)], dtype=np.float32)
    q, d = q8_quant(x, np)
    bf = bf16_rne(x, np)
    raw = bytearray(rng.randrange(256) for _ in range(4*36))
    for b in range(4):
        raw[b*36:b*36+4] = bytes((0 if b==0 else 127 if b==1 else 32,
                                   1, 27, 40))
    panel = project_panel(np.frombuffer(raw, np.uint8).reshape(1, 4, 36),
                          q, d, bf, np)
    need(abs(float(panel["q8_scalar_cpu"][0]) -
             float(panel["q8_fp64_math"][0])) < 5.e-5,
         "Q8 scalar map differs from GGUF mathematical reference")
    need(abs(float(panel["q8_dp4a_cpu"][0]) -
             float(panel["q8_fp64_math"][0])) < 5.e-5,
         "Q8 DP4A map differs from GGUF mathematical reference")
    bf_math = float(panel["bf16_fp64_math"][0])
    for name in VARIANTS:
        need(abs(warp_one_row(raw, bf, name) - bf_math) < 5.e-5,
             "BF16 warp lane-map error: " + name)
    need(all(abs(warp_one_row(raw, bf, n) -
                 warp_one_row(raw, bf, n)) == 0 for n in VARIANTS),
         "BF16 software model nondeterministic")
    try:
        q8_quant(np.full(256, np.nan, dtype=np.float32), np)
    except Blocked:
        pass
    else:
        raise AssertionError("accepted nonfinite activation")
    try:
        project_panel(np.zeros((1, 3, 36), dtype=np.uint8), q, d, bf, np)
    except Blocked:
        pass
    else:
        raise AssertionError("accepted DP4A K%256 != 0")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        try:
            check_sources(root)
        except Blocked:
            pass
        else:
            raise AssertionError("accepted untrusted kernel source")
    print("SELFTEST_PASS: 6 BF16 variants, Q8 scalar/DP4A, signed nibble, "
          "zero/reserved scales and rejection controls; CPU only")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--gguf", type=Path)
    ap.add_argument("--source-root", type=Path)
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return 0
    if any(v is None for v in (args.manifest, args.gguf, args.source_root)):
        ap.error("--manifest --gguf --source-root required for genuine replay")
    report = run_authentic(args.manifest, args.gguf, args.source_root)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Blocked, ImportError, OSError, ValueError, KeyError) as exc:
        print("FAIL_CLOSED_Q8_SAME_CAPTURE: " + str(exc), file=sys.stderr)
        raise SystemExit(2)
