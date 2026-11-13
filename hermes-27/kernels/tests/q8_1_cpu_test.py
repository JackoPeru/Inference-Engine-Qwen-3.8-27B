#!/usr/bin/env python3
"""CPU-only deterministic Q8_1 activation + raw GGUF NVFP4 contract tests.

Independent of CUDA runtime and llama model. Validates CUDA source formula
and packed block layout using Python standard library (no NumPy).
It does not execute CUDA and must not be presented as GPU validation.
"""
from __future__ import annotations

import math
import random
import struct

E2 = (0., .5, 1., 1.5, 2., 3., 4., 6.,
      0., -.5, -1., -1.5, -2., -3., -4., -6.)
E2_X2 = (0, 1, 2, 3, 4, 6, 8, 12,
         0, -1, -2, -3, -4, -6, -8, -12)


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f16bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def from_f16(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def gguf_scale(b: int) -> float:
    if b == 0 or b == 127:
        return 0.
    return (math.ldexp(float(b & 7), -9) if b >> 3 == 0 else
            math.ldexp(1. + (b & 7)/8., (b >> 3) - 7))


def cuda_away_from_zero(x: float) -> int:
    if not math.isfinite(x):
        raise ValueError("invalid quantization value")
    return int(math.copysign(math.floor(abs(x)+0.5), x))


def quant_ref(x: list[float]) -> bytes:
    assert len(x) == 32 and all(math.isfinite(t) for t in x)
    x = [f32(v) for v in x]
    maxval = max(abs(a) for a in x)
    # f32(a/127), not FP16-rounded d, must be used to compute q8 values.
    d_f32 = f32(maxval / 127.)
    q = [0 if maxval == 0 else
         cuda_away_from_zero(f32(f32(v) / d_f32)) for v in x]
    assert all(-127 <= v <= 127 for v in q)
    # Exact ggml_cuda warp_reduce_sum<float> implementation uses XOR.
    acc = x[:]
    for step in (16, 8, 4, 2, 1):
        acc = [f32(acc[i] + acc[i ^ step])
               for i in range(32)]
    return (struct.pack("<HH", f16bits(d_f32), f16bits(acc[0]))
            + struct.pack("<32b", *q))


def quant_independent(x: list[float]) -> tuple[list[int], float, float]:
    x = [f32(v) for v in x]
    amax = max(map(abs, x))
    d = f32(amax/127.)
    q = [0 if amax == 0 else
         (1 if val >= 0 else -1) * int(math.floor(abs(f32(val/d))+.5))
         for val in x]
    # The sum is not used by NVFP4 dot; the source stores it for Q8_1
    # generally, but the reference numeric projection depends only on d.
    return q, from_f16(f16bits(d)), math.fsum(x)


def gguf_decoded_dot(q8: bytes, w: bytes) -> float:
    """FP64 independent reference for one 64-weight GGUF block."""
    assert len(q8) == 72 and len(w) == 36
    res = 0.0
    for k in range(64):
        bi = k // 32
        d = from_f16(struct.unpack_from("<H", q8, bi*36)[0])
        qi = struct.unpack_from("<b", q8, bi*36+4+k % 32)[0]
        s = k // 16
        j = k % 16
        packed = w[4+s*8+(j % 8)]
        fp4 = E2[(packed >> 4) if j >= 8 else (packed & 15)]
        res += (d * qi) * (gguf_scale(w[s]) * fp4)
    return res


def gguf_integer_groups(q8: bytes, w: bytes) -> float:
    """Same integer grouping as the CUDA kernel, before FP32 reductions."""
    total = 0.0
    for s in range(4):
        d = from_f16(struct.unpack_from("<H", q8, (s//2)*36)[0])
        scale = gguf_scale(w[s])
        iqs = (s % 2)*16
        subtotal = 0
        for j in range(8):
            q0 = struct.unpack_from("<b", q8, (s//2)*36+4+iqs+j)[0]
            q1 = struct.unpack_from("<b", q8, (s//2)*36+4+iqs+j+8)[0]
            packed = w[4+s*8+j]
            subtotal += q0*E2_X2[packed & 15] + q1*E2_X2[packed >> 4]
        total += subtotal * (scale*.5) * d
    return total


def main() -> None:
    rng = random.Random(20261113)
    num_blocks = 0
    tests: list[list[float]] = [
        [0.]*32,
        [-0.]*32,
        [float(i-16)*.25 for i in range(32)],
        [float((i%5)-2) for i in range(32)],
        [1.0]+[0.]*31,
        [-1.0]+[0.]*31,
        [float(i%2)*(-1. if i%4 == 0 else 1.) for i in range(32)],
    ]
    for _ in range(3000):
        xs = [f32(rng.uniform(-2, 2)) for _ in range(32)]
        tests.append(xs)
    for xs in tests:
        packed = quant_ref(xs)
        assert len(packed) == 36
        q, stored_d, _ = quant_independent(xs)
        assert list(struct.unpack("<32b", packed[4:])) == q
        assert from_f16(struct.unpack_from("<H", packed)[0]) == stored_d
        num_blocks += 1
    print(f"PASS Q8_1 scale/round-away/packed 36B blocks: {num_blocks}")

    combos = 0
    for _ in range(1000):
        xs = [f32(rng.uniform(-3, 3)) for _ in range(64)]
        q8 = quant_ref(xs[:32]) + quant_ref(xs[32:])
        w = bytes(rng.randrange(256) for _ in range(36))
        actual = gguf_integer_groups(q8, w)
        expected = gguf_decoded_dot(q8, w)
        assert math.isclose(actual, expected, abs_tol=2e-10, rel_tol=2e-12), (
            actual, expected)
        combos += 1
    print(f"PASS E2M1*2 integer exact dot vs FP64 GGUF decoder: {combos} blocks")

    cases = [(0, 64), (1, 64), (16, 5120), (64, 5120), (256, 5120)]
    for T, K in cases:
        scratch = T*(K//32)*36 if T and K%64 == 0 else 0
        if T == 64:
            assert scratch == 368640
        print(f"Q8_1_TEMP T={T} K={K} bytes={scratch}")
    print("CPU_ONLY_Q8_1_PASS (CUDA runtime/DP4A/llama parity still unverified)")


if __name__ == "__main__":
    main()
