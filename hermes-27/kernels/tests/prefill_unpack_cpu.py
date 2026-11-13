#!/usr/bin/env python3
"""CPU-only GGUF NVFP4->BF16 vector unpack and tiled GEMM layout proof.

No CUDA, torch, cuBLAS, model file, service or third-party Python packages.
This checks the shuffle algebra and the exact matrix address/stride contract
against an independent reference, not the compiled GPU binary's execution.
"""
from __future__ import annotations

import math
import random
import struct


E2M1 = (0., .5, 1., 1.5, 2., 3., 4., 6.,
        0., -.5, -1., -1.5, -2., -3., -4., -6.)


def ue4m3(code: int) -> float:
    if code == 0 or code == 0x7F:
        return 0.
    e = code >> 3
    m = code & 7
    return math.ldexp(float(m), -9) if e == 0 else math.ldexp(1 + m / 8, e - 7)


def bf16_bits(value: float) -> int:
    u = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF


def bf16_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def reference_decode(block: bytes) -> list[int]:
    assert len(block) == 36
    output = [0] * 64
    for sub in range(4):
        sf = ue4m3(block[sub])
        for j in range(8):
            packed = block[4 + sub * 8 + j]
            output[sub * 16 + j] = bf16_bits(sf * E2M1[packed & 15])
            output[sub * 16 + j + 8] = bf16_bits(sf * E2M1[packed >> 4])
    return output


def vector_shuffle_decode(block: bytes) -> list[int]:
    """Simulate exactly CUDA warp32 lane/scale shfl mapping, CPU only."""
    assert len(block) == 36
    # Every warp lane loads byte 4+lane, broadcasts 4 scale bytes from
    # packed uint32 and writes one contiguous pair of BF16 values.
    packed_scale = int.from_bytes(block[:4], "little")
    lanes = list(block[4:])
    out = [0] * 64
    for lane in range(32):
        sub = lane >> 3
        pair = lane & 7
        src_lane = sub * 8 + (pair & 3) * 2
        q0 = lanes[src_lane]
        q1 = lanes[src_lane + 1]
        if pair >= 4:
            q0 >>= 4
            q1 >>= 4
        sf = ue4m3((packed_scale >> (sub * 8)) & 0xFF)
        out[2 * lane] = bf16_bits(sf * E2M1[q0 & 15])
        out[2 * lane + 1] = bf16_bits(sf * E2M1[q1 & 15])
    return out


def tiled_compute(xs: list[list[float]], weights: list[list[float]],
                  cap_rows: int) -> list[float]:
    """Simulate cuBLAS column-major N-stride Y with row-tiling BF16 W."""
    t_count = len(xs)
    n_count = len(weights)
    output = [math.nan] * (t_count * n_count)
    row_step = min(cap_rows, n_count)
    if row_step >= 128:
        row_step = row_step // 128 * 128
    assert row_step >= 1
    for base in range(0, n_count, row_step):
        for n in range(base, min(n_count, base + row_step)):
            for t in range(t_count):
                # cuBLAS output with ldc=N for column major [N,T], same
                # bytes as row-major [T,N] and base offset in each column.
                output[t * n_count + n] = math.fsum(
                    x * w for x, w in zip(xs[t], weights[n]))
    return output


def main() -> None:
    rng = random.Random(20261113)
    codes = 0
    for sf in range(256):
        for q in E2M1:
            v = ue4m3(sf) * q
            assert bf16_float(bf16_bits(v)) == v, (sf, q)
            codes += 1
    print(f"PASS GGUF UE4M3 x E2M1 -> BF16 exactly: {codes} code pairs")

    blocks = 0
    for scenario in ("zero", "max", "random"):
        for _ in range(2500):
            if scenario == "zero":
                block = bytes([0] * 36)
            elif scenario == "max":
                block = bytes([126] * 4 + [255] * 32)
            else:
                block = bytes(rng.randrange(256) for _ in range(36))
            expected = reference_decode(block)
            actual = vector_shuffle_decode(block)
            assert actual == expected, (scenario, expected, actual)
            blocks += 1
    print(f"PASS warp32 nibble/repacking bijection: {blocks} GGUF blocks")

    shapes = 0
    for N in (1, 7, 128, 130):
        K = 64
        xs = [[bf16_float(bf16_bits(rng.uniform(-2, 2)))
               for _ in range(K)] for _ in range(4)]
        weights = [[bf16_float(bf16_bits(rng.uniform(-1, 1)))
                    for _ in range(K)] for _ in range(N)]
        expected = tiled_compute(xs, weights, N)
        for cap in (1, 2, 3, 7, 127, 128, 129, 256):
            assert tiled_compute(xs, weights, cap) == expected, (N, cap)
            shapes += 1
    print(f"PASS column-major ldc=N tiled result and tail: {shapes} shapes/caps")

    N = 10240
    K = 5120
    for mib in (8, 32, 105, 178):
        cap = mib << 20
        rows = min(N, cap // (2 * K))
        if rows >= 128:
            rows = (rows // 128) * 128
        assert rows >= 1
        tiles = (N + rows - 1) // rows
        assert rows * 2 * K <= cap
        print(f"SCRATCH_PLAN capacity={mib}MiB, BF16_rows_per_tile={rows}, "
              f"sequential_tiles={tiles}, no_persistent_BF16_weights")
    print("CPU_ONLY_VALIDATION_PASS (compiled CUDA and runtime still need tests)")


if __name__ == "__main__":
    main()
