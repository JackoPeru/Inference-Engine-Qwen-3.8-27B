#!/usr/bin/env python3
"""Independent CPU/NumPy block-0 oracle for the current GGUF semantics.

Generates a versioned, checkpoint-bound artifact for test_p1_block0.cu.
No GPU, CUDA, torch, or old /tmp/e1_*.bin inputs are used.
"""
import argparse
import hashlib
import mmap
import os
import struct
import tempfile
from pathlib import Path

MAGIC = b"H27B0V2!"
IDS = (6205, 161421, 303, 220, 17, 1374, 9911, 1147, 27053, 6059, 13)
HID, FFN, T = 5120, 17408, len(IDS)
KD, VDD, HV, KHD = 2048, 6144, 48, 128
C = 2 * KD + VDD
SHAPES = (
    ("qkv", (T, C)),
    ("cv", (T, C)),
    ("beta", (T, HV)),
    ("gl", (T, HV)),
    ("o", (T, HV, KHD)),
    ("core", (T, HID)),
    ("y", (T, HID)),
)
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211


def fingerprint(mm):
    """Stable 64-bit identity check shared with test_p1_block0.cu."""
    size = len(mm)
    value = FNV_OFFSET
    for data in (mm[:min(4096, size)], mm[max(0, size - 4096):]):
        for byte in data:
            value = ((value ^ byte) * FNV_PRIME) & ((1 << 64) - 1)
    return value


class GGUF:
    def __init__(self, file, np):
        self.file = file
        self.mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        self.np = np
        self.pos = 0
        if self.take(4) != b"GGUF":
            raise ValueError("invalid GGUF magic")
        if self.u32() != 3:
            raise ValueError("only GGUF v3 supported")
        nt, nkv = self.u64(), self.u64()
        if nt > 5000 or nkv > 10000:
            raise ValueError("implausible GGUF counts")
        alignment = 32
        for _ in range(nkv):
            key, typ = self.string(), self.u32()
            value = self.kv_value(typ)
            if key == "general.alignment":
                alignment = int(value)
        if alignment < 1 or alignment > 4096 or alignment & (alignment - 1):
            raise ValueError(f"unsupported GGUF alignment: {alignment}")
        self.tensors = {}
        for _ in range(nt):
            key = self.string()
            nd = self.u32()
            if nd < 1 or nd > 4:
                raise ValueError(f"bad tensor rank: {key} rank={nd}")
            dims = tuple(self.u64() for _ in range(nd))
            typ, offset = self.u32(), self.u64()
            self.tensors[key] = (dims, typ, offset)
        self.data_offset = (self.pos + alignment - 1) // alignment * alignment

    def take(self, n):
        end = self.pos + n
        if n < 0 or end > len(self.mm):
            raise ValueError("truncated GGUF")
        result = self.mm[self.pos:end]
        self.pos = end
        return result

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def string(self):
        n = self.u64()
        return self.take(n).decode("utf-8")

    def kv_value(self, typ):
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if typ == 8:
            return self.string()
        if typ == 9:
            item_type, count = self.u32(), self.u64()
            if count > 1_000_000_000:
                raise ValueError("implausible GGUF metadata array")
            if item_type == 8:
                for _ in range(count):
                    self.string()
            elif item_type in sizes:
                self.take(count * sizes[item_type])
            else:
                raise ValueError(f"unknown GGUF array item type {item_type}")
            return None
        if typ not in sizes:
            raise ValueError(f"unknown GGUF metadata type {typ}")
        data = self.take(sizes[typ])
        if typ in (4, 5):
            return struct.unpack("<I" if typ == 4 else "<i", data)[0]
        return None

    def tensor(self, name, kind):
        np = self.np
        dims, typ, offset = self.tensors[name]
        expected_type = {"f32": 0, "bf16": 30, "nvfp4": 40}[kind]
        if typ != expected_type:
            raise ValueError(f"{name}: expected GGUF type {expected_type}, got {typ}")
        n = 1
        for dim in dims:
            n *= dim
        base = self.data_offset + offset
        if kind == "f32":
            return np.frombuffer(self.mm, dtype="<f4", count=n, offset=base).reshape(dims[::-1]).copy()
        if kind == "bf16":
            return np.frombuffer(self.mm, dtype="<u2", count=n, offset=base).reshape(dims[::-1])
        if len(dims) != 2 or dims[0] % 64:
            raise ValueError(f"{name}: invalid NVFP4 dimensions {dims}")
        ne0, ne1 = dims
        raw = np.frombuffer(self.mm, dtype=np.uint8, count=ne1 * (ne0 // 64) * 36,
                            offset=base).reshape(ne1, ne0 // 64, 36)
        scales = raw[:, :, :4]
        exp = (scales >> 3) & 15
        mantissa = scales & 7
        value = np.where(exp == 0, mantissa.astype(np.float32) * (2.0 ** -9),
                         (1.0 + mantissa.astype(np.float32) / 8.0) *
                         np.exp2(exp.astype(np.float32) - 7.0))
        scale = np.where((scales == 0) | (scales == 127), 0.0, value)
        lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
                       dtype=np.float32)
        packed = raw[:, :, 4:].reshape(ne1, ne0 // 64, 4, 8)
        decoded = np.empty((ne1, ne0 // 64, 4, 16), dtype=np.float32)
        decoded[..., :8] = lut[packed & 15]
        decoded[..., 8:] = lut[packed >> 4]
        return np.ascontiguousarray((decoded * scale[..., None]).reshape(ne1, ne0))


def bf16_to_f32(words, np):
    return (words.astype(np.uint32) << 16).view(np.float32)


def rms(x, weights, np):
    return x * np.reciprocal(np.sqrt(np.mean(x * x, axis=-1, keepdims=True) +
                                       np.float32(1e-6))) * weights


def sigmoid(x, np):
    return np.reciprocal(np.float32(1.) + np.exp(-x))


def silu(x, np):
    return x * sigmoid(x, np)


def reference(gguf, np):
    p = "blk.0."
    embed, = (gguf.tensor("token_embd.weight", "bf16"),)
    x = bf16_to_f32(embed[list(IDS)], np)
    h = rms(x, gguf.tensor(p + "attn_norm.weight", "f32"), np).astype(np.float32)
    del embed

    qkv = h @ gguf.tensor(p + "attn_qkv.weight", "nvfp4").T
    convw = gguf.tensor(p + "ssm_conv1d.weight", "f32").reshape(C, 4)
    v = np.zeros_like(qkv, dtype=np.float32)
    for j in range(4):
        lag = 3 - j
        if lag:
            v[lag:] += qkv[:-lag] * convw[None, :, j]
        else:
            v += qkv * convw[None, :, j]
    cv = silu(v, np).astype(np.float32)
    del v, convw

    z = (h @ gguf.tensor(p + "attn_gate.weight", "nvfp4").T).reshape(T, HV, KHD)
    beta = sigmoid(h @ gguf.tensor(p + "ssm_beta.weight", "nvfp4").T, np)
    alpha = h @ gguf.tensor(p + "ssm_alpha.weight", "f32").T
    dt = gguf.tensor(p + "ssm_dt.bias", "f32")
    A = gguf.tensor(p + "ssm_a", "f32")
    if A.shape != (HV,) or not np.all(np.isfinite(A)) or np.any(A > 0):
        raise ValueError("GGUF ssm_a must already be nonpositive -exp(A_log)")
    t = alpha + dt
    softplus = np.maximum(t, 0) + np.log1p(np.exp(-np.abs(t)))
    gl = A * softplus

    # Q/K: original 16 K groups repeated three times to 48 V heads, hh % 16.
    q = np.tile(cv[:, :KD].reshape(T, KD // KHD, KHD), (1, 3, 1))
    k = np.tile(cv[:, KD:2 * KD].reshape(T, KD // KHD, KHD), (1, 3, 1))
    values = cv[:, 2 * KD:].reshape(T, HV, KHD)
    qn = q * np.reciprocal(np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6))
    qn *= np.float32(1.0 / np.sqrt(float(KHD)))
    kn = k * np.reciprocal(np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6))
    state = np.zeros((HV, KHD, KHD), dtype=np.float32)
    out = np.empty((T, HV, KHD), dtype=np.float32)
    for i in range(T):
        state *= np.exp(gl[i])[:, None, None]
        updated = (values[i] - np.sum(state * kn[i, :, :, None], axis=1)) * beta[i, :, None]
        state += kn[i, :, :, None] * updated[:, None, :]
        out[i] = np.sum(state * qn[i, :, :, None], axis=1)

    gn = rms(out, gguf.tensor(p + "ssm_norm.weight", "f32"), np)
    gated = (gn * silu(z, np)).reshape(T, VDD)
    core = gated @ gguf.tensor(p + "ssm_out.weight", "nvfp4").T

    residual = x + core
    h2 = rms(residual, gguf.tensor(p + "post_attention_norm.weight", "f32"), np)
    gate = h2 @ gguf.tensor(p + "ffn_gate.weight", "nvfp4").T
    up = h2 @ gguf.tensor(p + "ffn_up.weight", "nvfp4").T
    y = residual + (silu(gate, np) * up) @ gguf.tensor(p + "ffn_down.weight", "nvfp4").T
    result = {"qkv": qkv, "cv": cv, "beta": beta, "gl": gl,
              "o": out, "core": core, "y": y}
    for name, shape in SHAPES:
        arr = result[name]
        if arr.shape != shape or not np.isfinite(arr).all():
            raise ValueError(f"{name}: bad shape/nonfinite: {arr.shape}")
    return result


def write_oracle(path, gguf, arrays, np):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack("<8sQQ11i", MAGIC, len(gguf.mm), fingerprint(gguf.mm), *IDS)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(header)
            for name, _ in SHAPES:
                raw = np.asarray(arrays[name], dtype="<f4", order="C").tobytes()
                out.write(raw)
                print(f"{name:5s} size={len(raw):8d} sha256={hashlib.sha256(raw).hexdigest()[:16]}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"CORRECTED ORACLE READY: {path} ({path.stat().st_size} bytes; CPU GGUF-only)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, default=Path(
        "/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf"))
    parser.add_argument("--out", type=Path, default=Path("/tmp/h27_block0_corrected_v2.bin"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error("--threads must be in 1..16")
    # CPU-only and bounded BLAS worker threads to coexist with other workloads.
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(args.threads)
    import numpy as np
    with args.gguf.open("rb") as file:
        gguf = GGUF(file, np)
        arrays = reference(gguf, np)
        write_oracle(args.out, gguf, arrays, np)
        gguf.mm.close()


if __name__ == "__main__":
    main()
