#!/usr/bin/env python3
"""Layer-0 numerical replay using an independently captured llama.cpp tensor.

Default CPU-only. GPU replay is an explicit opt-in after the prime grants GPU
exclusivity; CUDA binary is a private subordinate of THIS manifest verifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np

SCHEMA = "h27.reliability-layer-cb.v1"
INPUT = "attn_norm-0"
OUTPUT = "linear_attn_qkv_mixed-0"
CONV = "conv_output_silu-0"
TENSOR = "blk.0.attn_qkv.weight"
H = 5120
F = 10240
GGUF_TYPE_NVFP4 = 40
CHUNK = 32
WEIGHT_LUT = np.array(
    [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
    dtype=np.float64)
SCALE_LUT = np.array([
    0.0 if q in (0, 0x7f) else
    math.ldexp(float(q & 7), -9) if (q >> 3) == 0 else
    math.ldexp(float(8 + (q & 7)), (q >> 3) - 10)
    for q in range(128)
], dtype=np.float64)


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(detail)


def json_object_no_duplicate(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for k, v in pairs:
        require(k not in result, f"duplicate JSON key {k}")
        result[k] = v
    return result


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_sha(value: object, description: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value),
            f"{description}: missing/noncanonical SHA-256")
    return value


def read_manifest(directory: Path, model_override: Path | None,
                  exporter: Path | None):
    path = directory / "manifest.json"
    require(path.is_file(), f"missing manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"),
                          object_pairs_hook=json_object_no_duplicate)
    require(isinstance(manifest, dict), "manifest must be JSON object")
    require(manifest.get("schema") == SCHEMA, "unexpected capture manifest schema")
    require(manifest.get("status") == "CAPTURED_NOT_INDEPENDENTLY_COMPARED",
            "manifest changed status; review and update explicit contract")
    require(manifest.get("backend") == "llama.cpp", "capture is not llama.cpp")
    require(manifest.get("tensor_data_format") == "little-endian-fp32-ne0-fastest",
            "unexpected tensor encoding/strides")
    require(manifest.get("layer") == 0 and manifest.get("absolute_position") == 0,
            "this harness supports layer0 / absolute position 0 only")
    require(manifest.get("prompt_ids") == [6205], "unexpected token ID / prompt")
    require(manifest.get("greedy_argmax") == 161421,
            "unexpected captured continuation; check provenance")
    require(manifest.get("n_gpu_layers") == 80, "unexpected llama layer placement")
    require(manifest.get("callback_ask_count") == 3 and
            manifest.get("callback_ready_count") == 3,
            "incomplete callback capture")
    require(isinstance(manifest.get("tensors"), list) and
            len(manifest["tensors"]) == 3, "exactly 3 tensor capture entries required")
    expected = {INPUT: (H, H * 4), OUTPUT: (F, F * 4), CONV: (F, F * 4)}
    tensors = {}
    for t in manifest["tensors"]:
        require(isinstance(t, dict), "tensor manifest entry must be object")
        name = t.get("name")
        require(isinstance(name, str) and name in expected and name not in tensors,
                "missing, duplicate, or unexpected tensor capture")
        elements, nbytes = expected[name]
        require(t.get("file") == name + ".f32" and
                t.get("source_type") == "f32" and
                t.get("buffer") == "CUDA0" and
                t.get("shape") == [elements, 1, 1, 1] and
                t.get("source_strides_bytes") ==
                [4, nbytes, nbytes, nbytes] and
                t.get("elements") == elements and t.get("bytes") == nbytes,
                f"unexpected {name} buffer shape/bytes/strides")
        source = directory / t["file"]
        require(source.is_file() and source.stat().st_size == nbytes,
                f"{name} capture missing or truncated")
        digest = checked_sha(t.get("sha256"), f"{name} SHA")
        require(sha256(source) == digest, f"{name} SHA mismatch")
        tensors[name] = source.read_bytes()
    require(set(tensors) == set(expected), "missing tensor entry")
    recorded_model = Path(manifest.get("model_path", ""))
    require(recorded_model.is_file(), "manifest model path missing or invalid")
    model = model_override or recorded_model
    require(model.is_file() and os.path.samefile(model, recorded_model),
            "specified GGUF does not refer to manifest's exact model file")
    model_digest = checked_sha(manifest.get("model_sha256"), "GGUF SHA")
    require(sha256(model) == model_digest, "entire GGUF differs from llama capture")
    lib = Path(manifest.get("libllama_path", ""))
    require(lib.is_file(), "manifest libllama path is unavailable")
    lib_digest = checked_sha(manifest.get("libllama_sha256"), "libllama SHA")
    require(sha256(lib) == lib_digest, "llama shared-library SHA mismatch")
    exporter_digest = checked_sha(manifest.get("exporter_sha256"), "exporter SHA")
    if exporter is not None:
        require(exporter.is_file() and sha256(exporter) == exporter_digest,
                "exporter hash cannot be verified")
    print(f"CAPTURE_VERIFIED manifest={path} layer0 position0 prompt_ids=[6205] "
          f"backend=llama.cpp model_sha256={model_digest}")
    for name in (INPUT, OUTPUT, CONV):
        print(f"CAPTURE_VERIFIED {name} sha256={sha256(directory / (name + '.f32'))} "
              f"bytes={len(tensors[name])}")
    print(f"CAPTURE_VERIFIED libllama_sha256={lib_digest}")
    if exporter is None:
        print(f"EXPORTER_PROVENANCE_SHA recorded={exporter_digest} "
              "not independently verified: manifest has no exporter path")
    else:
        print(f"CAPTURE_VERIFIED exporter_sha256={exporter_digest}")
    return manifest, model, tensors


class GGUFReader:
    def __init__(self, model: Path):
        self.stream = model.open("rb")

    def close(self):
        self.stream.close()

    def read(self, count: int):
        require(0 <= count <= (1 << 29), "unreasonable GGUF read length")
        data = self.stream.read(count)
        require(len(data) == count, "truncated GGUF metadata")
        return data

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self):
        n = self.u64()
        require(n < 32 * 1024 * 1024, "GGUF string too large")
        return self.read(n).decode("utf-8")

    def skip_value(self, kind: int, depth: int = 0):
        sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4,
                 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        require(depth <= 1, "GGUF nested metadata arrays unsupported")
        if kind in sizes:
            self.read(sizes[kind])
        elif kind == 8:
            self.string()
        elif kind == 9:
            t = self.u32()
            n = self.u64()
            require(n < 10_000_000 and t != 9, "invalid metadata array")
            if t in sizes:
                self.stream.seek(n * sizes[t], os.SEEK_CUR)
            elif t == 8:
                for _ in range(n):
                    self.string()
            else:
                require(False, "unsupported metadata array element type")
        else:
            require(False, f"unsupported GGUF metadata value type {kind}")


def gguf_tensor(model: Path):
    reader = GGUFReader(model)
    try:
        require(reader.read(4) == b"GGUF", "invalid GGUF magic")
        require(reader.u32() == 3, "GGUF must be v3")
        nt, nk = reader.u64(), reader.u64()
        require(0 < nt < 5000 and nk < 10000, "implausible GGUF tensor/KV counts")
        alignment = 32
        for _ in range(nk):
            key, kind = reader.string(), reader.u32()
            if key == "general.alignment":
                require(kind in (4, 10), "invalid GGUF alignment type")
                alignment = reader.u32() if kind == 4 else reader.u64()
            else:
                reader.skip_value(kind)
        require(0 < alignment <= 256, "GGUF invalid alignment")
        selected = None
        for _ in range(nt):
            name = reader.string()
            ndim = reader.u32()
            require(1 <= ndim <= 4, "bad GGUF dimension count")
            dims = tuple(reader.u64() for _ in range(ndim))
            dtype, offset = reader.u32(), reader.u64()
            if name == TENSOR:
                require(selected is None, "duplicate selected tensor in GGUF")
                selected = (dims, dtype, offset)
        require(selected is not None, f"GGUF has no tensor {TENSOR}")
        dims, dtype, offset = selected
        require(dims == (H, F) and dtype == GGUF_TYPE_NVFP4,
                f"tensor shape/type wrong: GGUF ne={dims} type={dtype}")
        data_offset = ((reader.stream.tell() + alignment - 1) // alignment) * alignment
        blocks = H // 64
        nbytes = F * blocks * 36
        absolute_offset = data_offset + offset
        require(offset % alignment == 0 and absolute_offset >= data_offset and
                absolute_offset + nbytes <= model.stat().st_size,
                "selected GGUF tensor is out of file/alignment bounds")
        raw = np.memmap(model, dtype=np.uint8, mode="r",
                        offset=absolute_offset, shape=(F, blocks, 36))
        max_scale = int(raw[:, :, :4].max())
        require(max_scale < 128, "noncanonical UE4M3 high-bit scale in actual tensor")
        print(f"GGUF_TENSOR_VERIFIED {TENSOR} shape=({F},{H}) type=NVFP4 "
              f"offset={absolute_offset} bytes={nbytes} max_scale_byte={max_scale}")
        return raw, absolute_offset
    finally:
        reader.close()


def f32_to_bf16_rne(x: np.ndarray):
    require(x.dtype == np.dtype("<f4") and x.shape == (H,),
            "layer0 activation must be little-endian FP32 [5120]")
    require(bool(np.isfinite(x).all()), "nonfinite llama input activation")
    bits = x.view(np.dtype("<u4"))
    rounded = ((bits.astype(np.uint64) + 0x7fff +
               ((bits >> np.uint32(16)) & np.uint32(1))) >> np.uint64(16))
    bf16 = rounded.astype("<u2")
    bf16_f32 = ((bf16.astype("<u4")) << np.uint32(16)).view("<f4")
    require(bool(np.isfinite(bf16_f32).all()), "BF16 input overflow/NaN")
    return bf16, bf16_f32


def report_error(title: str, a: np.ndarray, b: np.ndarray):
    require(a.shape == b.shape == (F,), "numeric comparison dimensions wrong")
    delta = a.astype(np.float64) - b.astype(np.float64)
    worst = int(np.argmax(np.abs(delta)))
    rms = float(np.sqrt(np.mean(delta * delta)))
    ref_rms = float(np.sqrt(np.mean(b.astype(np.float64) ** 2)))
    print(f"{title} full_rows={F} max_abs={abs(delta[worst]):.12g} "
          f"rmse={rms:.12g} rel_rmse={rms / max(ref_rms, 1.e-30):.12g} "
          f"worst_row={worst} at_a={a[worst]:.12g} at_b={b[worst]:.12g}")


def cpu_reference(raw, original: np.ndarray, rounded: np.ndarray,
                  llama_qkv: np.ndarray):
    """Read exact GGUF scale/nibble bytes; sum in FP64 on all 10240 rows."""
    B = H // 64
    f32dot = np.empty(F, dtype=np.float64)
    bf16dot = np.empty(F, dtype=np.float64)
    fp32serial_bf16 = np.empty(F, dtype=np.float32)
    max_abs_sum = 0.0
    for start in range(0, F, CHUNK):
        stop = min(F, start + CHUNK)
        c = stop - start
        blocks = raw[start:stop]
        scales = blocks[:, :, :4]
        # Element for GGUF sub-block s, byte j: low nibble at k=16*s+j,
        # high nibble at k=16*s+j+8.
        q = blocks[:, :, 4:].reshape(c, B, 4, 8)
        weights = np.empty((c, B, 4, 16), dtype=np.float64)
        rs = SCALE_LUT[scales][:, :, :, np.newaxis]
        weights[:, :, :, :8] = WEIGHT_LUT[q & 15] * rs
        weights[:, :, :, 8:] = WEIGHT_LUT[q >> 4] * rs
        matrix = weights.reshape(c, H)
        p32 = matrix * original.astype(np.float64)
        pb = matrix * rounded.astype(np.float64)
        f32dot[start:stop] = np.sum(p32, axis=1, dtype=np.float64)
        bf16dot[start:stop] = np.sum(pb, axis=1, dtype=np.float64)
        # Pure diagnostic: a strictly serial FP32 accumulation in k order.
        # This is not a model of llama.cpp's integer-dot or HERMES warp tree.
        float_products = matrix.astype(np.float32) * rounded
        fp32serial_bf16[start:stop] = np.add.accumulate(
            float_products, axis=1, dtype=np.float32)[:, -1]
        max_abs_sum = max(max_abs_sum, float(np.max(np.sum(np.abs(pb),
                                                            axis=1, dtype=np.float64))))
    require(bool(np.isfinite(f32dot).all()) and bool(np.isfinite(bf16dot).all())
            and bool(np.isfinite(fp32serial_bf16).all()),
            "nonfinite CPU dot result")
    report_error("CPU_FP64_DOT_BF16_vs_FP32_INPUT", bf16dot, f32dot)
    report_error("CPU_FP32_SERIAL_vs_CPU_FP64_BF16", fp32serial_bf16, bf16dot)
    report_error("LLAMA_CAPTURE_vs_CPU_FP64_F32_INPUT", llama_qkv, f32dot)
    report_error("LLAMA_CAPTURE_vs_CPU_FP64_BF16_INPUT", llama_qkv, bf16dot)
    best_ref = int(np.argmax(llama_qkv))
    print(f"CPU_DIAGNOSTIC completed_rows={F} captured_qkv_argmax={best_ref} "
          f"max_sum_abs_products={max_abs_sum:.12g} "
          "LLAMA_EQUIVALENCE=UNPROVEN input capture and CUDA operator semantics require review")
    return f32dot, bf16dot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-dir", type=Path,
                   default=Path("/tmp/h27-layer0-llama-pos0-prime3"))
    p.add_argument("--model", type=Path)
    p.add_argument("--exporter", type=Path,
                   help="optional exporter executable to verify recorded exporter SHA")
    p.add_argument("--mode", choices=("cpu", "gpu"), default="cpu")
    p.add_argument("--gpu-exe", type=Path,
                   help="compiled gemv_layer0_compare.cu; GPU mode only")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--execute-gpu", action="store_true",
                   help="explicit action after prime grants exclusive GPU; never implicit")
    args = p.parse_args()
    try:
        if args.mode == "gpu":
            require(args.execute_gpu and args.gpu_exe is not None and
                    args.gpu_exe.is_file(), "GPU requires --execute-gpu and --gpu-exe")
            require(0 <= args.device < 128, "bad GPU device ID")
        else:
            require(not args.execute_gpu and args.gpu_exe is None and args.device == 0,
                    "CPU mode cannot use CUDA/GPU options")
        manifest, model, data = read_manifest(args.capture_dir, args.model, args.exporter)
        raw, start = gguf_tensor(model)
        orig = np.frombuffer(data[INPUT], dtype="<f4")
        llama = np.frombuffer(data[OUTPUT], dtype="<f4")
        require(orig.size == H and llama.size == F and
                bool(np.isfinite(llama).all()), "bad captured projection values")
        bf16_bits, rounded = f32_to_bf16_rne(orig)
        print(f"BF16_RNE_POS0 values={H} sha256={hashlib.sha256(bf16_bits.tobytes()).hexdigest()} "
              f"max_abs_input_rounding={np.max(np.abs(orig.astype(np.float64) - rounded.astype(np.float64))):.12g}")
        cpu_reference(raw, orig, rounded, llama)
        # Verify no manifest data changed while full GGUF/CPU processing ran.
        for item in manifest["tensors"]:
            require(sha256(args.capture_dir / item["file"]) == item["sha256"],
                    "capture changed during CPU replay")
        if args.mode == "cpu":
            print("CPU_ONLY_PASS all metadata/hash/input/FP64 checks; GPU_NOT_EXECUTED")
            return 0
        # GPU branch deliberately inaccessible in default CPU-only mode.
        with tempfile.TemporaryDirectory(prefix="hermes-gemv-layer0-") as td:
            temp = Path(td)
            input_f32 = temp / "attn_norm.f32"
            input_bf16 = temp / "attn_norm.rne.bf16"
            reference = temp / "llama_qkv.f32"
            input_f32.write_bytes(data[INPUT])
            input_bf16.write_bytes(bf16_bits.tobytes())
            reference.write_bytes(data[OUTPUT])
            cmd = [str(args.gpu_exe), "--gguf", str(model),
                   "--activation-bf16", str(input_bf16),
                   "--original-f32", str(input_f32),
                   "--llama-qkv", str(reference), "--device", str(args.device)]
            print("GPU_REPLAY_EXPLICIT invoking isolated compiled comparator "
                  "only after CPU hash+shape+FP64 acceptance", flush=True)
            run = subprocess.run(cmd, timeout=110, check=False)
            require(run.returncode == 0, f"isolated CUDA replay exit={run.returncode}")
        return 0
    except (OSError, UnicodeError, ValueError, TypeError, KeyError,
            subprocess.SubprocessError) as e:
        print(f"FAIL_OR_BLOCKED gemv_layer0_oracle: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
