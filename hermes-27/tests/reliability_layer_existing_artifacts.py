#!/usr/bin/env python3
"""CPU-only first-layer QKV analysis on previously saved exact-GGUF artifacts.

Verifies an actual llama.cpp cb_eval manifest, a *versioned* CPU GGUF
block0 oracle (not an HERMES runtime dump), and optionally a historical
E2 unversioned raw QKV dump (not authenticated as same-source/current).
Computes identical-W/identical-input FP64 dot products for FP32, BF16
and approximate GGML CUDA Q8_1 activation representations.

This does NOT execute either inference engine. Never promote a CPU
oracle or an old E2 dump into a current HERMES GPU trace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from block0_corrected_ref import (  # type: ignore[import-not-found]
    GGUF, IDS, MAGIC, SHAPES, T, bf16_to_f32, fingerprint,
)

SCHEMA = "h27.reliability-layer-existing-cpu-v1"
LAYER = 0
DIM = 5120
QKV = 10240
FIRST = 6205
HEADER = struct.Struct("<8sQQ11i")
EXPECTED_GGUF_SHA = "db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1"


class Blocked(RuntimeError):
    pass


def check(cond: bool, why: str) -> None:
    if not cond:
        raise Blocked(why)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(2 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def exact_manifest_tensor(manifest: dict, manifest_path: Path, name: str, n: int) -> np.ndarray:
    records = [x for x in manifest.get("tensors", []) if x.get("name") == name]
    check(len(records) == 1, f"exactly one {name} trace required")
    record = records[0]
    check(record.get("source_type") == "f32" and
          record.get("shape") == [n, 1, 1, 1] and
          record.get("elements") == n and record.get("bytes") == n*4,
          f"{name}: wrong source dtype or shape")
    filename = record.get("file")
    check(isinstance(filename, str) and filename == f"{name}.f32",
          f"{name}: unsafe or incompatible trace filename")
    f = manifest_path.parent / filename
    check(f.is_file() and f.stat().st_size == n*4, f"{name}: missing full FP32 trace")
    digest = sha256_file(f)
    check(digest == record.get("sha256"), f"{name}: trace SHA256 mismatch")
    x = np.fromfile(f, dtype="<f4")
    check(x.size == n and np.isfinite(x).all(), f"{name}: nonfinite or short")
    return x


def read_oracle(path: Path, gguf: GGUF) -> tuple[dict[str,np.ndarray], dict]:
    expected = HEADER.size + sum(math.prod(shape)*4 for _, shape in SHAPES)
    check(path.is_file() and path.stat().st_size == expected,
          f"corrected GGUF oracle expected {expected} bytes")
    with path.open("rb") as f:
        raw = f.read(HEADER.size)
        magic, model_size, fnv, *ids = HEADER.unpack(raw)
        check(magic == MAGIC, "old/incorrect oracle magic")
        check(model_size == len(gguf.mm) and fnv == fingerprint(gguf.mm),
              "oracle GGUF byte-size/FNV model fingerprint mismatch")
        check(tuple(ids) == IDS and ids[0] == FIRST,
              "oracle exact prompt token history mismatch")
        entries = {}
        for name, shape in SHAPES:
            a = np.fromfile(f, dtype="<f4", count=math.prod(shape)).reshape(shape)
            check(np.isfinite(a).all(), f"oracle {name}: nonfinite")
            entries[name] = a[0].copy()  # single token position 0 only
        check(not f.read(1), "oracle unexpected trailing bytes")
    return entries, {
        "path": str(path), "sha256": sha256_file(path), "header_magic": MAGIC.decode(),
        "gguf_size": model_size, "gguf_fnv_first_last_4096": f"{fnv:016x}",
        "prompt_ids": ids, "first_token": FIRST, "position": 0,
        "format": "CPU NumPy GGUF direct dequantized FP32; NOT runtime HERMES",
    }


def bf16_round(x: np.ndarray) -> np.ndarray:
    bits = np.asarray(x,dtype=np.float32).view(np.uint32)
    # round nearest even BF16, all source entries checked finite
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) &
            np.uint32(0xffff0000)).view(np.float32)


def q8_1_approx(x: np.ndarray) -> np.ndarray:
    check(x.size % 32 == 0, "Q8_1 requires multiple-of-32 activations")
    blocks = x.reshape(-1,32)
    mx = np.max(np.abs(blocks), axis=1, keepdims=True)
    scale = mx / np.float32(127)
    # ggml_cuda quantize.cu uses roundf (ties away from zero), NumPy
    # emulates ties away explicitly, but CUDA fp32 math may still differ.
    frac = np.divide(blocks,scale,where=scale!=0,
                     out=np.zeros_like(blocks,dtype=np.float32))
    rounded = np.copysign(np.floor(np.abs(frac)+np.float32(.5)),frac)
    q = np.clip(rounded,-127,127).astype(np.int8)
    return (q.astype(np.float32) *
            scale.astype(np.float16).astype(np.float32)).reshape(-1)


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    check(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
          "FP32 numeric comparison shape or finiteness mismatch")
    diff = a.astype(np.float64) - b.astype(np.float64)
    aa = a.astype(np.float64).ravel()
    bb = b.astype(np.float64).ravel()
    error = float(np.sqrt(np.mean(diff*diff)))
    return {
        "n": int(a.size), "max_abs": float(np.max(np.abs(diff))), "rmse": error,
        "mean_error_a_minus_b": float(np.mean(diff)),
        "fraction_abs_gt_1e_3": float(np.mean(np.abs(diff)>1e-3)),
        "pearson": float(np.corrcoef(aa,bb)[0,1]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--llama-manifest",type=Path,required=True)
    p.add_argument("--gguf",type=Path,required=True)
    p.add_argument("--corrected-oracle",type=Path,required=True)
    p.add_argument("--legacy-e2-qkv",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cpu-threads",type=int,default=2)
    args = p.parse_args()
    check(1<=args.cpu_threads<=4, "cpu threads out of range")
    check(args.llama_manifest.is_file() and args.gguf.is_file(),
          "missing llama manifest or GGUF")
    # NumPy is already imported by CLI, set thread budget before the process
    # is launched with OPENBLAS_NUM_THREADS to make it effective.
    manifest = json.loads(args.llama_manifest.read_text(encoding="utf-8"))
    check(manifest.get("schema")=="h27.reliability-layer-cb.v1" and
          manifest.get("backend")=="llama.cpp" and
          manifest.get("status")=="CAPTURED_NOT_INDEPENDENTLY_COMPARED" and
          manifest.get("layer")==LAYER and manifest.get("absolute_position")==0 and
          manifest.get("prompt_ids")==[FIRST] and
          manifest.get("model_sha256")==EXPECTED_GGUF_SHA,
          "llama model provenance, token or layer0/pos0 incompatible")
    l_norm = exact_manifest_tensor(manifest,args.llama_manifest,"attn_norm-0",DIM)
    l_qkv = exact_manifest_tensor(manifest,args.llama_manifest,
                                  "linear_attn_qkv_mixed-0",QKV)
    l_conv = exact_manifest_tensor(manifest,args.llama_manifest,
                                   "conv_output_silu-0",QKV)

    start = time.monotonic()
    hash_model = sha256_file(args.gguf)
    check(hash_model == manifest["model_sha256"],
          "physical GGUF file does not match llama SHA256 manifest")
    with args.gguf.open("rb") as f:
        gguf=GGUF(f,np)
        ora,oracle_manifest=read_oracle(args.corrected_oracle,gguf)
        embeddings=gguf.tensor("token_embd.weight","bf16")
        embedding=bf16_to_f32(embeddings[FIRST],np).copy()
        norm_w=gguf.tensor("blk.0.attn_norm.weight","f32")
        # RMS numeric control 1. Prime llama uses FP32 graph; CPU dequant
        # source has quant input BF16 and direct weights.
        norm_64=embedding.astype(np.float64) / math.sqrt(
            float(np.mean(embedding.astype(np.float64)**2))+1e-6)
        norm_64=(norm_64*norm_w.astype(np.float64)).astype(np.float32)
        # Control 2 follows reference NumPy FP32 arithmetic.
        norm_32=embedding / np.sqrt(np.mean(embedding*embedding,dtype=np.float32)
                                    + np.float32(1e-6))
        norm_32=(norm_32*norm_w).astype(np.float32)
        check(np.isfinite(norm_64).all() and np.isfinite(norm_32).all(),
              "nonfinite reconstructed GGUF norm")
        # Entire REAL NVFP4 weights, 10,240×5,120, bounded ~210 MiB.
        weights=gguf.tensor("blk.0.attn_qkv.weight","nvfp4")
        check(weights.shape==(QKV,DIM) and np.isfinite(weights).all(),
              "actual block0 QKV weight shape or finiteness mismatch")
        input_modes={
            "f32_llama_norm": l_norm,
            "bf16_from_llama_norm": bf16_round(l_norm),
            "q8_1_approx_from_llama_norm": q8_1_approx(l_norm),
            "f32_gguf_recomputed_norm": norm_64,
        }
        projections={k:np.empty(QKV,dtype=np.float32) for k in input_modes}
        for low in range(0,QKV,256):
            high=min(low+256,QKV)
            w=weights[low:high].astype(np.float64)
            for name,x in input_modes.items():
                projections[name][low:high]=(w @ x.astype(np.float64)).astype(np.float32)
        del weights

    sources={
      "llama": {
          "manifest":str(args.llama_manifest),
          "manifest_sha256":sha256_file(args.llama_manifest),
          "model_sha256":manifest["model_sha256"],
          "prompt_ids":[FIRST],"absolute_position":0,
          "layer":0,"runtime_provenance":"AUTHENTIC_GPU_CAPTURE_OLD_RUN",
          "tensors":{name:item["sha256"] for name in
                     ["attn_norm-0","linear_attn_qkv_mixed-0","conv_output_silu-0"]
                     for item in manifest["tensors"] if item["name"]==name},
      },
      "corrected_cpu_oracle":oracle_manifest,
      "physical_gguf": {"path":str(args.gguf),"sha256":hash_model,
                        "bytes":args.gguf.stat().st_size},
    }
    comparisons={
      "llama_attn_norm_vs_exact_gguf_fp64_norm":metrics(l_norm,norm_64),
      "llama_attn_norm_vs_exact_gguf_fp32_norm":metrics(l_norm,norm_32),
      "llama_qkv_vs_corrected_cpu_oracle_qkv":metrics(l_qkv,ora["qkv"]),
      "llama_conv_vs_corrected_cpu_oracle_conv":metrics(l_conv,ora["cv"]),
      "corrected_cpu_oracle_qkv_vs_f32_same_llama_norm":
          metrics(ora["qkv"],projections["f32_llama_norm"]),
    }
    for name,y in projections.items():
        comparisons[f"llama_qkv_vs_{name}"]=metrics(l_qkv,y)
        comparisons[f"corrected_oracle_qkv_vs_{name}"]=metrics(ora["qkv"],y)

    if args.legacy_e2_qkv:
        check(args.legacy_e2_qkv.is_file() and
              args.legacy_e2_qkv.stat().st_size==T*QKV*4,
              "E2 unversioned QKV size incompatible with historical T=11")
        legacy=np.fromfile(args.legacy_e2_qkv,dtype="<f4",count=QKV)
        check(np.isfinite(legacy).all(),"legacy E2 QKV nonfinite")
        sources["legacy_e2_qkv"]={
          "path":str(args.legacy_e2_qkv),
          "sha256":sha256_file(args.legacy_e2_qkv),
          "bytes":args.legacy_e2_qkv.stat().st_size,
          "format":"unversioned 11x10240 CPU architecture E2 HERMES_DUMPQKV",
          "provenance":"UNAUTHENTICATED_CHECKPOINT_AND_TOKEN_HISTORY",
          "current_runtime_hermes":False,
          "source":"architecture/src/model_fwd.cpp:171,230-237",
        }
        comparisons["legacy_e2_qkv_vs_llama_qkv"]=metrics(legacy,l_qkv)
        comparisons["legacy_e2_qkv_vs_corrected_cpu_oracle_qkv"]=metrics(legacy,ora["qkv"])
        comparisons["legacy_e2_qkv_vs_f32_same_llama_norm"]=metrics(
            legacy,projections["f32_llama_norm"])

    result={
      "schema":SCHEMA,
      "status":"HISTORIC_CPU_ORACLE_DIAGNOSTIC_REAL_RUNTIME_HERMES_QKV_MISSING",
      "not_release_gate":True,
      "model_sha256":hash_model,
      "same_token_position":[FIRST,0],
      "first_layer_stage":"attn_norm-0 -> linear_attn_qkv_mixed-0 -> conv_output_silu-0",
      "source_manifest":sources,
      "comparisons":comparisons,
      "elapsed_cpu_seconds":round(time.monotonic()-start,3),
      "conclusions":{
        "llama_qkv_runtime_present":True,
        "corrected_cpu_gguf_oracle_present":True,
        "current_hermes_gpu_qkv_runtime_capture_present":False,
        "first_divergent_current_hermes_kernel_proven":False,
        "root_cause_proven":False,
        "required_next":"Authentic current PP2 QKV after block0 NVFP4 GEMV, prior conv, "
                        "with same GGUF SHA256 and token[6205] pos0, plus attn_norm. "
                        "Compare both captured raw values with existing strict layer comparator.",
        "caveat":"Q8_1 CPU model uses exact source-style per32 amax, int8 rounding, "
                 "FP16 scale with FP64 downstream dot; llama CUDA uses DP4A "
                 "and different accumulation; this is NOT bitwise GPU simulation. "
                 "Historic E2 CPU cannot be treated as current PP2 GPU.",
      },
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    for name,record in comparisons.items():
        print(f"{name}: RMSE={record['rmse']:.9g} maxabs={record['max_abs']:.9g}")
    print("BLOCKED_CURRENT_HERMES_CAPTURE; CPU_DIAGNOSTIC_COMPUTED "+str(args.output))


if __name__=="__main__":
    try:
        main()
    except (Blocked, KeyError, ValueError, OSError) as exc:
        print("BLOCKED_CPU_ANALYSIS: "+str(exc),file=sys.stderr)
        sys.exit(2)
