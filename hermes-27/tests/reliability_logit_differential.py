#!/usr/bin/env python3
"""Read-only CPU diagnostic on real, same-token llama.cpp/HERMES FP32 traces.

No inference, no CUDA. Distinguishes correlated residual structure from a
single global affine logit transform. Optionally tests GGUF blk.0 activation
BF16-vs-Q8_1 quantization on exact same W and x, without reading full weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path


VOCAB = 248320
NAME = "prompt_{:d}_step_{:d}.f32"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_runs(root: Path, np):
    expected = (root / "llama-manifest.json", root / "hermes-manifest.json")
    if any(not f.is_file() for f in expected):
        raise ValueError("independent llama/HERMES manifests missing")
    llama, hermes = [json.loads(f.read_text(encoding="utf-8")) for f in expected]
    if (llama.get("schema") != "h27.fp32-traces.v1" or
        hermes.get("schema") != "h27.fp32-traces.v1" or
        llama.get("backend") != "llama.cpp" or hermes.get("backend") != "hermes-27" or
        llama.get("run_id") != hermes.get("run_id") or
        llama.get("gguf_sha256") != hermes.get("gguf_sha256") or
        llama.get("entries") != hermes.get("entries")):
        raise ValueError("mismatched/unknown model, backend, prompt, continuation, or provenance")
    entries = llama["entries"]
    if len(entries) < 4:
        raise ValueError("require four independent prompt captures")
    all_names = {NAME.format(i, j) for i, e in enumerate(entries)
                 for j in range(len(e["generated_token_ids"]))}
    for manifest in (llama, hermes):
        if all_names != set(manifest.get("files_sha256", {})):
            raise ValueError("trace file set differs from captured token histories")
        for name, original in manifest["files_sha256"].items():
            path = Path(manifest["dump_dir"]) / name
            if not path.is_file() or path.stat().st_size != VOCAB*4:
                raise ValueError(f"missing or malformed {path}")
            if sha256(path) != original:
                raise ValueError(f"captured trace modified: {path}")
    arrays = []
    for i, entry in enumerate(entries):
        for j in range(len(entry["generated_token_ids"])):
            name = NAME.format(i, j)
            ref = np.fromfile(Path(llama["dump_dir"]) / name, dtype="<f4")
            cur = np.fromfile(Path(hermes["dump_dir"]) / name, dtype="<f4")
            if ref.shape != (VOCAB,) or cur.shape != (VOCAB,) or not (
                    np.isfinite(ref).all() and np.isfinite(cur).all()):
                raise ValueError(f"invalid FP32 logits in {name}")
            if int(np.argmax(ref)) != int(np.argmax(cur)) or int(np.argmax(cur)) != (
                    entry["generated_token_ids"][j]):
                raise ValueError(f"captured identical greedy continuation violated in {name}")
            arrays.append((name, ref.astype(np.float64), cur.astype(np.float64)))
    return llama, hermes, arrays


def summarize_logits(arrays, np):
    summaries = []
    residuals = []
    for name, ref, cur in arrays:
        d = cur - ref
        residuals.append(d.astype(np.float32))
        a = ref.mean()
        b = cur.mean()
        cov = np.mean((ref-a)*(cur-b))
        slope = float(cov/np.mean((ref-a)**2))
        intercept = float(b-slope*a)
        corrected = cur - (slope*ref+intercept)
        p = np.exp(ref - ref.max())
        q = np.exp(cur - cur.max())
        p /= p.sum()
        q /= q.sum()
        mask = p>0
        kl = float(np.sum(p[mask]*np.log(p[mask]/np.maximum(q[mask],1e-300))))
        order = np.argsort(ref)
        top1, top2 = order[-1], order[-2]
        summaries.append({
            "file": name, "greedy": int(top1),
            "reference_top1_margin": float(ref[top1] - ref[top2]),
            "max_abs": float(np.max(np.abs(d))),
            "rmse": float(np.sqrt(np.mean(d*d))),
            "affine_fit_slope": slope, "affine_fit_intercept": intercept,
            "post_affine_rmse": float(np.sqrt(np.mean(corrected*corrected))),
            "residual_mean": float(d.mean()), "residual_std": float(d.std()),
            "corr_logits": float(cov/np.sqrt(np.mean((ref-a)**2) *
                                             np.mean((cur-b)**2))),
            "top10_overlap": int(len(set(np.argsort(ref)[-10:]) &
                                      set(np.argsort(cur)[-10:]))),
            "kl_llama_to_hermes": kl,
            "fraction_abs_gt_005": float(np.mean(np.abs(d) > .05)),
            "fraction_abs_gt_020": float(np.mean(np.abs(d) > .2)),
            "max_abs_index": int(np.argmax(np.abs(d))),
        })
    # Distinguish fixed vocabulary-position bias from prompt-varying state
    # drift. Pairwise normalized residuals and SVD use only 12 x 248320 (~12MB).
    mat = np.stack(residuals, axis=0).astype(np.float64)
    centered = mat - mat.mean(axis=1,keepdims=True)
    norm = np.linalg.norm(centered, axis=1,keepdims=True)
    similarity = centered @ centered.T / (norm @ norm.T)
    eigs = np.linalg.eigvalsh(centered @ centered.T)
    eigs = np.clip(eigs, 0, None)[::-1]
    all_off = similarity[np.triu_indices(len(arrays), k=1)]
    return {
        "counts": {"traces": len(arrays), "vocab": VOCAB},
        "traces": summaries,
        "residual_pairwise_corr": {
            "min": float(np.min(all_off)), "median": float(np.median(all_off)),
            "max": float(np.max(all_off)),
            "adjacent_same_prompt": [
                {"a": summaries[i]["file"], "b": summaries[j]["file"],
                 "correlation": float(similarity[i,j])}
                for i in range(len(summaries)) for j in range(i+1,len(summaries))
                if summaries[i]["file"].split("_step_")[0] ==
                   summaries[j]["file"].split("_step_")[0] and j == i+1
            ],
        },
        "residual_svd": {
            "rank_1_energy": float(eigs[0]/eigs.sum()),
            "rank_3_energy": float(eigs[:3].sum()/eigs.sum()),
            "rank_6_energy": float(eigs[:6].sum()/eigs.sum()),
        },
    }


def bf16_round(x, np):
    values = np.asarray(x, dtype=np.float32)
    bits = values.view(np.uint32)
    # IEEE BF16 round-to-nearest-even, except NaNs (input finite invariant).
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) &
            np.uint32(0xffff0000)).view(np.float32)


def q8_1_quant(x, np):
    values = np.asarray(x, dtype=np.float32)
    if values.size % 32:
        raise ValueError("Q8_1 activation length must be divisible by 32")
    blocks = values.reshape(-1,32)
    amax = np.abs(blocks).max(axis=1, keepdims=True)
    d = amax / np.float32(127.)
    # CUDA quantize.cu:86: roundf(xi/d), scale stored in half2(d,sum).
    # np.rint is ties-even rather than away; exact .5 ties are rare and this
    # is an approximate CPU isolate, NOT a bit-exact CUDA oracle.
    normed = np.divide(blocks,d,out=np.zeros_like(blocks),where=d!=0)
    qi = np.rint(normed).clip(-127,127).astype(np.int8)
    dq = qi.astype(np.float32) * d.astype(np.float16).astype(np.float32)
    return dq.reshape(-1)


def gguf_quant_probe(model: Path, ids: list[int], np, nrows: int):
    # Refer only to current user-supplied GGUF; no full 27B checkpoint load.
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from block0_corrected_ref import GGUF, bf16_to_f32
    with model.open("rb") as fd:
        gguf = GGUF(fd, np)
        tok = gguf.tensor("token_embd.weight", "bf16")
        norm = gguf.tensor("blk.0.attn_norm.weight", "f32")
        dims, typ, offset = gguf.tensors["blk.0.attn_qkv.weight"]
        if typ != 40 or dims != (5120, 10240):
            raise ValueError(f"unexpected blk0 qkv dimensions or type: {dims}, {typ}")
        # Evenly distribute output rows to avoid selecting just Q or K.
        row_ids = np.linspace(0, dims[1]-1, nrows, dtype=np.int64)
        scales_lut = np.array([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],
                              dtype=np.float32)
        weights = np.empty((nrows, dims[0]), dtype=np.float32)
        base = gguf.data_offset + offset
        row_bytes = (dims[0]//64)*36
        for i, ri in enumerate(row_ids):
            b = np.frombuffer(gguf.mm, dtype=np.uint8, count=row_bytes,
                              offset=base + int(ri)*row_bytes).reshape(-1,36)
            s = b[:,:4]
            e = (s>>3)&15
            mant = s&7
            sf = np.where(e==0, mant.astype(np.float32)*2**-9,
                          (1 + mant.astype(np.float32)/8)*np.exp2(e.astype(np.float32)-7))
            sf = np.where((s==0)|(s==127), 0,sf)
            pq = b[:,4:].reshape(-1,4,8)
            x = np.empty((b.shape[0],4,16),np.float32)
            x[...,:8] = scales_lut[pq&15]
            x[...,8:] = scales_lut[pq>>4]
            weights[i] = (x*sf[...,None]).reshape(dims[0])
        results = []
        for tokid in ids:
            emb = bf16_to_f32(tok[tokid],np)
            inv = np.float32(1)/np.sqrt(np.mean(emb.astype(np.float64)**2) + 1e-6)
            h = emb * inv * norm
            x_bf = bf16_round(h,np)
            x_q8 = q8_1_quant(h,np)
            x_f = h.astype(np.float32)
            def rms(v):
                return float(np.sqrt(np.mean(np.asarray(v,dtype=np.float64)**2)))
            y_f = weights.astype(np.float64) @ x_f.astype(np.float64)
            y_b = weights.astype(np.float64) @ x_bf.astype(np.float64)
            y_q = weights.astype(np.float64) @ x_q8.astype(np.float64)
            results.append({
                "token_id": tokid, "rows_sampled": int(nrows),
                "input_rms": rms(x_f),
                "activation_bf16_rmse": rms(x_bf-x_f),
                "activation_q8_1_rmse": rms(x_q8-x_f),
                "activation_bf16_vs_q8_rmse": rms(x_bf-x_q8),
                "block0_qkv_bf16_vs_q8_max_abs": float(np.max(np.abs(y_b-y_q))),
                "block0_qkv_bf16_vs_q8_rmse": rms(y_b-y_q),
                "block0_qkv_bf16_vs_fp32_rmse": rms(y_b-y_f),
                "block0_qkv_q8_vs_fp32_rmse": rms(y_q-y_f),
                "limitations": "approximate Q8_1 rounding; same NVFP4 decoded W sample, "
                               "not a validated llama.cpp CUDA layer trace",
            })
        # tok and the last block slice still export read-only mmap pointers.
        # Their lifetimes end at function return; closing explicitly here
        # would throw BufferError, while Python's GC releases the read-only
        # mapping safely when the arrays leave scope.
        return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scratch", required=True, type=Path,
                   help="path with manifests and intact llama-all/hermes-all dumps")
    p.add_argument("--gguf", type=Path, help="optional real GGUF for CPU blk0 quantization probe")
    p.add_argument("--sample-rows", type=int, default=256)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.sample_rows < 1 or args.sample_rows > 10240:
        p.error("sample-rows must be 1..10240")
    import numpy as np
    llama, hermes, traces = load_runs(args.scratch, np)
    report = {"schema": "h27.reliability-logit-differential.v1",
              "gguf_sha256": llama["gguf_sha256"],
              "run_id": llama["run_id"], "model_ref": "llama.cpp independent GGUF",
              "model_candidate": "HERMES-27", "gpu_used": False,
              "analysis": summarize_logits(traces,np)}
    if args.gguf:
        if not args.gguf.is_file() or sha256(args.gguf) != llama["gguf_sha256"]:
            raise ValueError("requested model GGUF missing or SHA256 mismatch")
        # Each case measures exact initial x/weights for prompt first token;
        # later tokens depend on prior 64-layer hidden and are not equivalent.
        ids=[item["prompt_ids"][0] for item in llama["entries"]]
        report["block0_quantization_probe"] = gguf_quant_probe(args.gguf,ids,np,args.sample_rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n",encoding="utf-8")
    stats=report["analysis"]
    print(f"CPU DIFF {len(traces)} traces, {VOCAB} logits each; "
          f"pairwise residual correlations: {stats['residual_pairwise_corr']}; "
          f"SVD: {stats['residual_svd']}")
    if args.gguf:
        for result in report["block0_quantization_probe"]:
            print("CPU BLK0",result)
    print("EVIDENCE",args.output)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        print(f"QA BLOCKED/FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
