#!/usr/bin/env python3
"""CPU-only authenticated GGUF layer0 pos0: Q8_1 -> conv -> GDN -> residual.

The only authenticated llama outputs here are attn_norm-0, QKV-0, conv_silu-0.
No downstream llama state/out tensors were captured. NEVER assert llama parity
past convolution. No CUDA, no live services, bounded NumPy / BLAS CPU threads.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys

SCHEMA = "h27.reliability-layer0-gdn-q8-cpu.v1"
HID, CC, KD, HV, KHD, VHD, VDD, GROUPS = 5120, 10240, 2048, 48, 128, 128, 6144, 16
MODES = ("q8_1", "bf16_rne", "fp32", "llama_conv_anchor_q8")
EPS = 1e-6


class Blocked(RuntimeError):
    pass


def check(ok, reason):
    if not ok:
        raise Blocked(reason)


def digest_array(a, np):
    arr = np.ascontiguousarray(a, dtype="<f4")
    check(bool(np.isfinite(arr).all()), "non-finite stage values")
    return {"shape": list(arr.shape), "dtype": "little-endian-fp32",
            "elements": int(arr.size),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
            "min": float(np.min(arr)), "max": float(np.max(arr)),
            "l2_rms": float(np.sqrt(np.mean(arr.astype(np.float64)**2)))}


def diff(a, b, np):
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    check(x.shape == y.shape and bool(np.isfinite(x).all()) and
          bool(np.isfinite(y).all()), "incompatible comparisons")
    d = x - y
    return {"shape": list(x.shape), "rmse": float(np.sqrt(np.mean(d*d))),
            "max_abs": float(np.max(np.abs(d))),
            "worst_idx": int(np.argmax(np.abs(d))),
            "fraction_abs_gt_1e_3": float(np.mean(np.abs(d)>1e-3))}


def f32(x, np):
    return np.ascontiguousarray(x, dtype=np.float32)


def guarded_matrix(gguf, name, kind, shape, np):
    check(name in gguf.tensors, "GGUF missing tensor "+name)
    dims, dtype, _ = gguf.tensors[name]
    want = {"f32": 0, "nvfp4": 40, "bf16": 30}[kind]
    check(dtype == want and tuple(dims[::-1]) == tuple(shape),
          f"GGUF {name}: required {kind} {shape}, actual {dtype} {dims}")
    matrix=gguf.tensor(name, kind)
    check(matrix.shape == tuple(shape) and bool(np.isfinite(matrix).all()),
          "invalid GGUF values "+name)
    return matrix


def matvec_panel(weights, inputs, np):
    """CPU FP64 matrix dot, FP32 output; multiple real activation precisions.

    Panel width 96 to bound scratch. Input dictionary values all shape [K].
    Weight values have been dequantized from actual GGUF using source parser.
    """
    n,k = weights.shape
    check(k in (HID, VDD), "invalid matrix inner dimension")
    x64 = {name: np.asarray(x, dtype=np.float64) for name,x in inputs.items()}
    for name,x in x64.items():
        check(x.shape==(k,) and bool(np.isfinite(x).all()),
              "bad matvec input "+name)
    ys={name:np.empty(n, dtype=np.float32) for name in x64}
    for low in range(0,n,96):
        high=min(low+96,n)
        panel=weights[low:high].astype(np.float64)
        for mode,x in x64.items():
            ys[mode][low:high]=(panel@x).astype(np.float32)
    return ys


def sigmoid(x, np):
    x=f32(x,np)
    return f32(1.0/(1.0+np.exp(-x)),np)


def silu(x, np):
    x=f32(x,np)
    return f32(x*sigmoid(x,np),np)


def softplus_source(x, np):
    """PP2 beta_gl_k: x > 20 ? x : logf(1 + expf(x))."""
    x=f32(x,np)
    return f32(np.where(x>np.float32(20), x, np.log(
        np.float32(1)+np.exp(np.minimum(x, np.float32(20))))),np)


def split_and_qk_norm(conv, np):
    """GGUF tiled [Q16, K16, V48] and GGML repeat_4d -> 48 heads."""
    check(conv.shape==(CC,), "wrong QKV convolution vector")
    q16=f32(conv[:KD].reshape(GROUPS,KHD),np)
    k16=f32(conv[KD:2*KD].reshape(GROUPS,KHD),np)
    v=f32(conv[2*KD:].reshape(HV,VHD),np)
    sums_q=np.sum(q16*q16,axis=1,dtype=np.float32)
    sums_k=np.sum(k16*k16,axis=1,dtype=np.float32)
    invq=f32(np.float32(1.0)/np.sqrt(sums_q+np.float32(EPS)),np)
    invk=f32(np.float32(1.0)/np.sqrt(sums_k+np.float32(EPS)),np)
    scale=np.float32(1.0/np.sqrt(np.float32(KHD)))
    q=f32(np.tile((q16*invq[:,None])*scale,(HV//GROUPS,1)),np)
    k=f32(np.tile(k16*invk[:,None],(HV//GROUPS,1)),np)
    check(q.shape==(HV,KHD) and k.shape==(HV,KHD) and v.shape==(HV,VHD),
          "GDN head reshape wrong")
    check(np.array_equal(q[0:16],q[16:32]) and
          np.array_equal(q[0:16],q[32:48]) and
          np.array_equal(k[0:16],k[16:32]) and
          np.array_equal(k[0:16],k[32:48]),
          "GDN head-repeat grouping incorrect")
    # The exact epsilon-adjusted squared norm identities of the normalized heads.
    q_n=np.sum(q.astype(np.float64)**2,axis=1)
    k_n=np.sum(k.astype(np.float64)**2,axis=1)
    predicted_q=(sums_q.astype(np.float64)/(sums_q.astype(np.float64)+EPS) / KHD)
    predicted_k=sums_k.astype(np.float64)/(sums_k.astype(np.float64)+EPS)
    check(np.max(np.abs(q_n[:16]-predicted_q))<3e-8 and
          np.max(np.abs(k_n[:16]-predicted_k))<3e-6,
          "Q/K normalization epsilon/scale check failed")
    return {"q_norm":q,"k_norm":k,"v":v,
            "q_l2_norm_sq":q_n,"k_l2_norm_sq":k_n}


def gdn_pos0(q,k,v,beta,gl,np):
    """Independent initial-zero-state recurrence, store rounded FP32 state.

    S_prev=0: S_post = outer(k, beta*v); o = S_post^T q.
    Verify equal to the source's double/FP32 recurrent-state numerical order.
    """
    check(q.shape==(HV,KHD) and k.shape==(HV,KHD) and
          v.shape==(HV,VHD) and beta.shape==(HV,) and gl.shape==(HV,),
          "GDN shape mismatch")
    check(bool((gl<=0).all()) and bool((beta>0).all()) and
          bool((beta<1).all()), "GDN gate sign or beta range violated")
    decay=f32(np.exp(gl),np)
    check(bool((decay<=1).all()) and bool((decay>=0).all()),
          "GDN gate must decay not grow")
    # Real code has a zero state and computes each vector-column update in
    # FP64 then rounds to FP32 state. At pos0 all old-state terms are zero.
    delta=v.astype(np.float64)*beta.astype(np.float64)[:,None]
    state=f32(k.astype(np.float64)[:,:,None] * delta[:,None,:],np)
    output=f32(np.einsum("hkv,hk->hv",state.astype(np.float64),
                        q.astype(np.float64)),np)
    # Pure mathematical simplification, independent of storage rounding.
    dot=np.sum(q.astype(np.float64)*k.astype(np.float64),axis=1)
    simpl=f32(dot[:,None]*beta.astype(np.float64)[:,None]*
              v.astype(np.float64),np)
    err=diff(output,simpl,np)
    check(err["max_abs"] <= 1e-5+1e-5*max(1.0,float(np.max(np.abs(output)))),
          "initial-state GDN algebraic identity failed")
    return {"state":state, "o":output, "gate_decay":decay,
            "pos0_recurrence_identity":err}


def gated_rms(o,z,wn,np):
    check(o.shape==(HV,VHD) and z.shape==(HV,VHD) and wn.shape==(VHD,),
          "gated RMS wrong dimensions")
    rms_sq=np.mean(o*o, axis=1, dtype=np.float32)
    inv=f32(np.float32(1)/np.sqrt(rms_sq+np.float32(EPS)),np)
    gated=f32(o*inv[:,None]*wn[None,:]*silu(z,np),np)
    check(bool(np.isfinite(gated).all()), "nonfinite gated RMS")
    return gated.reshape(VDD)


def inspect_source_contract(root: Path, llama_dir: Path, np):
    """Fail closed on changed *textual* source contract; record SHA256.

    Not a proof of identical rounding or actual GPU dispatch.
    """
    paths = {
      "pp2": root/"distributed/pp2_fwd.cu",
      "gdn_cuda": root/"kernels/src/gdn_step.cu",
      "llama_qwen35": llama_dir/"src/models/qwen35.cpp",
      "llama_delta": llama_dir/"src/models/delta-net-base.cpp",
      "llama_conv": llama_dir/"ggml/src/ggml-cuda/ssm-conv.cu",
    }
    fragments = {
      "pp2": ("g=hh%(KD/KHD)", "sqrtf(sq[0]+1e-6f)",
              "sqrtf(sk[0]+1e-6f)", "Wn[tid]*s",
              "x>20.f?x:logf(1.f+expf(x))", "ssm_a stores -exp(A_log)",
              "gdn_step_cuda", "d_x,d_core"),
      "gdn_cuda": ("Sh[i*V+j]*=dec", "double(Sh[i*V+j])*double(sk[i])",
                   "double(sk[i])*bv", "double(Sh[i*V+j])*double(sq[i])"),
      "llama_qwen35": ("build_gdn_l2_norm", "ggml_repeat_4d",
                       "build_norm_gated", "alpha_softplus", "ggml_sigmoid",
                       "ggml_silu(ctx0, gate)",
                       "return ggml_mul(ctx0, normalized, gated_silu)"),
      "llama_delta": ("ggml_exp(ctx0, g)", "ggml_mul(ctx0, s, g)",
                       "ggml_mul     (ctx0, s, k)", "ggml_sum_rows(ctx0, sk)",
                       "ggml_add(ctx0, s, kd)"),
      "llama_conv": ("x[(i + j) % d_conv] * w[j]",),
      "llama_gdn_norm": ("eps/n", "1.0f/sqrtf(n)", "ggml_rms_norm"),
    }
    paths["llama_gdn_norm"]=llama_dir/"src/models/models.h"
    sums={}
    for name,path in paths.items():
        check(path.is_file(), "source contract file missing: "+str(path))
        code=path.read_text(encoding="utf-8")
        for fragment in fragments[name]:
            check(fragment in code, f"source assumption changed: {path}:{fragment}")
        sums[name]={"path":str(path),
                    "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                    "assumptions_checked":list(fragments[name])}
    return sums


def run(manifest_path: Path, model_path: Path, root: Path,
        llama_source: Path, output: Path):
    import numpy as np
    sys.path.insert(0,str(root))
    sys.path.insert(0,str(root/"tests"))
    from scripts.block0_corrected_ref import GGUF, bf16_to_f32, rms
    import reliability_layer0_conv_q8_cpu as prev

    sources=inspect_source_contract(root,llama_source,np)
    try:
        manifest, captured=prev.validate_capture(manifest_path,model_path,np)
    except prev.Blocked as exc:
        raise Blocked("invalid authenticated llama capture: "+str(exc)) from exc
    captures=set(captured)
    check(captures=={"attn_norm-0","linear_attn_qkv_mixed-0","conv_output_silu-0"},
          "unexpected downstream llama capture provenance")
    tensors= {r["name"]:r["sha256"] for r in manifest["tensors"]}
    stages={mode:{} for mode in MODES}
    raw_stage={}
    def record(mode,key,a):
        arr=f32(a,np)
        stages[mode][key]=digest_array(arr,np)
        raw_stage[(mode,key)]=arr
        return arr

    with model_path.open("rb") as f:
        gguf=GGUF(f,np)
        emb=gguf.tensor("token_embd.weight","bf16")
        check(emb.shape[1]==HID and emb.shape[0]>6205,"GGUF embedding shape")
        embedding=f32(bf16_to_f32(emb[6205],np),np).copy()
        del emb
        norm_w=guarded_matrix(gguf,"blk.0.attn_norm.weight","f32",(HID,),np)
        h=f32(rms(embedding[None,:],norm_w,np).reshape(HID),np)
        norm_vs_llama=diff(h,captured["attn_norm-0"],np)
        check(norm_vs_llama["max_abs"]<1e-5,
              "model-derived RMS norm differs from authenticated llama input")
        x=captured["attn_norm-0"]
        modes={
          "q8_1":prev.quant_q8_1_per32(x,np),
          "bf16_rne":prev.round_bf16_rne(x,np),
          "fp32":x.copy()
        }
        for mode in MODES:
            record(mode,"norm_input_fp32",x)
        record("q8_1","input_q8_1_quantized",modes["q8_1"])
        record("bf16_rne","input_bf16_rne_quantized",modes["bf16_rne"])
        record("fp32","input_unrounded_fp32",modes["fp32"])

        w=guarded_matrix(gguf,"blk.0.attn_qkv.weight","nvfp4",(CC,HID),np)
        qkvs=matvec_panel(w,modes,np)
        del w
        gc.collect()
        for mode in modes:
            record(mode,"qkv_projected_fp32",qkvs[mode])
        record("llama_conv_anchor_q8","qkv_captured_llama",captured["linear_attn_qkv_mixed-0"])
        wconv=guarded_matrix(gguf,"blk.0.ssm_conv1d.weight","f32",(CC,4),np)
        convs={mode:prev.causal_conv_silu_pos0(y,wconv,np)
               for mode,y in qkvs.items()}
        for mode in modes:
            record(mode,"conv_output_silu",convs[mode])
        # Anchor at the only actual postconv llama tensor, gate/beta retain
        # the Q8-projected normalization. NEVER label later values llama.
        convs["llama_conv_anchor_q8"]=captured["conv_output_silu-0"].copy()
        record("llama_conv_anchor_q8","conv_output_silu_capture_anchor",
               convs["llama_conv_anchor_q8"])
        del wconv
        gc.collect()
        gate_w=guarded_matrix(gguf,"blk.0.attn_gate.weight","nvfp4",(VDD,HID),np)
        gate=matvec_panel(gate_w,modes,np)
        del gate_w
        gc.collect()
        beta_w=guarded_matrix(gguf,"blk.0.ssm_beta.weight","nvfp4",(HV,HID),np)
        beta_pre=matvec_panel(beta_w,modes,np)
        del beta_w
        gc.collect()
        gate["llama_conv_anchor_q8"]=gate["q8_1"].copy()
        beta_pre["llama_conv_anchor_q8"]=beta_pre["q8_1"].copy()

        alpha_w=guarded_matrix(gguf,"blk.0.ssm_alpha.weight","f32",(HV,HID),np)
        alpha=f32((alpha_w.astype(np.float64)@x.astype(np.float64)),np)
        A=guarded_matrix(gguf,"blk.0.ssm_a","f32",(HV,),np)
        dt=guarded_matrix(gguf,"blk.0.ssm_dt.bias","f32",(HV,),np)
        wn=guarded_matrix(gguf,"blk.0.ssm_norm.weight","f32",(VHD,),np)
        check(bool((A<0).all()),"GGUF ssm_a is not already -exp(A_log)")
        pre_dt=f32(alpha+dt,np)
        sp=softplus_source(pre_dt,np)
        gl=f32(A*sp,np)
        check(bool((gl<0).all()),"expected negative GDN gate")
        for mode in MODES:
            record(mode,"alpha_linear_f32",alpha)
            record(mode,"alpha_softplus",sp)
            record(mode,"gl_decay_exponent",gl)

        pos0_invariants={}
        for mode in MODES:
            record(mode,"attn_gate_z_pre_silu",gate[mode])
            b=record(mode,"beta_pre_sigmoid",beta_pre[mode])
            beta=record(mode,"beta_sigmoid",sigmoid(b,np))
            heads=split_and_qk_norm(convs[mode],np)
            q=record(mode,"q_norm_scaled",heads["q_norm"])
            k=record(mode,"k_norm",heads["k_norm"])
            v=record(mode,"v_from_conv",heads["v"])
            state=gdn_pos0(q,k,v,beta,gl,np)
            record(mode,"recurrent_state_pos0",state["state"])
            o=record(mode,"gdn_o",state["o"])
            g=record(mode,"gated_rms_silu",gated_rms(
                o,gate[mode].reshape(HV,VHD),wn,np))
            pos0_invariants[mode]={
                "l2_q_norm_sq_minmax":list(map(float,(heads["q_l2_norm_sq"].min(),
                                                        heads["q_l2_norm_sq"].max()))),
                "l2_k_norm_sq_minmax":list(map(float,(heads["k_l2_norm_sq"].min(),
                                                        heads["k_l2_norm_sq"].max()))),
                "beta_range":list(map(float,(beta.min(),beta.max()))),
                "decay_range":list(map(float,(state["gate_decay"].min(),
                                              state["gate_decay"].max()))),
                "post_state_vs_zero_history_algebra":state["pos0_recurrence_identity"],
                "qk_head_repeat_mod16_verified":True,
                "GDN_exponent_irrelevant_at_pos0_because_Sprev_zero":True,
                "llama_actual_postconv_capture_available":False,
            }

        wg=guarded_matrix(gguf,"blk.0.ssm_out.weight","nvfp4",(HID,VDD),np)
        out_inputs={
          "q8_1":prev.quant_q8_1_per32(raw_stage[("q8_1","gated_rms_silu")],np),
          "bf16_rne":prev.round_bf16_rne(
              raw_stage[("bf16_rne","gated_rms_silu")],np),
          "fp32":raw_stage[("fp32","gated_rms_silu")],
          "llama_conv_anchor_q8":prev.quant_q8_1_per32(
              raw_stage[("llama_conv_anchor_q8","gated_rms_silu")],np)
        }
        for mode in MODES:
            record(mode,"gated_linear_input",out_inputs[mode])
        out_proj=matvec_panel(wg,out_inputs,np)
        del wg
        gc.collect()
        for mode in MODES:
            core=record(mode,"ssm_out_projection",out_proj[mode])
            record(mode,"first_residual_embedding_plus_core",
                   f32(embedding+core,np))

    llama_comp={
      "norm_real_GGUF_vs_llama":norm_vs_llama,
      "q8_qkv_vs_llama":diff(raw_stage[("q8_1","qkv_projected_fp32")],
                             captured["linear_attn_qkv_mixed-0"],np),
      "q8_conv_vs_llama":diff(raw_stage[("q8_1","conv_output_silu")],
                              captured["conv_output_silu-0"],np),
      "bf16_qkv_vs_llama":diff(raw_stage[("bf16_rne","qkv_projected_fp32")],
                               captured["linear_attn_qkv_mixed-0"],np),
      "bf16_conv_vs_llama":diff(raw_stage[("bf16_rne","conv_output_silu")],
                                captured["conv_output_silu-0"],np),
      "fp32_qkv_vs_llama":diff(raw_stage[("fp32","qkv_projected_fp32")],
                               captured["linear_attn_qkv_mixed-0"],np),
      "fp32_conv_vs_llama":diff(raw_stage[("fp32","conv_output_silu")],
                               captured["conv_output_silu-0"],np),
    }
    q8_control={}
    for stage in ("conv_output_silu","beta_sigmoid","q_norm_scaled","k_norm",
                  "gdn_o","gated_rms_silu","ssm_out_projection",
                  "first_residual_embedding_plus_core"):
        baseline=raw_stage[("q8_1",stage)]
        q8_control[stage]={
            mode:diff(raw_stage[(mode,stage)],baseline,np)
            for mode in ("bf16_rne","fp32")
        }
    anchor_vs_q8={}
    for stage in ("gdn_o","gated_rms_silu","ssm_out_projection",
                  "first_residual_embedding_plus_core"):
        anchor_vs_q8[stage]=diff(
            raw_stage[("llama_conv_anchor_q8",stage)],
            raw_stage[("q8_1",stage)],np)

    result={
      "schema":SCHEMA,
      "status":"CPU_ONLY_POSTCONV_GDN_NUMERIC_MODEL_DOWNSTREAM_LLAMA_NOT_CAPTURED",
      "model_sha256":prev.EXPECTED_GGUF_SHA,
      "capture_manifest_sha256":prev.sha256_file(manifest_path),
      "capture_token":6205,"absolute_position":0,"layer":0,
      "llama_captured_names":sorted(captured),
      "llama_capture_tensors_sha256":tensors,
      "llama_lib_sha256":manifest["libllama_sha256"],
      "source_sha256_and_assumptions":sources,
      "first_uncertain_stage":"beta_sigmoid, attn_gate_z, q/k normalization and subsequent GDN: no authenticated llama tensors after conv_output_silu-0",
      "authenticated_llama_comparisons_only":llama_comp,
      "stage_digests_cpu_model_not_llama":stages,
      "verified_cpu_invariants":pos0_invariants,
      "controlled_modes_vs_q8_cpu_only_not_llama":q8_control,
      "captured_llama_conv_anchor_vs_q8_cpu_only":anchor_vs_q8,
      "limitations":[
        "No llama downstream tensor captures: cannot establish llama GDN parity",
        "MTP and whole-model logits not evaluated; current PP2 CUDA kernels not run",
        "CPU FP64 matvec and CPU F32 non-fused operators not bitwise CUDA",
        "Pos0 recurrence uses zero previous state; no evidence for pos>=1 temporal decay",
        "GGUF F32 alpha projection unquantized for all modes per Qwen35/PP2 source",
      ],
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",
                      encoding="utf-8")
    print("PROVENANCE_MATCH physical GGUF/llama capture+lib SHA validated")
    print("AUTH_LLAMA_ONLY",json.dumps(llama_comp))
    for stage in ("beta_sigmoid","q_norm_scaled","gdn_o","gated_rms_silu",
                  "ssm_out_projection","first_residual_embedding_plus_core"):
        print(stage, "BF16_vs_Q8",q8_control[stage]["bf16_rne"],
              "FP32_vs_Q8",q8_control[stage]["fp32"])
    print("FIRST_UNCERTAIN_LAYER0_STAGE=postconv; no captured llama downstream")
    print("CPU_GDN_INVARIANTS_PASS",output)
    return result


def selftest():
    import numpy as np
    # Relative f32 versus f64 zero-state GDN tensor invariants.
    q=np.zeros((HV,KHD),dtype=np.float32)
    k=np.zeros_like(q)
    v=np.zeros((HV,VHD),dtype=np.float32)
    q[:,0]=np.float32(1.0/np.sqrt(128.0))
    k[:,0]=1.
    v[:,0]=2.
    beta=np.full(HV,0.25,dtype=np.float32)
    gl=np.full(HV,-8.,dtype=np.float32)
    a=gdn_pos0(q,k,v,beta,gl,np)
    np.testing.assert_allclose(a["o"][:,0],np.full(HV,0.5/np.sqrt(128)),
                               atol=1e-7,rtol=1e-6)
    np.testing.assert_array_equal(a["o"][:,1:],
                                  np.zeros((HV,VHD-1),dtype=np.float32))
    np.testing.assert_array_equal(a["state"][:,0,0],
                                  np.full(HV,0.5,dtype=np.float32))
    # Altering gl cannot change pos0 initial-zero-state output.
    b=gdn_pos0(q,k,v,beta,np.full(HV,-0.1,dtype=np.float32),np)
    np.testing.assert_array_equal(a["o"],b["o"])
    norm=np.arange(1,CC+1,dtype=np.float32)
    heads=split_and_qk_norm(norm,np)
    assert heads["q_norm"].shape==(HV,KHD)
    assert np.array_equal(heads["q_norm"][0],heads["q_norm"][16])
    assert np.array_equal(heads["k_norm"][1],heads["k_norm"][33])
    assert np.allclose(silu(np.array([0.,1.,-1.],dtype=np.float32),np),
        np.array([0.,1./(1+math.exp(-1)),-1./(1+math.exp(1))],dtype=np.float32))
    assert np.allclose(softplus_source(np.array([25.,0.,-25.],dtype=np.float32),np),
         np.array([25.,math.log(2),math.log(1+math.exp(-25))],dtype=np.float32),
         atol=1e-6)
    try:
        gdn_pos0(q,k,v,beta,np.ones(HV,dtype=np.float32),np)
        raise AssertionError("positive growth gate allowed")
    except Blocked:
        pass
    print("SELFTEST_PASS initial zero GDN, gl independence, group indexing, softplus/gates")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path)
    p.add_argument("--gguf",type=Path)
    p.add_argument("--source-root",type=Path)
    p.add_argument("--llama-source-root",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--selftest",action="store_true")
    args=p.parse_args()
    if args.selftest:
        selftest()
        return 0
    if any(getattr(args,n) is None for n in
           ("manifest","gguf","source_root","llama_source_root","out")):
        p.error("all of --manifest --gguf --source-root --llama-source-root --out required")
    run(args.manifest,args.gguf,args.source_root,args.llama_source_root,args.out)
    return 0


if __name__=="__main__":
    try:
        sys.exit(main())
    except (Blocked, KeyError, ValueError, OSError, ImportError) as exc:
        print("BLOCKED_LAYER0_GDN_CPU: "+str(exc),file=sys.stderr)
        sys.exit(2)
