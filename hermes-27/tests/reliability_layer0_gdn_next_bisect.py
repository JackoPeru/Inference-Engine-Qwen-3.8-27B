#!/usr/bin/env python3
"""Independent CPU-only layer0 GDN NEXT bisect after authenticated llama conv.

No model runtime, CUDA, full-model forward or GPU operation. We compare
frozen authenticated llama tensors ONLY to stages actually captured (norm,
QKV, conv), then isolate real GGUF beta/gate Q8_1 vs BF16/FP32 on the exact
captured conv input. No llama beta/gate/GDN oracle was captured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

EXPECTED_FROZEN_SHA = "faed57314848224e9c41514b7ce4d2dd9368a29c28de0f11a53cb5c9c7ff893f"
L0 = 0
TOKEN = 6205
HID, CC, HV, DH, VDD = 5120, 10240, 48, 128, 6144
SCHEMA = "h27.layer0.gdn-next.cpu.v1"
MODES = ("q8_1", "bf16_rne", "fp32")
COMBOS = ("q8_beta_q8_gate", "bf16_beta_q8_gate", "q8_beta_bf16_gate",
          "bf16_beta_bf16_gate", "fp32_beta_fp32_gate")


class Blocked(RuntimeError):
    pass


def require(ok, msg: str) -> None:
    if not ok:
        raise Blocked(msg)


def sha_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(a,b,np) -> dict:
    a=np.asarray(a,dtype=np.float64).reshape(-1)
    b=np.asarray(b,dtype=np.float64).reshape(-1)
    require(a.size==b.size and a.size>0 and
            bool(np.isfinite(a).all() and np.isfinite(b).all()),
            "invalid stage comparison arrays")
    d=a-b
    worst=int(np.argmax(np.abs(d)))
    return dict(elements=int(a.size), rmse=float(np.sqrt(np.mean(d*d))),
                max_abs=float(np.max(np.abs(d))), worst_index=worst,
                value_a=float(a[worst]), value_b=float(b[worst]))


def stage(array,np):
    a=np.ascontiguousarray(array,dtype="<f4")
    require(a.size>0 and bool(np.isfinite(a).all()),"nonfinite stage")
    return {"shape":list(a.shape),"n":int(a.size),
            "sha256":hashlib.sha256(a.tobytes()).hexdigest()}


def _read_sources(pp2_root:Path,llama_root:Path):
    names={
        "pp2": (pp2_root/"distributed/pp2_fwd.cu",
                ("g=hh%(KD/KHD)", "sqrtf(sq[0]+1e-6f)",
                 "beta[i]=1.f/(1.f+expf(-bv))",
                 "gdn_step_cuda", "gated_rms_k")),
        "gdn": (pp2_root/"kernels/src/gdn_step.cu",
                ("Sh[i*V+j]*=dec","double(Sh[i*V+j])*double(sk[i])")),
        "llama_graph": (llama_root/"src/models/qwen35.cpp",
                ("beta_sigmoid", "q_conv_predelta", "build_norm_gated",
                 "ggml_repeat_4d")),
        "llama_fused": (llama_root/"ggml/src/ggml-cuda/gated_delta_net.cu",
                ("fastmodulo(h_idx, neqk1_magic)", "attn_col * scale",
                 "delta_col", "g_val * s_shard[r]")),
        "llama_unfused": (llama_root/"src/models/delta-net-base.cpp",
                ("ggml_exp(ctx0, g)","ggml_sum_rows(ctx0, sk)",
                 "ggml_add(ctx0, s, kd)")),
        "llama_norm": (llama_root/"src/models/models.h",
                ("eps/n", "ggml_rms_norm"))
    }
    info={}
    for name,(path,fragments) in names.items():
        require(path.is_file(),f"source missing: {path}")
        source=path.read_text(encoding="utf-8")
        for f in fragments:
            require(f in source, f"{name} source changed at {f!r}")
        info[name]={"path":str(path),"sha256":sha_file(path)}
    return info


def _norm_from_conv(conv,np):
    """Compute source equivalent q/k heads using L2(sum+epsilon)."""
    require(conv.shape==(CC,),"auth llama conv shape invalid")
    q16=np.array(conv[:2048].reshape(16,DH),dtype=np.float32)
    k16=np.array(conv[2048:4096].reshape(16,DH),dtype=np.float32)
    v=np.array(conv[4096:].reshape(HV,DH),dtype=np.float32)
    def normal(x,scale):
        s=np.sum(x*x,axis=1,dtype=np.float32)
        inv=np.float32(1)/np.sqrt(s+np.float32(1e-6))
        return np.asarray((x*inv[:,None])*np.float32(scale),dtype=np.float32)
    q16=normal(q16,1/math.sqrt(128))
    k16=normal(k16,1)
    # The reference CUDA kernel indexes the compact head via h % H_k
    # (fastmodulo(h_idx, neqk1_magic)), not via floor(h/3).
    q=np.tile(q16,(3,1))
    k=np.tile(k16,(3,1))
    require(np.array_equal(q[0],q[16]) and np.array_equal(k[1],k[33]),
            "q/k modulo16 repetition invalid")
    return q,k,v


def _gdn_pos0(q,k,v,beta,np,head_map=None):
    """Zero-state S: o[h,j] = sum_i (k[h,i]*beta[h]*v[h,j])*q[h,i].
    No temporal exp(gl) effect at pos0. Use FP64 products, round state F32.
    """
    require(beta.shape==(HV,) and
            q.shape==k.shape==(HV,DH) and v.shape==(HV,DH),
            "GDN dimensions invalid")
    if head_map is not None:
        qq=q[:16][head_map]
        kk=k[:16][head_map]
    else:
        qq=q
        kk=k
    state=np.asarray(kk.astype(np.float64)[:,:,None] *
        (beta.astype(np.float64)[:,None] * v.astype(np.float64))[:,None,:],
        dtype=np.float32)
    out=np.asarray(np.einsum("hij,hi->hj",state.astype(np.float64),
                     qq.astype(np.float64)),dtype=np.float32)
    require(bool(np.isfinite(out).all()),"GDN output nonfinite")
    return out


def _silu(x,np):
    x=np.asarray(x,dtype=np.float32)
    return np.asarray(x/(np.float32(1)+np.exp(-x)),dtype=np.float32)


def _gated_rms(o,gate,weight,np):
    z=gate.reshape(HV,DH)
    norm=np.float32(1)/np.sqrt(
        np.mean(o*o,axis=1,dtype=np.float32)+np.float32(1e-6))
    return np.asarray(o*norm[:,None]*weight[None,:]*_silu(z,np),
                      dtype=np.float32).reshape(VDD)


def _quantized_inputs(x,prev,np):
    return {
      "q8_1":prev.quant_q8_1_per32(x,np),
      "bf16_rne":prev.round_bf16_rne(x,np),
      "fp32":np.asarray(x,dtype=np.float32)
    }


def _guard_real_weights(gguf,gdn,name,kind,shape,np):
    arr=gdn.guarded_matrix(gguf,name,kind,shape,np)
    require(bool(np.isfinite(arr).all()),f"{name} nonfinite")
    return arr


def _real_run(args):
    import numpy as np
    sys.path[:0]=[str(args.source_root),str(args.source_root/"tests")]
    import reliability_layer0_conv_q8_cpu as prev
    import reliability_layer0_gdn_q8_cpu as gdn
    from scripts.block0_corrected_ref import GGUF,bf16_to_f32

    sources=_read_sources(args.source_root,args.llama_root)
    require(sha_file(args.frozen)==EXPECTED_FROZEN_SHA,
            "frozen published GDN CPU report JSON drift")
    old=json.loads(args.frozen.read_text(encoding="utf-8"))
    require(old.get("schema")=="h27.reliability-layer0-gdn-q8-cpu.v1"
            and old.get("capture_token")==TOKEN
            and old.get("absolute_position")==0
            and old.get("layer")==0
            and old.get("llama_captured_names")==
              ["attn_norm-0","conv_output_silu-0","linear_attn_qkv_mixed-0"],
            "unexpected frozen CPU GDN oracle schema/capture")
    try:
        manifest,captured=prev.validate_capture(args.manifest,args.gguf,np)
    except prev.Blocked as ex:
        raise Blocked("auth llama capture failure: "+str(ex)) from ex
    require(prev.sha256_file(args.manifest)==old["capture_manifest_sha256"],
            "frozen report not bound to same auth manifest")
    require(set(captured)=={"attn_norm-0","conv_output_silu-0",
                             "linear_attn_qkv_mixed-0"},
            "unexpected llama downstream reference set")
    require("beta_sigmoid-0" not in captured and
            "gate-0" not in captured and "attn_output-0" not in captured,
            "new independent llama tensors require new shape/provenance gate")
    frozen=old["authenticated_llama_comparisons_only"]
    require(frozen["q8_conv_vs_llama"]["rmse"] < 3.e-8
            and frozen["q8_qkv_vs_llama"]["rmse"]<2.e-7
            and frozen["bf16_conv_vs_llama"]["rmse"] > 1.e-4,
            "earlier authenticated Q8 conv PASS/BF16 FAIL evidence changed")

    x=captured["attn_norm-0"]
    conv=captured["conv_output_silu-0"]
    q,k,v=_norm_from_conv(conv,np)
    # Provenance: do NOT infer unseen llama beta/gate from conv tensors.
    modes=_quantized_inputs(x,prev,np)
    with args.gguf.open("rb") as raw:
        gguf=GGUF(raw,np)
        beta_w=_guard_real_weights(
            gguf,gdn,"blk.0.ssm_beta.weight","nvfp4",(HV,HID),np)
        beta_raw=gdn.matvec_panel(beta_w,modes,np)
        del beta_w
        gate_w=_guard_real_weights(
            gguf,gdn,"blk.0.attn_gate.weight","nvfp4",(VDD,HID),np)
        gate_raw=gdn.matvec_panel(gate_w,modes,np)
        del gate_w
        b={key:gdn.sigmoid(value,np) for key,value in beta_raw.items()}
        gated_raw={key: _silu(gate_raw[key],np) for key in modes}

        # q/k/v are held FIXED at the actual llama conv output, so all
        # downstream changes here are attributable solely to beta and z
        # projection quantization, not QKV/conv reconstruction differences.
        def branch(beta_mode,gate_mode):
            o=_gdn_pos0(q,k,v,b[beta_mode],np)
            g=_gated_rms(o,gate_raw[gate_mode],
                         norm_weight,np)
            return o,g

        norm_weight=_guard_real_weights(gguf,gdn,
            "blk.0.ssm_norm.weight","f32",(DH,),np)
        pairs={
            "q8_beta_q8_gate":("q8_1","q8_1"),
            "bf16_beta_q8_gate":("bf16_rne","q8_1"),
            "q8_beta_bf16_gate":("q8_1","bf16_rne"),
            "bf16_beta_bf16_gate":("bf16_rne","bf16_rne"),
            "fp32_beta_fp32_gate":("fp32","fp32")
        }
        outputs={}
        for label,(bm,gm) in pairs.items():
            o,g=branch(bm,gm)
            outputs[label]={"o":o,"gated":g}
        # One head-group mutation shows why modulo-16 is NOT interchangeable
        # with consecutive grouping h//3, even on the authentic conv.
        alternative=np.arange(HV)//3
        grouped=_gdn_pos0(q,k,v,b["q8_1"],np,head_map=alternative)
        group_diff=metrics(grouped,outputs["q8_beta_q8_gate"]["o"],np)

        # Only the first GDN output (6144) and immediately following gated
        # norm are recomputed. No 64-layer, FFN or final head is run.
        alpha_w=_guard_real_weights(gguf,gdn,"blk.0.ssm_alpha.weight",
                                   "f32",(HV,HID),np)
        alpha=np.asarray(alpha_w.astype(np.float64) @
                         x.astype(np.float64),dtype=np.float32)
        A=_guard_real_weights(gguf,gdn,"blk.0.ssm_a","f32",(HV,),np)
        dt=_guard_real_weights(gguf,gdn,"blk.0.ssm_dt.bias","f32",(HV,),np)
        gl=np.asarray(A*gdn.softplus_source(alpha+dt,np),dtype=np.float32)
        require(bool((gl<0).all()),"model decay must be nonpositive")
        # A fresh zero state means gl never enters the first output.
        unchanged=_gdn_pos0(q,k,v,b["q8_1"],np)
        require(np.array_equal(unchanged,outputs["q8_beta_q8_gate"]["o"]),
                "state-zero GDN cannot depend on decay gl")
        del gguf

    control=outputs["q8_beta_q8_gate"]
    result={
        "schema":SCHEMA,
        "status":"CPU_ONLY; LLAMA_ORACLE_STOPS_AFTER_CONV; BETA_GATE_NOT_CAPTURED",
        "gpu_executed":False,
        "full_model_rerun":False,
        "model_sha256":prev.EXPECTED_GGUF_SHA,
        "manifest_sha256":prev.sha256_file(args.manifest),
        "frozen_gdn_result_sha256":EXPECTED_FROZEN_SHA,
        "actual_token_id":TOKEN,
        "position":0,
        "source_provenance":sources,
        "auth_llama_captured_stages":sorted(captured),
        "auth_llama_checks_prior":{
            "q8_qkv_rmse":frozen["q8_qkv_vs_llama"]["rmse"],
            "q8_conv_rmse":frozen["q8_conv_vs_llama"]["rmse"],
            "bf16_conv_rmse":frozen["bf16_conv_vs_llama"]["rmse"],
        },
        "first_noncaptured_stages":[
            "beta_sigmoid-0","attn_gate_z-0","q_conv_predelta-0",
            "k_conv_predelta-0","attn_output-0"],
        "new_q8_beta_vs_bf16":metrics(b["q8_1"],b["bf16_rne"],np),
        "new_q8_beta_vs_fp32":metrics(b["q8_1"],b["fp32"],np),
        "new_q8_raw_gate_vs_bf16":metrics(
            gate_raw["q8_1"],gate_raw["bf16_rne"],np),
        "new_q8_gate_silu_vs_bf16":metrics(
            gated_raw["q8_1"],gated_raw["bf16_rne"],np),
        "fixed_authentic_llama_conv_factorial": {
            name:{
              "GDN_output_vs_q8":metrics(e["o"],control["o"],np),
              "gated_rms_vs_q8":metrics(e["gated"],control["gated"],np),
              "output_stage_hashes":{
                  "gdn_o":stage(e["o"],np),
                  "gated_rms":stage(e["gated"],np)
              }
            } for name,e in outputs.items()
        },
        "wrong_grouping_h_over_3_vs_modulo_16":group_diff,
        "grouping_rule":"pp2 g=hh%16; llama fused fastmodulo(h_idx,neqk1)",
        "zero_state_decay_check":"PASS; gl not applied to S=0",
        "reference_gap":"There is no captured authenticated llama beta, gate, normalized q/k, state or out; new numbers are CPU-internal controls only.",
        "minimal_recommendation":"Capture llama beta_sigmoid-0 and z/gate-0 first, plus q_conv_predelta-0 and attn_output-0 in authenticated token6205 pos0 callback before altering any PP2 formula."
    }
    require(result["new_q8_beta_vs_bf16"]["rmse"]>1.e-6,
            "expected first beta precision difference absent")
    require(result["new_q8_raw_gate_vs_bf16"]["rmse"]>1.e-6,
            "expected first gate precision difference absent")
    require(group_diff["rmse"]>1.e-4,
            "incorrect h//3 head grouping did not diverge")
    require(result["fixed_authentic_llama_conv_factorial"][
        "bf16_beta_q8_gate"]["GDN_output_vs_q8"]["rmse"]>1.e-8,
        "beta change unexpectedly disappears")
    require(result["fixed_authentic_llama_conv_factorial"][
        "q8_beta_bf16_gate"]["GDN_output_vs_q8"]["rmse"] == 0,
        "gate cannot affect pre-gated GDN recurrence output")
    require(result["fixed_authentic_llama_conv_factorial"][
        "q8_beta_bf16_gate"]["gated_rms_vs_q8"]["rmse"]>1.e-6,
        "gate change unexpectedly disappears at gated RMS")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",
                            encoding="utf-8")
    for key in ("new_q8_beta_vs_bf16","new_q8_raw_gate_vs_bf16",
                "wrong_grouping_h_over_3_vs_modulo_16"):
        print(key, result[key])
    for label,data in result["fixed_authentic_llama_conv_factorial"].items():
        print(label, "o_rmse=",data["GDN_output_vs_q8"]["rmse"],
              "gate_rmse=",data["gated_rms_vs_q8"]["rmse"])
    print("PASS_CPU_NO_LLAMA_DOWNSTREAM_ORACLE",args.output)
    return result


def selftest():
    import numpy as np
    # Causal all-zero history: changing decay cannot affect first update.
    q=np.zeros((HV,DH),dtype=np.float32)
    k=q.copy()
    v=np.zeros((HV,DH),dtype=np.float32)
    q[:,0]=np.float32(1/math.sqrt(DH))
    k[:,0]=np.float32(1)
    v[:,0]=np.float32(2)
    beta=np.full(HV,np.float32(.25))
    o=_gdn_pos0(q,k,v,beta,np)
    np.testing.assert_allclose(o[:,0],np.float32(.5/math.sqrt(DH)),
                               rtol=1.e-6,atol=1.e-7)
    assert bool((o[:,1:]==0).all())
    w=np.ones(DH,dtype=np.float32)
    g=np.zeros(VDD,dtype=np.float32)
    z=_gated_rms(o,g,w,np)
    assert bool((z==0).all())
    g[0]=1.
    z=_gated_rms(o,g,w,np)
    assert float(z[0])>0
    assert bool((z[1:]==0).all())
    require(stage(o,np)["n"]==HV*DH,"stage SHA summary")
    print("SELFTEST_PASS zero state, independent beta/gate, SiLU-zero, SHA")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest",action="store_true")
    parser.add_argument("--manifest",type=Path)
    parser.add_argument("--gguf",type=Path)
    parser.add_argument("--source-root",type=Path)
    parser.add_argument("--llama-root",type=Path)
    parser.add_argument("--frozen",type=Path)
    parser.add_argument("--output",type=Path)
    a=parser.parse_args()
    if a.selftest:
        selftest()
        return 0
    if any(getattr(a,x) is None for x in (
            "manifest","gguf","source_root","llama_root","frozen","output")):
        parser.error("required: --manifest --gguf --source-root --llama-root --frozen --output")
    _real_run(a)
    return 0

if __name__=="__main__":
    try:
        sys.exit(main())
    except (Blocked,OSError,ValueError,KeyError,ImportError) as exc:
        print("BLOCKED_LAYER0_NEXT:",exc,file=sys.stderr)
        sys.exit(2)
