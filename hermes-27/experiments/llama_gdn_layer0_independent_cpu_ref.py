#!/usr/bin/env python3
"""Source-derived CPU mathematical reference for Qwen35 layer0 GDN, token6205 pos0.

Only the input and convolution are authentic llama.cpp callback tensors. Every
beta, gate, normalized head, recurrent state and output below is independently
DERIVED on CPU from GGUF weights. This is not a llama post-convolution oracle,
fused CUDA arithmetic replay, GPU test or complete model evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

TOKEN, POS, LAYER = 6205, 0, 0
HIDDEN, GROUPS, HEADS, WIDTH, CHANNELS = 5120, 16, 48, 128, 10240
EPS = 1e-6
SCHEMA = "h27.llama-gdn-layer0-independent-cpu-ref.v1"
GGUF_SHA = "db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1"
MANIFEST_SHA = "49b0052027532e1e8572d9d865517b2beef92c0e8176a02adacd35efe46d540b"

# These are hashes of the INSTALLED llama.cpp sources, not PP2 kernels.
SOURCE_SHAS = {
    "src/models/qwen35.cpp": "ef973e9c789eb52ef27a77e0b1f36c6f41a378733a994d41b5924e3be012f671",
    "src/models/delta-net-base.cpp": "5e3b318953f854b3825dd7cfe4b483ca7935cd5f6680438a3ece4be9b4d5428f",
    "src/models/models.h": "d239ec2d363a5f2618eb91d486a3f53d9bf07b96be49b6d25b6c6688600d50a6",
    "ggml/src/ggml-cuda/gated_delta_net.cu":
        "fe5cfe4a35195fac999e8bd93d3ed18c68830096d4019bafa60e5da91d6ef4bf",
}
SOURCE_CONTRACT = {
    "src/models/qwen35.cpp": (
        'cb(beta, "beta_sigmoid", il)', "beta = ggml_sigmoid(ctx0, beta)",
        "ggml_softplus(ctx0, alpha_biased)", "model.layers[il].ssm_a",
        "build_gdn_l2_norm(ctx0, q_conv", "build_gdn_l2_norm(ctx0, k_conv",
        "ggml_repeat_4d(ctx0, q_conv", "ggml_repeat_4d(ctx0, k_conv",
        'cb(conv_output_silu, "conv_output_silu", il)',
        "build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il)",
        "ggml_silu(ctx0, gate)", "model.layers[il].wqkv_gate",
    ),
    "src/models/delta-net-base.cpp": (
        "const float scale = 1.0f / sqrtf(S_k)",
        "q = ggml_scale(ctx0, q, scale)", "g = ggml_exp(ctx0, g)",
        "s = ggml_mul(ctx0, s, g)", "sk = ggml_sum_rows(ctx0, sk)",
        "d = ggml_sub(ctx0, v, ggml_transpose(ctx0, sk))",
        "d = ggml_mul(ctx0, d, b)", "s = ggml_add(ctx0, s, kd)",
    ),
    "src/models/models.h": (
        "ggml_rms_norm(ctx, x, eps/n)", "1.0f/sqrtf(n)",
    ),
    "ggml/src/ggml-cuda/gated_delta_net.cu": (
        "fastmodulo(h_idx, neqk1_magic)",
        "(v_t[col] - g_val * kv_col) * beta_val",
        "g_val * s_shard[r] + k_reg[r] * delta_col",
        "attn_data[col] = attn_col * scale",
        "const float scale = 1.0f / sqrtf((float) S_v)",
    ),
}


class Blocked(RuntimeError):
    """Missing provenance, invalid shape or altered implementation contract."""


def require(ok, why):
    if not ok:
        raise Blocked(why)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_contract_text(key, text):
    for fragment in SOURCE_CONTRACT[key]:
        require(fragment in text, f"llama contract missing {key}: {fragment}")


def verify_installed_llama(llama_root):
    sha = {}
    for key, want in SOURCE_SHAS.items():
        file = llama_root / key
        require(file.is_file(), f"missing installed llama source {file}")
        data = file.read_bytes()
        check_contract_text(key, data.decode("utf-8"))
        got = hashlib.sha256(data).hexdigest()
        require(got == want, f"installed llama source SHA drift {key}: {got}")
        sha[key] = got
    return sha


def finite(array, shape, np, name):
    arr = np.asarray(array)
    require(arr.shape == shape and bool(np.isfinite(arr).all()),
            f"{name}: expected finite {shape}, got {arr.shape}")
    return arr


def q8_per32(input_vector, np):
    """GGML Q8_1 quantized input as a separate CPU scalar mathematical model.

    Each 32-component block stores a rounded FP16 scale and signed int8 codes.
    Rounding of the int8 codes uses nearest, ties away from zero.
    """
    x = np.asarray(input_vector, dtype=np.float32)
    require(x.ndim == 1 and x.size % 32 == 0 and bool(np.isfinite(x).all()),
            "invalid Q8_1 activation")
    out = np.empty_like(x)
    for i in range(0, x.size, 32):
        values = x[i:i + 32]
        high = float(np.max(np.abs(values)))
        if high == 0:
            out[i:i + 32] = 0
            continue
        scale = float(np.float32(high / 127.0))
        stored = float(np.float16(scale))
        require(math.isfinite(stored) and stored > 0, "Q8_1 stored scale invalid")
        for j, v in enumerate(values):
            ratio = float(np.float32(v / np.float32(scale)))
            code = min(127, int(math.floor(abs(ratio) + .5)))
            out[i + j] = np.float32(math.copysign(code, ratio) * stored)
    return out


def qk_normalize(conv, np, groups=GROUPS, heads=HEADS, width=WIDTH,
                 eps=EPS):
    """GGML [width,groups] Q/K and [width,heads] V, head index = h % groups.

    models.h: rms_norm(x, eps/width) / sqrt(width)
    = x / sqrt(sum_i(x_i^2)+eps), followed by fused delta-net Q scale.
    """
    require(heads % groups == 0 and width > 0 and eps >= 0,
            "unsupported Q/K head ratio, width or epsilon")
    conv = finite(conv, ((2*groups + heads)*width,), np, "conv")
    q0 = np.asarray(conv[:groups*width], dtype=np.float64).reshape(groups, width)
    k0 = np.asarray(conv[groups*width:2*groups*width],
                    dtype=np.float64).reshape(groups, width)
    v = np.asarray(conv[2*groups*width:], dtype=np.float64).reshape(heads,width)
    q0 /= np.sqrt(np.sum(q0*q0, axis=1, keepdims=True) + eps)
    k0 /= np.sqrt(np.sum(k0*k0, axis=1, keepdims=True) + eps)
    # Fused CUDA indexes compact Q/K by fastmodulo(h_idx, neqk1_magic).
    indices = np.arange(heads) % groups
    return q0[indices], k0[indices], v, indices


def sigmoid(x, np):
    a = np.asarray(x, dtype=np.float64)
    return 1 / (1 + np.exp(-np.clip(a, -80, 80)))


def source_gate(beta_pre, alpha, dt_bias, a_negative, np):
    """ggml sigmoid(beta); ssm_a * softplus(alpha+dt) in llama graph."""
    beta_pre, alpha = np.asarray(beta_pre, np.float64), np.asarray(alpha,np.float64)
    dt_bias, a_negative = np.asarray(dt_bias,np.float64), np.asarray(a_negative,np.float64)
    require(beta_pre.shape == alpha.shape == dt_bias.shape == a_negative.shape,
            "beta/alpha/A/dt head geometry mismatch")
    require(bool(np.isfinite(a_negative).all() and (a_negative < 0).all()),
            "ssm_a must be stored as negative exp(A_log)")
    beta = sigmoid(beta_pre, np)
    gate_exponent = a_negative * np.logaddexp(0., alpha + dt_bias)
    require(bool((gate_exponent < 0).all() and (beta > 0).all() and
                 (beta < 1).all()), "invalid delta-net beta/gate range")
    return beta, gate_exponent


def delta_step(q, k, v, beta, exponent, previous, np):
    """Mathematical delta rule in FP64; CPU-derived, not CUDA reduction order.

    S_decay = exp(g) * S_previous
    delta = beta * (v - S_decay^T k)
    S_next = S_decay + k (outer) delta
    output = S_next^T q / sqrt(width).
    The last factor is in both llama unfused scaling and fused CUDA output.
    """
    q = np.asarray(q,np.float64)
    k = np.asarray(k,np.float64)
    v = np.asarray(v,np.float64)
    require(q.ndim == 2 and k.shape == v.shape == q.shape,
            "delta-net Q/K/V shape mismatch")
    heads, width = q.shape
    beta = finite(beta,(heads,),np,"beta")
    exponent = finite(exponent,(heads,),np,"gate exponent")
    previous = finite(previous,(heads,width,width),np,"previous state")
    require(bool((beta >= 0).all() and (beta <= 1).all() and
                 (exponent <= 0).all()), "invalid beta or state decay")
    state_decay = np.asarray(previous,np.float64) * np.exp(exponent)[:,None,None]
    prediction = np.einsum("hij,hi->hj",state_decay,k)
    innovation = beta[:,None]*(v-prediction)
    state = state_decay+np.einsum("hi,hj->hij",k,innovation)
    output = np.einsum("hij,hi->hj",state,q)/math.sqrt(width)
    require(bool(np.isfinite(state).all() and np.isfinite(output).all()),
            "nonfinite GDN result")
    return output,state,innovation


def gated_rms(output, z, weights, np, eps=EPS):
    output = np.asarray(output,np.float64)
    z = finite(z,output.shape,np,"raw z")
    weights = finite(weights,(output.shape[1],),np,"ssm_norm")
    denom = np.sqrt(np.mean(output*output,axis=1,keepdims=True)+eps)
    return (output/denom) * weights[None,:] * (z * sigmoid(z,np))


def matrix_vector(gguf, name, kind, dims, vec, np):
    typ_code = {"f32":0,"nvfp4":40}[kind]
    require(name in gguf.tensors, f"missing GGUF {name}")
    physical, actual_typ, _ = gguf.tensors[name]
    require(tuple(physical[::-1]) == dims and actual_typ == typ_code,
            f"GGUF {name} bad shape or dtype: {physical}, {actual_typ}")
    weights = gguf.tensor(name,kind)
    finite(weights,dims,np,name)
    # Panel dot uses FP64 to represent mathematical GGUF matrix product;
    # ggml CUDA/FMA activation quantization and reduction order may differ.
    result = np.empty(dims[0],dtype=np.float32)
    v = np.asarray(vec,np.float64)
    require(v.shape == (dims[1],) and bool(np.isfinite(v).all()),
            "matvec activation shape/nonfinite")
    for start in range(0,dims[0],64):
        stop = min(dims[0],start+64)
        result[start:stop] = np.asarray(
            np.asarray(weights[start:stop],np.float64) @ v,np.float32)
    del weights
    return result


def gguf_rms_epsilon(gguf):
    """Read the pinned GGUF's F32 epsilon metadata, fail if absent or altered."""
    key=b"qwen35.attention.layer_norm_rms_epsilon"
    marker=struct.pack("<Q",len(key))+key+struct.pack("<I",6)
    start=gguf.mm.find(marker,24,gguf.data_offset)
    require(start >= 0 and start+len(marker)+4 <= gguf.data_offset,
            "missing GGUF qwen35 RMS epsilon F32 metadata")
    value=struct.unpack_from("<f",gguf.mm,start+len(marker))[0]
    require(math.isfinite(value) and abs(value-EPS) < 1e-12,
            f"GGUF RMS epsilon does not match source equations: {value}")
    return value


def describe(value, np):
    a = np.ascontiguousarray(value,dtype="<f4")
    require(a.size > 0 and bool(np.isfinite(a).all()), "invalid result stage")
    return {"shape":list(a.shape), "sha256":hashlib.sha256(a.tobytes()).hexdigest(),
            "min":float(a.min()),"max":float(a.max()),
            "l2_rms":float(np.sqrt(np.mean(a.astype(np.float64)**2)))}


def run(manifest_path, gguf_path, source_root, llama_root, output_path):
    import numpy as np
    source_root, llama_root = Path(source_root), Path(llama_root)
    sources = verify_installed_llama(llama_root)
    require(sha256(manifest_path) == MANIFEST_SHA,
            "auth llama manifest SHA mismatch")
    sys.path[:0] = [str(source_root),str(source_root/"tests")]
    from scripts.block0_corrected_ref import GGUF
    import reliability_layer0_conv_q8_cpu as capture_validator
    manifest,captured = capture_validator.validate_capture(
        Path(manifest_path),Path(gguf_path),np)
    require(set(captured) == {"attn_norm-0","linear_attn_qkv_mixed-0",
                             "conv_output_silu-0"} and
            manifest.get("prompt_ids") == [TOKEN] and
            manifest.get("absolute_position") == POS and
            manifest.get("layer") == LAYER and
            manifest.get("model_sha256") == GGUF_SHA,
            "not the three authenticated llama first-token captures")
    x = finite(captured["attn_norm-0"],(HIDDEN,),np,"auth llama norm")
    conv = finite(captured["conv_output_silu-0"],(CHANNELS,),np,
                  "auth llama conv")
    q,k,v,head_map = qk_normalize(conv,np)
    q8 = q8_per32(x,np)
    with Path(gguf_path).open("rb") as fh:
        gguf = GGUF(fh,np)
        checked_epsilon = gguf_rms_epsilon(gguf)
        beta_raw = matrix_vector(gguf,"blk.0.ssm_beta.weight","nvfp4",
                                 (HEADS,HIDDEN),q8,np)
        z_raw = matrix_vector(gguf,"blk.0.attn_gate.weight","nvfp4",
                              (HEADS*WIDTH,HIDDEN),q8,np)
        # The alpha matrix is GGUF F32; llama beta/alpha arithmetic may use
        # different backend reduction orders. Derive its mathematical value.
        alpha = matrix_vector(gguf,"blk.0.ssm_alpha.weight","f32",
                              (HEADS,HIDDEN),x,np)
        def vector(name):
            require(name in gguf.tensors and gguf.tensors[name][1] == 0 and
                    tuple(gguf.tensors[name][0][::-1]) == (HEADS,),
                    f"GGUF {name} bad F32 head vector")
            return np.asarray(gguf.tensor(name,"f32"),np.float64)
        neg_a = vector("blk.0.ssm_a")
        dt = vector("blk.0.ssm_dt.bias")
        require("blk.0.ssm_norm.weight" in gguf.tensors and
                gguf.tensors["blk.0.ssm_norm.weight"][1] == 0 and
                tuple(gguf.tensors["blk.0.ssm_norm.weight"][0][::-1]) ==
                (WIDTH,), "GGUF ssm_norm bad shape/type")
        rms_weights = gguf.tensor("blk.0.ssm_norm.weight","f32")
    beta, exponent = source_gate(beta_raw,alpha,dt,neg_a,np)
    zero = np.zeros((HEADS,WIDTH,WIDTH),dtype=np.float64)
    output,state,innovation = delta_step(q,k,v,beta,exponent,zero,np)
    # At position 0, previous state is zero and exponent cannot affect GDN.
    analytic = beta[:,None]*v * np.einsum("hi,hi->h",q,k)[:,None] / math.sqrt(WIDTH)
    require(np.max(np.abs(output-analytic)) < 1e-12,
            "zero-state GDN does not agree with independently factored formula")
    opposite_decay = np.full(HEADS,-9.,dtype=np.float64)
    other,_,_ = delta_step(q,k,v,beta,opposite_decay,zero,np)
    require(np.array_equal(output,other),"pos0 GDN depended on decay")
    gated = gated_rms(output,z_raw.reshape(HEADS,WIDTH),rms_weights,np)
    result = {
        "schema":SCHEMA,
        "status":"CPU_EQUATION_ONLY_NO_AUTHENTIC_LLAMA_POSTCONV_REFERENCE",
        "model_sha256":GGUF_SHA,
        "manifest_sha256":MANIFEST_SHA,
        "installed_llama_source_sha256":sources,
        "token":TOKEN,"absolute_position":POS,"layer":LAYER,
        "backend_cpu":"FP64_MATHEMATICAL_EQUATIONS_FROM_LLAMA_SOURCE",
        "authentic_llama_tensors":sorted(captured),
        "first_uncertain_stages":["beta_sigmoid-0","z-0",
                                  "q_conv_predelta-0","k_conv_predelta-0"],
        "head_mapping":"value_head % 16; 48 value heads to 16 Q/K heads",
        "q_scale":"1/sqrt(128) applied after delta-net contraction",
        "gguf_qwen35_rms_epsilon":checked_epsilon,
        "state_history":"assumed all-zero for first token at absolute pos0",
        "pos0_gamma_effect":"exactly zero because previous recurrent state is zero",
        "note":"CPU-derived tensors are NOT independently captured llama GDN tensors",
        "stages_cpu_derived_not_llama":{
            "q8_norm_input":describe(q8,np),
            "beta_pre_sigmoid":describe(beta_raw,np),
            "beta_sigmoid":describe(beta,np),
            "alpha":describe(alpha,np),
            "gamma_exponent":describe(exponent,np),
            "raw_z":describe(z_raw,np),
            "q_l2_normalized_unscaled":describe(q,np),
            "k_l2_normalized":describe(k,np),
            "v":describe(v,np),
            "delta_innovation":describe(innovation,np),
            "recurrent_state_pos0":describe(state,np),
            "gdn_output":describe(output,np),
            "gated_rms_silu":describe(gated,np),
        },
        "limits":[
            "No authentic llama beta, gate, q/k normalized, GDN or final output.",
            "CPU FP64 matrix products and recurrence are not CUDA reduction/rounding.",
            "Pos0 cannot test nonzero recurrent state or temporal decay.",
            "No PP2 GPU, llama GPU, full-logit or MTP accuracy test.",
        ],
    }
    require(np.array_equal(head_map, np.arange(HEADS)%GROUPS),
            "head mapping drift")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",
                           encoding="utf-8")
    print("CPU_EQUATION_PASS_NO_LLAMA_DOWNSTREAM_ORACLE",output_path)
    print("gdn_output_sha256",result["stages_cpu_derived_not_llama"]
          ["gdn_output"]["sha256"])
    print("gated_rms_sha256",result["stages_cpu_derived_not_llama"]
          ["gated_rms_silu"]["sha256"])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--gguf",type=Path,required=True)
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--llama-root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    run(args.manifest,args.gguf,args.source_root,args.llama_root,args.out)
    return 0


if __name__=="__main__":
    try:
        sys.exit(main())
    except (Blocked,KeyError,ValueError,OSError,ImportError) as exc:
        print("BLOCKED_LLAMA_GDN_CPU_REF:",exc,file=sys.stderr)
        sys.exit(2)
