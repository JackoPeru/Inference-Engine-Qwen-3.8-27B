# HERMES-27 × llama.cpp: independent numerical-differential investigation

**2026-11-13 | Read-only source / CPU analysis; no GPU jobs during this
follow-up.** Results here are diagnostic, not an acceptance-test waiver.
The mandatory independent FP32 release gate remains **FAIL / NO-GO**.
Do not regenerate or bless old E1/E2 fixtures.

## 1. Controlled evidence and what it actually proves

The existing independent runs used the same GGUF SHA256
`db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1`,
same token-ID input histories and two independently linked executables.
Their twelve 248320-element FP32 captures were re-read from the intact
scratch directories
`/tmp/h27-release-worker1-X3ZZUN/{llama-all,hermes-all}`.
Manifest backend, run-ID, prompt/continuation and all 24 trace SHA256
digests were verified again. Four prompts × three steps had the **same 12
greedy tokens**, including EOS, but all failed the strict independent
full-vocabulary RMSE criterion. Worst max abs 0.56496239 and RMSE
0.05635403.

A prior independent affine least-squares check found negligible reduction
of residual RMSE: Italian step 0 was 0.053513 → 0.052002. The new
`tests/reliability_logit_differential.py` measures the structure of the
unadjusted real residuals `r[v] = HERMES[v] - llama[v]`:

- Centered pairwise residual correlation across different traces:
  **min -0.0999; median 0.00506; max 0.2775**.
- Adjacent positions in one prompt: correlations range from **-0.0253
  to 0.2775**; they are generally small rather than a shared fixed pattern.
- SVD of the 12×248320 residual matrix after removing each vector's mean:
  first singular component explains **16.4%** of residual energy; first
  three explain 43.8%, first six 72.5%.

**Inference:** This is not well described by *one common vocabulary-position
bias* or one universal logit scale/offset. The residual appears largely
input/history-dependent. These statistics **do not distinguish** quantized
matrix multiplication noise from recurrent-state, RoPE, or norm errors;
they are not an architectural proof.

Machine-readable CPU evidence: scratch
`/tmp/h27-release-worker1-X3ZZUN/residual-quantization.json` (includes
the per-trace affine, KL, top-10 overlap, correlations, SVD and quantitative
quantization isolate).

## 2. Most concrete lead: NVFP4 × activation representation mismatch

**Verified source-level difference, not yet a proven sole cause.**

In the independent installed llama.cpp tree
`/opt/hermes/runtimes/llama-mainline`:

1. `ggml/src/ggml-cuda/mmf.cu:133-138` excludes quantized weights from
   the float-matvec path.
2. `ggml/src/ggml-cuda/ggml-cuda.cu:1858-1868` selects MMVQ for
   quantized single-column matvec when eligible.
3. `ggml/src/ggml-cuda/mmvq.cu:318-350` permits NVFP4 MMVQ for the
   single-token case; `mmvq.cu:1421-1506` explicitly converts the
   **F32 input activation** to **Q8_1** blocks before the dot product.
4. `ggml/src/ggml-cuda/quantize.cu:54-97` implements this as 32-value
   blocks with `d=amax/127`, `q=roundf(x/d)`, and a **FP16-stored**
   scale `d` (half2 stores another sum field).
5. `ggml/src/ggml-cuda/vecdotq.cuh:331-365` performs NVFP4×Q8_1
   integer DP4A products with decoded scales.

In HERMES:

- `distributed/pp2_fwd.cu` `f32_to_bf16_k` explicitly casts
  projected/normalized **F32 activations to BF16** before NVFP4 GEMV;
  decode `run_gdn_decode_layer` and `run_full_decode_layer` call this
  same path. The current scalar reference uses the same BF16 input.
- `kernels/src/nvfp4_gemv.cu:46-56,70-85,106-140` accepts BF16
  activations and dequantizes the same NVFP4 GGUF weights.

The **likely** independent CUDA matvec modes thus differ: llama
quantizes the *activation* into signed int8 per 32 values, while HERMES
rounds the activation to BF16. The llama MMVQ dispatcher was verified
in source; confirming its exact runtime dispatch (or a hypothetical
fused alternate path) requires a later profiled run, **not done here**.

### CPU isolated numerical experiment on the actual first layer

For each prompt's **first token only** (IT=6205, EN=8917,
chat=248045, short=6205), decoded the actual BF16 token embedding,
real `blk.0.attn_norm.weight` and 256 evenly spaced **real**
`blk.0.attn_qkv.weight` NVFP4 output rows. All three projections used
exactly the same independently dequantized FP32 weight matrix and
pre-normalized F32 activation: (a) unrounded F32, (b) BF16
round-to-nearest-even, (c) approximate llama Q8_1 quantization with
32-element per-block max scale stored in FP16. The output dot products
were computed in FP64 on CPU.

| Initial token | BF16 input vs F32 RMS | Q8_1 input vs F32 RMS | QKV BF16 vs Q8_1 RMS | QKV BF16 vs Q8_1 max abs |
|---|---:|---:|---:|---:|
| 6205 (IT/short) | 0.001574 | 0.005194 | **0.006003** | 0.01768 |
| 8917 (EN) | 0.001584 | 0.005103 | **0.005418** | 0.01463 |
| 248045 (chat) | 0.001612 | 0.005245 | **0.006090** | 0.02585 |

For these sampled rows, QKV BF16 vs ideal F32 input RMS was
0.00155–0.00206; Q8_1 vs ideal F32 RMS was 0.00502–0.00590.
Input BF16 vs Q8_1 RMS was 0.00535–0.00548.

**Interpretation:** The measurable difference is already large enough
that repeated, input-dependent effects over ~hundreds of projections
**could** contribute to a final 0.025–0.056 logit RMSE, consistent with
the observed weak cross-trace residual correlations. This is **not** a
demonstration that it explains the final discrepancy. The CPU emulator
uses NumPy ties-even rounding where the CUDA kernel uses `roundf`
ties-away; the exact quant kernel, scale multiplication order, FMA and
runtime dispatch need a genuine matched-input backend isolate.
These QKV samples are not direct llama.cpp GPU intermediate captures.

## 3. Other candidate numerical/semantic differences

### GDN recurrent update arithmetic — second-priority numeric lead

- llama.cpp `src/models/qwen35.cpp:355-454` computes `beta=sigmoid`,
  `g=ssm_a*softplus(alpha+dt)`, causal conv SiLU, L2 q/k, and uses
  `build_recurrent_attn`; `src/models/models.h:14-18` implements
  `rms_norm(x,eps/n)/sqrt(n)`, equivalent to true
  `x/sqrt(sum(x²)+eps)`.
- llama CUDA `ggml/src/ggml-cuda/gated_delta_net.cu:5-145` uses
  register FP32 states/warp reductions and a scaled Q readout;
  `src/models/delta-net-base.cpp:289-372` specifies the same
  `S←g*S+k*((v-g*(Sᵀk))*beta)` formula in the nonfused path.
- HERMES `kernels/src/gdn_step.cu:17-42` uses FP32 stored state,
  **FP64 accumulation per K-axis dot**, and a different reduction
  order. `distributed/pp2_fwd.cu:113-161` agrees on negative
  `ssm_a`, `hh%16`, L2 q/k and `1/sqrt(128)` Q scale.

Different rounding within recurrent updates is plausible. **No**
current evidence shows a wrong sign, wrong mapping or missing scale in
these inspected paths. To test numeric versus semantic mismatch,
inject **identical** initial `S` and `q,k,v,beta,g` into both
implementations, check `S` and `o` after each step, and measure
growth by layer/token. Do not compare different preprocessed inputs.

### Full attention RoPE / QK RMS / SDPA — stage to check, not culprit asserted

llama.cpp `src/models/qwen35.cpp:260-340` uses Q+gate
interleaved projection, per-head RMSNorm, 4 K/V heads, MRoPE
sections, sigmoid output gate and `1/sqrt(256)` KQ scale.
HERMES `distributed/pp2_fwd.cu` `q_norm_rope_pos_k`,
`k_norm_rope_pos_k`, `sdpa_gqa_decode_k` implements per-head norm,
32+32 split-half RoPE on first 64 dimensions (theta 1e7), 24:4
GQA and scale 1/16. The small-context text modes **appear
structurally compatible**, but equivalent MRoPE input positions
were not directly compared. A 1-token, position-0 test eliminates
nonzero RoPE angles and attention history to isolate other causes.

### Output norm and BF16 head / GGML accumulation — late-stage test

Both use GGUF `output_norm.weight` direct RMS multiplication and
`output.weight` BF16. HERMES `distributed/pp2_fwd.cu`
`head_bf16_k` accumulates in **double** (see original
helper near top of file). llama.cpp applies `build_norm` then
`build_lora_mm(model.output,...)`
(`src/models/qwen35.cpp:196-219`); runtime arithmetic
and output-head precision must be tested separately with **identical
final normalized hidden input**, not inferred from the logits alone.

## 4. Falsifiable differential experiment plan

**No GPU run has been authorized for this follow-up.** A later owner
can implement the experiments below in **new isolated test harnesses**
or opt-in trace hooks with explicit source ownership, without touching
the production default. Same checkpoint SHA256, *identical token IDs*,
separate process runs (the 19.69GB GGUF cannot be held twice in
16+16GiB), and bounded captures are prerequisites.

1. **Start at exactly one prompt token, pos=0.** Capture llama/HERMES
   final 248320-logit FP32 vectors for IDs 6205 and 248045, greedy
   `max_new_tokens=1`. If still >0.02 RMSE, residual cannot be
   solely a RoPE nonzero-position error, an incorrect KV history,
   or a later-token GDN carry mistake. If differences collapse,
   focus on positional/history semantics.
2. **Intercept llama.cpp intermediate tensors via existing native
   C API**, `llama_context_params.cb_eval`
   (`include/llama.h:392`), backed by
   `ggml_backend_sched_eval_callback` (`ggml/include/ggml-backend.h:
   305-321`). The callback is called with `ask=true` for selection
   and `ask=false` for an available computed tensor. Copy selected
   tensors into bounded host buffers using the GGML backend tensor
   read API; do not retain transient tensor pointers after callback.
   Confirm backend callback semantics and data transfer before use.
   For llama `src/models/qwen35.cpp` trace names:
   `attn_norm`, `linear_attn_qkv_mixed`,
   `conv_output_silu`, `q_conv_predelta`,
   `k_conv_predelta`, `v_conv_predelta`,
   `beta_sigmoid`, `gate`, `attn_output`,
   `linear_attn_out`, `post_ffn`, `result_norm`.
   On full attention blocks also `Qcur_normed`,
   `Kcur_normed`, `Qcur`, `Kcur`, `attn_pregate`.
3. **Obtain HERMES matching intermediates** using a future authorized
   opt-in trace (production files are outside this QA worker's
   write scope). Check in order:
   `embed -> blk0 attn_norm -> QKV -> conv -> beta/g/normalized qk
   -> GDN S/output -> block0 residual/post-FFN -> block3 Q/K/RoPE
   -> block31 output -> GPU PP boundary -> block32 input
   -> block63 post-FFN -> result_norm -> output head`.
   Each captured tensor must include name, layer, absolute position,
   dtype, shape, token history, checkpoint hash and backend mode.
4. **Isolate the first-layer quantization hypothesis using identical
   W and x**: capture llama's real `attn_norm` F32 output at block0,
   compare HERMES BF16+NVFP4 GEMV, llama CUDA NVFP4×Q8_1 MMVQ,
   a CPU reproduction of Q8_1 including `half2(d,sum)`, and an
   FP64 dequantized-W reference for a representative *and worst-case*
   sample of output rows. Confirm which input and arithmetic path
   matches each engine before asserting its contribution. The existing
   CPU 256-row test is a screening signal, not this controlled proof.
5. **Independent per-layer recurrence isolate:** take matching
   `q/k/v,beta,g` and a known FP32 `S0` from the first diverging
   layer. Execute llama fused GDN and HERMES GDN on the **same arrays**,
   compare `o` and new `S` after t=0,1,2,3. Run a zero-state
   single-token control and nonzero-state continuation. If the outputs
   match within measured backend numerical noise, move to the next
   stage; if not, distinguish FP32 warp reduction from FP64 accumulation
   by CPU emulation before altering semantics.
6. **At the first full-attention layer (block3)** compare pre-/post-
   RMS Q/K and then RoPE (position 0 and 1), attention scores/softmax
   and sigmoid gating. A mismatch *before* RoPE rules out RoPE as the
   original divergence.
7. **At final head** feed the *same* `result_norm[5120]` into the
   BF16 output matrix multiply of both engines. If same-input output
   logits agree closely, the difference originates in earlier hidden
   state propagation. If not, characterize the BF16 head arithmetic
   separately.
8. **Always retain two acceptance gates:** independent greedy
   token equality (currently PASS 12/12) and original independent
   full FP32 error (currently FAIL 12/12). Do not reduce thresholds
   from 0.20 max abs / 0.02 RMSE merely to turn the latter into PASS.
   Any legitimate precision-mode-specific release tolerance needs
   independently justified error budgets and new independent quality
   evidence rather than post-hoc fitted logits.

An especially diagnostic outcome would be: zero hidden mismatch before
block0 NVFP4 projection, ~0.006 QKV RMSE there, then growing drift
along all 64 layers. That would *support* mixed activation precision
as the mechanism. If instead block0 QKV agrees to near numerical noise
and the first large divergence appears at a GDN state, RoPE or head
boundary, then **reject the activation-quantization explanation
as the primary cause** and investigate the first divergent operation.

## 5. Reproduce offline diagnostics, safely

No CUDA, server access or inference required once the scratch artifacts
exist:

```bash
python3 tests/reliability_logit_differential.py \
  --scratch /tmp/h27-release-worker1-X3ZZUN \
  --output /tmp/h27-release-worker1-X3ZZUN/residual-analysis.json
```

Optional **read-only** GGUF block0 same-W/X quantization probe
(bounded 256 sampled output rows; Python NumPy required):

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 \
python3 tests/reliability_logit_differential.py \
  --scratch /tmp/h27-release-worker1-X3ZZUN \
  --gguf /home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf \
  --sample-rows 256 \
  --output /tmp/h27-release-worker1-X3ZZUN/residual-quantization.json
```

The script fails on altered/missing trace files, unmatched prompt histories,
or GGUF SHA mismatch; does not produce a release PASS decision. The optional
GGUF SHA256 calculation reads the entire ~19.7GB file sequentially and may
take longer than the trace-only calculation; that is disk/CPU IO, **not
inference**. It does not allocate GPU memory.
