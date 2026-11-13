# Layer 0, position 0: independent CPU Q8_1 → GDN → first residual

**2026-11-13 — isolated numerical follow-up; CPU only.** This report extends the independently validated Q8_1/GGUF convolution trace in `reports/RELIABILITY_LAYER0_Q8_CONV.md` to the complete **pre-FFN GDN path** of layer 0, token ID **6205**, absolute position **0**. **The only independently authenticated llama.cpp tensors are `attn_norm-0`, `linear_attn_qkv_mixed-0`, `conv_output_silu-0`. Downstream Q/K/beta/gate, recurrent state, output projection and first residual have NO llama capture.** Nothing here establishes numerical llama parity past convolution, current HERMES GPU parity, or an end-to-end release PASS.

## Verified outcomes

An independent CPU implementation using the exact GGUF weights successfully constructed and recorded **three consistent activation-precision paths** from the captured initial layer normalization through the initial-zero-state GDN update, gated RMS, raw GGUF `ssm_out.weight` and embedding residual. Source contract/dimension, model and trace SHA checks, selftest and CPU mathematical invariants all passed. All stages contain finite FP32 values.

| Stage / number of values | BF16-RNE vs CPU Q8_1 RMSE | FP32 unrounded vs CPU Q8_1 RMSE | BF16 max abs vs Q8 | Status |
| --- | ---: | ---: | ---: | --- |
| `beta_sigmoid`, 48 | 0.000600960 | 0.000543018 | 0.00158626 | CPU internal only |
| `q_norm_scaled`, 6,144 | 1.36260e-5 | 1.34095e-5 | 0.000130075 | CPU internal only |
| `gdn_o`, 6,144 | 2.10174e-5 | 1.35793e-5 | 0.000899971 | CPU internal only |
| `gated_rms_silu`, 6,144 | 0.000100266 | 0.0000997993 | 0.00251985 | CPU internal only |
| `ssm_out_projection`, 5,120 | **0.000698547** | **0.000647335** | **0.0183268** | CPU internal only |
| **First residual**, 5,120 | **0.000698547** | **0.000647335** | **0.0183268** | CPU internal only |

All comparisons in this table are **between locally reconstructed CPU variants**, not versus an unseen llama.cpp tensor. The FP32 control max-abs difference from Q8 at the first residual is **0.00311561** (index 1047); BF16 max abs **0.01832676** (index 3994). Controls quantize **each** real NVFP4 matvec input to their chosen convention (Q8 per 32, BF16 RNE, or unrounded FP32), including the `ssm_out` input; `ssm_alpha.weight` is an actual F32 matrix and uses the same original captured F32 `attn_norm` vector across modes.

A fourth control **anchors only the convolution** to the authentic llama `conv_output_silu-0` while retaining CPU Q8_1-projected gate/beta and CPU Q8_1 `ssm_out`. It differs from the Q8 oracle by:
- `gdn_o` RMSE **2.29222e-9**, max **1.19209e-7**;
- `gated_rms_silu` RMSE **3.68332e-9**, max **1.19209e-7**;
- `ssm_out_projection` and first residual **FP32 identical** (zero difference) after the Q8_1 input quantization, at this one position.

This establishes **internal stability/sensitivity** of the CPU pipeline against the already-tiny authenticated conv discrepancy. It does **not** authenticate those downstream CPU numbers as llama outputs.

## Inputs and provenance (fail closed)

| Evidence | Authenticated identity |
| --- | --- |
| Actual GGUF | `/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf` |
| Physical 19GB GGUF streamed SHA256 | `db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1` |
| llama capture manifest | `/tmp/h27-layer0-llama-pos0-prime3/manifest.json`; SHA256 `49b0052027532e1e8572d9d865517b2beef92c0e8176a02adacd35efe46d540b` |
| Capture strict identity | `h27.reliability-layer-cb.v1`; llama C API, single actual token 6205, layer0, absolute pos0, `n_gpu_layers=80`, CUDA0 FP32 ne0-fastest, original 3 stages only |
| Installed authenticated libllama SHA256 | `fd9e3e0810c6eca2e37abf443bcf15963671cbd0ee8489a37461a521094d4` |
| `attn_norm-0` SHA256 | `492ae3d622039e01dcce9c5d8bc32ff2986e1134d5c50fad47ff366e8561bfa3` (5,120 × F32) |
| `linear_attn_qkv_mixed-0` SHA256 | `4b32da962e03f2481aa869955cea378b7921418d920459e9e62cf9300c08184f` (10,240 × F32) |
| `conv_output_silu-0` SHA256 | `12513886591fd03cc1a28c2d671cb76217242339300037f1988da4031a9d56a2` (10,240 × F32) |
| Scratch machine-readable stage results | `/tmp/h27_w5_layer0_gdn_q8_result.json`, schema `h27.reliability-layer0-gdn-q8-cpu.v1` |
| Source code authored | **NEW** `tests/reliability_layer0_gdn_q8_cpu.py`; imports existing **unchanged** `tests/reliability_layer0_conv_q8_cpu.py` for authenticated capture, GGUF Q8 and causal conv |

An independent GGUF parser in unchanged `scripts/block0_corrected_ref.py:43-148` reads and decodes the real mmap'd model; the new test enforces actual tensor types and shapes, including:
- `attn_qkv.weight`: NVFP4, [10240,5120]; `ssm_conv1d.weight`: F32 [10240,4].
- `attn_gate.weight`: NVFP4 [6144,5120]; `ssm_beta.weight`: NVFP4 [48,5120]; `ssm_alpha.weight`: F32 [48,5120].
- `ssm_a`, `ssm_dt.bias`: F32 [48]; `ssm_norm.weight`: F32 [128]; `ssm_out.weight`: NVFP4 [5120,6144]; `token_embd.weight`: BF16, includes row 6205.

The same physical embedding and real `attn_norm.weight` reproduce the authentic llama `attn_norm-0` with **max abs 0** under this CPU norm implementation. Authenticated first 3 stages (full-array checks, not samples):

| Authentic llama capture comparison | CPU Q8 RMSE | Max abs | CPU BF16 RMSE | CPU FP32 RMSE |
| --- | ---: | ---: | ---: | ---: |
| Normalized 5,120 F32 | 0 | 0 | — | — |
| First NVFP4 QKV (10,240) | 1.57648647e-7 | 3.81469727e-6 | 0.00596864625 | 0.00570703076 |
| Depthwise conv+SiLU (10,240) | **2.07688537e-8** | **9.53674316e-7** | 0.000300890202 | 0.000204534249 |

Manifest source-type, tensor shape/bytes, CUDA0 origin, SHA256, actual libllama SHA256 and checkpoint SHA256 are checked prior to numerical modeling. A **tampered copy** of the manifest (`absolute_position=1`) was explicitly rejected with **exit code 2** and **no output JSON**; the genuine manifest run exited 0.

## Source-faithful chain and mathematical checks

The test reads and hashes the installed actual PP2/CUDA and independent llama sources. The source file hashes recorded in the machine-readable JSON are:

| Source | SHA256 |
| --- | --- |
| HERMES `distributed/pp2_fwd.cu` | `0d11b9a15974e56d6ef3003384127759b36a678b8c9fc79e3dcf277781fd145c` |
| HERMES `kernels/src/gdn_step.cu` | `a432d5ce57467320e232ec48ab19ca659c387394ad7e8a8ad70f5bd207b8d183` |
| llama mainline `src/models/qwen35.cpp` | `ef973e9c789eb52ef27a77e0b1f36c6f41a378733a994d41b5924e3be012f671` |
| llama mainline `src/models/delta-net-base.cpp` | `5e3b318953f854b3825dd7cfe4b483ca7935cd5f6680438a3ece4be9b4d5428f` |
| llama mainline `src/models/models.h` (GDN L2) | `d239ec2d363a5f2618eb91d486a3f53d9bf07b96be49b6d25b6c6688600d50a6` |

Production PP2/CUDA source hashes on the Windows shared worktree were independently checked to match those in the server's source tree; the CPU test did not copy or compile these sources. The test checks relevant **source expressions**, not just fixed expected output numbers. These textual checks provide evidence that the encoded formulas agree with the inspected source revision; they cannot prove the CUDA runtime's selected kernels or accumulation order.

1. **Gate/beta/alpha projection.** The same actual normalized 5120-vector is passed through GGUF `attn_gate` and `ssm_beta` NVFP4 projections, with chosen quantization before GEMV. The F32 `ssm_alpha` projection stays unquantized. PP2 `distributed/pp2_fwd.cu:122-139,520-530`; llama `qwen35.cpp:343-374`. Beta= sigmoid(beta_raw), softplus uses exact `x>20 ? x : logf(1+expf(x))` source branch, and `gl=ssm_a*softplus(alpha+dt)` (stored `ssm_a` must be negative, **not** `-exp(ssm_a)` again).
2. **Head grouping and Q/K L2.** From convolved [Q2048,K2048,V6144], 16 Q and 16 K heads of 128 are repeated **modulo16** to match the 48 V heads, exactly as PP2 `split_qkv_k` `:140-152` and llama `ggml_repeat_4d` `qwen35.cpp:427-444`. Normalize Q and K using `1/sqrt(sum(x²)+1e-6)`; scale Q further by `1/sqrt(128)`. PP2 `:153-166`, llama `models.h:14-17` algebraically identical. Test verifies repeated head [0,16,32] identities and all squared-norm identities.
3. **Recurrent GDN with initial S=0.** `S ← exp(gl)*S`; `d ← beta*(v - Sᵀk)`; `S ← S+k⊗d`; `o ← Sᵀq` (`kernels/src/gdn_step.cu:4-15,17-41`, llama `delta-net-base.cpp:319-370`). At **pos0**, with fresh zero state, the exact algebra simplifies to `S = k⊗(beta*v)` and `o = (q·k)*beta*v`. Because the previous state is zero, **decay `exp(gl)` cannot affect first-token output**. CPU state is stored in FP32 after FP64 outer multiplication; reduction to output uses FP64 then rounds to FP32 (similar to HERMES CPU/GPU recurrence but not bitwise GGML CUDA).
4. **Gated RMS.** Normalize the 128-component `o` vector **per V head**, using `sqrt(mean(o²)+1e-6)`, multiply directly by real `ssm_norm.weight[128]`, then multiply by SiLU(z) from `attn_gate`. PP2 `:167-178`, llama `qwen35.cpp:243-252,451-460`.
5. **Projection/residual.** Quantize the 6144 gated output (per selected mode), apply real raw GGUF NVFP4 `ssm_out.weight` [5120,6144], and compute `embedding[token6205] + core`. This is the **first GDN residual before post-attention RMS/FFN**, not final layer0 output. PP2 `:553-557`, llama `qwen35.cpp:455-466` plus parent residual graph.

The layer0-pos0 CPU test verifies the following invariants for the Q8 path:

| Invariant | Observed |
| --- | --- |
| 48 Q/K heads repeated from 16 groups | PASS, exact array equality |
| Q scaled squared L2 norm | min 0.007812474, max 0.007812500; expected ~1/128 |
| K squared L2 norm | min 0.99986634, max 0.99999949; expected <=1 with epsilon |
| Sigmoid beta range | 0.1657284 to 0.9953488 (strictly within 0..1) |
| Decay exp(gl) range, A negative | 0.00267598 to 0.99999970 (<=1) |
| Initial-state output vs algebraic `(q·k)*beta*v` | RMSE **3.922e-10**, max abs **2.98023e-8** over 6144 values |
| Changing `gl` on a zero state | Selftest produces exactly identical `o` |
| All stage FP32 values finite | PASS |

### Selected Q8 CPU per-stage SHA256 fingerprints

The numerical JSON records **SHA256/shape/dtype/min/max/RMS for every stage in all four modes**, including the 786,432-float recurrent state. Selected Q8 stages:

| Stage | FP32 elements | SHA256 |
| --- | ---: | --- |
| `norm_input_fp32` | 5120 | `492ae3d622039e01dcce9c5d8bc32ff2986e1134d5c50fad47ff366e8561bfa3` |
| `conv_output_silu` | 10240 | `1a01f415f524091f98ffcc68d6706466ea2c0387e7d6072d45c6d5a9ac1bcc8e` |
| `beta_sigmoid` | 48 | `9f4ae5142916ab7f093464b40f74a159732d2afa3595fb5e16e256a4f62dd262` |
| `q_norm_scaled` | 6144 | `dc78cab5a58b04ef1a26a9e5a90ddeea0f01feab229f6a23f695d3c35c367952` |
| `k_norm` | 6144 | `23abf7bdf849d77cffc22bf1ad679c183bec263b9d9c70b14fa7e640c1f225a6` |
| `recurrent_state_pos0` | 786432 | `46d70e4750666e49959cdadae712f10e99e4412faf172ca200911c4b066d285c` |
| `gdn_o` | 6144 | `a8b05a9a3024a0233f162167819480166173ad144c200c2e5ee401eb0030f0da` |
| `gated_rms_silu` | 6144 | `d486182927bb75b8d35c79156c9976e056f860559735528dce3867463c514d13` |
| `ssm_out_projection` | 5120 | `3d16d38a0e2e985ab6590dd6b928ec0cf7583345b8931d58006a4b0c2906e01c` |
| `first_residual_embedding_plus_core` | 5120 | `285006a03d1c0210db2ca5be0adf7596b5f6cb147d51c7106d0abcb97d752c3a` |

These are **CPU-stage reproducibility markers**, not hashes of unauthenticated llama results; only the original three captured tensors have independent llama provenance.

## Execution and negative controls

Only files added in this worker assignment:

- **NEW** `tests/reliability_layer0_gdn_q8_cpu.py`
- **NEW** `reports/RELIABILITY_LAYER0_GDN_Q8.md`

No existing convolution test, PP2, kernels, CMake, serving or ops files were edited. On the server the test ran as isolated source copy `/tmp/h27_w5_layer0_gdn_q8_cpu.py`; all output confined to `/tmp/h27_w5_layer0_gdn_q8_result.json`.

```bash
PY=/opt/hermes/runtimes/exllamav3-venv/bin/python
$PY -B /tmp/h27_w5_layer0_gdn_q8_cpu.py --selftest
# SELFTEST_PASS

OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
$PY -B /tmp/h27_w5_layer0_gdn_q8_cpu.py \
  --manifest /tmp/h27-layer0-llama-pos0-prime3/manifest.json \
  --gguf /home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf \
  --source-root /home/matteo/hermes-27-gh/hermes-27 \
  --llama-source-root /opt/hermes/runtimes/llama-mainline \
  --out /tmp/h27_w5_layer0_gdn_q8_result.json
# exit 0: CPU_GDN_INVARIANTS_PASS
```

A deliberate **tampered manifest** (`absolute_position=1` instead of 0) was created **only in /tmp**, then used with the same CPU command. The final diagnostic returned **exit 2 / BLOCKED_LAYER0_GDN_CPU**, and the invalid manifest produced **no result JSON**. Selftest also rejects invalid `gl>0`, incorrect head shape and incorrect zero-state mathematical behavior.

## First unsupported stage / concrete next experiment

**First unsupported independent llama comparison after the verified QKV/conv capture:** `attn_gate_z`, `ssm_beta`, `q_conv_predelta/k_conv_predelta` and their normalization, `beta_sigmoid`, `a_softplus`, `dnet_add_ar_state`, `attn_out_norm`, `linear_attn_out` and residual. The installed llama graph `qwen35.cpp:343-466` names these callback stages, but **none exists in the authenticated three-tensor capture**. There is no numeric proof that its fused/unfused GDN path matches this independent CPU recurrence or that its state starts at precisely the assumed zero history at all positions.

The minimal future falsifiable check is an authenticated **new** independent llama capture of `z`, `beta_sigmoid`, `alpha_softplus`, `q_conv_predelta`, `k_conv_predelta`, `state_predelta`, `dnet_add_ar_state`, `attn_out_norm` and `linear_attn_out`, all bound to the same GGUF SHA and token pos0. Compare each against this report's stage SHA and full FP32 numeric arrays (or regenerate and dump in a separate authorized diagnostic). If the first mismatch appears in gate/beta GEMV, isolate actual llama Q8_1 dispatch; if only after recurrence, inspect fused state mapping/accumulation. Then capture **positions 1–4** to test the real accumulated S state and `gl`, because pos0 algebra cannot validate temporal decay. The corresponding current HERMES GPU stage manifests would require separate GPU-exclusive approval and PP2 maintainer ownership.

**No release gate has been relaxed.** This is an independent, checkpoint-bound CPU mathematical/source test—not proof of llama downstream parity, HERMES current GPU correctness or whole-model FP32-logit acceptance. The independent final-logit numerical gate remains **NO-GO**.
