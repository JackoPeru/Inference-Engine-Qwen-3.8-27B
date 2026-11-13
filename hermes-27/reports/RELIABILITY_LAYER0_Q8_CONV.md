# HERMES-27 — independent Q8_1 → depthwise conv+SiLU numerical bisect (layer 0 / pos 0)

**Date:** 2026-11-13. **Method:** authenticated real llama.cpp callback tensors + exact checkpoint GGUF weights, CPU-only/NumPy. **Result: PASS for the isolated layer0/position0 Q8_1 convolution reproduction; NOT a production HERMES GPU or full-logit gate.** No GPU, live service requests, edits to existing tests, PP2, serving or CMake.

## Decision

Independent Q8_1 quantization of the *actual* captured llama attention-normalized FP32 input, projection through the *actual* GGUF `blk.0.attn_qkv.weight`, and convolution through the real `blk.0.ssm_conv1d.weight` with three causal zero inputs reproduces the **entire 10240-element** `conv_output_silu-0` tensor of llama.cpp to **RMSE 2.07688537e-8, max absolute 9.53674316e-7**.

- By comparison, BF16-rounded activations on exactly the same GGUF matrix and conv weight yield conv RMSE **0.0003008902025**; Q8_1 reduces this error by **~14,488×**.
- FP32-unrounded activations yield conv RMSE **0.0002045342494**; Q8_1 reduces this error by **~9,848×**.
- Using the genuine captured llama QKV without recomputing the projection, the independent causal CPU conv reproduces its captured conv to **RMSE 1.06705943e-8, max abs 4.76837158e-7**. This strong isolated control confirms **zero-history positional indexing/tap 3 and SiLU** for the first token.
- Q8_1-projected QKV itself reproduces the captured llama QKV to **RMSE 1.57648647e-7**, consistent with worker-1's previous CPU independent check.
- No Q8_1 production kernel was tested or enabled. This result **does not prove** subsequent-position convolution cache, downstream GDN state, other layers, final logits, or the current PP2 GPU QKV values.

## 1. Provenance and explicit gates

| Provenance | Verified |
| --- | --- |
| Model | `/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf` |
| **Physical model SHA256** (streamed actual file) | `db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1` |
| llama capture directory | `/tmp/h27-layer0-llama-pos0-prime3` |
| manifest SHA256 | `49b0052027532e1e8572d9d865517b2beef92c0e8176a02adacd35efe46d540b` |
| Capture identity | `schema=h27.reliability-layer-cb.v1`, `backend=llama.cpp`, `CAPTURED_NOT_INDEPENDENTLY_COMPARED`, `n_gpu_layers=80`, **token 6205**, absolute **pos 0**, layer **0**, CUDA0 tensors, format little-endian FP32 |
| Actual installed libllama SHA256 | `fd9e3e0810c6eca2e37b57abf443bcf15963671cbd0ee8489a37461a521094d4` (matched path and manifest) |
| Exporter SHA256 | `2259be0c502f5488fee09c8826a16986cb437167192cd352d0e408801c5ef241` (**manifest-reported**, independent exporter executable path not supplied by capture) |
| `attn_norm-0` | FP32 [5120,1,1,1], 20,480B, SHA256 `492ae3d622039e01dcce9c5d8bc32ff2986e1134d5c50fad47ff366e8561bfa3` |
| `linear_attn_qkv_mixed-0` | FP32 [10240,1,1,1], 40,960B, SHA256 `4b32da962e03f2481aa869955cea378b7921418d920459e9e62cf9300c08184f` |
| `conv_output_silu-0` | FP32 [10240,1,1,1], 40,960B, SHA256 `12513886591fd03cc1a28c2d671cb76217242339300037f1988da4031a9d56a2` |
| Actual conv implementation source | `/opt/hermes/runtimes/llama-mainline/ggml/src/ggml-cuda/ssm-conv.cu`, SHA256 `684ef5e8180ba88bb1f389310b28f8a694cb107a924492c43682a5c5c1754424` |
| Full numerical manifest of **this** CPU run | `/tmp/h27_w5_layer0_conv_q8_result.json` (schema `h27.reliability-layer0-q8-conv-cpu.v1`) |

The new `tests/reliability_layer0_conv_q8_cpu.py` rejects incompatible capture schema, backend, status, prompt token, absolute position, layer, CUDA0 buffer provenance, source FP32 dtype/strides, duplicate or missing stage, file length or hash, physical model SHA, wrong libllama SHA, unexpected GGUF QKV/conv tensor types/shapes and nonfinite tensor values. Physical checkpoint SHA256 is rechecked before running the CPU projection. The GGUF embedding token 6205 plus real `blk.0.attn_norm.weight` reproduces actual captured `attn_norm-0` with **maxabs 0** in this test.

The test **reads** (does not execute) the installed llama convolution source and checks the ring indexing `x[(i+j)%d_conv]*w[j]`, the sliding-state update and `ggml_cuda_op_silu_single` invocation. The exact model's `blk.0.ssm_conv1d.weight` is confirmed GGUF F32 dims [4,10240], NumPy shape [10240,4]; QKV GGUF tensor is NVFP4 shape [10240,5120]. No synthetic weights or historical unmanifested E2 artifacts were used for numerical conclusions.

## 2. Mathematical and source contract

At absolute position 0 the depthwise 4-tap convolution has **three zero causal predecessors**, hence for every channel `c∈[0,10239]`:

```
conv_raw[0,c] = 0 * Wc[c,0] + 0 * Wc[c,1]
              + 0 * Wc[c,2] + qkv[0,c] * Wc[c,3]
conv_output_silu[0,c] = conv_raw[0,c] / (1 + exp(-conv_raw[0,c]))
```

llama's `src/models/delta-net-base.cpp:449-474` builds the convolution state from 3 preceding values plus `qkv_mixed`; `ggml/src/ggml-cuda/ssm-conv.cu:29-56` implements four oldest-to-newest coefficients and optional fused SiLU. PP2's prefill `distributed/pp2_fwd.cu:98-105` and decode `:1068-1078` use the same positional order at pos0; `:402-411` saves 3 raw QKV values for continuation. **The current result covers pos0 only**, not whether a restarted/rewound decoder uses the right cached state.

For projection, the exact captured llama `attn_norm-0` vector (5120 FP32) is transformed into 3 strictly isolated activation controls with the **same** GGUF NVFP4 weights:
- `q8_1`: groups of 32, `amax/127` FP32, round-to-nearest with ties away from zero to signed int8, scale stored as **FP16**, then dequantized FP32. The CPU projection sums in FP64 over each 128-row panel and casts FP32. This mirrors ggml CUDA quantization semantics but does **not** reproduce the exact CUDA DP4A accumulation order.
- `bf16_rne`: FP32→BF16 round-to-nearest-even→FP32 before the same panel matvec (models PP2 projection-input precision, **not** a PP2 GPU capture).
- `fp32`: no activation quantization before the same GGUF matrix matvec.

Independent source references: `/opt/hermes/runtimes/llama-mainline/ggml/src/ggml-cuda/mmvq.cu:1421-1426,1500-1506` for F32 input/Q8_1 quantization; `ggml/src/ggml-cuda/vecdotq.cuh:331-361` for NVFP4×Q8_1 dot. Existing worker-1 report `reports/RELIABILITY_LAYER_BISECT.md:53-88` independently established the per32 Q8_1 projection. The present task **adds** the full real GGUF depthwise conv+SiLU and the direct-llama-QKV conv control.

## 3. Measured full 10,240-element comparisons

All numbers are candidate minus authentic llama **captured** tensor, computed from entire arrays (no row sampling or aggregate-only subset).

| Input mode | QKV vs llama: RMSE | QKV maxabs | conv+SiLU vs llama: RMSE | conv maxabs |
| --- | ---: | ---: | ---: | ---: |
| Direct llama QKV (*conv-only positive control*) | **0** | **0** | **1.06705943e-8** | **4.76837158e-7** |
| **CPU GGUF NVFP4×Q8_1** | **1.57648647e-7** | **3.81469727e-6** | **2.07688537e-8** | **9.53674316e-7** |
| CPU GGUF NVFP4×BF16-RNE | 0.00596864625 | 0.0366220474 | 0.000300890202 | 0.00882434845 |
| CPU GGUF NVFP4×FP32 | 0.00570703076 | 0.0352149010 | 0.000204534249 | 0.00703954697 |

The maximum absolute Q8 conv error is at **channel 4585** (llama 8.998784065, CPU 8.998785019); the max Q8 QKV error is also at **channel 4585** (llama 32.22339630, CPU 32.22340012). Zero components >1e-3 for the two Q8 comparisons; all outputs finite.

**This is strong evidence that the layer0 pos0 conv stage has no additional macroscopic semantic mismatch once the projection inputs follow llama Q8_1**, because a direct llama-QKV conv control matches at ~1e-8 RMSE and the independently projected Q8 conv matches at ~2e-8. The larger BF16 and unrounded-FP32 controls do **not** recover llama's conv. It is therefore incorrect to blame a layer0 conv tap-index reversal on these captures.

**Do not claim a whole-model root cause from this stage.** There may be subsequent Q8_1 precision amplification, different GDN recurrence, attention RoPE/KV, FFN, or head precision. No authentic HERMES current GPU layer0 output was captured by this task. Neither the 12-trace independent llama full-logit release threshold nor service readiness was retested. Existing release remains NO-GO.

## 4. Test execution and repeatability

Only new owned source file:

`tests/reliability_layer0_conv_q8_cpu.py`

Runtime copy used for isolated CPU execution:

`/tmp/h27_w5_layer0_conv_q8_cpu.py`

The selftest covers BF16-RNE, true Q8_1 FP16 stored-scale behavior (unit input reconstructs **0.99993896484375**, not 1), Q8 zero group, invalid input rejection and conv pos0 zero-history/tap-order validation:

```bash
PY=/opt/hermes/runtimes/exllamav3-venv/bin/python
$PY -B /tmp/h27_w5_layer0_conv_q8_cpu.py --selftest
# SELFTEST_PASS
```

Full authenticated replay command, with a source copy of this new test in `/tmp` (copied from local owned test; no runtime code or production path modified):

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/opt/hermes/runtimes/exllamav3-venv/bin/python -B /tmp/h27_w5_layer0_conv_q8_cpu.py \
  --manifest /tmp/h27-layer0-llama-pos0-prime3/manifest.json \
  --gguf /home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf \
  --source-root /home/matteo/hermes-27-gh/hermes-27 \
  --llama-conv-source /opt/hermes/runtimes/llama-mainline/ggml/src/ggml-cuda/ssm-conv.cu \
  --out /tmp/h27_w5_layer0_conv_q8_result.json
# Exit 0: CONV_SEMANTICS_MATCH=True Q8_CONV_MATCH=True
```

No new GPU execution, model load through llama runtime, service restart, checkpoint mutation, use of unauthorized source edits or CMake changes occurred.

## 5. Next falsifiable test; ownership decision

A new **GPU-exclusive grant** and PP2 owner's cooperation would be required to capture **current** PP2 layer0 `attn_norm`, pre-conv QKV and post-conv output with exact GGUF SHA, same token6205 pos0, then compare those values against this independently reproduced Q8 path and against genuine llama tensors. Next also capture token at **pos1–4** (so conv cache state is nonzero) and the first GDN recurrence for the same actual prefix. If PP2's BF16 QKV matches the BF16 oracle while llama matches Q8, the proven first-stage cause transfers to current PP2. Until such evidence, this is **a CPU-only source/semantics isolation** and no production correction or numeric tolerance change is warranted.
