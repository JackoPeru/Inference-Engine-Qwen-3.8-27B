# HERMES-27 independent layer bisection — llama.cpp callback harness

**Status: CPU/offline harness compiled and synthetic callback verified.
Real 27B layer captures NOT RUN; production release remains NO-GO.**
2026-11-13. No GPU access was requested or used for this task.

## Deliverables and ownership

Only new isolated QA source files were authored:

- `tests/reliability_layer_llama_cb.cpp`: independent llama.cpp C API
  layer-0/other-layer tensor capture via `llama_context_params.cb_eval`.
- `tests/reliability_layer_sha256.hpp`: standalone streaming SHA256 used
  to tag the actual checkpoint, linked libllama, capture executable and
  each bounded FP32 tensor.
- `tests/reliability_layer_compare.py`: offline fail-closed
  same-model/same-position cross-engine tensor comparator.
- `reports/RELIABILITY_LAYER_BISECT.md`: this report.

No production, CMake, PP2, kernels, serving, tokenizer, or other
workers' files were modified.

## C API callback contract: source verified

Installed independent llama.cpp:
`/opt/hermes/runtimes/llama-mainline/`.

| Item | Exact source | Consequence |
|---|---|---|
| Callback entry | `include/llama.h:392-393` (`cb_eval`, `cb_eval_user_data`) | Set in `llama_context_params` prior to `llama_init_from_model` |
| Callback signature | `ggml/include/ggml-backend.h:305-318` | `bool callback(ggml_tensor*,bool ask,void*)` |
| Selection and delivery | `ggml/src/ggml-backend.cpp:1798-1831` | `ask=true` selects node; scheduler computes its split and synchronizes backend; `ask=false` receives computed node; returning false there stops iteration |
| Host transfer | `ggml/include/ggml-backend.h:93` and `ggml/src/ggml-backend.cpp:350-362` | `ggml_backend_tensor_get` copies from the correct backend; requires tensor data and owning buffer |
| View ownership | `ggml/src/ggml-backend.cpp:350-352` | For views the buffer comes from `view_src`; never dereference `tensor->data` on CPU when GPU-backed |
| Tensor naming | `src/llama-context.cpp:2600-2604` | Graph labels become `<name>-<layer>`; global labels without layer suffix |
| Actual Qwen35 graph labels | `src/models/qwen35.cpp:227-455` | `attn_norm-0`, `linear_attn_qkv_mixed-0`, `conv_output_silu-0`, `beta_sigmoid-0`, `gate-0`, `attn_output-0`, `linear_attn_out-0`, `post_ffn-0`; full-attention stages include `Qcur_normed-3`, `Kcur_normed-3`, `Qcur-3`, `Kcur-3`, `attn_pregate-3` |

These names are **verified in installed source**, not yet observed
in a 27B runtime callback. The default capture set is restricted
according to GDN layers (layer % 4 != 3) and full attention
(layer % 4 == 3). The `--include` option provides narrow,
fail-closed selection in case graph fusion elides individual nodes.

### Backend read and boundedness

The callback reads a selected node only during `ask=false` (after
the scheduler's own backend synchronization). It follows the
`view_src` chain to establish a valid buffer and verifies
`tensor->data`, `ne[]`, `nb[0]`, raw backend extent and per-tensor
and total output budgets **before allocating the host output**.

Each logical `ne[0]` row is copied separately using
`ggml_backend_tensor_get(t,buf,offset,bytes)` and strides
`nb[1..3]`. This handles noncontiguous GGML views without treating
GPU pointers as host memory or spilling padding into a captured
tensor. Supported source types: F32, F16, BF16. Stored files:
**little-endian FP32, `ne[0]` fastest**, all entries finite.
The callback rejects unsupported types, unallocated buffers,
unexpected duplicate labels, truncated views, stale output files,
too many captured tensors, or budget overruns. A blocked callback
sets an error and the driver refuses to write a successful manifest.

Defaults: maximum **2 MiB per tensor**, **32 MiB total**,
**32 selected tensors**. A synthetic CPU test exercises a strided
`2×4` view, correct FP32 layout and file SHA256, no callback
for another layer, duplicate-label rejection, and 16-byte budget
rejection on a 32-byte output. The test does **not** execute the
27B network.

## Compile and offline checks (already executed on real Ubuntu host)

The installed library directory is
`/opt/hermes/runtimes/llama-mainline/build-cuda/bin`.

```bash
LLAMA=/opt/hermes/runtimes/llama-mainline
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" -Itests \
  tests/reliability_layer_llama_cb.cpp \
  -L"$LLAMA/build-cuda/bin" -Wl,-rpath,"$LLAMA/build-cuda/bin" \
  -lllama -lggml -lggml-base -lggml-cpu -ldl \
  -o /tmp/reliability_layer_llama_cb

/tmp/reliability_layer_llama_cb --inspect-plan
/tmp/reliability_layer_llama_cb --selftest
python3 tests/reliability_layer_compare.py --selftest
```

Result: **g++ -Werror compilation PASS**, native linked binary;
`--inspect-plan` PASS; `--selftest` PASS for synthetic backend
selection, real GGML CPU tensor get, strided view, FP32, SHA256,
duplicate rejection, budget fail-closed; Python comparator
synthetic PASS including detecting first difference and rejecting
tampered file. `--capture` lacking `--gpu-exclusive` was
verified to **exit code 2** before model load.

Evidence is in temporary server scratch
`/tmp/h27-layer-bisect-worker1/`:
`reliability_layer_llama_cb` and `tests/reliability_layer_*`.
No real model inference was executed in this task.

## Future exact GGUF experiment — requires GPU-exclusive grant

Launch **only after** prime releases/assigns both GPUs, on a new empty
directory. The `--gpu-exclusive` flag is an opt-in guard; it
does not reserve GPU resources by itself. Never launch while the
main HTTP engine or a different worker holds either GPU.

```bash
mkdir /tmp/h27-layer0-llama-pos0
LLAMA=/opt/hermes/runtimes/llama-mainline
LD_LIBRARY_PATH="$LLAMA/build-cuda/bin" \
  /tmp/reliability_layer_llama_cb \
    --capture --gpu-exclusive \
    --model /home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf \
    --llama-lib "$LLAMA/build-cuda/bin/libllama.so" \
    --out-dir /tmp/h27-layer0-llama-pos0 \
    --token-id 6205 --layer 0 --n-gpu-layers 80 --threads 4 \
    --include attn_norm,linear_attn_qkv_mixed,conv_output_silu,beta_sigmoid,gate,post_ffn
```

The harness performs **exactly one `llama_decode`** with numeric token
6205 at absolute position 0, no tokenizer or chat template, no greedy
continuation. It binds the captures to checkpoint SHA256, executable
SHA256, runtime-loaded **libllama** path and SHA256 (checked against
`dladdr`), token history and position, source dtype/shape/strides,
backend buffer, FP32 output digest and first-token greedy argmax.
The output manifest `manifest.json` has
`schema=h27.reliability-layer-cb.v1` and explicitly says
`CAPTURED_NOT_INDEPENDENTLY_COMPARED`. Missing selected stages
are an error (no fake all-PASS).

**Not exercised:** the actual callback's runtime selection/copy with
this 27B model on GPU. Graph scheduling or fusion may change the
availability of some selected nodes. A guarded `BLOCKED` in
that case is a finding, not grounds for weakening the comparison.

## Exact HERMES matching requirements

The production author, not this QA worker, must expose an **opt-in**
read-only HERMES trace compatible with the same one-token decode
configuration and use no hidden re-normalization.

| llama tensor | HERMES tensor boundary to capture | Critical mapping |
|---|---|---|
| `attn_norm-0` | After block0 `rms_f32_k`, before first NVFP4 GEMV (F32 5120) | First layer, same BF16 token embedding and direct norm |
| `linear_attn_qkv_mixed-0` | Raw block0 `attn_qkv.weight` GEMV output **before** depthwise convolution (F32 10240) | First matched NVFP4 projection; source activation quantization differs (Q8_1 vs BF16) |
| `conv_output_silu-0` | After block0 `conv_silu_k` (F32 10240) | Single-token state starts at zero; conv kernel width4 |
| `beta_sigmoid-0` | After `ssm_beta` and sigmoid (F32 48) | Do not compare raw pre-sigmoid beta |
| `gate-0` | After `ssm_a * softplus(alpha + dt_bias)` (F32 48) | GGUF already contains negative `ssm_a`; no second exp |
| `attn_output-0` | Before gated RMSNorm, GDN recurrence output (F32 6144) | Distinguish normalized Q/K repetitions and state stride |
| `linear_attn_out-0` | Block0 GDN output projection **before** first residual add (F32 5120) | Compare equivalent projection stage |
| `attn_residual-0` | Block0 x after attention residual (F32 5120) | Input to post-attention norm |
| `attn_post_norm-0` | Block0 post-attention RMSNorm (F32 5120) | Input to block0 FFN |
| `post_ffn-0` | Block0 x after FFN residual add (F32 5120) | First complete block checkpoint |

**Shape handling:** llama uses GGML `ne[0]` fastest, often
`[dim,1,1,1]` at token 0. HERMES's internal `[T,dim]`
flat array should be exported with **semantic shape `[dim,1,1,1]`**
without transposition for T=1. No implicit reshape, head permutation,
rescaling or broadcasting is allowed in the strict comparator.
`q_conv_predelta` and `k_conv_predelta` in llama have only
16 distinct key-group heads; HERMES currently expands to 48
value heads using `hh%16`. Their raw 4D arrays are **not
elementwise shape-equivalent** until the production author exposes
an unexpanded 16-group view (or a separately reviewed explicit
mapping-aware comparator). Do not turn that shape mismatch into PASS.

For the 3rd block (first full attention), capture separate per-head
Q/K before and after norm/RoPE. For pipeline split regression, compare
block31 `post_ffn` with GPU0 transfer and block32 `attn_norm`;
do not mistake a legitimate norm output for a raw PP boundary.

### Independent CPU comparison (after both real captures)

Have the HERMES author provide a **separate** manifest with
`schema=h27.reliability-layer-hermes.v1`,
`backend=hermes-27`, `status=CAPTURED_NOT_INDEPENDENTLY_COMPARED`,
and the same `model_sha256`, `prompt_ids=[6205]`,
`absolute_position=0`, `layer=0`, tensor
`name/file/shape/elements/bytes/sha256` fields.

```bash
python3 tests/reliability_layer_compare.py \
  --llama-manifest /tmp/h27-layer0-llama-pos0/manifest.json \
  --hermes-manifest /tmp/h27-layer0-hermes-pos0/manifest.json \
  --stages attn_norm,linear_attn_qkv_mixed,conv_output_silu,post_ffn \
  --max-abs 0.001 --max-rmse 0.0002 \
  --out /tmp/h27-first-divergence.json
```

If the HERMES trace uses different approved labels, add
`--aliases /tmp/mapping.json` with keys such as
`{"linear_attn_qkv_mixed":"blk0_qkv_before_conv"}`.
The comparator checks shape/size, full raw file SHA256, nonfinite
values, and same model/prompt/position/layer before numerical analysis.
It reports the **first differing *captured* stage**, not the
first differing kernel globally. Missing mandatory stages or
unmatched shape/identity return **2 BLOCKED**. At least one
out-of-threshold stage returns **1 DIFFERENCE**. All captured
stages within those *diagnostic* tolerances return 0 but explicitly
**not a model release PASS**.

The original full-logit independent gate remains separately
**FAIL** (12/12 real GGUF traces RMSE 0.025–0.056,
worst max abs 0.565). The new layer tracer is designed to establish
**where** the numerical difference begins, not to infer the root
cause from logits or relax a threshold post hoc.
