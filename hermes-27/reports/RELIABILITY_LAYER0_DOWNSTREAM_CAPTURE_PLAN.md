# HERMES-27 / independent llama.cpp layer-0 downstream callback capture

**Status (2026-11-14): CPU/OFFLINE source inspection, standalone C API
compile and deterministic integrity/tamper tests PASS. No model
inference or GPU workload was executed for this task.** The Q8_1
and DP4A code is frozen/published and not modified.

## Question and source-of-truth

The Q8_1 layer-0 QKV numerical experiment matched independent
llama.cpp QKV for the same normalized input and original GGUF
matrix. This does not establish where downstream conv/GDN/residual
data first differ, and cannot prove full-model logits. Next capture
must compare authentic callback tensors from llama.cpp for one
contiguous causal sequence with token IDs **6205 at position 0**,
then **161421 at position 1**. Token 161421 is the previously
captured pos0 greedy output; neither token may be substituted.

Installed local source independently audited, **read-only**:

- `/opt/hermes/runtimes/llama-mainline/src/models/qwen35.cpp`
  (`graph::graph`, `build_qkvz`, `build_layer_attn_linear`);
- `/opt/hermes/runtimes/llama-mainline/src/models/delta-net-base.cpp`
  (`llm_build_delta_net_base::build_recurrent_attn`);
- `/opt/hermes/runtimes/llama-mainline/include/llama.h`
  (`llama_context_params.cb_eval`, `llama_batch_init`);
- earlier independently authenticated layer-0 capture manifest
  `/tmp/h27-layer0-llama-pos0-prime3/manifest.json`, token 6205
  position0, model SHA256
  `db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1`.

Read-only SHA256 provenance for the **installed source** inspected
(these hashes identify code, but do not independently prove which
source commit produced the installed binary):

- qwen35.cpp: `ef973e9c789eb52ef27a77e0b1f36c6f41a378733a994d41b5924e3be012f671`
- delta-net-base.cpp: `5e3b318953f854b3825dd7cfe4b483ca7935cd5f6680438a3ece4be9b4d5428f`
- llama.h: `7279a2065cf7b7a081be9138dbe2be1bf8e0ce527e7df74429a7a982ae877a47`

Layer0 is a recurrent Gated Delta Net layer; Qwen35's
`hparams.is_recr(0)` follows the recurrent branch. The exact
tensor name handed to `cb(t, "stage", 0)` is `stage-0`.
These are **actual code callback labels**, not guesses based
on GGUF weight names.

## Actual intermediate labels and expected shape

Expected GGML `ne[0..3]` is for **one sequence of two tokens**
at absolute positions [0,1], using model dimensions:
hidden=5120, K/Q projection per-head=128, k-heads=16,
value-heads=48, value-dim=6144, conv-dim=10240.
Values below are **source-derived expected shapes**, not
measurements of a two-token runtime graph (not executed).
The manifest records and validates actual shapes at capture time;
unknown layouts fail closed instead of being silently reinterpreted.

| Exact callback tensor | Expected `ne[0..3]` for T2 | Location / interpretation |
|---|---|---|
| `attn_norm-0` | [5120,2,1,1] | qwen35 graph, normalized input |
| `linear_attn_qkv_mixed-0` | [10240,2,1,1] | QKV projection before 4-tap conv |
| `conv_output_raw-0` | [10240,2,1,1] | ggml_ssm_conv before SiLU |
| `conv_output_silu-0` | [10240,2,1,1] | SiLU of raw conv |
| `beta_sigmoid-0` | [1,48,2,1] | per-head beta after sigmoid |
| `gate-0` | [48,2,1,1] | alpha/dt softplus multiplied by SSM A, before reshape |
| `q_conv-0`, `k_conv-0` | [128,16,2,1] | direct pre-norm Q/K views |
| `v_conv-0` | [128,48,2,1] | V view after conv |
| `q_conv_predelta-0`, `k_conv_predelta-0` | [128,16 **or** 48,2,1] | L2-normalized Q/K; may repeat to 48 heads depending fused-GDN configuration |
| `v_conv_predelta-0` | [128,48,2,1] | V passed to GDN |
| `attn_output-0` | [128,48,2,1] | **post recurrent GDN computed output**, callback in delta-net-base.cpp |
| `final_output-0` | [6144,2,1,1] | post gated RMS normalization and reshape |
| `linear_attn_out-0` | [5120,2,1,1] | post SSM output projection before residual |
| `attn_residual-0` | [5120,2,1,1] | **first layer0 residual** (SSM projection + layer input) |
| `attn_post_norm-0` | [5120,2,1,1] | norm after first residual |
| `ffn_out-0`, `post_ffn-0` | [5120,2,1,1] | optional after FFN / residual |
| `state_predelta-0` | [128,128,48,1] | **pre-GDN recurrent state**, optional; ~3,145,728 FP32 bytes |
| `new_state-0` | implementation-dependent state shape | **only cb in non-fused `!keep` path**; not a stable fused-GDN stage |

The tracer defaults to 14 selected, bounded stages:
attn_norm, linear_attn_qkv_mixed, conv_output_raw,
conv_output_silu, beta_sigmoid, gate, q_conv_predelta,
k_conv_predelta, v_conv_predelta, attn_output, final_output,
linear_attn_out, attn_residual, attn_post_norm.
The optional `--include` list overrides the defaults.
The callback function returns `false` for unrelated stages.

For conventional per-token tensors, manifest `token_axis=1`
when `ne[1]=2`; for beta, Q/K/V and GDN post-attention
4D heads, `token_axis=2` when `ne[2]=2`.
Recurrent state has `token_axis=-1` and cannot be safely
interpreted as a pos0/pos1 output tensor.

### Exact GDN visibility limitation / blocker

`qwen35.cpp` calls `build_recurrent_attn` and then
`build_norm_gated`. In `delta-net-base.cpp`, the non-fused
`!keep` branch explicitly callbacks `attn_output` and
`new_state`. In the fused rollback-snapshot branch,
`ggml_gated_delta_net(ctx0,...)` returns a combined temporary
tensor `gdn_out`, but **there is no `cb(gdn_out,...)`**.
It creates `ggml_view_4d(...,gdn_out,...)` for the actual per-token
output and callbacks it as **`attn_output-0`**. The trailing fused
state snapshots are separately copied into the recurrent cache,
with **no dedicated named cb() on the snapshot view or copy**.

Thus a C API `cb_eval` filter can capture the real GDN
**output** and pre-GDN input/state, but cannot directly request
the raw fused `gdn_out` (including updated state snapshots)
by a nonexistent label. More aggressive capture would require an
**independently built instrumented llama.cpp scratch binary**
that explicitly names/callbacks the GDN output/state views, or a
separately verified state-access API that exports exactly the same
bytes. Neither is authorized in this task. Disabling fusion or
changing `n_rs_seq` merely to expose state can change computation;
do not treat that trace as equivalent without independent proof.

## New isolated deliverables (only permitted new paths)

1. `tests/reliability_layer0_downstream_callback_trace.cpp`:
   standalone C API tracer using installed libllama/libggml.
2. `tests/reliability_layer0_downstream_callback_sha256.hpp`:
   self-contained streaming CPU SHA256 helper; copied from an existing
   previously checked independent tracer, without changing the original.
3. `tests/reliability_layer0_downstream_callback_verify.py`:
   independent capture verifier and 17 strict tamper fixtures.
4. `reports/RELIABILITY_LAYER0_DOWNSTREAM_CAPTURE_PLAN.md`:
   this audit, build, admission plan and limitations.

### Capture safety and provenance

- **No implicit inference**: running without one of `--selftest`,
  `--inspect-plan`, `--inspect-runtime` or `--capture`
  fails. `--capture` rejects missing explicit `--gpu-exclusive`.
  The latter is a mandatory CLI interlock, **not** proof of prime
  approval; the operator still needs actual exclusive authorization.
- **Token positions**: one single `llama_batch_init(2,0,1)`
  batch with both tokens belonging to the same sequence 0;
  positions 0/1, logits enabled for both. This is not two
  independent fresh-context token0 forwards.
- **Authentic tensors**: GGML scheduler callback ask=true selects
  exact named node, ask=false copies once after backend readiness.
  `ggml_backend_tensor_get` gathers noncontiguous views by rows,
  converts F32/F16/BF16 to canonical finite FP32 (ne0-fastest).
- **Bounded memory**: default 2 MiB maximum per captured tensor,
  16 MiB total, up to 32 captures; hard cap options 8 MiB per
  tensor and 64 MiB total. `state_predelta` exceeds 2 MiB
  and must be explicitly selected with 4 MiB per tensor;
  `new_state` may be missing depending on fused graph path.
  No raw GGUF weights or huge whole-GDN state are copied by default.
- **Fail closed**: rejects missing stage names, duplicate callbacks,
  non-FP32/F16/BF16 types, nonfinite output, over-budget views,
  missing 2-token axis, existing destination files, invalid
  dimensions, missing logits or unmatched model/runtime hashes.
  Failed captures do **not** write `manifest.json`; partial tensor
  files left behind by a failure must be discarded as invalid.
- **Identity**: complete **19,694,360,544-byte** model file hashed
  with SHA256 before model load (not merely 4 KiB samples);
  independently anchored 64-hex expected model/libllama hashes
  required. Actual loaded libllama path is checked with `dladdr`
  and compared to the requested filename. All loaded
  `libggml*`/`libllama*` ELF objects are SHA256-hashed
  via `dl_iterate_phdr` and matched exactly against an
  out-of-band operator-supplied runtime hash lock.
- **Manifest**: source shapes/strides/types, exact file bytes/hash,
  token axis, expected token IDs/positions, both greedy outputs,
  model hash/size, actual libllama and all runtime library
  paths/hashes, SHA256 of the standalone exporter and runtime lock.
- **Independent verification**: `verify.py --verify` hashes the
  **entire physical GGUF again**, all shared libraries, exporter,
  lock and FP32 payloads; checks exact tensor shape×4 bytes,
  layouts, axes, finite values, requested stage set and budgets.
  Integrity hashes prove identity, **not** llama-vs-HERMES accuracy.

### Runtime library hashes seen in offline inspection

Observed from `trace --inspect-runtime`, CPU-only, after offline
linking against the already installed CUDA-enabled llama libraries.
Use only while those exact installed binaries remain unchanged:

    libggml-base.so.0.25.3 63be6a671ca0bdc97d5ceb4ba4406f9a267bff4ae9ad9a9b5ba4b84ace24c876
    libggml-cpu.so.0.25.3 01e92fe889efdd9da239d43dd7154a9fbb66f52193d77c45144fb849dbb0a191
    libggml-cuda.so.0.25.3 27dff5b6ded0e179b6fbe4504e7c1f0f5751598cdd35759000bbe4df30b20de3
    libggml.so.0.25.3 cd3766cae482625d6e8a309a6536b00c8285c80aa986b04095f85b51af99f794
    libllama.so.0.5.0 fd9e3e0810c6eca2e37b57abf443bcf15963671cbd0ee8489a37461a521094d4

Runtime locks must come from independently trusted values, not simply
from unverified names. A loaded library set that changes during model
initialization, including a new dlopened libggml backend, fails
unless the trusted lock accurately covers the final loaded set.
No fallback silently accepts newly loaded untrusted binaries.

## Reproduce CPU/OFFLINE evidence (does not run a model)

After prime syncs these four files into `/home/matteo/hermes-27`:

    root=/home/matteo/hermes-27
    llama=/opt/hermes/runtimes/llama-mainline
    s=/tmp/h27-layer0-downstream-worker2
    mkdir -p "$s"
    cp "$root/tests/reliability_layer0_downstream_callback_trace.cpp" "$s/"
    cp "$root/tests/reliability_layer0_downstream_callback_sha256.hpp" "$s/"
    g++ -std=c++20 -O2 -Wall -Wextra -Werror \
      -I"$llama/include" -I"$llama/ggml/include" -I"$s" \
      "$s/reliability_layer0_downstream_callback_trace.cpp" \
      -L"$llama/build-cuda/bin" \
      -Wl,-rpath,"$llama/build-cuda/bin" \
      -lllama -lggml -lggml-cpu -lggml-base -ldl -pthread \
      -o "$s/trace"
    "$s/trace" --selftest
    "$s/trace" --inspect-plan
    "$s/trace" --inspect-runtime
    python3 "$root/tests/reliability_layer0_downstream_callback_verify.py" --selftest
    # Negative test: must exit nonzero with NO model load
    "$s/trace" --capture

**Executed offline and passed on the installed llama-mainline build**:
G++ C++20 `-Werror` compile/link, C++ synthetic CPU callback/strided
view/SHA256/budget/duplicate/runtime-lock tests, `--inspect-plan`,
`--inspect-runtime`, Python verifier 17/17 independent tamper
rejections (tensor byte/size/NaN, shape/axis, path traversal, digest,
token/position, callback count, stage mismatch, lib hash, model
bytes/content, exporter bytes and lock changes).
No GGUF model inference or GPU work was attempted.

## Possible real capture — BLOCKED pending prime-exclusive grant

This section is a **plan, not an executed command**. Only the prime
may authorize and schedule it; avoid running concurrently with
Hermes inference/Comfy or other workers.

    root=/home/matteo/hermes-27
    s=/tmp/h27-layer0-downstream-worker2
    out=/tmp/h27-layer0-downstream-capture-prime
    model="$root/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf"
    lib=/opt/hermes/runtimes/llama-mainline/build-cuda/bin/libllama.so.0.5.0
    model_sha=db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1
    lib_sha=fd9e3e0810c6eca2e37b57abf443bcf15963671cbd0ee8489a37461a521094d4
    # Create a NEW EMPTY directory, do not overwrite previous evidence.
    mkdir "$out"
    # runtime.lock must be prepared with trusted runtime hashes shown
    # above, then separately checked against the installed ELF files.
    lock=/tmp/h27-layer0-downstream-runtime.lock
    timeout 180 "$s/trace" --capture --gpu-exclusive \
      --model "$model" --llama-lib "$lib" --runtime-lock "$lock" \
      --expected-model-sha "$model_sha" --expected-libllama-sha "$lib_sha" \
      --out-dir "$out" --token0 6205 --token1 161421 \
      --n-gpu-layers 80 --max-total-mib 16 --max-tensor-mib 2
    python3 "$root/tests/reliability_layer0_downstream_callback_verify.py" \
      --verify "$out/manifest.json" --runtime-lock "$lock" \
      --expected-model-sha "$model_sha" \
      --expected-libllama-sha "$lib_sha" --tracer-exe "$s/trace"

Failure or missing callbacks = **BLOCKED**, not an implied capture.
The token-position trace can subsequently be compared with genuine
Hermes PP2 tensors captured from *identical* checkpoint, input,
positions, state initialization and backend-specific head ordering.
No tolerance reduction or runtime routing is authorized by this task.
