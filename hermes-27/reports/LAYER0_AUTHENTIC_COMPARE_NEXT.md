# HERMES-27 / physical authenticated layer0 callback comparator

**State: CPU/OFFLINE, 2026-11-14.** New isolated verifier and test suite
prepared without GPU, model inference or changes to production/PP2/serving,
CMake, installed llama libraries or any service. **23/23 deterministic
synthetic tests PASS**. The authentic llama.cpp *single-token* pos0 callback
manifest was verified against the physical full GGUF and physical library:
only **attn_norm**, **linear_attn_qkv_mixed** and **conv_output_silu**
are present; downstream llama callbacks and the corresponding real HERMES
scratch v2 manifest remain absent. Hence **no authentic HERMES-versus-llama
downstream numerical PASS or first-divergence claim is possible yet**.

## Exclusive new files

- `tests/reliability_layer0_compare_authentic.py`
  CPU-only Python 3 standard-library comparator. It does not import llama,
  torch, GPU runtimes or source-side verifier modules and has no device
  or inference actions. All comparable stages are 100% raw FP32 data
  paired with exact physical SHA-256 and verified time/shape provenance.
- `tests/test_reliability_layer0_compare_authentic.py`
  23 independent fixture/negative tests. Two isolated, disjoint synthetic
  GGUF/library/exporter/HERMES/llama fixture trees have distinct hashes;
  no genuine checkpoint file is modified or shadowed.
- This new report. There are no edits to existing tests,
  `experiments/pp2_layer0_trace_scratch_patch.py`,
  `experiments/verify_hermes_layer0_trace.py`, existing callback tracers,
  model weights or production code.

## Physically verified real data

Authenticated single-token llama source:
`/tmp/h27-layer0-llama-pos0-prime3/manifest.json`.
The read-only command below completed with exit **0** on the Ubuntu host,
and recomputed the SHA-256 over the *entire* real 19,694,360,544-byte GGUF:

```bash
python3 tests/reliability_layer0_compare_authentic.py \
  --inspect-llama \
  --llama-manifest /tmp/h27-layer0-llama-pos0-prime3/manifest.json \
  --model /home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf
```

Verified `model_sha256`:
`db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1`.
Verified exact `libllama.so.0.5.0` physical SHA:
`fd9e3e0810c6eca2e37b57abf443bcf15963671cbd0ee8489a37461a521094d4`.
Manifest prompt `6205`, absolute position **0**, layer **0**,
greedy next token **161421**; callback ask/ready 3/3, source backend CUDA0,
little-endian FP32, ne0-fastest, exact captured bytes/shape/stride and
independently recomputed tensor SHA-256:

| Authentic v1 llama tensor | Elements | SHA-256 |
| --- | ---: | --- |
| `attn_norm-0.f32` | 5120 | `492ae3d622039e01dcce9c5d8bc32ff2986e1134d5c50fad47ff366e8561bfa3` |
| `linear_attn_qkv_mixed-0.f32` | 10240 | `4b32da962e03f2481aa869955cea378b7921418d920459e9e62cf9300c08184f` |
| `conv_output_silu-0.f32` | 10240 | `12513886591fd03cc1a28c2d671cb76217242339300037f1988da4031a9d56a2` |

Its exporter SHA is recorded but the v1 manifest has **no exporter file
path**. Inspection therefore reports `exporter_verified=false`, not
a fabricated fully authenticated exporter identity. The v1 capture lacks
`beta_sigmoid`, `gate`, `attn_output`, `final_output`,
`linear_attn_out`, `attn_residual`. It is **not** a full downstream
oracle. The 2-token llama downstream tracer source is present, but no
real authenticated 2-token callback manifest was located under the known
scratch capture directories during this audit. The new HERMES scratch v2
source and its verifier exist; **no physically authenticated two-token
HERMES capture manifest with runtime token/position sidecars was found**.
No synthetic values were represented as actual llama/HERMES data.

## Supported manifest contracts

**llama v1** `h27.reliability-layer-cb.v1`:
position 0 only, prompt `[6205]`, 3 known named callbacks,
actual FP32 tensor `[ne0,1,1,1]`, exact file bytes/hash, physical full
GGUF hash and libllama hash. No speculative pos1 slicing allowed. Optional
`--llama-exporter` checks a physical exporter against the manifest's
recorded SHA; absent this path, the result declares the limitation.

**llama extended two-token callback schema**
`h27.reliability-layer0-downstream-cb.v1` (the prepared downstream
tracer's actual schema; not a guessed `h27.reliability-layer-cb.v2`
alias): contiguous `[6205,161421]`, absolute positions `[0,1]`,
all requested callback stages/counters, original source dtype
F32/F16/BF16 converted to canonical *captured FP32*, four-dimensional
GGML ne[0..3], stride[0] matching the original scalar type, exact
`token_axis` per stage, strict source-derived head layout, original
tensor byte count/SHA and finite data. This format **requires**
`--llama-exporter` and `--llama-runtime-lock`, both physically hashed;
every declared installed libllama/libggml runtime binary is rehashed
and matched against the independent runtime lock. No inference run
is performed by this verifier.

**HERMES scratch v2**
`h27.hermes-scratch-layer0-capture.v2` from
`experiments/verify_hermes_layer0_trace.py`: backend
`hermes-27-scratch`,
status `RUNTIME_DECODE_ARGUMENTS_RECORDED_NOT_LLAMA_COMPARED`,
recorded runtime tokens `[6205,161421]` and positions `[0,1]`.
It requires *both* exact physical `p0_context.txt` and
`p1_context.txt` content with token/position/sequence, matching hashes.
It also requires the **22** physical stage tensor files, all
`p{0,1}_{stage}.f32`, matching manifest per-stage sizes, shapes,
physical SHA-256, finite FP32 values and the same GGUF identity.
The external original PP2 source, patched scratch source and scratch
`hermes_generate` binary must be separately supplied; the comparator
hashes all three and checks that the original lacks the tracing hook
and the scratch patch contains exactly its expected two hook occurrences.
Missing context, runtime source, binary or physical token identity
is a **BLOCKED** result.

## GGML axes and stage correspondence (not based on flat length alone)

For llama extended batch, normal vector captures use
`ne=[width,2,1,1]` and `token_axis=1`.
The per-head `beta_sigmoid-0` is `[1,48,2,1]` and
`attn_output-0` is `[128,48,2,1]`, both
`token_axis=2`. For position `p`, the verifier calculates
the slice offset as `p * product(ne[:token_axis])`, then insists on
width **48** or **6144** respectively before comparing against
the verified separate HERMES `p{p}_stage.f32` buffer.
For head-bearing callbacks, `[6144,2,1,1]` cannot silently
stand in for `[128,48,2,1]`; wrong axis or head shape blocks.

The nine source-grounded semantic stages, in causal order:

| Stage | HERMES F32 per position | llama extended GGML ne[0..3] | Token axis |
| --- | ---: | --- | ---: |
| attn_norm | 5120 | [5120,2,1,1] | 1 |
| linear_attn_qkv_mixed | 10240 | [10240,2,1,1] | 1 |
| conv_output_silu | 10240 | [10240,2,1,1] | 1 |
| beta_sigmoid | 48 | [1,48,2,1] | 2 |
| gate | 48 | [48,2,1,1] | 1 |
| attn_output | 6144 | [128,48,2,1] | 2 |
| final_output | 6144 | [6144,2,1,1] | 1 |
| linear_attn_out | 5120 | [5120,2,1,1] | 1 |
| attn_residual | 5120 | [5120,2,1,1] | 1 |

These mappings are source-derived *candidate operator equivalences*;
matching shape/head layout alone cannot prove kernel semantics.
In particular HERMES `q_norm_scaled` and `k_norm` are **not**
silently matched to llama `q_conv_predelta` or
`k_conv_predelta`, which can differ in normalization/repetition.
Llama `attn_post_norm`, `ffn_out` and `post_ffn` have no
corresponding HERMES v2 capture. A request for these stages returns
`BLOCKED_UNSUPPORTED_STAGE`, never artificial zero error or PASS.
Raw fused GDN state outputs do not expose a reliable callback on the
installed llama graph; the first comparable post-GDN stage is the
named `attn_output` view, not an invented raw state tensor.

## Numerical and error semantics

Every requested comparable stage calculates **all FP32 elements**,
max absolute error, RMSE (double-precision `math.fsum` over squared
elementwise differences), worst flattened element and both values.
Diagnostic thresholds default to max abs <= 0.001 and RMSE <= 0.0002;
they are **not** release tolerances and can be specified as explicit
positive finite arguments. Comparison is only allowed on verified
GGUF, token ID, absolute position, physical tensor SHA, axis, head
layout and matching widths.

`first_differing_captured_stage` is the first observed mismatch in
the requested stages. `first_differing_verified_prefix_stage`
is set only if all causally earlier comparable stages were inspected
without a missing or unsupported gap. Requested stages must be in
causal order; a sparse subset cannot claim the first overall divergence.
If a llama downstream stage is not physically captured, the result is
`BLOCKED_MISSING_LLAMA_DOWNSTREAM` (exit **2**) even when the three
early captured stages match. Missing HERMES v2, unsupported stage and
invalid provenance also exit **2**. An observed, fully captured
numerical difference exits **1**. Exit **0** is
`CAPTURED_STAGES_MATCH_ONLY` for the *explicitly selected* complete
subset, never a claim of 64-layer or full-logit parity.

All numeric rows and block statuses are emitted as JSON on standard
output. The program does not create or overwrite source, capture
manifest, numerical output file or checkpoint.

## Reproduce the isolated CPU test and authentic v1 inspection

```bash
PYTHONPATH=tests python3 -m unittest -v test_reliability_layer0_compare_authentic

python3 tests/reliability_layer0_compare_authentic.py \
  --inspect-llama \
  --model /home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf \
  --llama-manifest /tmp/h27-layer0-llama-pos0-prime3/manifest.json
```

In a remote scratch copy of the **same new two Python files**,
`python3 -m unittest -v test_reliability_layer0_compare_authentic.py`
produced **23 passed in 1.382 seconds**, no failures.
Tests covered two independently constructed disjoint fixtures,
both two-token positions, ne1/ne2 extraction, beta and attention
head shapes, first mismatching QKV stage, first mismatch at
position1 only, sparse stage prefix restrictions, missing
llama downstream, non-equivalent stage refusal, wrong token
position or tokens, wrong head layout, tampered physical hashes,
truncated tensor, changed runtime context, missing exporter
or runtime lock, wrong source/model provenance, duplicate JSON
keys and fully rehashed NaN payloads.

## Exact future CPU-only authentic two-token invocation

Only after prime produces a real HERMES v2 manifest and real
llama extended two-token callback files (or an independently
accepted runtime-equivalent capture), use actual source and
binary paths recorded in the manifests. Placeholder paths below
are **not claims of existing artifacts**:

```bash
python3 tests/reliability_layer0_compare_authentic.py \
  --model /home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf \
  --llama-manifest /tmp/<LLAMA_2TOKEN_CAPTURE>/manifest.json \
  --llama-exporter /tmp/<LLAMA_TRACER_EXECUTABLE> \
  --llama-runtime-lock /tmp/<LLAMA_RUNTIME_SHA_LOCK> \
  --hermes-manifest /tmp/<HERMES_V2_CAPTURE>/manifest.json \
  --hermes-dir /tmp/<HERMES_V2_CAPTURE> \
  --hermes-source /tmp/<ORIGINAL_PP2_SOURCE>/pp2_fwd.cu \
  --hermes-patched-source /tmp/<SCRATCH_PP2_SOURCE>/pp2_fwd.cu \
  --hermes-binary /tmp/<SCRATCH_BUILD>/hermes_generate \
  --position 0
```

Repeat with `--position 1` only when the two-token contiguous
llama callback manifest and HERMES v2 runtime context both prove
actual token 161421 at absolute position 1. **No GPU required**
for validation/comparison. If prime only supplies HERMES v2 while
the llama downstream capture is still missing, the tool can
compare only a selected early-stage subset against the real
single-token llama v1 and must block downstream claims.

Remaining blockers are physical real HERMES v2 files with
runtime context sidecars and independent real llama downstream
callback files, not absence of comparator code. No full-model
inference should be scheduled by this tool or inferred from
its synthetic tests.
