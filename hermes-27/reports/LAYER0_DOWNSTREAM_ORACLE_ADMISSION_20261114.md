# HERMES-27 — strict downstream llama oracle admission, CPU/OFFLINE

**Frozen 2026-11-14, worker10.** The bounded eight-callback llama guard now
fails closed for stale or aliased capture files, invalid GGML scalar strides,
noncontiguous token identities, and a claim of numerical oracle readiness.
**Ubuntu CPU tests: 32/32 PASS (20 existing + 12 new).** No GPU, model
inference, installed service, ingress or Git state was modified. This
document records a **diagnostic source/identity gate**, not an oracle,
accuracy comparison, full-vocabulary logit parity or throughput measurement.

## Genuine, physically checked inputs

- Genuine historical single-token llama callback manifest:
  `/tmp/h27-layer0-llama-pos0-prime3/manifest.json`, physical SHA256
  `49b0052027532e1e8572d9d865517b2beef92c0e8176a02adacd35efe46d540b`.
  Ubuntu directory has exactly four regular files: the manifest and
  `attn_norm-0.f32`, `linear_attn_qkv_mixed-0.f32`, and
  `conv_output_silu-0.f32`. The strict source verifier checked all three
  pinned tensor SHA/bytes. It records **token 6205**, absolute **position 0**,
  layer0 and greedy next token **161421**. It does **not** record a position-1
  layer trace, downstream recurrent output or post-FFN tensor. Its exporter
  has a recorded digest without a physical exporter path: this old result
  does not establish an independently rehashed historical exporter binary.
- The physically installed `Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf` on
  Ubuntu is **19,694,360,544 bytes**, rehashed completely with `sha256sum`
  (observed exit0):
  `db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1`.
  Hashing the GGUF is a file identity check only; it does not execute the model.
- Installed llama mainline source SHA256, checked with `sha256sum` on Ubuntu:
  `src/models/qwen35.cpp` =
  `ef973e9c789eb52ef27a77e0b1f36c6f41a378733a994d41b5924e3be012f671`;
  `src/models/delta-net-base.cpp` =
  `5e3b318953f854b3825dd7cfe4b483ca7935cd5f6680438a3ece4be9b4d5428f`;
  `include/llama.h` =
  `7279a2065cf7b7a081be9138dbe2be1bf8e0ce527e7df74429a7a982ae877a47`.
  The frozen downstream tracer `tests/reliability_layer0_downstream_callback_trace.cpp`
  SHA256 is `ecaebd72739a2cc443147139b899e58717b532d9bea80da65cb8d11204e46429`,
  helper `reliability_layer0_downstream_callback_sha256.hpp`
  SHA256 `eebe48b6c3518f84c8928b9613e13cb9f92bdb9fb7ceaae06a890c38bf2c37df`,
  and independent validator `reliability_layer0_downstream_callback_verify.py`
  SHA256 `2d9cc3da5fcb454672efcd63444f55e3da38ad57094f0a5f1bf3112e5087ae10`.
  Source hashes by themselves cannot prove executable build provenance.

## Exactly what the installed llama graph exposes

Read-only source inspection on Ubuntu, with independently pinned source
hashes: `qwen35.cpp` around lines 170–196 and 355–480;
`delta-net-base.cpp` around lines 552–583. Source tags below are a
**capture request plan**, not proof that the runtime actually returned
these tensors. Full position 1 requires **one single llama_decode batch**
with `tokens=[6205,161421]`, `positions=[0,1]`, sequence ID 0 for both,
and both logits enabled. Two one-token calls with reset state are invalid.

| Phase / real callback | GGML `ne0,ne1,ne2,ne3` if 2-token, one sequence | token axis | HERMES v2 counterpart |
| --- | --- | ---: | --- |
| attn input `attn_norm-0` | `[5120,2,1,1]` | 1 | `attn_norm` |
| QKV projection `linear_attn_qkv_mixed-0` | `[10240,2,1,1]` | 1 | `linear_attn_qkv_mixed` |
| convolved SiLU `conv_output_silu-0` | `[10240,2,1,1]` | 1 | `conv_output_silu` |
| recurrent beta `beta_sigmoid-0` | `[1,48,2,1]` | 2 | `beta_sigmoid` |
| decay gate `gate-0` | `[48,2,1,1]` | 1 | `gate` |
| Q after L2 norm `q_conv_predelta-0` | `[128,16,2,1]` or `[128,48,2,1]` | 2 | **no proven equivalence** to `q_norm_scaled` |
| K after L2 norm `k_conv_predelta-0` | `[128,16,2,1]` or `[128,48,2,1]` | 2 | **no proven equivalence** to `k_norm` |
| V for GDN `v_conv_predelta-0` | `[128,48,2,1]` | 2 | no separate HERMES v2 V snapshot |
| fused recurrent result `attn_output-0` | `[128,48,2,1]` | 2 | `attn_output` |
| gated normalized result `final_output-0` | `[6144,2,1,1]` | 1 | `final_output` |
| SSM output projection `linear_attn_out-0` | `[5120,2,1,1]` | 1 | `linear_attn_out` |
| residual `attn_residual-0` | `[5120,2,1,1]` | 1 | `attn_residual` |
| post-attn normalization `attn_post_norm-0` | `[5120,2,1,1]` | 1 | no HERMES v2 capture |
| FFN `ffn_out-0` | `[5120,2,1,1]` | 1 | no HERMES v2 capture |
| post-FFN residual `post_ffn-0` | `[5120,2,1,1]` | 1 | no HERMES v2 capture |

The frozen tracer's `--inspect-plan` advertises all 22 declared source
labels. Its default capture requests 14 labels, including V and
`attn_post_norm`, but does **not** request `ffn_out` or `post_ffn`;
the prior minimal guard requests only eight. The existing guard deliberately
retains this bounded **eight-stage diagnostic** policy. It checks that the
seven additional labels above exist in the compiled tracer's inspect plan,
and reports them as **not admitted** in its result. A possible
`ffn_out` ambiguity is visible in installed `qwen35.cpp`: both
`build_layer_ffn` and its caller label the result
`cb(cur, "ffn_out", il)` (two distinct source sites). The strict C++ callback
receiver rejects duplicate ready callbacks with that name. It remains
unverified whether the physical graph emits one or two ready callbacks in a
particular configuration; do not label a full-FFN trace complete without a
genuine runtime witness.

The raw fused GDN `gdn_out` state is not a straightforward named ready
callback. `delta-net-base.cpp` exposes a `[128,48,2,1]` view as
`attn_output-0`, while the state snapshots are held in the fused output
layout. Source availability of `new_state` on the non-fused branch does
not establish a captured fused state oracle.

For GGML `ne0`-fastest output, position `p` starts at
`p * product(ne[:token_axis])`; the guard admits `[128,48,2,1]`,
never a flattened `[6144,2,1,1]` reinterpretation of `attn_output`.
Q/K callback head count can be 16 or 48: source applies L2 normalization,
then conditionally repeats heads depending on fused-GDN flags. HERMES's
`q_norm_scaled` and `k_norm` are flat 6144-vectors and do not prove
head repetition, scale, normalization placement or causal mapping.
The frozen comparator correctly blocks Q/K as unsupported. Even if the
byte widths agree, **do not compare or treat them as a numerical oracle**.

## New fail-closed edge in the owned guard

`experiments/llama_layer0_downstream_capture_guard.py` now:

1. Requires an **exact physical directory inventory** before and after
   capture, including rejecting stale `.partial` entries, hidden files,
   symlink entries, nested folders, and aliased manifests. Existing pos0
   admission requires exactly four files; after the eight-stage capture
   requires exactly eight named FP32 files and `manifest.json`.
2. Anchors the old manifest to its pinned digest, original three tensor
   hashes and dimensions, original model and installed libllama paths, and
   validates the old source scalar stride `nb0=4`.
3. Requires actual postcapture FP32 source data format, legal GPU-layer
   count, F32/F16/BF16 type, source `nb0=4/2`, positive complete strides
   and backend label. The preexisting metadata geometry-check API remains
   available to synthetic CPU unit tests but is **never** an oracle verdict;
   real postcapture necessarily sets `strict_source_contract=True`.
4. Guards against manifest mutation between the independent physical hash
   verifier and its final structural check. Captures must have exact
   `[6205,161421]`, `[0,1]`, contiguous-batch flag, expected phase0
   greedy `161421`, all eight requested callback ready names and true
   token axes. Its authenticated physical source, runtime lock and exporter
   checks remain in force.
5. Emits explicit `numerical_oracle_admission:
   BLOCKED_NO_AUTHENTIC_CROSSENGINE_PROOF` and
   `qk_semantic_mapping: BLOCKED_UNPROVEN_HEAD_REPETITION_AND_Q_SCALE`,
   including when the limited physical stage capture is admitted as
   `VERIFIED_CAPTURE_STAGES_ONLY`. No state in this tool asserts
   model-wide accuracy. Source and executable hashes alone cannot
   establish reproducible-build provenance; approval must verify the
   actual compiled executable independently.

## CPU proof (actual Ubuntu run)

Isolated directory:
`/tmp/h27-llama-oracle-w10-20261114-1157`. A read-only `scp`
copy of the guard and two test files was placed in new temporary scratch.
No repository, runtime service, GPU or installed llama file was changed.
The command:

```bash
python3 -m unittest discover -v \
  -s /tmp/h27-llama-oracle-w10-20261114-1157/experiments/tests \
  -p 'test_*.py'
```

**Observed exit 0: `Ran 32 tests in 0.005s`, `OK`.**
This is **20 previously published structural tests + 12 new provenance
negative tests**; zero skips on Ubuntu. The new tests checked physical
genuine historical pos0 SHA/files/identity, installed source SHA and duplicate
`ffn_out` labels, malformed token and position mapping, isolated fake
short GGUF, Q/K and GDN axis ambiguity, wrong strides, missing callbacks,
stale/hidden/nested/symlink paths and explicit blocked oracle status.
Synthetic fixture data is used only to exercise rejection logic; it is
not described or delivered as a llama inference result.

Frozen Windows source and independently matching Ubuntu scratch file hashes:

```text
experiments/llama_layer0_downstream_capture_guard.py
  96797bc499858502077d5ebf42992521e19d469b44a0e34fe5ff65c8183cb2fa
experiments/tests/test_llama_downstream_capture_provenance_cpu.py
  b582260ab2994f614ab398da1f912b6ee2b21a8081135f71f1a1ccb41176fdb8
existing untouched experiments/tests/test_llama_layer0_downstream_trace_contract_cpu.py
  78093964133359d097621a22640f586da1761e7fa337133a21a19a82e9e10aaa
```

**Remaining status: BLOCKED_ORACLE.** To establish numerical evidence,
the prime needs a verified physical ingress/GPU ownership grant, a genuinely
compiled tracer whose exporter SHA and full runtime set are independently
approved, a single real two-token llama batch with complete requested stages,
and an authentic physically checked two-token HERMES v2 with matching
runtime sidecars. Post-attn/FFN and full logits require further genuine
capture paths; the existing v2 HERMES tracer ends at `attn_residual`,
and the llama tracer currently records greedy IDs rather than full logit
vectors. No complete cross-engine oracle is available.
