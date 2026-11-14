# Source-authentic layer0 post-FFN scratch tracer: independent compile receipt

Date: 2026-11-14. **Offline preparation PASS. Numerical release NO-GO.**

## Scope and provenance

The new source-only patcher extends the previous SHA-pinned three-marker
HERMES layer0 tracer without replacing the published CUDA engine. It adds
`attn_post_norm`, `ffn_out`, and `post_ffn` (each 5120 FP32 values)
for both absolute positions 0 and 1, using the **same** frozen token
identity `6205,161421`, request state and original v2 context sidecars.
The extra trace files reside only in a distinct canonical
`<original-scratch-dir>-postffn` directory. No reference llama downstream
tensors exist yet, and no new traces were acquired.

- Immutable Git baseline:
  `75ed5e406ef6ebdf57d08d0facbe8f04629e6019`.
- Published baseline `distributed/pp2_fwd.cu` SHA256:
  `6faccbda1aaf568deb7767393a53c5ea89a025a97fbc9746a0fc97a61e29469e`.
- Frozen prior 3-marker patcher
  `experiments/pp2_layer0_trace_scratch_patch.py` SHA256:
  `80c0be0c1e29faf5db837da306af59ee729d4207c9bc21022163cfdef049c886`.
- New additive patcher SHA256:
  `4fa1bb4ead989d8297624881d04002c07e2ed4387d3e8642e8feb960628c18ff`.
- New source/negative CPU test SHA256:
  `98a34cafbcb1f93c06ae38939273456594aac81b54714253ca7bda3447e102a4`.
- Worker's detailed report:
  `LAYER0_POSTFFN_SCRATCH_CAPTURE_NEXT.md`.

## Independent Ubuntu execution

Scratch directory:
`/tmp/h27-postffn-integrated-75ed5e4-20261114`.
Only the three new files were overlaid on `git archive`; exact hashes and
both original source/patcher identities were verified.

1. `python3 -B -m unittest -v
   experiments.tests.test_layer0_postffn_scratch_capture_source_cpu`:
   **18/18 PASS** (synthetic negative fixtures, no model/inference).
2. On the original published `distributed/pp2_fwd.cu`,
   `pp2_layer0_postffn_scratch_patch.py --source ... --output ...`
   generated one previously nonexistent scratch file; exact output SHA256
   `4bcf20061bc01e3d69e5b1aff0d54e6396de94cb2943cca3437a2ed3f052cc03`.
   The generated scratch source includes the original three BASE markers and
   exactly three additional POST markers, without changing the real checkout.
3. Genuine CUDA 12.8.93 SM120a `nvcc -std=c++20
   -arch=sm_120a --expt-relaxed-constexpr -c` on the generated
   source: **PASS**, object SHA256
   `283bc7aebd15840502f9025b2c60a580fb2567490a598fd20e530a63f4e07df5`.
4. In **only** the disposable source archive, replaced the source copy with
   the generated scratch source and built both genuine
   `hermes_server` and `hermes_generate` with CUDA 12.8 Release and
   `HERMES_REGISTER_GPU_TESTS=OFF`: **full compile/device-link PASS**.
   No GPU kernels were launched.

Binary SHA256:

- Scratch instrumented `hermes_server`:
  `0719f32a3e2fb6dab4cdbbae88f2d4a3202d218293e607942a2d90a8456fdcfd`.
- Scratch instrumented `hermes_generate`:
  `213bf69027f44569c2eb0ad800f65d1f7fe13d8ef05368051963ad8f42b40891`.

## Boundaries

This commit adds **only tracer tooling, tests and reports**. The instrumented
`pp2_fwd.cu` was never installed, committed or used on HERMES GPUs.
Any resulting raw sidecars require exact original model+binary+source hashes,
real exclusive device authorization, genuinely captured llama.cpp
same-token/same-position downstream callback tensors, verified shapes/head
axes, and independent full 248320-vocabulary comparisons before becoming
numerical evidence. Synthetic sidecars are explicitly `NOT_ORACLE`.
Production and physical GPU release remain NO-GO.
