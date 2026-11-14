# HERMES-27 layer0: additive post-FFN scratch source contract

**2026-11-14 — worker10: CPU source-only PROVISIONAL; CUDA compile BLOCKED,
GPU/model capture NOT EXECUTED, independent oracle NOT AVAILABLE.**
The change consists of exactly two new Python source/test files plus this
report. No edits were made to production PP2, the published 3-marker v2
tracer/verifier, llama callback tooling, comparator, Git, services or ingress.
Worker14 owns the production PP2 forward source. Do not merge or run this
candidate until the prime has reconciled worker14's source freeze and reviewed
the newly compiled scratch binary.

## Frozen inputs actually inspected (Windows workspace)

- Published HEAD as supplied by prime:
  `3daac943c0e7ef8f46c5b24c94181c7a8ac1fbbf`. The Windows workspace has
  no usable Git metadata, so this is a **prime-reported HEAD**, not a local
  `git rev-parse` assertion; no Git operations were performed.
- `distributed/pp2_fwd.cu`, **137,729 bytes**, physically SHA256
  `6faccbda1aaf568deb7767393a53c5ea89a025a97fbc9746a0fc97a61e29469e`.
  New patcher refuses even a one-byte change to this input. This source may
  be invalidated by a later worker14 snapshot; the tool MUST remain blocked
  until updated under a separate review if its SHA changes.
- Existing `experiments/pp2_layer0_trace_scratch_patch.py`,
  **11,404 bytes**, SHA256
  `80c0be0c1e29faf5db837da306af59ee729d4207c9bc21022163cfdef049c886`.
  Its untouched `patched_source` composes the base 3-marker v2 witness
  first; the new patcher checks this independent physical SHA before importing
  it and never changes the published patcher on disk.
- GGUF expected checkpoint (not rehashed during this task):
  `/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf`,
  full length **19,694,360,544 bytes**, expected SHA256
  `db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1`.
  The new CPU sidecar validator requires that length/hash in its real mode;
  synthetic tests skip the 19.7GB input **only** with a distinct, explicitly
  synthetic status. This check cannot attest what a future binary loaded.

## Exact, source-derived stage and pointer provenance

The published `run_gdn_decode_layer` is at `pp2_fwd.cu:1476-1558`,
with `T=1` and only layer0 relevant:

| Intended layer0 tensor | Device pointer in arena | Source production point | FP32 geometry |
|---|---|---|---|
| `attn_post_norm` | `d_h = (float*)(ws+0)` | `rms_f32_k(..., d_h, d_x, wPost, ...)` line 1541 | 5120 x 4 = 20,480 bytes |
| `ffn_out` | `d_down = (float*)(ws+490240)` | FFN down projection `gemv_batch(...,d_down,...)` line 1547 | 5120 x 4 = 20,480 bytes |
| `post_ffn` | `d_x` input/output hidden | residual `add_k(...,d_x,d_down,...)` line 1548 | 5120 x 4 = 20,480 bytes |

Each pointer is physically available at the selected **source hook** after
the second residual `add_k` and before the original stream sync at line
1549. `d_h` is not overwritten by the intervening FFN projection kernels;
`d_down` is the additive FFN output; `d_x` is the post-FFN residual. The
old v2 probe records `attn_residual` **before** post-norm and FFN and
cannot substitute it for these later values.

The old source already defines `L0TraceScope(tok,pos,&ds)` in the exact serial
decode path. In the v2 probe:

- A global monotonically increasing trace sequence admits **only** positions
  0 and 1, exact runtime IDs **6205** then **161421**, and the same
  `DecodeState` pointer, with no concurrent first request.
- The actual layer0 pre-FFN probe emits its exclusive raw
  `p0_context.txt` or `p1_context.txt` after the 11 source-derived
  snapshots and checks GPU0. The new patcher copies the already validated
  scratch path/sequence into TLS **only if** the additional postFFN
  environment variable is present; it does not infer ID/position from
  filenames or CLI order.
- Before any post-FFN output, the new helper rechecks the current TLS
  invocation `state`, exact runtime token/absolute-position,
  `g_l0_trace_sequence == seq+1`, GPU0, stream0 and the **exact physical
  original v2 sidecar bytes**. An invalid/missing context fails closed.
- Async/sliding and nonzero stream are refused. All copies follow
  `cudaStreamSynchronize(0)` and use one bounded fixed-size
  `std::array<float,5120>` on the host; no dynamically sized recovery
  vector is added. Non-finite values, CUDA copy errors, short writes,
  missing/aliased scratch directories, unknown leftovers and duplicates
  reject the capture. Each tensor and sidecar uses exclusive `fopen("wbx")`.

### Explicit scratch-only opt-in and separated raw directories

Only a newly compiled scratch binary from this source transformation can
include the additional hook. It requires **both** the original v2
`HERMES_SCRATCH_LAYER0_TRACE_DIR` and the additional
`HERMES_SCRATCH_LAYER0_POSTFFN_DIR`; the latter must be the exact **canonical
path** `<old-root>-postffn`, a separately pre-existing, initially empty
directory under the same `/tmp/h27-*` namespace. The old v2 directory
remains exactly **24** files (22 FP32 + two context); the new directory is
exactly **8** files after two calls (six 5120-FP32 tensors + two context
sidecars). This preserves the unchanged old strict v2 verifier, which would
otherwise reject unknown extra files.

The extra environment variable by itself cannot produce post-FFN outputs:
the old v2 pre-hook must have successfully bound an actual token/position
and emitted its matching context first. The two valid post snapshots use
the same one-request serialized GDN state. Source guards still contain
**exactly three** occurrences of the old v2 marker and three of the
additional postFFN marker. No default production route is modified.

## New source-only extension and CPU inspector

Owned file `experiments/pp2_layer0_postffn_scratch_patch.py`:

1. Requires the exact production source SHA and the exact frozen 3-marker
   patcher SHA, reconstructs the v2 source **in memory** and validates
   its shape/ownership/context guards.
2. Requires single, exact helper/context anchor sites and **one** precise
   `gemv_batch(d_down) -> add_k(d_x,d_down) -> sync` anchor **inside the
   serial GDN function**, not the similarly named full-attention path.
   Missing, duplicated, changed or reordered critical source anchors fail
   closed. It refuses to repatch a derived source and refuses to write over
   existing output or outside a new canonical temp scratch subtree.
3. Adds only the postFFN helper, a TLS root/sequence arm at the original v2
   context write, and the postFFN call after the FFN residual. It checks
   composition for three old and three new marker occurrences.
4. Provides a separate **CPU bytes-only** parser:
   `validate_post_sidecars(base, post, model=..., synthetic_test=False)`.
   It verifies exact paired raw directory inventories, both token sidecars
   byte-for-byte, expected 11 earlier stage widths, three new stage widths,
   finite little-endian F32 values and SHA256 of all six post tensors.
   Genuine-mode checks the **complete physical GGUF** SHA/length.
   It never labels synthetic metadata as a real callback or independent
   oracle; real mode explicitly says
   `CPU_RAW_BYTES_CHECKED_NOT_INDEPENDENT_ORACLE`.

The composite scratch `pp2_fwd.cu` computed deterministically on Windows
is **152,175 bytes**, SHA256
`4bcf20061bc01e3d69e5b1aff0d54e6396de94cb2943cca3437a2ed3f052cc03`.
This composite was produced **in memory** and in short-lived OS temp files
for tests. It was not deployed, uploaded or executed.

## Exact CPU test evidence

New `experiments/tests/test_layer0_postffn_scratch_capture_source_cpu.py`
contains 18 separate deterministic `unittest` methods:

- Source SHA/pointer geometry, exact source-location ordering, same
  request/token/position root and CUDA-stream source assertions.
- Fail-closed input SHA mismatch, changed/duplicated FFN producer hook,
  removed context write, missing `DecodeState` fence, async preflight
  mutation, marker repatching and altered frozen patcher.
- Fresh scratch-only CLI output and rejection of overwrite.
- Synthetic two-position metadata that **cannot** become a genuine oracle;
  missing/extra/stale/hidden/nested files, swapped token IDs/positions,
  truncated/NaN/Inf post data, corrupted pre-v2 tensors and fake GGUF.

Actual commands on **Windows**, with Python interpreters already installed
locally (no SSH):

```powershell
& 'C:\Users\matte\AppData\Local\Programs\Python\Python312\python.exe' -m unittest discover -v -s experiments/tests -p 'test_layer0_postffn_scratch_capture_source_cpu.py'
& 'C:\Users\matte\AppData\Local\Programs\Python\Python313\python.exe' -m unittest discover -v -s experiments/tests -p 'test_layer0_postffn_scratch_capture_source_cpu.py'
```

**Final rerun Python 3.12: 18/18 PASS, exit0, 0.644s.**
**Final rerun Python 3.13: 18/18 PASS, exit0, 0.624s.**
No GPU inferencing or installed GGUF load was performed. An unprivileged
Windows runner forbids creation of symlinks (WinError 1314), so that
individual mutant's **dynamic symlink branch did not run**: the test
instead checks the explicit fail-closed symlink predicate in source.
The remaining filesystem negative cases ran normally. No Unix symlink
runtime rejection or ASAN/UBSAN was claimed.

Files SHA256 (Windows physical source freeze, before prime publication):

```text
NEW experiments/pp2_layer0_postffn_scratch_patch.py
    4fa1bb4ead989d8297624881d04002c07e2ed4387d3e8642e8feb960628c18ff
NEW experiments/tests/test_layer0_postffn_scratch_capture_source_cpu.py
    98a34cafbcb1f93c06ae38939273456594aac81b54714253ca7bda3447e102a4
composed scratch-only C++ source (in memory, not a tracked file)
    4bcf20061bc01e3d69e5b1aff0d54e6396de94cb2943cca3437a2ed3f052cc03
```

## BLOCKERS — not a numerical oracle

1. **No CUDA12.8 compilation or linker test.** The Windows machine lacks
   `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe`
   and no SSH or remote build was authorized for this follow-up. A real
   isolated CUDA12.8 compile-only and source-to-binary provenance receipt
   remain required. CPU source transformation is not proof the generated
   C++ binary builds or its helper executes correctly.
2. **No real 2-token HERMES v2/postFFN sidecars or llama downstream
   callbacks.** The tests contain artificial metadata/F32 bytes only.
   Source shows that CUDA pointers exist; it cannot prove kernel results,
   actual GDN state evolution, host/device ordering or absolute correctness.
3. **Not a release gate.** The v2 sidecar records token/position inside the
   instrumented HERMES runtime, not an independently observed llama call.
   Neither a source hash nor even a full physical GGUF SHA proves which
   checkpoint a newly compiled binary actually loaded without separately
   anchored runtime attestation. No full-vocabulary logits are recorded,
   and independent llama postnorm/FFN capture is absent (its source has
   two `ffn_out` labels requiring real callback uniqueness review).
4. **Ownership and update boundary.** Prime must reconcile
   worker14's final `pp2_fwd.cu` SHA with the exact pin above before
   even attempting an isolated CUDA build. If changed, this patcher
   must **BLOCK**, not silently adapt. No GPU grant is presently
   established; do not execute the scratch binary, touch services, or
   treat any mock fixture as an oracle.

**Verdict: PASS_CPU_SOURCE_CONTRACT / BLOCKED_CUDA_BUILD_AND_REAL_ORACLE.**
