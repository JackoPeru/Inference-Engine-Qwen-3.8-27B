# Real GGUF NVFP4 GEMV oracle (Scalar / Warp4 / Packed4)

Status: **isolated test source; CUDA execution requires prime's explicit GPU
clearance**. No production code or CMake changes. No latency or model-quality
claims are derived from this harness.

## Files / source of truth

- `gemv_real_common.hpp`: host-only GGUF tensor loader, exact input capture
  decoder, independent double-precision E2M1/UE4M3 dequant + dot, FP32 error
  bounds, fail-closed inputs and provenance.
- `gemv_real_cpu.cpp`: CPU-only selftest, real GGUF tensor inspection and full
  FP64-reference generation. Does **not** initialize CUDA.
- `gemv_real_q8_cpu.cpp`: independent CPU-only Q8_1-like activation
  quantization + FP64 dot diagnostic, for comparison with the BF16 path.
- `gemv_real_compare.cu`: calls the **actual** three kernel APIs:
  `nvfp4_gemv_cuda_variant(..., Scalar/Warp4)` in
  `kernels/src/nvfp4_gemv.cu`, and
  `nvfp4_gemv_warp_experiment(..., PackedU32x4)` in
  `kernels/src/nvfp4_gemv_warp_variants.cu`.

All paths below run from the repository root on the Linux host with access to
the exact GGUF. Build in an isolated `/tmp` directory, not in production.

```bash
model=/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf
mkdir -p /tmp/hermes27-gemv-real-oracle

# CPU build and CPU-only tests are permitted without GPU access.
g++ -O2 -std=c++20 -Imodel/include -Ikernels/tests \
    model/src/gguf.cpp kernels/tests/gemv_real_cpu.cpp \
    -o /tmp/hermes27-gemv-real-oracle/gemv_real_cpu
/tmp/hermes27-gemv-real-oracle/gemv_real_cpu --selftest
/tmp/hermes27-gemv-real-oracle/gemv_real_cpu \
    --gguf "$model" --tensor blk.26.ssm_out.weight --inspect-weights

# Independent CPU Q8_1 quantization simulator (does not touch CUDA).
g++ -O2 -std=c++20 -Imodel/include -Ikernels/tests \
    model/src/gguf.cpp kernels/tests/gemv_real_q8_cpu.cpp \
    -o /tmp/hermes27-gemv-real-oracle/gemv_real_q8_cpu
/tmp/hermes27-gemv-real-oracle/gemv_real_q8_cpu --selftest

# Compile-only; do not run the resulting GPU binary until prime grants GPU.
nvcc -O3 -std=c++20 -arch=sm_120a \
    -Imodel/include -Ikernels/include -Ikernels/tests \
    model/src/gguf.cpp kernels/src/nvfp4_gemv.cu \
    kernels/src/rmsnorm.cu kernels/src/nvfp4_gemv_warp_variants.cu \
    kernels/tests/gemv_real_compare.cu \
    -o /tmp/hermes27-gemv-real-oracle/gemv_real_compare
```

The CPU selftest checks finite F32→BF16 round-to-nearest-even including ties
and negative zero, scale and nibble decoding, and a known FP64 dot product.
The real GGUF inspect mode validates the complete scale bytes of one original
weight tensor and its bounds. It does **not** invent an activation or claim
to have executed any GEMV. The exact model checkpoint SHA-256, compiler,
binary hash, CUDA driver/runtime, command and source revision should be
recorded by the operator for a reproducible GPU report; the printed FNV-64
hashes of only the selected tensor and activation are diagnostic, not secure
checkpoint identifiers.
Build against the same checked-out `nvfp4_gemv.cu` and
`nvfp4_gemv_warp_variants.cu` as the intended deployment; a remote server
checkout may lack the experimental Packed4 source or contain older kernels.
Copy the exact current source into an isolated build directory if needed,
and do not silently substitute stale binaries or overwrite live source.

## Real activation input — two supported modes

### 1. Direct BF16 (preferred)

Use `--activation-bf16 path` containing exactly `H*2` bytes of little-endian
BF16 **captured from the model immediately before the chosen GEMV**, not a
random or synthetic buffer. It must match `--tensor` and the selected token,
step, layer and operator. Capture in the producing CUDA stream, complete the
copy before the workspace is reused, and retain provenance (prompt token IDs,
decode position, tensor name, checkpoint identity). A direct BF16 capture hook
is not yet part of production code. Do not attribute a user-supplied BF16 file
to a live inference unless that provenance is established independently.
If the original FP32 vector before this same cast is also available, pass
`--original-f32 path` (exactly `H*4` little-endian bytes). Every element of
`RNE(F32)` must equal the supplied BF16 bits; otherwise the program aborts.
With an original FP32 vector it also computes the independent double-precision
matrix-vector product twice, before and after BF16 activation rounding.

### 2. Current `HERMES_L26` log (zero production edits)

Existing `distributed/pp2_fwd.cu:1093-1118` computes `d_g` FP32, invokes
`f32_to_bf16_k(d_gb, d_g, VDD)` at line 1108, then executes GEMV with
`d_gb` and real `blk.26.ssm_out.weight` at line 1109. Under
`HERMES_L26=1` and **legacy synchronous decode only**, lines 1111-1118
append to `/tmp/l26_dec.bin` one record per executed layer26 step:

| Data in record | Float count | Bytes |
| --- | ---: | ---: |
| `d_g` FP32 | VDD=6144 | 24576 |
| `d_core` FP32 | HID=5120 | 20480 |
| `d_o` FP32 | HV×VHD=48×128=6144 | 24576 |
| `d_x` FP32 | HID=5120 | 20480 |
| **Total, no header** | **22528** | **90112** |

`--l26-dec path --record N` reads only the first 6144 F32 values at offset
`N*90112`, then converts to the **BF16 operand that the preceding kernel
would produce**, using nearest-even:

```cpp
uint16_t bf16 = uint16_t((f32_bits + 0x7fffU + ((f32_bits >> 16) & 1U)) >> 16);
```

This conversion is valid for finite IEEE F32 values and matches the
`__nv_bfloat16 h = (__nv_bfloat16)inp[i]` conversion in
`distributed/pp2_fwd.cu:89-94` with the CUDA default round-nearest-even mode.
It is **not** equivalent to simply `f32_bits>>16` (truncation). The harness
rejects nonfinite values and refuses to use L26 for any other tensor or shape.
The file must consist of a positive integer number of 90112-byte records.
`--record` is mandatory because the log is opened with `"ab"` and may
contain stale records from prior runs. Record indices begin at zero.

For strict proof that reconstructed bytes match the **actual device** `d_gb`,
obtain a separately gated direct BF16 capture of `d_gb` at line 1108 on the
same stream and position. Supply `--confirm-bf16 path`; the harness requires
exact equality of every BF16 input bit and aborts if any differ. Without this
file the harness reports `capture_confirmed=no-reconstructed`: it can compare
all three kernels on the same reconstructed real activation, but the exact
device input has not been independently verified. Production async decode
explicitly rejects `HERMES_L26`; do not force this debug switch into async.

**No inference or capture generation is authorized by this audit.** If the
capture file is absent, only run the CPU selftest and GGUF inspector.

## Running against captured input (after GPU permission)

```bash
model=/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf
capture=/tmp/l26_dec.bin
test -s "$capture" || { echo 'BLOCKED: real L26 capture missing'; exit 2; }

# CPU-side full-row reference; useful before the GPU reservation.
/tmp/hermes27-gemv-real-oracle/gemv_real_cpu \
    --gguf "$model" --tensor blk.26.ssm_out.weight \
    --l26-dec "$capture" --record 0

# Additional CPU-only Q8_1-like vs BF16 vs FP32 activation comparison.
/tmp/hermes27-gemv-real-oracle/gemv_real_q8_cpu \
    --gguf "$model" --tensor blk.26.ssm_out.weight \
    --l26-dec "$capture" --record 0

# Only with explicit prime permission. GPU index refers to the target host.
/tmp/hermes27-gemv-real-oracle/gemv_real_compare \
    --gguf "$model" --tensor blk.26.ssm_out.weight \
    --l26-dec "$capture" --record 0 --device 0

# Alternative direct input: replace --l26-dec/--record with:
# --activation-bf16 /path/to/real_x_6144_bf16.bin
# Optional paired pre-cast input: --original-f32 /path/to/real_d_g_6144_f32.bin
# Optional L26 device-bit confirmation: --confirm-bf16 /path/to/d_gb_6144.bin
```

The harness reads **only one original NVFP4 tensor** from the GGUF, maximum
256 MiB for memory safety. It never rewrites/quantizes/repackages weights.
The exact same loaded `W` and `X` device buffers are passed to each kernel;
the harness compares both buffers bit-for-bit against the host originals at
the end. It checks all `F` output rows, not a sample, and computes independent
CPU FP64 dot products over the exact GGUF scale/nibble layout. Every scale
byte `>=0x80`, bad tensor/type/dimension/file length, NaN/Inf or CUDA failure
causes a nonzero exit. No heuristic fallback is silently treated as a pass.
If the original FP32 vector is available, the isolated CUDA harness also
casts **every real input value** to BF16 with the same `__nv_bfloat16` cast
expression as production and requires bit-exact agreement with its CPU RNE
reconstruction. This checks actual CUDA conversion semantics, but it still
does not replace a direct readback of the production `d_gb` allocation.

Each variant runs **10 consecutive times**. Before each call, 16 sentinel
words are set before and after its output; the test checks their exact bits
and all outputs finite. Repeated results must be bitwise identical to that
variant's first run. Each variant is independently compared to the complete
FP64 reference. All three variant pairs are compared directly; the argmax
across a single projection's output `F` must match. No kernel timing occurs.

Default diagnostic/gate thresholds on individual rows are:

- Empirical: `abs(output - CPU_FP64) <= 1e-4 + 2e-4*abs(CPU_FP64)`.
- Floating-point envelope: `abs(output - CPU_FP64) <= gamma_n *
  sum_k(abs(x[k]*w[k])) + 2*ulp(float(CPU_FP64))`, with `u=2^-24`,
  `gamma_n = n*u/(1-n*u)`, `B=H/64`, and
  `n=max(192*ceil(B/256)+8,16*ceil(B/4)+5)`. This is a conservative bound
  for the compared FP32 pair accumulations and tree reductions under normal
  IEEE operations. GPU FTZ/compiler choices can invalidate a theoretical
  assumption, so any real violation is reported, not waved away.
- Direct variant pair: `abs(A-B) <= 1e-4 + 2e-4*max(abs(A),abs(B))`.

**Both** individual numerical gates, all pair gates and the determinism,
finiteness, guard, immutability and output-argmax checks must pass.
The program reports per-variant/pair maximum absolute difference, RMSE,
worst row, and maximum ULP distance to FP64 (ULP is diagnostic around zero,
not a separate success criterion). Near-cancellation can produce larger ULP
counts even for small absolute differences. These thresholds are deliberately
stricter than the FP32 envelope in some cases and may fail on legitimate
roundoff; investigate the flagged row rather than increasing tolerances by
default.

**Activation precision diagnostic:** when the input is a layer26 FP32 capture
or a direct BF16 input paired with `--original-f32`, CPU code computes two
independent FP64 dot products over the **same raw GGUF weights**:
`dot(W, original_FP32_d_g)` and `dot(W, BF16_RNE(original_FP32_d_g))`.
It prints `FP32_VS_BF16_DOT` with maximum absolute projection difference,
RMSE, RMSE relative to the RMS of the BF16 projection, and worst row.
This isolates activation-rounding error at that GEMV. It does not establish
the cause of a llama.cpp/HERMES logit discrepancy, because both models'
preceding hidden states and subsequent layers require independent comparisons.
If original FP32 is unavailable the diagnostic reports `unavailable`.

## Additional llama.cpp Q8_1-like CPU diagnostic

The inspected llama.cpp CUDA source at
`ggml/src/ggml-cuda/quantize.cu:5-50` quantizes each group of 32 FP32
activation elements by `amax=max(abs(x))`, `d=amax/127.f`,
`q=roundf(x/d)` (zero if amax=0), and stores `d` rounded to FP16 in
`make_half2(d,sum)`. Its `vecdotq.cuh:331-357` NVFP4×Q8_1 path uses an
integer DP4A dot and the FP16 `d` scale. The isolated
`gemv_real_q8_cpu` command models that **activation representation** with
nearest-even FP16 scale and signed int8 values, then uses the independent
FP64 GGUF dot. It compares three vectors evaluated on **the same captured
HERMES F32 activation**: original F32, HERMES BF16-RNE and Q8_1-like.
It does not execute llama's CUDA reduction and cannot substitute for llama's
actual intermediate F32 input. The actual llama input may already differ from
HERMES before this GEMV. The simulator will reject nonfinite values, a
rounded FP16 scale overflow, invalid signed int8 values, and missing original
F32. This is a representation error budget, not a logit-oracle PASS.

## Real-checkpoint, real-activation run on 2026-11-13

`HERMES_L26=1 HERMES_NVFP4_GEMV=scalar` was enabled for **one short run
of the existing scratch** `/tmp/hermes27-prime-integ/build/hermes_generate`
using the model at the path above, fixed seed IDs
`6205,161421,303,220,17,1374,9911,1147,27053,6059,13`,
and `--max-new-tokens 2`. It used legacy synchronous decode,
`HERMES_PP2_ASYNC_STREAMS=0`, no MTP. It generated `271,248068` and wrote
12 consecutive layer26 capture records (11 seed, one subsequent decode).
File `/tmp/l26_dec.bin` was 1,081,344 bytes (12×90112), SHA-256:
`8da888003e5737aa1a0d8b1b01e5c94fb5de1607733d811a84197c4ba1acd04d`.
This `/tmp` file is ephemeral, not a checked-in artifact.

Selected real tensor `blk.26.ssm_out.weight`: H=6144, F=5120,
17,694,720 raw GGUF bytes, absolute offset 10,884,365,664, diagnostic
FNV64 `6cad8852d373fee4` in the 19,694,360,544-byte checkpoint. Tests
used one weight buffer and the **same** reconstructed BF16 activation buffer
for all kernels. The isolated CUDA cast probe produced exactly the same
6,144 BF16 bit patterns as CPU nearest-even, at both record positions.

| Real record | Scalar↔Warp4 max abs | Scalar↔Packed4 max abs | Warp4↔Packed4 max abs | Maximum individual deviation from CPU FP64 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 4.768372e-7 | 1.192093e-7 | 4.768372e-7 | 5.722832e-7 (Warp4) |
| 11 | 2.384186e-7 | 8.940697e-8 | 2.384186e-7 | 1.322478e-7 (Warp4) |

For **both records**, each variant passed 10 bit-for-bit repeat runs, all
5,120 FP32 output values finite, 16 output guard words on each side,
unchanged input/weight GPU buffers, zero FP64/empirical/pair bound failures,
and identical projection argmax across all three kernels. All three pair
comparisons and all CPU FP64 comparisons covered every output row. This
experiment required a granted exclusive GPU slot; that slot was released
immediately afterward, with GPU memory/usage returned to the observed
486/1289 MiB and 0% baseline. The **actual production d_gb allocation was
not independently copied**, although CUDA's exact conversion expression was
verified against each real F32 input with a standalone CUDA kernel.

CPU-only activation representation results for those same two real inputs:

| Real record | Comparison with same raw GGUF weights, CPU FP64 | Max abs projection difference | RMSE projection difference |
| --- | --- | ---: | ---: |
| 0 | BF16-RNE versus original F32 | 0.000717769 | 0.0000802462 |
| 0 | Q8_1-like versus original F32 | 0.002803268 | 0.0005090941 |
| 0 | Q8_1-like versus BF16-RNE | 0.002930454 | 0.0005204823 |
| 11 | BF16-RNE versus original F32 | 0.001148036 | 0.0002848153 |
| 11 | Q8_1-like versus original F32 | 0.006035369 | 0.0015196183 |
| 11 | Q8_1-like versus BF16-RNE | 0.005901576 | 0.0015350287 |

Q8_1-like scales were nonzero in all 192 groups of 32 for both inputs.
The difference between CPU simulated Q8_1 and BF16 projection is much
larger than direct Scalar↔Warp4↔Packed4 rounding differences on this single
matrix, but **does not prove causality** for full-model logits. Validate
the actual llama.cpp input activation, numeric path and subsequent layer
propagation before making any quantitative accuracy claim across engines.

## What success proves and does not prove

A PASS verifies these three GEMV kernel entry points on **one real checkpoint
tensor and one same-input BF16 replay**. It does not establish 64-layer
accumulated logits parity, long-context/near-tie behavior, concurrency
correctness, correctness against llama.cpp, or decode performance. The
previous three-step scalar↔Warp4↔Packed4 full-vocabulary drift therefore
remains a separate model-level investigation. Repeat this harness on
real inputs from multiple operators/layers and steps before attributing
model-level drift exclusively to FP32 reduction order. Additional capture
points need explicit tagged BF16 instrumentation in the producing stream;
this test does not modify the serving/inference path to add them.
