# Q8_1 activation x raw NVFP4 GGUF: isolated accuracy experiment

**State: CPU reference PASS, CUDA12.8 standalone compilation/link PASS,
DP4A T1 and T64 real-GPU strict A/B PASS as reported by prime.**
Experiment remains opt-in only; production full-model quality and
multiuser throughput are NOT established. No default integration.
No PP2/serving/CMake modification. No production readiness claim.

## Why Q8_1 matters

Worker-1 captured independent llama.cpp layer0 tensors for token 6205,
position 0, GGUF SHA-256:
db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1.

- Source capture: /tmp/h27-layer0-llama-pos0-prime3/manifest.json
- Input: attn_norm-0.f32, 5120 little-endian FP32 values.
- Output: linear_attn_qkv_mixed-0.f32, 10240 little-endian FP32 values.
- Real weight: blk.0.attn_qkv.weight, GGUF NVFP4, N=10240 K=5120,
  29,491,200 bytes.

On the identical input and weights, CPU Q8_1 reference reproduced
llama QKV (RMSE=1.57648647082e-7, maxabs=3.81469726562e-6).
Unquantized FP32 and BF16 CPU projections diverged from llama much
more (RMSE ~0.005707 and ~0.005969 respectively). This is evidence
that exact activation Q8_1 quantization can explain a layer0 QKV
difference; it is NOT proof of full-model quality or fast performance.

## Source semantics and implementation

Reference source inspected:
/opt/hermes/runtimes/llama-mainline/ggml/src/ggml-cuda/quantize.cu,
function quantize_q8_1; common.cuh float warp_reduce_max/sum.

For each 32 FP32 activation values:

1. Compute amax=max(abs(x)) with CUDA warp XOR shuffles.
2. Set d_fp32=amax/127.0f.
3. Set q[i]=int8(roundf(x[i]/d_fp32)), zero when amax==0.
   roundf uses ties away from zero, NOT ties to even.
4. Store d=FP16(d_fp32), s=FP16(warp_XOR_sum_FP32(x));
   then 32 signed int8 values. Q8_1 block is exactly 36 bytes.

Q8 integer quantization uses the **unrounded** FP32 d. Using the
stored rounded FP16 scale when computing q changes the result.
The sum field s is retained for Q8_1 compatibility but is not used
by the NVFP4 projection.

NVFP4 GGUF weights stay unchanged: 4 UE4M3 scale bytes and 32 packed
E2M1 bytes per 64 weights. E2M1*2 takes integer values
{0,1,2,3,4,6,8,12} with sign. GPU accumulates int32 products of
these integers and the signed Q8 activations within scale subgroups,
then multiplies by (0.5 * GGUF UE4M3 scale * stored FP16 Q8 d).
The output is FP32 T-by-N. FP32 reduction order is not bitwise
identical to llama.cpp's DP4A/VDR path.

**Important input contract:** this experimental API consumes the
FP32 layer activations *before* PP2 BF16 conversion. Passing BF16
or reinterpreting a uint16 buffer as float is invalid.

## Exact new file scope

- kernels/src/nvfp4_gemv_q8_1_experimental.cu
- kernels/include/hermes/nvfp4_gemv_q8_1_experimental.hpp
- kernels/tests/q8_1_cpu_test.py
- kernels/tests/q8_1_real_capture_cpu.cpp
- kernels/tests/q8_1_gpu_bench.cu
- kernels/tests/q8_1_dp4a_cpu_test.py
- kernels/tests/q8_1_dp4a_gpu_results.json
- kernels/tests/Q8_1_EXPERIMENT.md

No production source/CMake routing modifications.

## CPU/offline tests completed

Python standard-library q8_1_cpu_test.py passed **3007**
Q8_1 36-byte block quantization tests and **1000** random GGUF
E2M1x2 grouped integer-dot vs independent FP64 decoded weight tests.
There is no CUDA usage in this test.

The standalone CPU C++ oracle built using g++ and libhermes_gguf.a
and compared same actual GGUF plus authentic worker-1 llama capture:

    Q8_1_REFERENCE n=10240 input_amax=3.789217
    max_abs=3.81469726562e-06 rmse=1.57648647082e-07
    rel_rmse=4.95713357977e-08
    PASS real GGUF Q8_1 CPU numerical projection only

These match worker-1's independent CPU result exactly. The model
hash printed by the standalone C++ checker is a **declared capture
identifier**, not a fresh rehash of the 19.7GB file.

## Reproducible offline build — does not initialize a GPU

    root=/home/matteo/hermes-27
    s=/tmp/h27-q8-1-worker2
    mkdir -p "$s/src" "$s/include/hermes" "$s/tests"
    cp "$root/kernels/src/nvfp4_gemv_q8_1_experimental.cu" "$s/src/"
    cp "$root/kernels/include/hermes/nvfp4_gemv_q8_1_experimental.hpp" "$s/include/hermes/"
    cp "$root/kernels/tests/q8_1_gpu_bench.cu" "$s/tests/"
    cp "$root/kernels/tests/q8_1_real_capture_cpu.cpp" "$s/tests/"
    python3 "$root/kernels/tests/q8_1_cpu_test.py"
    g++ -std=c++20 -O3 -I"$root/model/include" \
      "$s/tests/q8_1_real_capture_cpu.cpp" "$root/build/libhermes_gguf.a" \
      -o "$s/q8_1_real_capture_cpu"
    model="$root/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf"
    sha=db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1
    "$s/q8_1_real_capture_cpu" "$model" /tmp/h27-layer0-llama-pos0-prime3 "$sha"
    nvcc=/usr/local/cuda-12.8/bin/nvcc
    "$nvcc" -std=c++20 -O3 -arch=sm_120a -I"$s/include" \
      -c "$s/src/nvfp4_gemv_q8_1_experimental.cu" -o "$s/q8.o"
    "$nvcc" -std=c++20 -O3 -arch=sm_120a \
      -I"$s/include" -I"$root/model/include" \
      -c "$s/tests/q8_1_gpu_bench.cu" -o "$s/bench.o"
    "$nvcc" -std=c++20 -O3 -arch=sm_120a -o "$s/q8_bench" \
      "$s/bench.o" "$s/q8.o" "$root/build/libhermes_gguf.a"
    "$s/q8_bench" --help

## Future GPU tests: DO NOT RUN without explicit prime-exclusive grant

First verify T1 on identical authentic llama X/W, then synthetic
T64 only if T1 passes. All T>1 rows after the first use deterministic
perturbations of the real llama input, NOT new llama positions.

    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
    timeout 60 /tmp/h27-q8-1-worker2/q8_bench \
      --gguf "$model" --capture /tmp/h27-layer0-llama-pos0-prime3 \
      --T 1 --device 1 --loops 10
    timeout 60 /tmp/h27-q8-1-worker2/q8_bench \
      --gguf "$model" --capture /tmp/h27-layer0-llama-pos0-prime3 \
      --T 64 --device 1 --loops 10

The benchmark checks all Q8 packed block q, FP16 d and FP16 s
against independent CPU quantization; selected FP64 CPU output samples;
the authentic full QKV output for position0; CUDA event quantization,
GEMV and combined latency; and device allocation delta. The local
diagnostic projection gate maxabs<=0.001 / RMSE<=0.0001 only detects
obvious bugs, NOT a full-model quality release gate.

## Memory / throughput caveats

Q8 scratch is **5760 bytes per token** at K=5120:

| T | Q8 scratch (bytes) |
|---|---:|
| 1 | 5,760 |
| 16 | 92,160 |
| 64 | 368,640 |
| 128 | 737,280 |
| 256 | 1,474,560 |

Scratch excludes original GGUF W, FP32 X and Y, model KV/GDN,
other PP2 buffers/CUDA context. No duplicate persistent weight copy.
This simple per-token output GEMV reads the same raw GGUF weight
matrix per token; it has no Tensor Core batching and may be slower
than the existing optimized paths. No throughput gain is claimed.

Even a GPU QKV match would not prove the complete model. Strict
full-model logits/argmax, state continuation, and 1/4/8 user
throughput/VRAM tests remain mandatory. Previous BF16 full-model
T64 argmax failures must NOT be ignored or used to loosen thresholds.

## New follow-up: explicit batched DP4A NVFP4 x Q8_1 experiment

Prime reported independent real-GPU quality PASS for the ORIGINAL
Q8 kernel at T1 and T64 (authentic first-token llama layer0 QKV
maxabs 7.629e-6). However, T64 Q8 GEMV was **58.006 ms** in a
separate run vs raw Warp4 **9.056 ms**. The old Q8 GEMV kernel is
not a viable throughput path, even if numerically correct.

The original quantizer, gemv_q8_1_nvfp4, and baseline projection
API are preserved unchanged. NEW opt-in entrypoints are:

- gemv_q8_1_nvfp4_dp4a_batched(x_q8, rawW, y, T, N, K, stream)
- project_f32_q8_1_nvfp4_dp4a_batched(x_f32, rawW, q8_scratch, y, T, N, K, stream)

The SAME original quantize_f32_q8_1 function supplies activation
Q8 blocks. The new projection maps one warp to one GGUF output row,
reads four contiguous GGUF 64-weight blocks per iteration, and
reuses those FP4 bytes over up to **8 tokens**. Each warp lane reads
four consecutive packed FP4 bytes, extracts 4 lower and 4 upper
E2M1 nibbles, converts each to the EXACT signed E2M1*2 int8, and
executes two native signed DP4A operations per token.
Q8 scales remain stored FP16. Output is FP32 row-major.
The kernel requires K divisible by 256 and four-byte-aligned
GGUF pointers; other layouts fail closed without fallback.

This changes the intended raw weight rereads for T64 from 64 to
8 per row, an algorithmic traffic reduction of 8x. Prime's
T64 same-process GPU benchmark now demonstrates a speedup over
the **original Q8 GEMV for this one projection**, not an actual
8x DRAM bandwidth result or an end-to-end model improvement. The
FP32 summation order differs from llama CUDA and the old Q8 kernel,
so no numerical equality is presumed. The entire raw GGUF weight
matrix stays resident in its original representation, with
**zero additional weight copies**. T64/K5120 Q8 temporary scratch
remains 368,640 bytes. This approach does not use Tensor Cores.

### CPU/offline gate, completed

- q8_1_cpu_test.py PASS: 3007 Q8 signed int8 quantization blocks
  and 1000 random GGUF integer dot comparisons.
- q8_1_dp4a_cpu_test.py PASS: 120 signed nibble/DP4A mapping
  checks, 999 independent numeric outputs across 18 shapes
  K=256/512/1024 and T=1/7/8/9/16/64 including tails,
  ten-repeat deterministic simulation.
- Standalone CUDA 12.8 SM120a NVCC compile and link PASS.
- Executable help (no GPU initialization) PASS.

Offline build (DOES NOT RUN GPU):

    root=/home/matteo/hermes-27
    s=/tmp/h27-q8-dp4a-worker2
    mkdir -p "$s/src" "$s/include/hermes" "$s/tests"
    cp "$root/kernels/src/nvfp4_gemv_q8_1_experimental.cu" "$s/src/"
    cp "$root/kernels/include/hermes/nvfp4_gemv_q8_1_experimental.hpp" "$s/include/hermes/"
    cp "$root/kernels/tests/q8_1_gpu_bench.cu" "$s/tests/"
    python3 "$root/kernels/tests/q8_1_cpu_test.py"
    python3 "$root/kernels/tests/q8_1_dp4a_cpu_test.py"
    nvcc=/usr/local/cuda-12.8/bin/nvcc
    "$nvcc" -std=c++20 -O3 -arch=sm_120a -I"$s/include" \
      -c "$s/src/nvfp4_gemv_q8_1_experimental.cu" -o "$s/q8_dp4a.o"
    "$nvcc" -std=c++20 -O3 -arch=sm_120a \
      -I"$s/include" -I"$root/model/include" \
      -c "$s/tests/q8_1_gpu_bench.cu" -o "$s/bench.o"
    "$nvcc" -std=c++20 -O3 -arch=sm_120a -o "$s/q8_bench" \
      "$s/bench.o" "$s/q8_dp4a.o" "$root/build/libhermes_gguf.a"
    "$s/q8_bench" --help

### Reproducible GPU acceptance: every rerun requires a fresh prime grant

    model=/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf
    capture=/tmp/h27-layer0-llama-pos0-prime3
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
    timeout 60 /tmp/h27-q8-dp4a-worker2/q8_bench \
      --gguf "$model" --capture "$capture" --T 1 \
      --device 1 --loops 10 --dp4a-ab
    # T64 only after T1 PASS and while prime retains exclusive grant.
    timeout 60 /tmp/h27-q8-dp4a-worker2/q8_bench \
      --gguf "$model" --capture "$capture" --T 64 \
      --device 1 --loops 10 --dp4a-ab

The flag --dp4a-ab is MANDATORY for this experimental path.
Without it, the benchmark remains on the original Q8 path.
For each shape, compare **all T*N outputs** against the original
same-Q8 GEMV and all 10,240 authentic llama QKV outputs for token0.
Run 10 DP4A repeats, require **bitwise identical** FP32 output
arrays and all finite values. For T64 the additional 63 input
rows are deliberately synthetic perturbations, NOT authentic llama
positions, and cannot prove full-model accuracy.

Additional strict A/B threshold (tighter than earlier exploratory
0.001/0.0001): max abs vs original Q8 <=1e-5, RMSE <=2e-6;
authentic llama maxabs <=7.62939453125e-6, RMSE <=2e-6;
CPU FP64 QKV maxabs <=1e-5, RMSE <=2e-6.
**Any violation exits nonzero before DP4A latency is reported**;
no gate is loosened. After quality PASS, CUDA events measure Q8
quantization, original and DP4A GEMV, and full quant+GEMV pipelines
in the same process. Even this local PASS would NOT authorize
production, PP2 routing or full-model quality parity.

### Prime's real-GPU A/B results (13 November 2026)

Prime executed the bounded CUDA benchmark against the genuine GGUF
layer0 QKV matrix on an RTX 5060 Ti. The original Q8_1 GEMV and
the DP4A candidate were timed **in the same process**. Both T1 and
T64 passed the unmodified strict projection A/B gate. These results
were reported by prime; raw GPU console logs are not embedded here.

| Same-process CUDA GEMV latency | T1 | T64 |
|---|---:|---:|
| Original Q8_1 | 0.928848 ms | 57.990173 ms |
| Experimental DP4A batched Q8_1 | **0.804874 ms** | **6.722262 ms** |
| Speedup vs original Q8_1 | 1.154x | 8.627x |

For T64, all **655,360 FP32 outputs** (64 x 10,240) compared
against the original Q8_1 path with maximum absolute difference
**7.6293945e-6** and RMSE **3.07e-7**. Ten DP4A
repetitions were **bitwise identical**. The real first-token
DP4A output compared with the independent llama.cpp QKV capture:
maximum absolute difference **3.8146973e-6**, RMSE **1.81917e-7**
over all 10,240 columns.

Important provenance: only the first token uses a genuine llama.cpp
capture. At T64, the other 63 activation rows are deterministic
synthetic perturbations of that capture. The above GPU checks do NOT
establish all 64 real llama positions or full-model logit parity.

Prime confirmed post-test GPU memory returned to the prior idle
baseline: GPU0 **486 MiB**, GPU1 **1,289 MiB** used; both GPU
slots were released. These are idle-memory readings, **not**
the DP4A peak allocation. T64 Q8 activation scratch is
**368,640 bytes** (360 KiB); no persistent extra GGUF weight copy.

The previously reported **9.056 ms Warp4 T64** measurement was
from a **separate run**. It is not a controlled same-process A/B
against the 6.722262 ms DP4A measurement. Therefore, do not
claim an end-to-end PP2 or multiuser speedup from these timings.

**Decision remains NO-GO for default routing or production.**
Keep the original Q8_1 GEMV unchanged for A/B/rollback.
The strict full-model logits/argmax and stateful 1/4/8-user
quality/throughput gates have NOT been completed for DP4A.
Previously observed BF16 full-model T64 argmax mismatches remain
unresolved and are not grounds to relax any threshold.
