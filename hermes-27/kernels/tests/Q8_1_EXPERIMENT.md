# Q8_1 activation x raw NVFP4 GGUF: isolated accuracy experiment

**State: CPU reference PASS, CUDA12.8 standalone compilation/link PASS.**
GPU numeric/latency measurements pending **prime-exclusive GPU grant**.
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
