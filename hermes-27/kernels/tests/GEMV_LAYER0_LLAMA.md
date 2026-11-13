# Real llama.cpp layer-0 QKV capture versus HERMES NVFP4 GEMV

Status (2026-11-13): CPU input checks, SHA-256 and full 10,240-row
FP64 oracle PASS. CUDA Scalar/Warp4/Packed4 compiled for CUDA 12.8 /
SM120a and real captured-input GEMV replay PASS on GPU0 after prime
explicitly granted an exclusive slot. This GPU slot was released and
the verified post-run VRAM baseline is GPU0 486 MiB, GPU1 1289 MiB,
both 0% utilization. No production code or CMake changes.

## Exact capture provenance

Read-only capture: /tmp/h27-layer0-llama-pos0-prime3/manifest.json

Manifest schema h27.reliability-layer-cb.v1, backend llama.cpp,
status CAPTURED_NOT_INDEPENDENTLY_COMPARED, layer=0, absolute_position=0,
prompt_ids=[6205], greedy continuation 161421, n_gpu_layers=80,
callback_ask_count=callback_ready_count=3.
Tensor encoding: little-endian FP32, ne0 fastest, CUDA0. All original
captured tensors were hashed and checked against manifest:

| Name | Float elements | Bytes | Verified SHA-256 |
| --- | ---: | ---: | --- |
| attn_norm-0.f32 | 5120 | 20480 | 492ae3d622039e01dcce9c5d8bc32ff2986e1134d5c50fad47ff366e8561bfa3 |
| linear_attn_qkv_mixed-0.f32 | 10240 | 40960 | 4b32da962e03f2481aa869955cea378b7921418d920459e9e62cf9300c08184f |
| conv_output_silu-0.f32 | 10240 | 40960 | 12513886591fd03cc1a28c2d671cb76217242339300037f1988da4031a9d56a2 |

GGUF checkpoint (19,694,360,544 bytes) **entire-file SHA-256** verified:
db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1.
Its original blk.0.attn_qkv.weight tensor is NVFP4, H=5120, F=10240,
29,491,200 bytes, absolute file offset 5,114,324,960, maximum scale byte
0x19=25 and no scale bytes >=0x80. The actual llama library file hash
fd9e3e0810c6eca2e37b57abf443bcf15963671cbd0ee8489a37461a521094d4
also matches manifest. Exporter hash
2259be0c502f5488fee09c8826a16986cb437167192cd352d0e408801c5ef241
is recorded but the manifest lacks its executable path; exporter identity
was NOT independently checked. Use the optional --exporter parameter if
that executable becomes available.

## Source and acceptance contract

The two NEW sources are:

1. kernels/tests/gemv_layer0_oracle.py: required manifest verifier and
   default CPU-only independent GGUF projection oracle.
2. kernels/tests/gemv_layer0_compare.cu: isolated CUDA subordinate only,
   launched through the Python verifier when prime grants GPU.

The Python script checks manifest schema/position/prompt/strides/types,
exact tensor lengths and SHA-256, entire model SHA-256, libllama SHA-256,
GGUF tensor geometry, actual NVFP4 scale domain and finite captured inputs.
All input digests are rechecked after CPU numerical processing. It mmaps
only the needed 29.5 MB GGUF tensor and does not load the entire checkpoint.

The independent CPU FP64 oracle dequantizes every 36-byte NVFP4 block:
4 unsigned UE4M3 scales (one scale per 16 weights); 8 bytes for each
sub-block (low nibble maps to k=16*s+j and high nibble to k=16*s+j+8).
It computes the 10,240 real F32-input and BF16-input FP64 dot products
and a serial CPU FP32 accumulation of the BF16 products (diagnostic only).
Finite F32 input converts to BF16 with ties-to-even using:

    uint16_t bf16 = uint16_t((f32_bits + 0x7fffU + ((f32_bits >> 16) & 1U)) >> 16);

BF16-RNE capture SHA-256:
5d88ad4e5c0a88bf08af31cc37708e4745b991fe481d30fbfad2ddbc02614600.

CPU results, measured on the verified position-0 llama capture:

| Compared full 10240-row projection | Max abs | RMSE |
| --- | ---: | ---: |
| FP64 with BF16-RNE input vs FP64 with original F32 input | 0.0177226965403 | 0.00202888471726 |
| CPU FP32 serial BF16 dot vs FP64 BF16 dot | 0.00002804398537 | 0.000000745336826 |
| Captured llama QKV vs CPU FP64 original F32 dot | 0.0352148329298 | 0.00570703138739 |
| Captured llama QKV vs CPU FP64 BF16-RNE dot | 0.0366220474243 | 0.00596864406834 |

These are numerical diagnostics, not llama-versus-HERMES parity passes.
The llama CUDA path may use Q8_1 activation quantization and a different
integer reduction. Another worker has separately reported close agreement
of the Q8_1 FP64 model and llama QKV; the present harness has NOT repeated
that independent result. The captured tensor is named attn_norm-0, but
the claim that it equals the exact GEMV input for every backend and mode
requires independent model-graph verification.

## Reproduce CPU verification (no GPU)

    python3 kernels/tests/gemv_layer0_oracle.py --mode cpu \
      --capture-dir /tmp/h27-layer0-llama-pos0-prime3 \
      --model /home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf

Requires Python 3 and NumPy. The script defaults to CPU, does not initialize
CUDA, and fails closed for any missing/modified capture, incorrect digest,
shape, dtype, unexpected position or invalid quantization metadata.

## CUDA build-only (allowed without GPU ownership)

    mkdir -p /tmp/h27-gemv-layer0-build
    /usr/local/cuda-12.8/bin/nvcc -O3 -std=c++20 -arch=sm_120a \
      -Ikernels/include -Imodel/include -Ikernels/tests \
      model/src/gguf.cpp kernels/src/nvfp4_gemv.cu \
      kernels/src/rmsnorm.cu kernels/src/nvfp4_gemv_warp_variants.cu \
      kernels/tests/gemv_layer0_compare.cu \
      -o /tmp/h27-gemv-layer0-build/gemv_layer0_compare

This is a scratch build. It reuses the unchanged current Scalar/Warp4/
Packed4 CUDA sources, and the read-only existing gemv_real_common.hpp.
The server checkout may lack the newer Packed4 source; use read-only
copies of the matching source revision in scratch if necessary.
Build PASSed with CUDA 12.8 SM120a from source copies under
/tmp/hermes-gemv-layer0-worker8. Only preexisting unused-variable warning
at nvfp4_gemv.cu:66 (rm) occurred. The compiled executable was subsequently
used for one specifically authorized, bounded GPU0 replay.

## GPU acceptance — previously executed after explicit prime grant

    timeout 110s python3 -u kernels/tests/gemv_layer0_oracle.py \
      --mode gpu --execute-gpu \
      --capture-dir /tmp/h27-layer0-llama-pos0-prime3 \
      --gpu-exe /tmp/h27-gemv-layer0-build/gemv_layer0_compare \
      --device 0

The verified Python wrapper is the mandatory GPU entrypoint. The standalone
CUDA binary does not authenticate the manifest itself. After CPU proof,
the wrapper writes only verified, temporary copies of F32 input, BF16-RNE
input and captured llama QKV, then invokes a 10x-each three-kernel replay
on one shared raw W and one shared BF16 X. Requirements:

- 5120 CUDA F32->BF16-RNE conversions bitwise identical to CPU.
- Scalar, Warp4, Packed4 each run 10 times with bit-identical outputs,
  16 sentinel words before and after, 10240 finite FP32 results each.
- Every output row compared with independent CPU FP64 BF16 projection;
  both 1e-4+2e-4*abs(FP64) and conservative FP32 rounding envelope pass.
- All three kernel variant pairs compared across all 10240 rows with
  1e-4+2e-4*max(abs(A),abs(B)) per-row bound.
- W/X device buffers unchanged; all CUDA errors fail closed.
- Each kernel also reports 10240-row max abs, RMSE, worst row against
  captured llama QKV. Such metrics are NOT a llama semantics PASS.

The bounded run completed with exit 0. Results used the exact same real
weights and activation for each of the three variants. GPU BF16-RNE
matched all 5120 input values bitwise. Scalar, Warp4, Packed4 each had
10 identical repeat outputs, all 10240 rows finite, 16+16 guard words
intact, input/weights unchanged, zero per-row FP64/empirical errors,
and zero pairwise threshold violations.

| Real GPU path | Max abs against CPU FP64 BF16 | RMSE against CPU FP64 BF16 |
| --- | ---: | ---: |
| Scalar | 0.000002473592758 | 0.0000001288717297 |
| Warp4 | 0.000003665685654 | 0.0000001390041257 |
| Packed4 | 0.000003233551979 | 0.0000001330723781 |

Every variant pair max abs was 0.000003814697266; pairwise RMSE
Scalar/Warp4 0.0000001760872467, Scalar/Packed4 0.0000001810412395,
Warp4/Packed4 0.0000001730431115. Each pair had zero threshold
violations.

| Real GPU path | Max abs against captured llama QKV | RMSE against captured llama QKV |
| --- | ---: | ---: |
| Scalar | 0.03662204742 | 0.005968642314 |
| Warp4 | 0.03662204742 | 0.005968645941 |
| Packed4 | 0.03662204742 | 0.005968646774 |

For all three, worst llama discrepancy was output row 3604. Since the
variants agree with the same BF16-input CPU FP64 oracle within about
3.7e-6, their 0.00597-RMSE difference from the original llama QKV
capture is not explained by a wrong NVFP4 nibble mapping in these
kernels on this input. This test alone does not prove which distinct
activation quantizer or upstream operator causes the llama discrepancy.
The independent llama Q8_1 reference, if accepted, should be considered
as separate evidence with its exact captured source and reduction rules.

The BLK0 numerical experiment does not prove 64-layer logit correctness.
To claim llama CUDA equivalence, first establish matching GGUF/position,
the actual input/quantizer representation, and equivalent operator outputs.
The worker released both GPUs immediately after the bounded run and
verified the post-run baseline of 486 MiB / 1289 MiB, both 0%.
Future executions still require a new explicit GPU grant.
