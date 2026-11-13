#pragma once

// Standalone research-only prototype: GGUF NVFP4 T-token prefill via CUTLASS
// W4A4. Never included by production PP2 unless deliberately integrated.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime_api.h>

namespace hermes::prefill_prototype {

// The CUTLASS SFA swizzle is owned by Nvfp4GemmW4A4::repack_sfa_cpu.
// Resolve it once on the host, rather than reimplement a second CUTLASS layout.
// map[t*(K/16)+b] gives the device SFA byte offset for activation group (t,b).
// Returns false when the layout is unsupported or not a bijection.
bool build_sfa_scatter_map(int T, int K, std::vector<uint32_t>& map,
                           size_t& sfa_bytes, std::string& error);

// BF16 X[T,K] -> packed E2M1 activations (T*K/2 bytes) and interleaved UE4M3
// SFA. X, offset_map, Aq and SFA all reside on the active GPU.
// SFA must be zero-initialized once to handle any padding bytes.
// Each group owns 8 packed bytes and one scale destination; no atomics.
// Enqueues only. No sync, allocation or hidden host transfer.
cudaError_t quantize_bf16_device(
    const uint16_t* x_bf16, const uint32_t* offset_map, uint8_t* aq,
    uint8_t* sfa, int T, int K, cudaStream_t stream = 0);

// Reference to current PP2 default: repeated scalar raw-GGUF GEMV per token
// (same launch/grid/ordering as kernels/src/nvfp4_gemv.cu). Y[T,N] FP32.
// A local baseline copy keeps this prototype independent of kernel API drift.
cudaError_t repeated_gguf_gemv(
    const uint16_t* x_bf16, const uint8_t* raw_gguf_weights, float* y,
    int T, int N, int K, cudaStream_t stream = 0);

} // namespace hermes::prefill_prototype
