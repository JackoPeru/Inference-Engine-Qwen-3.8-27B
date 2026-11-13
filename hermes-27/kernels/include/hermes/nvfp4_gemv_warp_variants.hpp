#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

namespace hermes {

// Experimental, standalone alternatives to production NVFP4 Warp4.
// Raw GGUF [F,H]: H/64 consecutive blocks per row, each 36 bytes:
//   scale UE4M3[4], then E2M1[32] (q[j] low nibble at k=s*16+j,
//   high nibble at k=s*16+j+8). Input BF16; output FP32.
// All alternatives leave the production entry points unchanged.
enum class Nvfp4WarpExperiment : uint8_t {
  PackedU16x2 = 0,    // 2 GGUF blocks / warp iteration, 16 lanes/block
  PackedU32x4 = 1,    // 4 GGUF blocks / warp iteration, 8 lanes/block
  Unroll2 = 2,        // 2 GGUF blocks interleaved, original one-row warp
  WarpSplit2 = 3,     // 2 warps cooperating per row
  WarpSplit4 = 4,     // 4 warps cooperating per row
  Warp8Rows = 5       // 8 independent row warps / CTA
};

// No allocation or synchronization. Returns launch status, validates aligned
// inputs and dimensions; benchmark uses cudaEventSynchronize for timing.
cudaError_t nvfp4_gemv_warp_experiment(const uint16_t* x,
    const uint8_t* GGUF_W, float* y, int H, int F, Nvfp4WarpExperiment variant,
    cudaStream_t stream = 0);

} // namespace hermes
