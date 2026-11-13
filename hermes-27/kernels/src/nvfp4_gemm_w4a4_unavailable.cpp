// Fail-closed CPU linker shim for the optional CUTLASS W4A4 experiment.
//
// HERMES-27's production NVFP4 GEMV and BF16-exact prefill do not need
// CUTLASS. Preserve the old diagnostic ABI for P1 tests and PP2 upload
// statistics without silently enabling activation requantization.
#include "hermes/nvfp4_gemm.hpp"

namespace hermes {
size_t Nvfp4GemmW4A4::sfa_bytes(int, int) { return 0; }
size_t Nvfp4GemmW4A4::sfb_bytes(int, int, int) { return 0; }
void Nvfp4GemmW4A4::repack_sfa_cpu(const uint8_t*, uint8_t*, int, int) {}
void Nvfp4GemmW4A4::quant_activation(const uint16_t*, uint8_t*, uint8_t*, int, int) {}
void Nvfp4GemmW4A4::build_B_from_gguf(const uint8_t*, uint8_t*, uint8_t*, int, int) {}
size_t Nvfp4GemmW4A4::workspace_bytes(int, int, int) { return 0; }
int Nvfp4GemmW4A4::workspace_ensure(Workspace&, int, int, int) { return -1; }
void Nvfp4GemmW4A4::workspace_release(Workspace& ws) {
  // There is no valid workspace under this build. Never free an unknown
  // pointer; callers may own it. An enabled CUTLASS build uses its own API.
  ws.ptr = nullptr;
  ws.bytes = 0;
}
int Nvfp4GemmW4A4::run(const uint8_t*, const uint8_t*, const uint8_t*,
                          const uint8_t*, const uint16_t*, uint16_t*, int,
                          int, int) { return -1; }
int Nvfp4GemmW4A4::run_ws(const uint8_t*, const uint8_t*, const uint8_t*,
                             const uint8_t*, const uint16_t*, uint16_t*, int,
                             int, int, void*, size_t, cudaStream_t) { return -1; }
const char* Nvfp4GemmW4A4::backend_name() {
  return "disabled (set HERMES_BUILD_W4A4_PROTOTYPE=ON with CUTLASS)";
}
#ifdef HERMES_ENABLE_DEVICE_QUANT
int Nvfp4GemmW4A4::quant_activation_device(const uint16_t*, uint8_t*,
                                              uint8_t*, int, int, cudaStream_t) {
  return -1;
}
#endif
} // namespace hermes
