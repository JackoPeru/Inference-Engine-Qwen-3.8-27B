#pragma once
#include <cstdint>
// SwiGLU MLP: g = silu(x @ Wg), u = x @ Wu, y = (g*u) @ Wd
// Qwen3.8-27B: h=5120, f=17408. Wg,Wu: [h,f] row-major (h rows of f), Wd: [f,h].
// BF16 bits. CPU ref float; CUDA fuses gate/up/act, then down proj.
namespace hermes {
void mlp_cpu(const uint16_t* x, const uint16_t* Wg, const uint16_t* Wu,
             const uint16_t* Wd, uint16_t* y, int h, int f);
#ifdef __CUDACC__
void mlp_cuda(const uint16_t* x, const uint16_t* Wg, const uint16_t* Wu,
              const uint16_t* Wd, uint16_t* y, int h, int f);
#endif
}
