#pragma once
// HERMES-27 W4A4 NVFP4 GEMM (prefill path) via CUTLASS SM120 block-scaled MMA.
// GEMM config adapted from NVIDIA CUTLASS example 79a (BSD-3, (c) NVIDIA).
// Y[T,N] BF16 = Aq[T,K] NVFP4 @ Bq[K,N col-major] NVFP4 (alpha=1, beta=0).
// A = activations (quantized per chunk), B = weights (GGUF row-major nibbles
// shuffled to col-major + SFB interleave, once per matrix at load).
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
namespace hermes {
// Manageria CUTLASS (BSD NVIDIA, vedi nvfp4_gemm_w4a4.cu). Tutti i puntatori = device.
struct Nvfp4GemmW4A4 {
  // Byte size of interleaved scale tensor for logical [M, K]
  static size_t sfa_bytes(int M, int K);
  // GGUF flat scales [M, K/16] UE4M3 -> interleaved device-ready (CPU repack)
  static void repack_sfa_cpu(const uint8_t* flat, uint8_t* out, int M, int K);
  // Host-side activation quant (GPU fused kernel = M3 optimization).
  // X_bf16 host [T,K] -> Aq nibbles (row-major via layout_A) + interleaved SFA.
  // Aq must be zeroed (size T*K/2); SFA size = sfa_bytes(T,K).
  // NOTE (hot path): host-side; richiede H2D di Aq/SFA per-call. Vedi
  // quant_activation_device (disabilitato di default) per il path fused futuro.
  static void quant_activation(const uint16_t* X_bf16, uint8_t* Aq, uint8_t* SFA, int T, int K);
  // One-time weight upload: GGUF row [N rows of 64B-blocks] -> B col-major
  // nibbles (via layout_B) + SFB interleave. Bq zeroed (size N*K/2).
  // gguf_row: pointer to N*K/64*36 bytes (row r at r*(K/64*36)).
  // NOTE (c): chiamare UNA volta al caricamento per matrice; Bq/SFB device
  // persistenti. Nessun lavoro GGUF nel percorso caldo run()/run_ws().
  static void build_B_from_gguf(const uint8_t* gguf_rows, uint8_t* Bq, uint8_t* SFB, int N, int K);
  static size_t sfb_bytes(int T, int N, int K);

  // (a) Workspace persistente per il percorso caldo.
  // Allocare una volta (ensure), riusare per ogni run_ws, liberare a shutdown.
  // Nessun cudaMalloc/cudaFree/cudaDeviceSynchronize dentro run_ws().
  struct Workspace {
    void* ptr = nullptr;
    size_t bytes = 0;
  };
  // Byte richiesti per (M_T,N,K). 0 = nessun workspace necessario.
  static size_t workspace_bytes(int M_T, int N, int K);
  // Assicura capacita` per (M_T,N,K): alloca/cresce una sola volta. 0 ok, <0 errore CUDA.
  static int workspace_ensure(Workspace& ws, int M_T, int N, int K);
  // Libera e azzera. Chiamare una volta a shutdown.
  static void workspace_release(Workspace& ws);

  // Run: D[T,N] = Aq[T,K] @ Bq[K,N as (N,K) colmajor] ; SFA/SFB interleaved. C may be null (beta=0).
  // Returns 0 ok, <0 unsupported/not-implemented for this shape.
  // COMPAT/COLD: alloca workspace per-call + cudaDeviceSynchronize. Preferire run_ws().
  static int run(const uint8_t* Aq, const uint8_t* SFA,
                 const uint8_t* Bq, const uint8_t* SFB,
                 const uint16_t* C, uint16_t* D, int M_T, int N, int K);
  // Hot path: usa workspace persistente del chiamante, nessuno malloc/free,
  // nessuna sync globale; enqueue su stream dato (0=default). workspace puo`
  // essere nullptr se workspace_bytes()==0. Ritorna 0 ok, <0 errore.
  static int run_ws(const uint8_t* Aq, const uint8_t* SFA,
                    const uint8_t* Bq, const uint8_t* SFB,
                    const uint16_t* C, uint16_t* D, int M_T, int N, int K,
                    void* workspace, size_t ws_bytes,
                    cudaStream_t stream = 0);
  static const char* backend_name();

#ifdef HERMES_ENABLE_DEVICE_QUANT
  // (b) Device-side activation quant, bit-esatto vs quant_activation(host).
  // X_d device BF16 [T,K] -> Aq_d nibbles + SFA_d interleaved (device).
  // Aq_d deve essere azzerato (T*K/2); SFA_d size sfa_bytes(T,K).
  // DISABILITATO di default (vedi .cu): abilitare con -DHERMES_ENABLE_DEVICE_QUANT=1
  // solo dopo verifica tiny host-vs-device su GPU1. MAI rompere la build.
  static int quant_activation_device(const uint16_t* X_d, uint8_t* Aq_d, uint8_t* SFA_d,
                                     int T, int K, cudaStream_t stream = 0);
#endif
};
}
