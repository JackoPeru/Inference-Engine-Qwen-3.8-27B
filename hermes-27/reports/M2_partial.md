# M2 parziale — Math single-GPU + TP2 (2026-11-12, in corso)

## Toolchain SM120 (verificata, non distruttiva)
- Toolkit: `cuda-nvcc-12.8` + `cuda-cudart-dev-12.8` + `libcublas-dev-12.8` e `13.2` da repo NVIDIA (`cuda-keyring`), solo toolkit. `nvcc --version` 12.8.93 e 13.2.86 OK. Produzione invariata (GPU0 208MB free, GPU1 ~3.3GB).
- CUTLASS: `thirdparty/cutlass` checkout **v4.2.2** (main ha bug host `subbyte_reference.h: __nv_atomic_load_n` sia con 12.8 che 13.2). Build `79a_blackwell_geforce_nvfp4_bf16_gemm` con CUDA 12.8 arch `120a`: LINK OK.
- NVFP4 Tensor Core reali su 5060 Ti (GPU1, `CUDA_VISIBLE_DEVICES=1`):
  - `--m=512 --n=512 --k=512`: Passed, 0.006ms, 44.5 TFLOPS
  - `--m=2048 --n=2048 --k=2048`: Passed, 0.067ms, **257 TFLOPS**
  - Prova che il percorso GPU esegue vere MMA `block_scale ... e2m1`, non dequant BF16.

## Kernel propri (C++20 + CUDA, `hermes_kernels`, arch 120a)
- `kernels/src/rmsnorm.cu`: 1 blocco/riga, reduction shared, BF16 in/out, eps 1e-6. Test hidden 5120: max err 0.0 PASS.
- `kernels/src/rope.cu`: RoPE standard coppie, BF16, `rotary_dim` param (hd 256, rd 64 = partial 0.25, pos 1000, theta 10M). Test: err 0.0 PASS.
- mRoPE Qwen3.8 (sections [11,11,10], interleaved, partial 0.25): nucleo sopra riusato per-sezione nel layer architecture; mapping completo in M2.3.
- `ctest`: 3/3 PASS (gguf_synth, rmsnorm, rope) su GPU1 senza toccare produzione.

## Manca per M2 done
- Gated DeltaNet (chunked vs token-by-token), full attention causale, MLP gated, output head, dequant on-the-fly W4A4, TP2 shard + inferenza corretta, confronto logit vs runtime stesso checkpoint, sampling. Gated DeltaNet non ancora implementato → M2 NON dichiarata conclusa.

Prossimo: DeltaNet chunked kernel + single-layer forward vs riferimento, poi TP2.
Riproduci: `cmake -S . -B build && cmake --build build -j6 && CUDA_VISIBLE_DEVICES=1 ctest --output-on-failure`
