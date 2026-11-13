# HERMES-27 Architettura (indipendente, Qwen3.8-27B only)

```
GGUF NVFP4-MTP (mmap) -> model/ (parser strict + IWeightBackend)
  -> distributed/ (TP2 shard weights/comm; PP2 shard layers; host-pinned, no P2P)
  -> memory/ (arena per-GPU, KV paged FP8, DeltaNet state separato, CoW prefix)
  -> kernels/ (SM120: NVFP4 W4A4 CUTLASS, RMSNorm, RoPE mRoPE, MLP gated, attn)
  -> architecture/ (qwen3_5: 48x GatedDeltaNet + 16x full_attn, MTP NextN block64)
  -> scheduler/ (continuous batching, chunked prefill, admit/drop, fairness)
  -> speculation/ (MTP proposal/verify/rollback, K=1..5 adattivo)
  -> serving/ (OpenAI-compat, streaming SSE, /health /metrics)
```

- core C++20 + CUDA SM120, Python solo test/bench/piano controllo.
- Ordine ott: (1) throughput multiutente, (2) prefill/TTFT, (3) decode singolo.
- Non wrapper/fork: riuso solo CUTLASS/cuBLAS/primitive matematiche. Riferimento llama.cpp solo per formato.
- Correttezza prima: parser sintetico, GEMM vs FP32/BF16, forward vs runtime stesso checkpoint, DeltaNet chunked vs token-by-token, MTP greedy identity, sampling preserva distribuzione.
- Stato attuale: M0 audit+baseline OK, M1 parser+dump+synth in corso, GGUF download 19.6GB background.
