# PP2 admission: FP32 KV state (2026-11-13)

`pp2batch::admit_check(N, Tctx, detail)` now rejects `Tctx < 1`, `Tctx > DecodeState::MAX_CONTEXT` (currently **8192**), and `N` outside 1..16. These are input/policy limits, **not experimentally validated throughput limits**.

The old `distributed/P5_BATCH.md` and `reports/VRAM_BUDGET_V2.md` assumed FP8 KV at 16 KiB/token/GPU and mentioned 32K contexts. Those are **historical projections**; they do not describe the current `DecodeState` implementation. In `pp2_fwd.cu`, each GPU holds 8 full-attention layers, with two FP32 caches (`Kc` and `Vc`) per layer, each storing `1024` floats per token:

```
KV per token per GPU = 8 layers * 2 (K,V) * 1024 * 4 bytes = 65,536 bytes (64 KiB)
GDN+conv per GPU    = 24 layers * (48*128*128 + 10240*3) * 4 = 78,446,592 bytes
Workspace            = 568,064 bytes/GPU
GPU0 step buffers    = 4 + 5120*4 bytes
GPU1 step buffers    = (5120*2 + 248320)*4 bytes
```

The admission calculation uses these actual state-buffer sizes, then adds **all N states** to a conservative static PP2 baseline. P3 reported **9856/11470 MiB** steady on GPU0/GPU1. The operational cap is 15360 MiB/GPU; a further 1024 MiB/GPU is reserved for runtime variance. This leaves a projected allowance of **4480 MiB on GPU0 and 2866 MiB on GPU1** for request states. The baseline already contains one warm decoder state, so counting N full states again is deliberately conservative.

The function is a static preflight estimate. It does not measure free GPU memory or predict latency/throughput. Production must continue checking `cudaMemGetInfo` and `decode_init` before accepting each request; mixed-context traffic may be rejected conservatively because `admit_check` estimates N requests at the new request's context size.

## CPU-only regression test

Compile the same implementation and test sources in an **isolated scratch directory** with a C++ compiler, without `nvcc` or initializing CUDA:

```bash
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -O2 \
  -DHERMES_ADMISSION_CPU_ONLY -x c++ \
  pp2_batch.cu test_pp2_batch.cu -o admission_cpu
./admission_cpu
```

Include `pp2_batch.hpp` and `pp2_fwd.hpp` from the same source revision in the scratch directory. The regression checks invalid request counts, zero and out-of-range capacities, and budget boundaries (1/4/5 requests at 8192, 4 at 1024). Expected failure of 5@8192 comes from projected **2941.7 MiB** GPU1 state versus **2866 MiB** allowance. This is a **policy test**, not a GPU measurement.

`test_pp2_batch admission` runs the same CPU-only assertions in a normal CUDA-linked build, before any model upload; modes `g1`, `g2` and `all` retain their existing CUDA benchmarking behavior. Only execute those CUDA workload modes after GPU scheduling coordination.
