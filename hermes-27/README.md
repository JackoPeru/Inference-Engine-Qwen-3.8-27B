# HERMES-27 — independent Qwen3.8-27B inference engine

Experimental C++20/CUDA inference runtime for the Qwen3.8-27B Uncensored
NVFP4-MTP GGUF checkpoint, targeting **2× RTX 5060 Ti 16 GiB (SM120)**.

**Current state (2026-11-13):** PP2 64-layer GPU prefill, incremental decode,
greedy CLI, and native HTTP chat serving exist and pass short real-model smoke
tests. This is **not yet a production-ready multiuser replacement** for TabbyAPI.
True fused GPU continuous batching, fast prefill, long-context performance,
full MTP verification/sampling and TP2 remain open. Never substitute the
historical E1/E2 golden files for independent GGUF reference checks.

## Build on the Hermes Ubuntu host

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
cmake --build build -j4 --target hermes_generate hermes_server test_pp2_batch test_mtp_gpu
```

Requires CUDA 12.8+ for SM120, CMake, C++20, and CUTLASS under
`thirdparty/cutlass`. The HTTP target currently also relies on cpp-httplib
and nlohmann/json headers; see `CMakeLists.txt`. Model weights are excluded.

## Run the CLI

```bash
export HERMES_NVFP4_GEMV=warp4  # optional optimized GEMV; omit to use scalar default
./build/hermes_generate \
  --model "$HOME/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf" \
  --prompt-ids 6205,161421,303,220,17,1374,9911,1147,27053,6059,13 \
  --max-new-tokens 12
```

JSONL records go to stdout; diagnostics and the per-request `TIMING` record
go to stderr. Set `HERMES_NVFP4_GEMV=scalar` to force the original GEMV.

## HTTP serving (isolated tests only)

```bash
HERMES_NVFP4_GEMV=warp4 ./build/hermes_server \
  --model "$HOME/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf" \
  --tokenizer /path/to/matching/tokenizer.json \
  --gguf-dump ./build/gguf_dump \
  --host 127.0.0.1 --port 18127 --active 2 --queue 2 --context 128
```

`hermes_server` exposes `/health`, `/metrics`, `/v1/models`, and a limited
OpenAI-style `/v1/chat/completions` endpoint with SSE and cancellation. Run
`python3 serving/smoke.py --base-url http://127.0.0.1:18127` against an
**isolated** test instance. Do not bind over TabbyAPI or disrupt GPU Manager.
The current default context limit is 128 tokens, configurable up to 8192;
prompt processing is still token-by-token and long contexts are slow.

## Correctness and benchmarks

- `tests/gate_math_gguf.py`: inexpensive semantic/layout preflight against
  the actual checkpoint (`ssm_a` convention and GDN head ordering).
- `tests/compare_llama.py`: independent greedy comparison with llama.cpp on
  the exact same checkpoint/token IDs. Requires a separately running
  llama-server using the model.
- `scripts/check_generate.py`: PP2 greedy/EOS/state-isolation smoke.
- `benchmarks/WARP4_AB_2026-11-13.md`: measured real-model scalar vs Warp4
  performance with commands and limitations.
- `benchmarks/HEAD_AB_2026-11-13.md`: optional BF16 output head, numerical
  and complete-engine A/B timings.
- `benchmarks/PROFILING_GUIDE.md`: GPU profiling procedure.
- `kernels/tests/README_nvfp4_prefill_prototype.md`: **research only**,
  CUTLASS SM120 W4A4 batched prefill microbench on the real GGUF. Device
  activation quantization and weight repacking work, but the measured ~4.5%
  relative projection error has **not** passed a full-model logit check.
  Neither the benchmark nor the prefill prototype is wired into PP2.
- `tests/bench_ab_engine.py`: repeatable full-engine token-equality and
  timing benchmark across the four scalar/Warp4/head/async combinations;
  use `--self-test` without GPUs or run with `--engine`/`--gguf` during an
  exclusive GPU window.

Three strictly optional CUDA switches can be combined:

- `HERMES_NVFP4_GEMV=warp4` selects the optimized raw-GGUF GEMV.
- `HERMES_HEAD_BF16=warp4` selects an FP32 warp-per-vocabulary-row output
  projection, replacing the original FP64 scalar dot product.
- `HERMES_DECODE_SKIP_LAYER_SYNC=1` omits redundant per-layer device-wide
  synchronization in the decode path only (other GPU/host boundaries stay).

All three are disabled by default. Inference remains experimental: a full
matrix of long-context, concurrency, streaming, and sampling tests is not
complete. CPU reference, CUDA kernel, HTTP and benchmark results must identify
their exact checkpoint and source revision.
