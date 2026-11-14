# HERMES-27 — stateful prefill / NVFP4 exact-kernel CPU audit (2026-11-14)

**Outcome: CPU negative-oracle PASS; CUDA kernel UNCHANGED; physical numeric and throughput gates still NO-GO.** This is a source and CPU audit of the current Windows mirror, not a GPU run or a claim of GGUF/model parity. User's optimization order remains multiuser aggregate throughput, prompt prefill, then individual decode.

## Source provenance and edit scope

The Windows `/saas/Inference Engine Qwen 3.8 27B/hermes-27` mirror has no `.git`; the prime's reported published GitHub HEAD is `6da5f5334b17e7fc839296b8c213220d7d79e480`. The kernel SHA matches the separate prior tracked-SHA inventory under the parent directory; the current PP2/server SHA values differ from that inventory, so no Git commit correspondence is inferred for their current contents. Other agents may own ongoing PP2/server changes.

| Path | Current SHA256 | Access |
| --- | --- | --- |
| `kernels/src/nvfp4_prefill_exact.cu` | `9842b9b7a776e7d6be8a3a43e561dfc717d7752ad78e46a71dc3c9813a633a51` | **unchanged** |
| `kernels/include/hermes/nvfp4_prefill_exact.hpp` | `c3cf9110b5f6e7780a9c344aaa56849835a502dfd143e95ab88ecf32c0a3d8fe` | read-only |
| `distributed/pp2_fwd.cu` | `0b8b8bcbcb6d60ee1c49fb240d7e925772fafc96def5c03151fe02b0d50b93ed` | read-only |
| `serving/server.cpp` | `ad2522dc7297bb9e3826a617778ec580e591d917e6bde3796adc3146e596a7f4` | read-only |
| `distributed/test_prefill_chunk_cpu.py` | `d07d0575e63ca799beee4f5f26771b627c89cd3a9a4e616b568087335defb7ac` | read-only |
| **NEW** `kernels/tests/test_prefill_exact_state_invariants_cpu.py` | `0eed4c989e7982c703bcc8ab3349c15df774994f8e2b34fcb8e1ca4cb0eb9830` | owned |

No kernel edit was justified by the available CPU evidence. The exact NVFP4 BF16 decode/GEMM path itself does not implement prompt state. The only new files are this report and the CPU test above. No changes to Git, model, services, GPU, CMake, PP2 or HTTP serving.

## Source-anchored invariants across first and later chunks

- **First chunk, `pos0=0`:** `pp2_fwd.cu:525-527` seeds `DecodeState::CS[li]` from the final three *pre-convolution* QKV rows. The zero-left-padding cases T1/T2 are handled by `seed_conv_tail_k` at lines 458-464. `d_S=seed->S[li]` reuses the request-owned GDN recurrence storage (lines 509-514).
- **Later chunks, `pos0>0`:** `conv_silu_chunk_k` reads the three persisted QKV samples; the subsequent `append_conv_tail_k` advances their tail, including short chunks T1/T2 (lines 109-120 and 466-483, invoked at 532-535). `gdn_step_cuda` advances S in token order for all T positions (lines 549-552). Neither object is owned or changed by `nvfp4_prefill_exact.cu`.
- **KV and RoPE:** the full-attention branch positions normalized Q/K with absolute `pos0`, appends K/V at `pos0` (lines 615-631), attends to prior+current cached positions for later chunks (653-661), and updates `klen=pos0+T` only after a successful layer completion (674-680). `prefill_chunks_into_decode_state` rechecks 16 KV lengths after every chunk and destroys a failed request state (1976-2009).
- **FFN per position:** gate, up, activation and down projections use the complete T dimension on every GDN and full-attention layer (GDN 567-575, full 666-673). The `last_only` optimization applies **after all layers** to output normalization/head (903-925). Skipping FFN for earlier positions would corrupt later layer inputs, even if only final-token logits are returned.
- **T boundary:** one-shot RAW seeded T1..64 returns `T*248320` logits; legacy chunked RAW handles T65..8192 in bounded T<=64 chunks, at offsets 0,64,128 and with one vocabulary row returned **per chunk**. Therefore T64=one call, T65=64+1, T129=64+64+1. Serving gates the latter to legacy `active=1`; chunked BF16 is explicitly rejected (`pp2_fwd.cu:1919-1922`). No N>1 multiuser chunked throughput is established.
- **CUDA ordering within the experimental BF16 slot:** `ReusableBF16WeightScratch::enqueue` uses a private per-slot mutex/handle. A later stream waits on the previously recorded `ready_event_` before the vector GGUF weight dequantization and GEMM; it records a new event after the last tile (kernel lines 312-371). Intermediate tiles reuse scratch in the same stream. The cuBLAS output leading dimension remains full **N**, with destination pointer `y_fp32+base`; using a tile-sized leading dimension would corrupt Y[T,N]. Error paths poison the slot and teardown drains the event or, if its generation is unknown, synchronizes the device (374-411). These are source invariants, **not GPU runtime witnesses**.

## Test evidence (CPU only)

Command (Windows Python 3.13, repository root):

```
python kernels/tests/test_prefill_exact_state_invariants_cpu.py
```

Actual stdout, exit 0:

```
PREFILL_STATE_INVARIANTS_CPU_PASS state=37 tiled=18 source=18 GPU_GGUF_PARITY=NOT_TESTED
```

The new test uses an independent absolute-position convolution, closed-form recurrent GDN sum, causal KV softmax and per-position nonlinear FFN oracle; it compares stream/chunk execution at T1/2/3/63/64/65/128/129 with chunk sizes 1/3/7/64. All **32** valid combinations preserve output, GDN state, 3-token convolution tail and KV length. It also verifies that five deliberately broken variants **are detected**: reset convolution history, reset GDN state, reset KV, reset absolute position, and skip earlier FFN positions. An independent toy BF16-compatible matrix shape (K64, N129) checks T64/65/129 row tiling for rows-per-tile 1/7/128 and detects a deliberately wrong output leading dimension. 18 source guards assert critical PP2/slot contracts and stream/event ordering. Python syntax verification passed; its generated bytecode file was removed.

The **pre-existing**, independently run `distributed/test_prefill_chunk_cpu.py` also exited 0:

```
CHUNK_CPU_PASS conv=91 attention_rope=64 kv_append=42 static=20 GPU_GGUF_PARITY=NOT_TESTED
```

These toy arithmetic witnesses cannot establish actual GGUF CUDA numerics, FFN kernel bit accuracy, CUDA event execution, layer-to-layer parity, next-token continuation or production readiness. The source guards may need an intentional update when another owner changes the PP2 contract.

## Cost and numerical constraints

`pp2_fwd.cu:485-680` has 48 GDN layers with **22** per-layer temporary `cudaMalloc` allocations each, plus 16 full-attention layers with **18** allocations each in the seeded path. Source count: `48*22 + 16*18 = 1,344` mallocs and corresponding frees **per chunk**, excluding frame/head staging and persistent decode state. The same 64 layers each include `cudaDeviceSynchronize` (at least 64 explicit layer barriers per chunk). Thus T64 entails >=1,344 layer allocations/64 barriers, T65 >=2,688/128, and T129 >=4,032/192. These are counts from source structure, not profiling or CUDA timing; error paths can terminate early.

For a T64 chunk, the source's allocation formulas total `64*568,064 = 36,356,096 B` (~34.67 MiB) of GDN per-layer temporaries, or `64*477,184 = 30,539,776 B` (~29.12 MiB) of full-layer temporaries; successive layers release them, so the figures must **not** be summed into a peak. Every full-attention layer also transfers a T*12,288 FP32 Q raw buffer to host and a T*6,144 FP32 gate buffer back to device; at T64 these D2H/H2D copies sum to ~72 MiB across the 16 full layers, plus host CPU gate extraction and mandatory synchronization. The T64 full-vocabulary output is `64*248320*4 = 63,569,920 B`; the T65+ chunk API returns only `248320*4 = 993,280 B` at each chunk endpoint. Measured bottleneck attribution is still absent.

The experimental BF16 route uses a separate 8 MiB weight scratch per GPU **per seeded request**, tiled by output row. This caps duplicate weight memory but can require numerous unpack+GEMM launches and serializes sharing of a slot across streams. It also runs a complete original RAW reference pass for every BF16 quality check (source 1806-1882), so it is not yet a demonstrated throughput acceleration. In existing real-GGUF reports (`distributed/PREFILL_RUNTIME_INTEGRATION.md:96-115`), T2/T11/T64 BF16 full-logit gates **failed**, with T64 max_abs 5.14553571 and one argmax change; the strict 1e-3 gate remains in place. The same report records earlier RAW seeded state/continuation parity at T2/11/64; it is historical evidence, not a rerun here.

## Admission decision and next gate

**NO-GO** for promoting the BF16 exact-GEMM route, stateful chunked prefill to concurrent serving or a claimed GPU prefill speedup. No new kernel defect was demonstrated by the available independent CPU witnesses; therefore the CUDA kernel was left unchanged. A strict real-GGUF validation when GPU ownership is explicitly granted must compare every full-model logit and per-layer GDN/conv/KV state against an independent reference at T64/65/129, then continuation and concurrent N1/N2/N4/N8 stress under the same workload. CUDA 12.8 compilation was unnecessary this turn because the kernel/source interface did not change; no physical GPU or service was accessed.
