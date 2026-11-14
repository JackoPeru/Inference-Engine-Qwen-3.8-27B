# HERMES-27 MTP K>1 target atomicity — CPU/offline follow-up

Date: 2026-11-14. Scope: `distributed/mtp_gpu.cu`, `distributed/mtp_gpu.hpp`, new `distributed/tests/test_mtp_batched_rollback_*`, this report. **NO-GO for K>1.** The Windows `/saas/Inference Engine Qwen 3.8 27B/hermes-27` folder has no `.git`; the Ubuntu production `main` HEAD `efad3f42` supplied by the prime was not reconfirmed here because SSH/network and repository writes are outside this worker's grant. No GPU inference, production deployment, Git changes, or source edits to PP2/serving/batch/CMake were performed.

## Source-anchored result

1. `mtp_gpu.hpp:43-46` explicitly exports `target_batch_rollback_supported() == false`. The MTP API (`:88-114`) accepts **one** draft input at a time, tracks its own GPU1 K/V history, and has no target verification, EOS, target snapshot, commit or multi-step transaction interface. `detail::can_append/commit_append/rewind_metadata` (`:50-85`) enforce consecutive bounded positions and sticky failure poison. Those guards do not rollback target GDN/conv/KV.
2. `mtp_gpu.cu:202-362` computes a BF16 draft one step at a time using host target hidden and the previous token; it writes one draft K/V row (`:307-310`) and commits only **draft** metadata after GPU synchronization and logits download (`:354-361`). Any failure through its checked CUDA path jumps to a sticky poisoned state (`:363-381`). `init/reset/rewind/free_state` hold the state mutex (`:155-198`). These are reasonable **draft-only** protections, not a target accepted-prefix transaction.
3. Actual `pp2::DecodeState` (`pp2_fwd.hpp:46-86`) owns 64-layer `S`, `CS`, `Kc`, `Vc`, `klen`, workspace, position-dependent decode behavior, and optional asynchronous streams/sliding KV. `decode_step()` updates these in place (`pp2_fwd.hpp:124-126`). Its public API (`:120-128`) has no snapshot/restore, tentative target branch, compare-and-commit, or accepted-prefix fence. Rewinding only K/V *lengths* would preserve speculative mutations to GDN recurrent matrices and conv tails. Source inspection confirms there is no legitimate way to make target K>1 atomic solely by modifying the MTP draft files.
4. Actual integration `tools/hermes_mtp_generate.cpp:269-300` emits the target greedy token and, after a draft proposal, calls authoritative `pp2::decode_step(current, target, pos, ...)`; it compares candidates for diagnostics and never stages rejected tokens in the target. `:151-160` rejects non-finite target logits. `:272-274` stops at target EOS. This is strict **serial K=1 observation**; it cannot validate K>1 target throughput or rollback. The old `distributed/MTP_GPU.md` sweep used fixed `h_last` and full-prefill verification; it is historical and does not supersede this gate.
5. CUDA ordering: `draft_step` stages an embedding through a **synchronous** GPU0 device-to-host copy (`mtp_gpu.cu:250-254`), then performs GPU1 allocations, default-stream kernel/copies and a device-wide `cudaDeviceSynchronize()` (`:257-357`) before returning. The per-state mutex protects calls sharing one live draft state; no per-request explicit CUDA stream is supplied to MTP. Its device-wide sync can serialize other GPU1 workloads, and the MTP mutex does **not** fence PP2's shared uploaded-weight lifetime or other PP2 states/streams. There is no GPU run here proving concurrent stream safety or throughput, and no source-only change can establish such a proof.

## Independent CPU state-transition test

**Added** `distributed/tests/test_mtp_batched_rollback_atomicity_cpu.cpp` (standalone C++20, no CUDA/PP2 linked). The file includes the actual MTP and PP2 public headers and compile-time rejects an accidental change of the target-batch capability flag to true; it also checks all required PP2 layer state arrays exist. It models 48 recurrent GDN+conv layers, 16 full-attention K/V layers with actual token/position-dependent values, absolute positions, a request ID, an epoch, and terminal state. This is an **independent toy policy oracle**, not the Qwen numerical oracle or the implementation of a production target transaction.

Unlike the earlier `test_mtp_rollback_contract.cpp` toy `Transaction::resolve()`, which eagerly appends the correction token, the new serial reference preserves the crucial actual `decode_step` contract: the last **emitted** correction/bonus token is **not yet processed** by target recurrent/KV state. A verified EOS terminates without processing EOS or later speculative inputs. The model checks the difference on first, middle and last rejection, all-accepted K=1..4, EOS at candidate/bonus positions, fake draft EOS, cancellation, deadline, device error, stale snapshots, request/epoch changes and eight interleaved requests. It includes intentional bad-policy witnesses for KV-length-only rollback, retained rejected K/V values, residual GDN/conv, eager correction, unverified EOS and one-token logit-position drift.

Build/run on actual Windows workspace, Visual Studio 2022 MSVC compiler:

```cmd
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /std:c++20 /EHsc /W4 /WX /Zc:__cplusplus /Fo:"%TEMP%\mtp_batched_atomicity.obj" /Fe:"%TEMP%\mtp_batched_atomicity.exe" distributed\tests\test_mtp_batched_rollback_atomicity_cpu.cpp
"%TEMP%\mtp_batched_atomicity.exe"
```

Result, **exit 0**: `PASS MTP_BATCHED_ATOMICITY_CPU_ONLY assertions=22761 target_K_gt_1=BLOCKED GPU=NOT_RUN throughput=NOT_MEASURED`.

AddressSanitizer compile and run, also **exit 0**, same 22,761 checks:

```cmd
cl /nologo /std:c++20 /EHsc /W4 /WX /Od /Zi /fsanitize=address /Zc:__cplusplus /Fo:"%TEMP%\mtp_batched_atomicity_asan.obj" /Fd:"%TEMP%\mtp_batched_atomicity_asan.pdb" /Fe:"%TEMP%\mtp_batched_atomicity_asan.exe" distributed\tests\test_mtp_batched_rollback_atomicity_cpu.cpp
"%TEMP%\mtp_batched_atomicity_asan.exe"
```

The local Windows CUDA compiler `nvcc` and CUDA toolkit were unavailable, so no CUDA 12.8 compile was attempted. MTP `.cu`/`.hpp` deliberately unchanged: there is no bounded source-only correction that can honestly turn PP2's absent target transaction into a functioning atomic K>1 verifier. All tests here are CPU model checks and compile-time public API guards; **zero** performance or real GPU correctness claims.

## Freeze manifest

| File | Status | SHA-256 |
|---|---|---|
| `distributed/mtp_gpu.cu` | existing, unchanged | `5abaf14ccc3f6aee832b444319b9a82594a3cde99be52fc88f5d2caf8f206c17` |
| `distributed/mtp_gpu.hpp` | existing, unchanged | `03741b69c810769a330dac7b9fb9523da04d3d04467f163268b947afb86cf2e3` |
| `distributed/pp2_fwd.hpp` | read-only dependency | `97649e4027ca8db47bed7f0323101edffbe3ddf68ea547f2534e9501d7696b2a` |
| `tools/hermes_mtp_generate.cpp` | read-only caller | `6d03814bf4ab0a50f520c03e30da51401be98b79ce7a12ab8e7eff2471bf3f4a` |
| `distributed/tests/test_mtp_batched_rollback_atomicity_cpu.cpp` | **new** | `462bbc81faf18976ad64b891d26049e9cb0860e8c6be5dbf58e11c92acc5d1db` |

## Required production gates before enabling any K>1

PP2's owner must first add an explicit per-request, generation/epoch-fenced tentative target state and atomic resolution, including all 48 GDN S matrices, convolution tails, full-attention K/V content and lengths, absolute position, output/pending token, CUDA streams and in-flight events. Forced verification errors, EOS at every position, cancellation and stale completion must leave the live target prefix unchanged. Differential testing against independent target-only greedy decode must check full logits, emitted tokens and every committed state after accepting 0..K proposals; only then can end-to-end aggregate throughput be measured. Until those interfaces and gates exist, retain `target_batch_rollback_supported() == false` and do not route K>1 to serving.
