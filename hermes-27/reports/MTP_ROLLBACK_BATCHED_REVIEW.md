# HERMES-27 MTP — batched verification and rollback audit (CPU/OFFLINE)

**Date:** 2026-11-14. **Decision:** **BATCHED SPECULATIVE MTP K>1 BLOCKED / NOT IMPLEMENTED.** The existing GPU MTP draft/verification path is a **serial, target-authoritative greedy K=1 probe**, not parallel target verification. This assignment made only bounded, safe **draft-side** metadata/failure/concurrency corrections and verified their host policies with CPU/mock tests. No GPU access or live inference was used; GPUs remain exclusive to prime. No target state, serving, scheduler, CMake or production route was changed.

## 1. Exact source-level evidence — no real target batch commit

| Component and file | Source evidence | Correct scope of claim |
|---|---|---|
| `tools/hermes_mtp_generate.cpp:269-299` (READ-ONLY) | Each iteration calls `mtp_gpu::draft_step(...)`, then **one** `pp2::decode_step(current, target, pos)`, compares draft and target argmax, emits **only** the target token | One PP2 target step per generated token. K=1 can never derive a target-step throughput reduction. No speculative target rollback is exercised. |
| `distributed/test_mtp_gpu.cu:141-174` | The previous GPU test loops `draft_step` then `decode_step`; `mtp_gpu::rewind` only precedes a replay of **draft** logits; the test compares two independently advanced **serial** target states | Successful past real-GPU A/B confirmed greedy serial correctness and draft KV replay, **not** K>1 target state rollback. This review updated only misleading annotations and report labels. No new GPU run. |
| `distributed/pp2_fwd.hpp:50-77` (READ-ONLY) | A `pp2::DecodeState` holds per-layer `S[64]` recurrent GDN, `CS[64]` convolution tails, `Kc[64]`, `Vc[64]`, `klen[64]`; persistent work buffers, `ever_advanced`, opaque `async_ctx` | Restoring only `klen`, or only draft KV length, cannot restore full target state or undo in-place recurrent updates. Scope, ownership and async completion also require fencing. |
| `distributed/pp2_fwd.hpp:80-102` (READ-ONLY) | `prefill_into_decode_state` seeds a **fresh** target state from a bounded prefix, and `prefill_chunks_into_decode_state` also demands an **empty** state and may invalidate on failure | These are **not** continuation-based batched verification from a live decoder's committed prefix, nor a selective accepted-prefix transaction. |
| `distributed/pp2_fwd.hpp:104-112` (READ-ONLY) | `decode_init`, `decode_step` and `decode_free`: `decode_step` modifies GDN/CS/KV in place | No public `clone_state`, `begin_tentative_batch`, `verify_K`, `commit_accepted_prefix`, or `restore_all_layers` API. |
| `distributed/mtp_gpu.cu:296-305,344-369` | Draft appends GPU1 K/V for the single MTP NextN layer and increments **draft** metadata after synchronization. Rewind only changes draft visible length | These buffers are not PP2 target per-layer states. Existing MTP cannot restore target S/CS/K/V or target RoPE/position after speculative target steps. |
| `tools/hermes_generate.cpp` (READ-ONLY) | Production CLI explicitly reports MTP off and does not route speculative validation into serving | No default MTP route changed. |

**Threshold and EOS contract:** In greedy mode, acceptance is equality of the draft token with **the target argmax at the same position**, not top5/confidence; target token must always win on rejection. EOS must stop emission and **prevent any further speculative target mutation**, and cancellation/deadline/obsolete session must not commit stale results. Greedy token equality is not a valid acceptance algorithm for arbitrary temperature/top-p sampling (distribution correction would be required). MTP source has no end-to-end EOS/target rollback API: EOS policy currently belongs to the external serial caller. The CPU mock tests EOS and cancellation policies, **not production implementation**.

The architectural blocker is independent of speed: the public API cannot expose a consistent temporary target branch and atomically commit a verified prefix over all **64** transformer layers. Implementing that interface is PP2-owner work outside this assignment, not a safe change to `mtp_gpu.cu`.

## 2. Concrete owned-file correctness defects fixed

1. **Logical rewind-to-zero lost absolute RoPE origin.** Before: `mtp_gpu::rewind(st,0)` set `first_pos=-1`, allowing a later `draft_step` at any nonnegative absolute position despite not calling reset (a replay position jump). After: `detail::rewind_metadata` preserves `first_pos` for all valid `keep_len`, **including zero**; subsequent draft append must repeat the **original first absolute position**, while explicit `reset()` is the sole start-new-prompt operation. Positions are bounded to `[0,MAX_CONTEXT-1]`, never silently wrapped into RoPE int. **This is draft history only**, not a PP2 position rollback.
2. **Partial CUDA failure previously allowed replay of a potentially corrupted draft KV slot.** `draft_step` can update Kc and fail on Vc copy, a later kernel or D2H. The old `len` remained unchanged, permitting `reset`/`rewind` and retry without proving K/V consistency. After: any failure through the CUDA cleanup path sets a sticky `poisoned=true`. `init` on that still-initialized state, `draft_step`, `rewind` and append guards refuse reuse; legacy `reset()` **does not clear poison**. Recovery requires `free_state` and a new `init`, with fresh buffers. Invalid early host argument rejection does not mutate draft GPU state.
3. **Owned GPU pointers were copyable.** `MtpState` previously permitted a shallow copy of GPU allocations, risking aliasing/double free across requests. It is now explicitly **noncopyable and nonmovable** with a host `std::mutex`. All public draft state lifecycle and host mutation operations (`init`, `reset`, `rewind`, `free_state`, `draft_step`) hold the per-state lock. Distinct requests may have distinct independent states, but **no concurrency speedup is claimed**. Callers must keep each object alive through every call and must not directly mutate public metadata outside the lock.
4. **No explicit fail-closed target batch capability flag.** `mtp_gpu::target_batch_rollback_supported()` is now a constexpr **false**. It cannot enable production; callers can use it to refuse K>1 until PP2's actual target transaction API exists. `distributed/test_mtp_gpu.cu` now reports `draft_KV_rewind_gpu=PASS; target_batch_rollback=UNSUPPORTED; TPS claim=NONE`, rather than the ambiguous `rollback_gpu=PASS`. Its generation algorithm was not changed.

**Important compatibility:** Adding a mutex and poison bit changes the **ABI/size/layout of `mtp_gpu::MtpState`**. Any caller/library built with the old header must be **recompiled** with the new header before executing even the opt-in K=1 path. This audit only compiled the new `.cu` and the existing standalone CLI translation unit, without linking/executing live inference. Prime must coordinate any build/release; do not mix old CLI objects with new MTP library objects.

**Residual limits:** a per-state mutex protects only operations on the **same live `MtpState` object**. It does not create session/epoch fences for upstream requests, guarantee shared-device CUDA-stream scheduling, make direct external writes to public fields safe, or add PP2 target rollback. `free_state` has a legacy `void` return and does not expose CUDA deallocation errors; recovery after a persistent CUDA device error may require process/device-level handling. All GPU changes are **COMPILE-ONLY** in this task, not empirically validated on hardware.

## 3. Test-first CPU-only evidence

**New isolated test:** `distributed/tests/test_mtp_rollback_contract.cpp` (links no CUDA/PP2/MTP production library). It includes the **actual owned** `mtp_gpu.hpp` and read-only `pp2_fwd.hpp`. CPU policy tests call the **same inline bookkeeping functions** `detail::can_append`, `detail::commit_append`, `detail::rewind_metadata`, `detail::reset_metadata` that the GPU implementation now calls under its per-state lock.

**RED before implementation:** standalone `g++ -std=c++20` failed with nonexistent guarded-bookkeeping functions, no poison state, and two positive copyability traits where noncopyability was required. After the owned implementation, it compiles and runs.

**GREEN:** `2,377` deterministic assertions with `PASS CPU_ONLY_MTP_MOCK_ROLLBACK checks=2377 target_batch=BLOCKED no_gpu=true`, repeated with AddressSanitizer+UndefinedBehaviorSanitizer, exit 0. No GPU use, no actual PP2 target calls.

Test matrix:

| Mock/host check | What passed | What it does **not** prove |
|---|---|---|
| True production draft metadata | Append must be contiguous; `rewind(0)` preserves origin; only reset starts a new origin; no negative/out-of-context positions; capacity, malformed `first_pos`, invalid rewind and sticky poison rejected | Kernel K/V contents after hardware failure (no GPU run) |
| Ownership/concurrency | `MtpState` copy/move blocked at compile time; guarded host metadata increments from 8 threads × 32 steps produce no lost updates; independent states do not share history or poison | Live CUDA stream ordering, races in external client or upstream PP2 state |
| 64-layer target **mock** | Simulates 48 layers with GDN digest S + depthwise-conv three-value tails, and 16 full-attention layers with actual token-valued K and V vectors; keeps **per-layer absolute position**, request scope, terminal status and output prefix | **Not** the actual GDN matrices/PP2 kernel state |
| Full transaction policy **mock** | Correctly restores full target state on 0..4 accepted tokens, correction after first/middle/last rejection, all accepted, terminal/EOS, cancelled request, session-scope mismatch and stale snapshot | No real PP2 snapshot/restore exists |
| Negative rollback | Taking a target mock after 3 tentative mutations and truncating **only** visible K/V lengths leaves GDN S, convolution histories, KV **values** and position wrong vs authoritative serial teacher replay | No claim about unknown hardware kernel behavior |
| Deterministic bounded matrix | 256 pseudorandom cases (K=1..4), plus 8 separately interleaved requests, each compared to a fresh serial target-only mock state over all layers | No measured speculative accuracy or speed |

The CPU mock deliberately defines the **required** transaction policy, rather than pretending to call a PP2 batch interface. Only the draft metadata tests exercise shared helpers from the GPU path; production PP2 target rollback remains unimplemented.

### Reproduction (CPU/offline scratch only)

```bash
# Existing worktree content was copied to scratch; no new persistent CMake.
cd /tmp/hermes_w5_mtp_rollback_offline
g++ -std=c++20 -O2 -Wall -Wextra -Werror -pedantic -pthread \
    tests/test_mtp_rollback_contract.cpp -o test_mtp_rollback_contract
./test_mtp_rollback_contract
# PASS CPU_ONLY_MTP_MOCK_ROLLBACK checks=2377 target_batch=BLOCKED no_gpu=true

g++ -std=c++20 -O1 -g -Wall -Wextra -Werror -pedantic -pthread \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    tests/test_mtp_rollback_contract.cpp -o test_mtp_rollback_san
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
    ./test_mtp_rollback_san
# PASS CPU_ONLY_MTP_MOCK_ROLLBACK checks=2377 target_batch=BLOCKED no_gpu=true

# CPU-only compilation of CUDA source (NO kernel launch or GPU admission):
cd /tmp/hermes_w5_mtp_20261113
/usr/local/cuda/bin/nvcc -O2 -std=c++20 -arch=sm_120 -Xcompiler=-Wall \
    -Idistributed -c distributed/mtp_gpu.cu \
    -o /tmp/hermes_w5_mtp_rollback_offline/mtp_gpu.o
/usr/local/cuda/bin/nvcc -O2 -std=c++20 -arch=sm_120 -Xcompiler=-Wall \
    -Idistributed -c distributed/test_mtp_gpu.cu \
    -o /tmp/hermes_w5_mtp_rollback_offline/test_mtp_gpu.o
# Both passed.
```

The separately owned, existing `tools/hermes_mtp_generate.cpp` was **not edited**; compiling it as a CPU-only host object with the updated header and `g++ -std=c++20 -Wall -Wextra -Werror` also passed. The existing real-GPU K=1 A/B results from November 13 are **historical baseline only**, not re-run evidence after the new guard/mutex changes.

## 4. Changed-file/source manifest (for prime integration)

The current local files were independently hashed after compilation. The only authorized changes are:

| Relative path | Type | SHA256 |
|---|---|---|
| `distributed/mtp_gpu.hpp` | modified — pure C++ guards/noncopyable state/lock/blocked target-batch flag | `03741b69c810769a330dac7b9fb9523da04d3d04467f163268b947afb86cf2e3` |
| `distributed/mtp_gpu.cu` | modified — lock, guarded append/rewind/reset, CUDA poison fail-stop | `5abaf14ccc3f6aee832b444319b9a82594a3cde99be52fc88f5d2caf8f206c17` |
| `distributed/test_mtp_gpu.cu` | modified — only truthful draft-only rollback comments/label | `e25d27bac1845dd83bf95d074edf8da527eee417826cd22cb7343ff35e457b56` |
| `distributed/tests/test_mtp_rollback_contract.cpp` | **NEW** CPU test and bounded mock | `663a11b6226ddf1ed2fbb0124275941ddb0c20c1223cdae70a8bda2d9114427b` |
| `reports/MTP_ROLLBACK_BATCHED_REVIEW.md` | **NEW** this evidence/manifest | self-hash intentionally omitted (report created last) |

Scratch binaries: `test_mtp_rollback_contract` SHA256 `ceddb1cdae829fec847fcba37a6000cf5f7128f2c6b717751f064d51b1a15bf1`; sanitized `test_mtp_rollback_san` SHA256 `2b6116edc1414a8c6d06c4e5dcceb57181130921a27a19c7aeed09127420a07d`; nvcc MTP object `93228c76c6f3ec3b4db46b4b67255cdad677c53a7c523bf11067bc9fcd7fcef0`; nvcc test object `8ff2d09a0ef6788a0d27e7c8e27bb7a2ada9afa8e6fab924191832e58cd8c4e6`.

All source changes reside in the shared workspace, **not automatically committed or published**. The Windows project folder is not a Git checkout (`git status` reports no repository), so there was no Git stage/commit. Prime must integrate source files into its tracked worktree, ensure ABI-consistent recompilation, and rerun any real GPU gates under a new explicit exclusive grant.

## 5. Admission requirements to unblock true batched MTP

For safe opt-in K>1, the PP2 owner must first expose a target transaction interface that (1) snapshots/restores **every** per-request per-layer GDN S, CS tails, full-attention K/V data **and lengths**, position/absolute-RoPE cursors, workspace and async completion boundaries; (2) verifies K tentative tokens from the live **committed** prefix without contaminating it; (3) commits exactly the accepted prefix plus the target correction/EOS atomically with session/epoch fencing; and (4) fails closed on cancel, timeout, device error and partial batch, with no cross-request state reuse. An engine capable of simply serially replaying each accepted target token remains **correct** but **not accelerated**.

These required cross-owner APIs must be implemented/tested by prime/PP2 worker, **not** by this MTP-only worker. Only after real checkpoint multi-step output equality, state-by-state rollback parity including forced first/middle/last rejection and EOS across concurrent requests, and measured end-to-end tokens/s should a throughput claim or production route be considered.

**Final gate: target batched speculative decoding remains disabled and blocked; draft metadata CPU regression PASS.**
