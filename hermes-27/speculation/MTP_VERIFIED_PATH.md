# HERMES-27 MTP: strictly verified iterative greedy path

Status (2026-11-13): **real-checkpoint GPU K=1 iterative correctness PASS on two prompts, independent target-only A/B PASS, GPU rewind/replay PASS. No speedup: the guarded K=1 CLI is slower than target-only in its measured run.** This implementation is a correctness gate for MTP/NextN `blk.64`, not a batched speculative target verifier.

## Findings from the original code

1. `test_mtp_gpu.cu`'s previous `k3` and `sweep` used one frozen `h_last` for K=3/4 draft positions, even though MTP's input contract combines the **target** hidden of the previous token with the currently processed token embedding. This is not a validated autoregressive draft stream.
2. The old sweep verified with `pp2::prefill(P + candidates)` (recomputes the complete prefix) and never updated/rolled back the target's persistent GDN/conv/KV state. Counting `(accepted + 1) / (draft_ms + full-prefill_ms)` as an effective tokens/s result is misleading: it does not represent an executable stateful decoding pipeline.
3. `MtpState::len` tracked cache entries but not their absolute RoPE positions; a caller could skip a position or reuse the wrong cached context. Cache capacity and shared score array were fixed to 128, despite PP2 now supporting 8192 target positions. GPU failures after KV append could leave a changed KV payload; only the host `len` could be considered authoritative.
4. `pp2::decode_step` does not return the draft input hidden through its function result, but **`pp2::DecodeState::x1buf`**, a public GPU1 buffer, holds the target hidden immediately before `output_norm` after the step. The new test compares this GPU buffer to the documented `pp2::prefill(..., &h_last)` output for the same prompt (numeric tolerance, plus greedy top1). An undocumented semantic change to `x1buf` must cause the MTP path to stop.
5. Weights `blk.64.nextn.{enorm,hnorm,eh_proj,shared_head_norm}`, full-attention and FFN `blk.64.*`, `output.weight` and `token_embd.weight` are now strictly checked for expected GGUF dtype (F32=0/BF16=30), dimensions and resident GPU (GPU0 embeddings; GPU1 remaining weights). Missing or incompatible weights fail closed.

## Executable greedy verification policy

For every **committed** input token at absolute position `pos>=1`:

1. Use the **target** hidden captured after `decode_step(previous_token, ...)`, together with the committed `token[pos]`, to advance the MTP draft cache with `draft_step(hidden_previous, token[pos], mtp_state, pos)`.
2. Run authoritative `pp2::decode_step(token[pos], target_state, pos)` and compare `argmax(draft_logits)` to `argmax(target_logits)` for the token at position `pos+1`.
3. Use **only** the target token, whether the MTP proposal matches or not. This preserves bit-for-bit target greedy token *choices* (subject to the target backend determinism). Continue from the target-committed state, not from an unverified candidate. Rejected proposals are measured as rejections, never emitted.
4. Warm the draft KV history by processing prompt positions `1..P-1` with each preceding target hidden. There is no synthetic hidden for position `-1`; the first MTP cache position is `1`. This is explicitly checked as a draft modeling assumption on real prompts.
5. `MtpState` enforces sequential absolute positions and supports logical `rewind(keep_len)` after speculative draft writes. The standalone test rewinds the most recent GPU entry, recomputes it from the same target hidden and verifies all logits plus greedy argmax. Rejected tokens never advance the **target** state and thus do not need a target rollback in this K=1 verification path.

**Acceptance:** for greedy generation, a proposal is accepted if and only if its token ID **equals** the argmax of the target logits for the same position. Top-5 overlap, confidence scores and heuristic thresholds are diagnostic only and must not become acceptance criteria. All returned logits are checked for finiteness. No sampling (`temperature`, `top-k`, `top-p`) is implemented; using greedy acceptance for sampling would bias the target distribution.

## Why K>1 and speedup are blocked

The current PP2 public API supports only `decode_step(tok, state, pos)` (mutates target state for one token) or `prefill(ids)` (recomputes a whole prefix from scratch). There is no API to batch-verify K candidates starting from a committed incremental GDN/conv/KV prefix and then **commit exactly the accepted prefix**, nor any API to clone/rollback all target recurrent state safely. Replaying each accepted token with `decode_step()` is correct but still performs a full target decoding step per token; rescanning `P + K` with `prefill()` does not fix that cost. With the single NextN layer, advancing proposals past the first step also lacks a fresh target hidden for unverified future tokens; using one frozen hidden must be treated as experimental and cannot authorize acceleration claims.

The guarded CLI `tools/hermes_mtp_generate.cpp` therefore deliberately implements **K=1 per-step verified proposals**. It is runnable with existing PP2 public APIs, can replay an independent target-only sequence, and reports accepted/rejected counts and measured times separately. These times describe overhead and are **not speculative acceleration**. Until a stateful batch verification/commit interface is implemented in PP2 by its owner, MTP cannot legitimately claim an increase above target-only decoded tokens/sec.

## Validation and commands

Source paths in exclusive worker ownership:

- `distributed/mtp_gpu.cu` and `.hpp`: guarded draft state, weight validation, dynamic score array, KV rewind.
- `distributed/test_mtp_gpu.cu`: strict prompt warmup, target hidden comparison, independent target-only incremental decoding over 5 new tokens, draft acceptance, real GPU rewind/replay. Legacy `k3/sweep` removed.
- `tools/hermes_mtp_generate.cpp`: token-ID greedy CLI, optional independent target-only replay, rollback verification and prefill-hidden check.

The new CLI/test are intentionally **not** added to the production `CMakeLists.txt` by this worker. In an isolated scratch CMake build, link them against `hermes_mtp`, `hermes_pp2`, `hermes_kernels` and `hermes_gguf` (CUDA device linking is required). Example commands *only after GPU exclusive access is granted*:

```bash
timeout 180 ./test_mtp_gpu_scratch verified
timeout 180 ./test_mtp_gpu_scratch verified-b
timeout 180 ./hermes_mtp_generate_scratch \
  --model /home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf \
  --prompt-ids 6205,161421,303,220,17,1374,9911,1147,27053,6059,13 \
  --max-new-tokens 5 --compare-target-only --verify-rollback --verify-prefill-hidden
```

Exit code 0 requires successful checks. A draft-token disagreement is an ordinary measured **rejection**, not a test failure; a target-stream disagreement, non-finite logits, missing weights, wrong cache positions or failed rollback verification terminates nonzero. Long-context mode uses a variable KV allocation (`capacity` 1..8192; 8 KiB per cached draft token) and dynamic attention score shared memory. It has **not** been validated at long context with real weights.

## Results of real GPU verification (2026-11-13)

SM120 GPU pair: 2× RTX 5060 Ti 16GB, real `Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf`. Isolated scratch folder `/tmp/hermes_w5_mtp_20261113/build`, current project PP2 source copied and linked there. Commands above completed with `timeout 120`, exit 0. No live services restarted or production sources modified by the experiment.

| Test | Prompt length | Verified new target tokens | MTP accepted proposals | Rejected | Full target logits | Draft GPU rewind | Hidden vs prefill |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `test_mtp_gpu_scratch verified` | 11 | 5/5 target-only identical | 3/4 | 1 | max abs 0 | max abs 0 | max abs 0 |
| `test_mtp_gpu_scratch verified-b` | 13 | 5/5 target-only identical | 2/4 | 2 | max abs 0 | max abs 0 | max abs 0 |

The test also compared target full-prefill logits against incremental target logits over the entire vocabulary: max abs 0 on both prompts. MTP prompt-history warmup matched teacher argmax on 6/10 (A11) and 8/12 (B13), respectively. Rejections are visible and the next target-only state remains identical on every following step. The five emitted A11 greedy tokens were `[271,248068,198,760,1156]`.

The standalone `hermes_mtp_generate_scratch --compare-target-only --verify-rollback --verify-prefill-hidden` also exited 0 with `target_only_matched=true`, outputting exactly the A11 IDs above. Timings from that run: model upload 4.36s; prompt seed 5255.048 ms; **verified MTP generation 1957.605 ms vs target-only generation 1787.733 ms** for five emitted tokens. The MTP path took approximately **9.5% longer** in this comparison; no speedup or speculative throughput is asserted. MTP draft time, including prompt warming, was 470.663 ms. Generation timings and seed timings are separate; `MTP_SUMMARY` prints them separately.

Resident model weights: GPU0 8.770 GiB and GPU1 9.561 GiB. With two simultaneous target decode states initialized for the strict test, GPU0 used ~9948 MiB and GPU1 ~11563 MiB. GPU usage returned to normal idle allocations (486 MiB/1289 MiB) after the tests, and GPU exclusivity was released.

Build verification: `nvcc -std=c++20 -O2 -c distributed/mtp_gpu.cu`, `nvcc -std=c++20 -O2 -c distributed/test_mtp_gpu.cu`, and `g++ -std=c++20 -Wall -Wextra -Wconversion -c tools/hermes_mtp_generate.cpp` all passed offline. A complete scratch CMake build including CUDA device linking built both runnable binaries successfully. System prebuilt `libhermes_pp2.a` was initially stale and lacked newer public overloads; the scratch binary used an up-to-date PP2 **source copy**, with no writes to production build/services.

## Release gate

Before any default enablement or speed claim: validate `x1buf` versus prefill hidden, finite draft logits, positional KV replay, and exact generated token IDs vs independent target-only on multiple real prompts including forced draft rejections. Then implement a proper batched target verification + accepted-prefix state commit/rollback interface in PP2 and measure accepted tokens per target call and end-to-end tokens/sec on the actual checkpoint. Preserve the current target-only path as rollback.
