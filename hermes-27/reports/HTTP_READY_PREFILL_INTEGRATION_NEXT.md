# HERMES-27 — HTTP ready/prefill interoperability: focused CPU-only gate

**14 November 2026 — worker-8 scope freeze. PASS 8/8 new focused tests; production GPU and release remain NO-GO.**

## Provenance, ownership and isolation

Writable scope: ONLY NEW `serving/tests/test_http_ready_prefill_interop_cpu.py` and NEW `reports/HTTP_READY_PREFILL_INTEGRATION_NEXT.md`. No changes to server.cpp, batch/fwd, mock PP2, CMake, Git, services or GPUs. The sole HTTP interactions were with newly launched fake-PP2 processes listening on private loopback in disposable CPU scratch. No live model/inference endpoint was contacted.

- Requested published source baseline: Git commit **efad3f42f9d996aeae5ba6280c6209668ead2e8d**. Initially verified as both HEAD and origin/main on Ubuntu with a clean working tree.
- Original checkout: `/home/matteo/hermes-27-gh/hermes-27`. Windows shared source has no .git. The Ubuntu checkout later advanced concurrently to clean HEAD **6da5f5334b17e7fc839296b8c213220d7d79e480**, but the **test build was pinned by git archive efad3f42**, not by whichever HEAD was current.
- Fresh CPU-only scratch: `/tmp/h27-worker8-ready-prefill-interop-efad3f42`. Archived real `serving/server.cpp`, `serving/tokenizer_bridge.py`, `distributed/pp2_batch.cu/.hpp`, `distributed/pp2_fwd.hpp` and existing PP2 fake/test helpers. Local GGUF was an empty `fake.gguf`; tokenizer was the real tokenizer.json fixture, but no model weights were read.
- Exact compiled source SHA-256: server.cpp **ad2522dc7297bb9e3826a617778ec580e591d917e6bde3796adc3146e596a7f4**, pp2_batch.cu **279c76c0c502d10f540c0f16dd754bbadc4f746692a44d3e4bedfbb85135fb06**, prod_sched_mock_pp2.cpp **cd7f2e953f7c167186116a7764929b4ab8ea37799c293b2d26663aeda3993c37**.
- Fresh GCC C++20 CPU fake-server compile/link PASS with `-DHERMES_ADMISSION_CPU_ONLY` and vendored cpp-httplib. Executable SHA-256 **a42c71fdd75a54130d54ad2a7a654b32c8fe7617058db7fd315f4368868ba956**. No CUDA device/runtime used.

## Earlier receipts: inspected, NOT rerun

The published `reports/HERMES27_INTEGRATED_CPU_CUDA_GATE_2026-11-14.md` reports **90/90** integrated serving HTTP CPU-mock checks and **7/7** terminal HTTP/SSE fault tests, as previously executed in a separate gate. Relevant unchanged suites were inspected, including prod_sched_ready_test.py, prod_sched_prefill_test.py, prod_sched_chunked_test.py, prod_sched_state_test.py and their PP2 CPU fake. **Neither the preexisting 90-case gate nor the seven terminal tests were rerun** by this worker. Their results are not counted as today's eight new tests.

## Real CLI and scheduler contract, source anchors

All anchors refer to the exact pinned 1773-line server.cpp.

- Lines 107–139: the actual one-shot RAW seed flag is **--batched-prefill**; the legacy 65+ token path is **--chunked-prefill** (which also sets batched); the ready mode is **--continuous-scheduler --parallel-steps --ready-completion**. There is **no --raw-prefill or --legacy-chunked CLI alias**; each must be rejected.
- Lines 1475–1506: ready requires parallel, and parallel requires continuous plus HERMES_PP2_ASYNC_STREAMS=1 while forbidding batched; chunked requires legacy only, --active 1, and a separate exact HERMES_PP2_CHUNKED_PREFILL=1 opt-in. Therefore ready+RAW/chunked is intentionally not an admissible combined execution mode.
- Lines 1508–1529: sliding-KV cannot join ready mode; BF16 activation seed rejected by serving.
- Lines 432–461 and 1098–1100: only one ContinuousScheduler is allocated in continuous mode and the server run loop returns into its dedicated continuous path; legacy queue/prefill path cannot run as a second scheduler.
- Lines 1023–1095: legacy RAW or chunked seed is applied to its original untouched DecodeState once. Continuation consumes the generated token at absolute position T; no prompt replay.

## NEW actual focused tests: 8/8 PASS

New test `serving/tests/test_http_ready_prefill_interop_cpu.py` has **367 lines**, ten negative startup matrix entries and additional positive, concurrency and fault-recovery checks:

| Method | Result | Actual witness |
| --- | --- | --- |
| test_invalid_ready_plus_seed_flags_fail_before_model_upload | PASS | Ten combinations rejected with exit code 2 before MODEL ready/LISTEN: ready alone; no continuous; no async; ready+batched RAW; ready+chunked; continuous+legacy chunked; ready+sliding; RAW+BF16; unsupported --raw-prefill; unsupported --legacy-chunked |
| test_ready_env_does_not_seed_without_flag_mixed_T64_T65 | PASS | Two concurrent T64/T65 ready requests, two output tokens each, with CHUNKED_PREFILL env present but no chunked flag; exactly two opens/closes, **131 regular steps**, correct absolute-position multiset, zero prefill/chunked calls and zero seed counters |
| test_raw_seed_allowed_separately_legacy_and_continuous_without_parallel | PASS | T11/3-token SSE with legacy and nonparallel continuous separately: one open, one RAW seed, step positions 11/12, one close, one finish/usage/DONE |
| test_legacy_chunk_seed_one_state_without_parallel_owner | PASS | T65/3-token SSE legacy-only: one open, one chunked seed, steps positions 65/66, one close, correct usage and one terminal |
| test_ready_late_cancel_after_visible_token_one_owner | PASS | Ready T1 with 150ms CPU-only decode; cancel endpoint 202 only after receiving a visible generated SSE token. Exactly one cancelled finish/usage/DONE, one open/close and no repeated step positions |
| test_chunked_late_cancel_after_seed_no_replay_or_double_close | PASS | Legacy chunked T65, cancel 202 after first generated SSE token. One seed, one open/close, no step below 65, no replay, only one cancelled SSE terminal |
| test_ready_sse_oom_single_terminal_then_normal_completion | PASS | CPU-only HERMES_TEST_FAIL_TERMINAL_USAGE_ONCE failure after local finish frame construction but before publish. First SSE has one error+[DONE], no success finish/usage; second is normal; exact two independent open/close states, no repeated tokens |
| test_chunked_sse_oom_does_not_reseed_or_fake_success | PASS | CPU-only HERMES_TEST_FAIL_TERMINAL_QUEUE_ONCE injection after T65 seed during terminal queue commit. First SSE error+[DONE] only; second normal; exactly two seeds, two states, no duplicate terminal or prompt replay |

**Verification history must not be rewritten:** first invocation yielded 6 passing methods and **2 failing matrix subcases**. The newly authored harness expected "unknown option" for missing-argument `--raw-prefill` and `--legacy-chunked`, but the actual CLI correctly reported "missing value for ..." with code 2. The test was corrected **only** by passing a dummy value `1` after each unsupported switch, which exercises the expected "unknown option" rejection. This was a **test assertion error**, not a production bug or bypass.

**Final real run: `Ran 8 tests in 7.409s — OK`; exit code 0.** Python `-m py_compile` also PASS on the isolated Ubuntu CPU scratch. No unchanged full suite was rerun.

## Reproduction (isolated CPU fake only)

In the existing isolated scratch after the described source-pinned GCC fake PP2 build, the new test was run with:

`/opt/hermes/runtimes/exllamav3-venv/bin/python serving/tests/test_http_ready_prefill_interop_cpu.py --server /tmp/h27-worker8-ready-prefill-interop-efad3f42/build/hermes_server --model /tmp/h27-worker8-ready-prefill-interop-efad3f42/fake.gguf --gguf-dump /tmp/h27-worker8-ready-prefill-interop-efad3f42/gguf_dump_fake.py --tokenizer /opt/hermes/models/Qwen3.8-27B-Uncensored-EXL3-3.07bpw/tokenizer.json --python /opt/hermes/runtimes/exllamav3-venv/bin/python`

Run it only against the CPU-mock executable built with HERMES_ADMISSION_CPU_ONLY, not a production CUDA server. The test starts and stops only its own ephemeral loopback fake HTTP processes.

## Remaining limitations and ownership decision

- **No proven product bug** from this focused CPU integration witness. Therefore no ownership request to edit the published server.cpp was made.
- Ready+RAW/chunked prefill remains **deliberately unsupported**: this gate proves fail-closed rejection rather than combined GPU execution or throughput gains. The name `--raw-prefill` is not a supported CLI synonym.
- The fake traces do not include request IDs on `step`. Multiuser T64/T65 proves exact total position multiset and two distinct allocations, while single-request late-cancel witnesses rule out repeated state positions; this does **not** constitute a CUDA concurrent-state race proof.
- Cancellation cannot interrupt an already started backend step, and the CPU fake cannot validate real GPU kernel cancellation latency, CUDA stream state poisoning or physical memory ownership.
- OOM uses deterministic mock-only terminal failure hooks; no claim is made about global host memory exhaustion or actual VRAM OOM.
- No real GGUF inference, numeric reference logits, GPU p50/p95/aggregate throughput, physical ingress drain, soak, source change after efad or release readiness was tested. Published production release status stays **NO-GO**.
- The archive was intentionally pinned at efad3f42 even though Ubuntu main subsequently moved; prime should rerun these targeted tests after a later integrated source snapshot if its serving/scheduler semantics changed.

## Frozen fingerprints

- New `serving/tests/test_http_ready_prefill_interop_cpu.py` SHA-256 **074711b03ffaa68a4b7c99776bf930964a1a79071b2e67aa3287508bee75bbb1**, physically identical on Windows and Ubuntu scratch.
- New report: `reports/HTTP_READY_PREFILL_INTEGRATION_NEXT.md` (this file; hash recorded separately at freeze).
- Fake executable SHA-256 **a42c71fdd75a54130d54ad2a7a654b32c8fe7617058db7fd315f4368868ba956**; fixed source fingerprints above.

**Scoped integration verdict: PASS 8/8 CPU/offline fake HTTP; full-model/GPU/release gate remains NO-GO.**
