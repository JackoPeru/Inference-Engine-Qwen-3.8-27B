# HERMES-27: clean integrated build and release blockers

Date: 2026-11-13. **Decision: NO-GO for daily production.** All results below
distinguish clean-source compilation and CPU contracts from checkpoint-wide
numerical correctness, CUDA throughput, and production cutover.

## Reproducible clean source provenance

- Engine source snapshot: GitHub `main` commit
  `1779fbe7be07e8dda7503bf9955594de029c5250`, confirmed equal to
  `origin/main` before the new diagnostic-only commits.
- Created via `git archive HEAD hermes-27`, not an untracked or modified
  Windows worktree and not the older runtime tree.
- Build directory on the Ubuntu server:
  `/tmp/h27-prime-integrated-Xa44Hz/build`.
- CUDA 12.8.93, NVIDIA SM120a. `HERMES_REQUIRE_SERVER=ON`.
- Experimental diagnostic build toggles enabled: real async, real standard
  prefill, real chunked prefill, exact BF16 prefill bench, GEMV oracle, MTP
  CLI and stream prototype. **GPU test registration OFF**; heavyweight
  checkpoint CPU test registration OFF; W4A4 CUTLASS prototype OFF.
- Normal live services were never stopped, restarted or reconfigured.
  Existing `hermes-tabby` and `hermes-gpu-manager` stayed active.

## Fresh build and CPU gates from this clean snapshot

All ten requested targets built successfully: `hermes_server`,
`hermes_generate`, `test_async_multiuser`, `test_prefill_chunk_real`,
`test_prefill_integration_real`, `hermes_mtp_generate`,
`prefill_exact_bench`, `gemv_real_compare`, `gemv_real_cpu`, and
`gemv_real_q8_cpu`. The fresh `gguf_dump` dependency and
`test_gguf_synth` also built. NVCC emitted unused-variable warnings
but no build failures. The earlier stale-header `DecodeState::async_ctx`
error did **not** reproduce.

| Gate | Fresh result | Scope |
| --- | --- | --- |
| `ctest --output-on-failure -j2` | **10/10 PASS** | Registered host-only tests |
| `bash serving/tests/prod_sched_cpu_gate.sh` | **70/70 PASS** | Real serving source and scheduler with mock CPU PP2 |
| `verify_release_static.py` | **PASS_SCOPED_GATES_ONLY** | Real GGUF math/layout and 10 invalid CLI args, before GPU upload |
| `verify_release_contexts.py` | **129/1024/8192 exact token fixtures PASS** | Real tokenizer + freshly built `gguf_dump`; NO GPU inference |
| `python -m unittest discover -s ops/tests` | **113/113 PASS** | Staged lease/deployment proposal; OFFLINE simulation only |

For the last row, use
`/opt/hermes/runtimes/exllamav3-venv/bin/python -B`, which has FastAPI.
System `python3` lacks FastAPI and cannot import
`test_manager_lease_runtime` (an environment dependency, not a runtime
assertion failure). The ops proposal was still staged and uncommitted when
the suite was run; it must be reviewed and committed independently.

Output evidence:

- `/tmp/h27-prime-integrated-static-20261113.json`
- `/tmp/h27-prime-integrated-contexts-20261113.json`
- `/tmp/h27-layer-bisect-worker1/real_cpu_firstqkv_compare.json`

The first two are produced from the clean snapshot. Paths under `/tmp`
are ephemeral and should never be treated as persisted release binaries.

## Independent numerical evidence: first layer

The identical 19,694,360,544-byte GGUF has SHA256
`db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1`.
An actual independent llama.cpp `cb_eval` capture at layer0,
token 6205, absolute position 0, was compared with independently
decoded exact GGUF NVFP4 weights and *the same captured norm vector*.

| CPU projection compared to llama QKV (10240 values) | RMSE | Max absolute |
| --- | ---: | ---: |
| FP32 activation | 0.005707039 | 0.03521442 |
| BF16-rounded activation | 0.005968646 | 0.03662157 |
| GGML-style per-32 Q8_1 activation quantization | **1.5765e-7** | **3.815e-6** |

The first norm layer matches to within FP32 rounding. Q8_1 activation
quantization accounts for the measured **CPU vs independent llama**
first-QKV difference for this one input. It does **not** prove that the
current HERMES PP2 GPU output matches llama, or that 64-layer logits
will pass. Full-vocabulary independent release comparison still FAILS
12/12 traces (previous worst max abs ~0.565, RMSE ~0.056). No thresholds
were relaxed, and no Q8_1 experimental kernel is enabled in production.
See `reports/LAYER0_INDEPENDENT_NUMERICS_2026-11-13.md` and
`reports/RELIABILITY_LAYER_BISECT.md`.

## CUDA / performance: existing measurements, not re-run here

The committed `benchmarks/CHUNKED_PREFILL_REAL_2026-11-13.md` already
documents exact same-engine last logits, GDN/conv/KV and continuation
for T129 and T1024. The T1024 prefill itself took approximately 106 s;
T8192 remains unvalidated end-to-end, and exactly T8192 leaves no
position for the next token at current MAX_CONTEXT=8192.

The committed `benchmarks/HTTP_SEEDED_PREFILL_AB_2026-11-13.md` already
measures short-prompt N1/N2/N4/N8 real HTTP requests: seeded opt-in
approximately 0.96-0.98 aggregate completion tok/s at N4/N8, versus
about 1.11 tok/s for the original async nonseeded baseline. Those are
short-load wall-clock completion rates, not steady-state decode rate or
a viable daily multiuser throughput. Tests in this clean-integration
pass did **not** run CUDA inference while worker GPU ownership was
under review; old measurements are cited, not silently re-labelled.

## Mandatory blockers before enabling production

1. Capture current real HERMES GPU block0 QKV on exactly the llama input,
   implement and independently verify any Q8_1 parity correction, then
   repeat the strict full-vocab 12/12 gate. Same-engine parity alone is
   insufficient.
2. Eliminate performance collapse at N4/N8; benchmark true multiuser
   sustained throughput, p50/p95 TTFT/TPOT and VRAM pressure after a
   proven modification, not by selectively reporting N1.
3. Complete bounded long-context generation and continuation, including
   cancellation/timeouts; T8192 input currently cannot continue.
4. Complete watchdog-fenced GPU Manager lease, independent ingress
   drain and 2GPU/PID-owner checks, live rollback rehearsal, and
   production-specific security review. The lease patch was tested
   offline but has **not** been installed.
5. Run soak, error/recovery, independent logits and cutover preflight
   against the exact shipping source with authorized GPU exclusivity.

**Release decision: NO-GO.** Keep the previous TabbyAPI installation and
GPU Manager active. Do not equate clean builds, CPU mocks, or
single-projection precision with product acceptance.
