# HERMES-27 — Integrated numerical, concurrency and safety gate (revision 2)

**14 November 2026 — release: NO-GO.** Scope: independent CPU/host
validation, source hardening, CUDA 12.8 compile-only, read-only physical
ingress/GPU inventory, and verified commits to GitHub. Nothing here is a
full-model CUDA correctness or real user-throughput certificate.

## Reconciliation, provenance and publication

This advancement began with Ubuntu Git `HEAD == origin/main ==
efad3f42f9d996aeae5ba6280c6209668ead2e8d`, clean, and a separate
Windows shared workspace without a `.git` directory. Worker scopes were
separated; frozen SHA256 source was copied to new Ubuntu scratch archives
and checked before integration. The GitHub target remained
`JackoPeru/Inference-Engine-Qwen-3.8-27B`.

Committed to `main` in order; **each push was followed by a matching
local/remote HEAD check**:

| Commit | Technical contribution |
| --- | --- |
| `6da5f533` | Repair independent layer0 comparator for the current strict three-marker scratch tracer |
| `31927f88` | Private no-overwrite publishing of optional full-vocabulary logit evidence |
| `67147811` | Strengthen llama downstream callback provenance; never promote synthetic tensors as an independent oracle |
| `9b844406` | Ready-scheduler fairness/credit/backpressure CPU witness for N2/N4/N8 |
| `7e836763` | Strict synthetic MTP K>1 acceptance/rollback contract, still disabled in production |
| `6b96d0d5` | Ready-completion/RAW/chunked HTTP backend interoperability tests |
| `3e549547` | Real-GGUF/llama position-0 same-capture Q8_1 scalar, DP4A and BF16 source-math comparison |
| `17a8c8ee` | Stateful prefill source, chunk boundaries and tiled GEMM CPU reference witnesses |
| `536b4b79` | Refuse unexpected Tabby restart during Hermes GPU-owning deployment phases; preserve two external-authority security XFAIL |
| `b5e0f254` | Reject short/oversized nonempty full-prefill reference logits before out-of-bounds access |

No model GGUF, hidden activation, binary build, key, credential, cache or
raw tensor dump was committed. Existing deployed `/home/matteo/hermes-27`,
TabbyAPI, GPU Manager and services were not touched.

## Numerics: independent, bounded conclusions

The authentic independent llama callback manifest at
`/tmp/h27-layer0-llama-pos0-prime3/manifest.json` remains SHA256
`49b0052027532e1e8572d9d865517b2beef92c0e8176a02adacd35efe46d540b`;
the real 19,694,360,544-byte GGUF remains
`db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1`.
Existing independent actual callback bytes cover `attn_norm`, mixed QKV
and convolved SiLU at input token 6205, absolute position 0 only.

The new same-capture **CPU mathematical/source emulation** used the same
captured FP32 input, real GGUF NVFP4 weights and all 10,240 authentic
llama QKV output elements:

| Method relative to genuine llama QKV | RMSE | Maximum absolute error |
| --- | ---: | ---: |
| Q8_1 DP4A lane/order CPU emulation | 1.86908227e-7 | 3.81469727e-6 |
| Q8_1 scalar order CPU emulation | 2.87955345e-7 | 5.72204590e-6 |
| Independent Q8_1 NVFP4 FP64 mathematical oracle | 1.57648647e-7 | 3.81469727e-6 |
| BF16 RNE NVFP4 FP64 mathematical oracle | 0.00596864625 | 0.0366220474 |

Six BF16 warp index/accumulation patterns were source-emulated on **six**
authentic GGUF output rows, never claimed as GPU warp execution.
No real CUDA kernel timing or accuracy was measured. Both existing
Q8 full-model 3/3 differential failures and the independent **248,320-logit
full-model 12/12 FAIL** remain unresolved. Do not infer 27B parity from
the first QKV stage.

The comparator itself formerly rejected the *real current* scratch
transformation because it expected two marker occurrences instead of
three. Its RED was reproduced with a **synthetic** manifest, then its exact
source transform and marker audit passed **27/27** focused CPU tests
including hash-repinned tamper negatives. The downstream llama guard
passed **32/32** CPU source/provenance tests across original and new
contracts. However there is **no genuine contiguous two-token llama
beta/gate/GDN/FFN oracle or physical HERMES v2 sidecar**. The Q/K head
repeat/scale and FFN callback duplication semantics remain unproven.

## Stability, concurrency and prefill

- Full-vocabulary dump previously could overwrite existing output and
  truncate a prior partial. CPU RED demonstrated overwrite. A private
  `0600`, owner-only `0700` directory and exclusive/no-replace
  hard-link publication passed **21** GCC ASAN/UBSAN assertions and
  **7/7** compiled CLI negative option tests.
- A second real CPU **RED** in the optional `--verify-prefill` path showed
  `AddressSanitizer: heap-buffer-overflow` when a mock backend returned
  `T*248320-1` nonempty logits. Checking exact `T*248320` length
  **before** iterator slice creation passed **4/4** dynamic ASAN/UBSAN
  mock cases, and a full CUDA 12.8.93 `hermes_generate` build/device
  link plus existing **7/7** CLI tests passed. Final CLI binary SHA256
  `44dbfcce62060820db1d750ac3eca770e3408c96052f26ea24d7e3200ef9b177`.
- The real ready-completion scheduler with fake backend passed **253**
  GCC ASAN/UBSAN CPU assertions for N2/N4/N8, fairness, token credit
  preservation, queued cancellation/expiry and slow callback behavior.
  Late admission can remain blocked while an A step is outstanding,
  and callback delivery remains pump-serial: these are source-observed
  limitations, not measured GPU throughput regressions.
- Separate integrated HTTP CPU real-server/mock-PP2 checks passed **8/8**
  for ready/RAW/chunked startup incompatibilities, no double seed or
  repeated request states, cancellation after first SSE token, and
  terminal OOM error+[DONE]. The previous unchanged HTTP **90/90+7/7**
  suite receipt was not redundantly rerun.
- Prefill independent CPU closed-form state/chunk source checks passed
  `state=37 tiled=18 source=18`; T64/65/129 and 1/3/7/64 chunk sizes,
  causal KV, absolute RoPE, GDN/conv and per-position FFN were tested
  as toy witnesses. This cannot certify actual full-GGUF CUDA parity.
  Source analysis found **1,344 temporary cudaMalloc/free pairs and
  >=64 explicit layer device synchronizations per chunk**. No reliable
  GPU time or safe allocation-hoist speedup is claimed.
- A separate MTP K1..4 rollback policy witness passed **22,761** CPU
  assertions on a synthetic target. Production target batched atomic
  snapshot/rollback is absent, so MTP K>1 remains **disabled**.

## Deployment: one fail-closed fix, external controls missing

Read-only Ubuntu process/GPU observation at **2026-11-14 11:03:06 UTC**
showed GPU0/GPU1 21/26 MiB, no reported compute apps, and inactive
`hermes-tabby`/`hermes-gpu-manager`, **but live direct, localhost,
gateway and Tailscale listeners remained**. Idle does not grant GPU
ownership or establish HTTP/SSE all-ingress drain.

Three strict deployment contracts were initially **XFAIL**: replay of
an older HMAC-authentic journal across restart, one signed ingress proof
reused for multiple lease epochs, and acceptance of an unexpected Tabby
restart during Hermes-watch. The third was reproduced RED and corrected
locally in `ops/production_deploy.py::lease_guard` for `hermes-start`
and `hermes-watch`. After the fix, **10 tests: 8 PASS, 2 XFAIL**;
`ops.tests.test_production_deploy` **20/20 PASS**; 20/20 entirely
simulated cutover/recovery scenarios PASS. The two remaining XFAIL
require a privileged external monotonic authority and lease-specific
nonce-consuming ingress proof. A local guard cannot enforce a physical
systemd/Tabby restart interlock or attest concurrent owners.

No lease was granted, process killed, real HTTP ingress changed, or
CUDA model test run. Historical 12/12 numerical FAIL remains the
decisive independent numerical blocker.

## Required final GO sequence (still pending)

1. Independently attest all-ingress drain and HTTP/SSE inflight=0 across
   direct, Tailscale and gateway routes. Bind externally trusted
   anti-rollback epoch, lease nonce, PID/start-time and both GPU UUIDs
   to a reversible ownership grant, with crash/boot fencing.
2. Only after that authorization, capture actual same-GGUF contiguous
   HERMES/llama `[6205,161421]`, positions `[0,1]` tensor stages
   through beta/GDN/gate/attention/post-norm/FFN; bisect and correct the
   *effective CUDA path*, then pass the unchanged 12/12 full-vocabulary
   independent comparison.
3. Compare actual full-model legacy vs experimental Q8/RAW/chunked
   throughput **N1/N2/N4/N8**, TTFT, token TPOT or accurately labeled
   chunk TTFC, p50/p95, VRAM, fairness, cancellation and backpressure.
   Validate T8192+ GPU continuation against a truly equivalent sliding
   window oracle; do not conflate eviction with unlimited history.
4. Prove target-batched atomic K>1 verification and actual speedup before
   enabling MTP. Run real soak, recovery and reversible deployment gates
   before any public service switch.

**Final verdict: substantial CPU/offline/compilation progress; true
GPU/numerical/performance and physical deployment gates remain BLOCKED,
therefore HERMES-27 remains NO-GO.**
