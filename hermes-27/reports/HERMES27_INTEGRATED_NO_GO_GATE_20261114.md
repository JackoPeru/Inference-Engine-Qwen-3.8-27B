# HERMES-27 — integrated CPU/CUDA source gate (2026-11-14)

**Release verdict: NO-GO.** Hardware execution, production cutover,
all-ingress drain and independent full-model numerical parity remain
unapproved/unproven. This report does not convert synthetic CPU tests
or CUDA compilation into a physical model validation.

## Provenance

Published GitHub main at this report draft:
`925534d6970223ec8d9d989f021f81f07a8e6cbe`.
Ubuntu checkout `/home/matteo/hermes-27-gh`, separate Windows/shared
workspace `/saas/Inference Engine Qwen 3.8 27B/hermes-27`.
Model file `Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf`, 19,694,360,544 B,
pinned SHA256
`db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1`.
Authenticated prior llama pos0 manifest SHA256
`49b0052027532e1e8572d9d865517b2beef92c0e8176a02adacd35efe46d540b`
has only `attn_norm-0`, `linear_attn_qkv_mixed-0`,
`conv_output_silu-0`.

## Completed and actually verified

| Priority | Integrated change / concrete evidence | Boundary |
| --- | --- | --- |
| Multiuser throughput | Ready completion, bounded opt-in callback delivery, per-job acknowledgment/fail-stop, fairness/terminal recovery, HTTP `--ready-callback-delivery`; 265 CPU adversarial checks, HTTP SSE/Unicode/stop/cancel/usage, CTest 6 scheduler targets | No physical N1/N2/N4/N8 gain demonstrated |
| Prefill | Request-private GPU0/GPU1 scratch pools with lifetime-safe teardown; two pooled end-of-layer waits now stream-local, unpooled behavior preserved; CPU allocation/fault/source mutants and CUDA device-link PASS | 64 stream waits/chunk remain; no true latency benchmark |
| Serving | Original legacy SSE `bad_alloc` SIGABRT reproduced/fixed; readiness after ready-pump poison reports HTTP 503, `ready=false`, metric 0; stuck tokenizer writer RED original/GREEN 3/3 CPU, bounded total nonblocking write and 503 `tokenizer_unavailable` | No physical multiuser cancellation/OOM/driver-recovery test |
| Long context | Independent 8192-window/deque/RoPE/GDN-state oracle 8/8 CPU; strict standalone `--sliding-kv` and HTTP opt-ins and CPU T8192+2/3 mock | Real CUDA T8192+ numerical parity and windowed llama oracle absent |
| Numerical | Authenticated pos0 QKV Q8_1 and conv traces; source-derived independent llama GDN pos0 CPU reference 10/10, strictly bounded tracer v2 and additive post-FFN scratch tracer 18/18 CPU+CUDA12.8 full compile/device link | Cannot identify first real postconv CUDA mismatch; independent 12/12×248320 full-vocab gate still FAIL |
| Deployment | Synthetic signed ingress six-route/2-GPU/zero-inflight antirollback CPU checks 23/23, external lease highwater/nonce staging 22/22, unchanged production two XFAIL | No privileged physical signer, durable independent issuer/store or authenticated GPU/ingress authority |
| Build and reliability | Exact `36b4b113` immutable Git archive CUDA12.8 `hermes_server` + `hermes_generate` full device-link PASS, all **14/14 host CTest PASS** after building all required mock executables; synthetic SSE profiler CTest 1/1 PASS | No CUDA kernel launched; host tests cannot prove VRAM or throughput |

### Further independent audit — keep existing fail-closed barriers

An independent source-only audit of the `pp2_fwd.cu` pooled path found
that replacing all **64 per-chunk stream waits** with two GPU-stage
barriers is **not currently proven safe**. The pooled scratch allocator
may replace/free a prior layer's slot, full attention requires synchronous
device-to-host gate vectors before host CPU/H2D work, selected diagnostics
copy tensors back to host, full-layer `klen` must not be committed before
the layer finishes, and several CUDA return paths were not designed for
deferred error checking. It also identified a source-level risk from the
cached GEMV variant versus a live environment admission check. No
speculative CUDA optimization was merged; the published layer barriers
remain in force. Speedup is **not** claimed.

The tokenizer deadline correction has a real CPU RED/GREEN test, but a
separate read-only audit identified unresolved production availability
risks: each request can still consume separate 15-second write and read
deadlines for two attempted transactions plus a child restart, so a
**whole-transaction and shutdown deadline** is not enforced. The Python
tokenizer bridge's external `gguf_dump` subprocess also lacks an explicit
timeout and process-descendant recovery fence. These findings require
separate deterministic CPU witnesses and narrowly scoped fixes before
assuming bounded real-world recovery under every child failure.

### Two independent synthetic lease ledgers are not a single proof

After the initial external-authority and six-route ingress tests, a new
strict CPU-only composition stages both unchanged validators into **one
abstract parent atomic update**. It rejects separately consumed nonce
ACKs, mismatched boot/lease/epoch/fence/route manifest, cross-role signer
mixups, lost commit ACK, replay and double invocation. The 20 new tests
plus 22 external-authority and 23 physical-ingress tests pass **65/65**
on an isolated Linux source snapshot, with a public CTest 1/1 PASS.
Every synthetic success is still `status=BLOCKED`,
`production_admissible=False`, `manager_lease_proof=None`.
This design requires an external privileged **shared atomic store**,
physical signer/observer and explicit crash recovery/rotation; those do
not exist in the current production system. See
`AUTHORITY_INGRESS_COMPOSITION_INTEGRATED_20261114.md`.

Relevant published commits, in chronological progression:
`727d9cd`, `8a82913`, `44559347`, `fe95ae70`,
`de15d66`, `5e98136`, `a5c079bf`, `ebf8a28e`,
`3ca223f`, `3daac943`, `4ec788a`, `75ed5e4`,
`ff11245`, `7b72713`, `2a6bac6`, `36b4b11`, `925534d`.
The exact changes can be verified with `git log --oneline`;
short SHAs here are navigation only, not substitutes for source hashes.

## Latest physical read-only facts (not an owner grant)

At 2026-11-14 13:49:40 UTC, both GPU UUIDs matched earlier observations:
`GPU-5a3820fd-8771-64b3-7808-1404b1d71a85` (21 MiB, 0%) and
`GPU-8777b420-7899-af2c-f518-d0b12a56c82c` (26 MiB, 0%).
No compute apps were reported. `hermes-tabby.service` and
`hermes-gpu-manager.service` had `MainPID=0`, inactive.
Multiple loopback/direct/Tailscale listeners remain, including 8030,
9119, 8650, 8645, 8001 and others. A separate abandoned **CPU mock**
at `127.0.0.1:58805`, PID 2716931, was independently verified as
our isolated test executable and cleaned up **gracefully by SIGTERM**;
it was not a production model. Idle GPU utilization is **not**
authorization or proof of zero ingress.

## Strict release gate not yet met

1. Independently prove every physical Tabby/HERMES ingress and restart
   path is fenced and HTTP/SSE queues/inflight are zero, including direct,
   Tailscale and local clients; prove both GPU owner identities and
   PID/starttime/boot for both exact UUIDs.
2. Establish a real privileged signer and antirollback highwater/nonce
   consumption authority, crash/restart/rollover evidence, signed
   all-route observation, and reversible ownership rollback. A local mock
   signed dictionary never suffices.
3. With an explicit exclusive GPU maintenance authorization, capture
   the genuine same-request HERMES layer0 and independent llama.cpp
   downstream callbacks for token IDs `[6205,161421]`,
   absolute positions `[0,1]`, same original GGUF SHA and exact
   binary/source/capture provenance. Bisect the first real mismatch in
   beta, z/gate, Q/K, GDN, residual, postnorm, FFN and logits, fixing
   only the effective CUDA path based on evidence.
4. Pass 12/12 **independent full 248320-logit vocabulary** comparisons
   without changing thresholds or inventing oracle tensors.
5. Physically verify long-context T8192+, absolute RoPE, chronological
   KV ring/window semantics, recurrent GDN+conv persistence and rollback
   against an independent oracle with the **same window definition**.
6. Measure full-model aggregate throughput first (N1,N2,N4,N8),
   then prefill and individual decode, with TTFT, TPOT or qualified TTFC,
   p50/p95, VRAM, fairness, backpressure, cancellation, shutdown,
   long-running soak and legacy-vs-opt-in A/B on identical conditions.
7. Keep **MTP K>1 disabled** until a real batched target verifier,
   atomic KV/GDN/conv accept/reject/EOS rollback, independent parity and
   positive full-model measured speedup are demonstrated.
8. Only then evaluate an explicitly authorized reversible staged cutover,
   preserving `/home/matteo/hermes-27`, TabbyAPI, GPU Manager and every
   unrelated listener/service.

There has been **no model launch, GPU inference benchmark, service restart,
production deployment, networking change or GPU context acquisition** in
this CPU/offline integration series.
