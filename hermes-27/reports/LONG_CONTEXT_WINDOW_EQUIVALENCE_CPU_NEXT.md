# HERMES-27: long-context window equivalence, CPU-only — 2026-11-14

**Result: 8/8 PASS on Windows and 8/8 PASS on isolated Ubuntu CPU scratch. Full CUDA/GGUF numerical parity is BLOCKED.** Worker-17 changed only the two new files assigned: distributed/tests/test_sliding_window_equivalence_cpu.py and this report. No engine/header, serving, CLI, CMake, Git, service, ingress or GPU state was changed.

## Frozen provenance

Published Ubuntu repository: /home/matteo/hermes-27-gh on main. Initially observed HEAD 5c74e211e19a7b4a6f3d05a7a62b63430aedb718; subsequently observed HEAD 95fcd54384c5cb137eeea8549b45738fd9e2e670. Source hashes below remained identical at both observations. The Ubuntu test was run from frozen copies in /tmp/h27-w17-window-equiv-20261114, not from the active Windows fwd file being changed by worker-14.

| File | SHA-256 frozen |
| --- | --- |
| Published distributed/pp2_fwd.cu | 0b8b8bcbcb6d60ee1c49fb240d7e925772fafc96def5c03151fe02b0d50b93ed |
| Published distributed/pp2_fwd.hpp | 97649e4027ca8db47bed7f0323101edffbe3ddf68ea547f2534e9501d7696b2a |
| Published kernels/src/gdn_step.cu | a432d5ce57467320e232ec48ab19ca659c387394ad7e8a8ad70f5bd207b8d183 |
| New distributed/tests/test_sliding_window_equivalence_cpu.py | f6272d747b373457949d469fcd003da42f0bad8e20e5f4cc74466d9f9cc7de29 |
| Published serving/server.cpp | ad2522dc7297bb9e3826a617778ec580e591d917e6bde3796adc3146e596a7f4 |
| Published tools/hermes_generate.cpp | 62b6e852121f1dc0abe3da71df766e8f40bf253462dfd5ab54781893c7a9868b |

The Windows fwd changed concurrently: observed SHA256 ccaf8f23b2c2886e6f5ca3bb1f2692304163aa1b12747a02c689a9d4b1e14c99 initially, then 87364d1f21062a6dd2c009ca15efe16cde7d92636a35a0c576b08f855edbf71e. These are NOT substituted for the published CUDA source snapshot. Windows tests load whichever source is present at Python process start.

## Independent reference and tested invariants

1. **Full 8192 window at four checkpoints.** A genuine chronological deque(maxlen=8192) reference and an independently implemented modulo-addressed ring retain two separate full-attention layer identities: layer 3 on device 0 and layer 35 on device 1. At absolute p8191, p8192, p8193 and p16384, all 8192 visible logical positions and absolute tags are compared before commit. Expected sets: 0..8191; 1..8192; 2..8193; 8193..16384. The to-be-evicted physical slot still contains p-8192 until staged attention completes; only then are K/V written at p % 8192. Eight distinct full layers per device are source-verified; numerical attention models one representative full layer per device.
2. **24 query heads, 4 KV groups, absolute RoPE.** Each checkpoint compares all 24 query heads with the appropriate one of four K/V groups over the *complete* resident 8192 entries, with nonzero deterministic toy Q/K/V. Oracle uses two-pass max-subtracted softmax; candidate uses separate stable online softmax. Each checks two value coordinates with absolute tolerance 2e-10. Sparse toy head vectors have four nonzero coordinates 0,1,32,33 and 252 zero coordinates; they preserve the kernel's 256-dimensional RMS denominator, rotary pairs and base 10^7, and its 1/16 attention scale. A negative witness explicitly rotates the new query at slot zero instead of absolute p8192 and observes different attention.
3. **Persistent GDN and convolution.** All 48 independent reduced-size GDN state tensors S[1,2,2] and conv tails CS[2,3] (24 per device) advance at every absolute position to p16384. Every checkpoint verifies their cumulative absolute step count, most recent three *pre-convolution* inputs and nonzero state. For one representative GDN layer per device, layers 0 and 32, an independent complete-history replay recomputes decay, gated delta update and outputs from position zero and compares state/output within 5e-12. Resetting at window wrap yields a numerically distinguishable output. Real GDN uses 48 heads and 128x128 state, so this is a reduced-dimension proof only.
4. **Two-device poison rather than rollback.** A mocked post-wrap GPU0 commit succeeds; GPU1 K is overwritten but its V commit fails. Per-layer positions become different, mixed K/V remains in the GPU1 slot and both stages are poisoned. Retries, stale or skipped absolute positions and duplicate commits reject. This demonstrates why a physical partial failure cannot be safely retried without reinitialization; it is not a CUDA fault-injection experiment.
5. **Actual source and five negative mutants.** Source assertions inspect the PP2 source/header plus GDN CPU source. They pin GQA 24/4 grouping, chronological mapping, staged new K/V, absolute RoPE frequency and paired positions, stable-score scaling, completion-before-K/V-overwrite, persistent CS shifting and GDN decay, per-request streams and GPU1 completion-before-unpoison. Five mutants must fail: Q rotary base, K rotary pair, conv CS shift, GDN decay and GPU1 event wait. The first local run failed two of these negative tests because their replacement matched earlier prefill functions rather than decode functions. Scoped mutation selection was corrected and all tests subsequently passed. No production edit was made to fix the test.
6. **Deliberate mismatch to full history.** The discarded oldest value has a nonzero contribution to full-history uniform attention, but contributes exactly zero to the 8192-token bounded result. Bounded window parity must be evaluated against a *sliding* oracle. It cannot equal an unrestricted 8193+ history in general.

The numerical witnesses use synthetic sparse vectors, Python double precision and independent two-pass/online reductions. They do NOT execute the real CUDA kernel, packed weights, logits, full model, large 256-wide dense tensors, or an authentic llama.cpp reference. Actual FP32 rounding, cross-stream races, GPU0/GPU1 fault injection, cancellations and throughput are untested.

## Executed verification

| Environment | Command | Result |
| --- | --- | --- |
| Windows PowerShell, Python 3.12 | C:\Users\matte\AppData\Local\Programs\Python\Python312\python.exe -B distributed/tests/test_sliding_window_equivalence_cpu.py | **8/8 PASS**, 16.641 s, exit 0 |
| Ubuntu remote isolated CPU scratch, Python 3.12.3 | python3 -B /tmp/h27-w17-window-equiv-20261114/distributed/tests/test_sliding_window_equivalence_cpu.py | **8/8 PASS**, 6.836 s, exit 0 |

Ubuntu scratch includes the exact copied published PP2 fwd/header/GDN and new test hash in the table above; the working checkout was not modified. No GCC, nvcc, model or GPU run was performed because the only new executable is Python.

## Read-only HTTP and CLI sliding-admission audit

**HTTP already exposes a strict experimental sliding continuation beyond absolute position 8192.** Published serving/server.cpp:1508-1522 requires both explicit --sliding-kv and HERMES_PP2_SLIDING_KV=1, legacy --chunked-prefill, --context 8192, --active 1 and no continuous/parallel/ready scheduler. Lines 1643-1650 reject prompt length greater than cfg.context but allow prompt.size()+max_tokens-1 up to the explicit absolute position limit (1<<20) in sliding mode. Lines 1373-1384 continue to require each request's max_tokens <=8192. Lines 962-973 and 1146-1147 allocate and admit against fixed KV physical capacity 8192 independent of total generated positions. This means a prompt with 8192 tokens plus at least two completion tokens may reach absolute decode position 8192 in the experimental mode; it is **not** a claim that a physical run succeeded. Prompts longer than 8192, multiple active users and continuous scheduling remain intentionally excluded. No HTTP source change is needed to make this guarded mode syntactically available.

**The standalone generator CLI still blocks it.** Published tools/hermes_generate.cpp:202-204 unconditionally rejects prompt.size()+max_new_tokens-1 greater than kMaxContext=8192, and :308-309 sets decode capacity to that total. No --sliding-kv CLI flag selects fixed 8192 KV capacity and an independent absolute-position budget. To expose longer continuation would require an explicit parser opt-in, guarded capacity=8192, separate prompt<=8192 and total absolute position<1<<20 checks, environment/stream-owner requirements and compatible seeded-prefill validation. Loosening the single old guard would permit unsafe capacity/OOB states. **No CLI change was made.**

## Independent verification questions and revised assessment

- Does every grouped-query head see the exact chronological 8192-key suffix at both wraps? **Yes in two independent reduced-dimension CPU models**, 24 query heads, four KV groups, two device/layer identities.
- Are RoPE rotations absolute while GDN/conv state continues past a KV eviction? **Yes for source contract and CPU reference**, with targeted negative counterexamples; GPU behavior remains to be established.
- Can a partial GPU1 failure be rolled back safely? **No**. The test witnesses non-atomic cross-stage mutation and requires poison/reinitialize.
- Is bounded 8192 attention equivalent to unlimited history? **No**. An explicit evicted-value counterexample establishes the expected divergence.
- Is the full model safe for release or long-context multiuser inference? **Not established. NO-GO** until exclusive GPU physical full-model numerical verification, cancellation/fault evidence and aggregate throughput measurement.

Reviewed decision: **PASS CPU/source-only; BLOCKED CUDA/real-checkpoint parity, performance and production GO.** Project priority remains aggregate multiuser throughput > prefill > individual decode, while experimental sliding currently permits only one active HTTP request.
