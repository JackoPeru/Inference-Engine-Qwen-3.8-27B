# One-transaction authority/ingress composition: immutable Linux CPU receipt

Date: 2026-11-14. **CPU contract PASS; physical release BLOCKED / NO-GO.**

## Provenance and exclusive scope

- Git source snapshot:
  `925534d6970223ec8d9d989f021f81f07a8e6cbe`, archived under
  `/tmp/h27-authority-composition-git925-20261114/src/hermes-27`.
- Overlay strictly three new worker-frozen artifacts and a CMake CPU-test
  registration. No production serving, PP2, deployment manager, GPU,
  ingress listener or lease-core code changed.
- Frozen `ops/staging_authority_ingress_composition.py` SHA256:
  `20d9c41a2875a72efedea405df7712c3a394e67acec89a029a5c4893117548b8`.
- Frozen `ops/tests/test_authority_ingress_composition_cpu.py` SHA256:
  `f9b3bff7ee5e106387349d143a204aebd8a78c36b419984a4deb2c6fb16a1379`.
- Worker source/report:
  `reports/AUTHORITY_INGRESS_COMPOSITION_CPU_NEXT.md` SHA256:
  `3543120b3d3823086e399848532749ba0c8106b699f164b21cfdb115636bc57a`.
- `CMakeLists.txt` registers
  `host_authority_ingress_composition_cpu` as
  `cpu-only;source-contract;no-gpu` with source-root cwd and a bounded
  120-second timeout. This test does not need to build a binary or run a
  CUDA kernel.

## Actual Linux verification

Executed Python 3.12 `-B -m unittest` on the isolated snapshot with all
three suites: **65/65 PASS**:

- 20 new cross-ledger composition synthetic CPU cases;
- 22 previously published external monotonic authority CPU cases;
- 23 previously published ingress/GPU physical attestation **synthetic**
  CPU cases.

CMake Release configured with nvcc CUDA 12.8.93,
`HERMES_REQUIRE_SERVER=ON`,
`HERMES_REGISTER_GPU_TESTS=OFF`, without launching CUDA code.
The new public CTest
`host_authority_ingress_composition_cpu`: **1/1 PASS**, 0.11 s.
The unchanged real CUDA targets were already compiled and linked on the
prior exact published source; no unnecessary rebuild was performed for
this new-only Python/CMake registration.

## What this proves — and what it does not

Both previous staging validators consume distinct ledgers. Even if both
separately acknowledge the *same* signed nonce/highwater, their operations
are not one atomic commit across failures. The new composition deliberately
requires an initially pristine ledger pair and a *single* injected parent
`AtomicCompositionStore.atomic_update` transaction. Within the callback
it cryptographically checks same manager boot/lease/epoch/fence/nonce and
manifest identity, stages both original validators with separate
signer-verifier objects, and refuses replay, split commits, forged ACKs,
clock expiry and cross-protocol signed-claim drift.

All positive fixtures use in-memory synthetic signers/stores and return
`status=BLOCKED`, `production_admissible=False`,
`manager_lease_proof=None`. The new module is **not a privileged verifier
or producer**, cannot rotate/renew an existing epoch, and refuses all
non-pristine history until an independently attested recovery protocol
exists.

No physical six-route all-ingress fence, HTTP/SSE zero inflight, complete
GPU UUID/PID/starttime/driver-context ownership, root signer or durable
external antirollback highwater was proven. No production deployment,
GPU model inference or network change occurred.

**Verdict: synthetically consistent CPU contract only. Physical release
and MTP K>1 remain NO-GO.**
