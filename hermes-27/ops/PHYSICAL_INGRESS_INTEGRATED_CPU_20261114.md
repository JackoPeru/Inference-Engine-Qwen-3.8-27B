# Physical ingress attestation staging — independent Ubuntu integration

Date 2026-11-14. Production verdict: **NO-GO / BLOCKED**.

## Frozen provenance

- Independent Git source snapshot `ff1124571231bd92b3679af7bb0899274eb8b4cb`, archived into `/tmp/h27-physical-ingress-contract-ff11245-20261114/src/hermes-27`.
- New `ops/ingress_attestation_contract.py` SHA256 `43e37d02a2e9ff6b21f8603ea95d617e055a73a52b0244224d054e63bf80dab7`.
- New `ops/tests/test_physical_ingress_attestation_contract_20261114.py` SHA256 `7b788fab2462c94d2475d1e4d525590841611f09b76955f41b7e3a73b4c88768`.
- Original staging dependencies unchanged: `manager_lease_external_authority_staging.py` SHA256 `02980fc28cae6be33ea5b886058fc36b1425a23fe8c52e501ef80dcaf85d3fe3`; `ingress_drain_gate_staging.py` SHA256 `30635c873310f93bbf5ea1fe33a182757b2971cfeb0590fa8ccca31dc81b9b48`; `gpu_owner_attestor_staging.py` SHA256 `a2d6ceaa90654cbf707f3e48cc71c4e0cd6d4fb70488a30584aa9428c824e0f3`.

## Verification

The worker provided 23 Windows Python 3.12 and 3.13 tests. The prime independently copied the three frozen files into the immutable Linux snapshot and executed:

- `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest -v ops.tests.test_physical_ingress_attestation_contract_20261114`: **23/23 PASS**.
- Existing `ops.tests.test_external_monotonic_proof_staging`: **22/22 PASS**, no change.

The new contract pins six ingress/control routes, individual route signed zero HTTP/SSE/queue/open-connection counters, listener PID/start-ticks and manifest generation, the two GPU UUID/PCI/process/birth/boot identities, role-separated signer inputs, one-shot epoch/nonce and monotonic ledger, freshness limits and crash/rollback denial. The tests reject unknown routes, counter overflows, bool-as-int, missing/role-swapped signatures, wrong GPU context owners, stale replay and consumed nonces. Artificial successful fixtures are explicitly synthetic; `production_admissible=False` and `manager_lease_proof=None` are unconditional.

## Physical blockers

No root/driver-backed all-listener observer or restart/admission fence, complete HTTP/SSE inflight snapshot, genuine GPU UUID PID/starttime owner attestor, independent privileged role-signing keys or durable external antirollback authority exists. This is contract design and synthetic CPU testing, **not** evidence that all ingress paths have been drained or the two GPUs transferred. Original production lease replay/nonce XFAIL remain open. No model, CUDA kernel, service or networking was touched.
