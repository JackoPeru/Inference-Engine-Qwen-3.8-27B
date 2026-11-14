# HERMES-27 — atomic authority / ingress composition (CPU-only staging)

**Date:** 2026-11-14. **Verdict:** **SYNTHETIC CPU contract PASS; production BLOCKED / NO-GO.**

Owned new files only:

- `ops/staging_authority_ingress_composition.py`
- `ops/tests/test_authority_ingress_composition_cpu.py`
- `reports/AUTHORITY_INGRESS_COMPOSITION_CPU_NEXT.md` (this report)

The published `ops/manager_lease_external_authority_staging.py`,
`ops/ingress_attestation_contract.py`, and their 22/23-test suites
were read and exercised without modification. The assignment identifies
Git `36b4b113ea74fb45749215b4dba4541456153bf1` as published main;
the Windows `/saas` source copy has no Git metadata, so this is
**prime-supplied provenance, not a Git verification from this worker**.
No Git/SSH/network/GPU/NVML/systemd/live ingress/services were touched.

## Frozen SHA-256 (Windows source copy)

| Path | SHA-256 |
| --- | --- |
| NEW `ops/staging_authority_ingress_composition.py` | `20D9C41A2875A72EFEDEA405DF7712C3A394E67ACEC89A029A5C4893117548B8` |
| NEW `ops/tests/test_authority_ingress_composition_cpu.py` | `F9B3BFF7EE5E106387349D143A204AEBD8A78C36B419984A4DEB2C6FB16A1379` |
| read-only `ops/manager_lease_external_authority_staging.py` | `02980FC28CAE6BE33EA5B886058FC36B1425A23FE8C52E501EF80DCAF85D3FE3` |
| read-only `ops/ingress_attestation_contract.py` | `43E37D02A2E9FF6B21F8603EA95D617E055A73A52B0244224D054E63BF80DAB7` |
| read-only `ops/tests/test_external_monotonic_proof_staging.py` | `4459185BEC06C01BBDD6A1C3AB017FA89F46EFB9BC7D970985A266C70DF54028` |
| read-only `ops/tests/test_physical_ingress_attestation_contract_20261114.py` | `7B788FAB2462C94D2475D1E4D525590841611F09B76955F41B7E3A73B4C88768` |

This report's own final SHA is recorded in the worker handoff after writing.

## Independent-source audit: why simple proof handoff fails

The existing external authority has `LedgerState` with independent
`highwater`, `phase`, `spent_nonces` and an
`ExternalAtomicStore.atomic_update` API
(`manager_lease_external_authority_staging.py:63-98,271-294,311-388`).
`StagingAuthority.reserve` consumes the reservation nonce,
`consume_ingress` advances that state to `drain-verified`.

The physical ingress validator has a separate `ConsumptionState`
highwater/nonce ledger and a separate
`AtomicConsumptionStore.atomic_update`
(`ingress_attestation_contract.py:125-145,341-349,352-431`).
`assess` cryptographically validates an authority-signed inventory,
all six route proofs and a GPU inventory, then consumes its nonce.
Each validator reports `production_admissible=False` and
`manager_lease_proof=None`.

Even if the same synthetic nonce has been independently consumed in
**both** stores, neither committed operation proves both decisions
belong to **one atomic transaction**. Independently signed or
independently consumed evidence is therefore not transferable. A naive
`assess(...)` followed by `consume_ingress(...)` could leave one
store committed and the other not after a crash. A boolean PASS from
either operation cannot authorize GPU allocation.

## Strict proposed composition contract

`CompositionState` holds both *published* immutable ledger states
and a paired manifest/nonce record. `AtomicCompositionStore` defines
one external, serializable atomic update interface without supplying a
production backend. Under **one injected outer atomic callback**,
`compose` performs:

1. Deep-copy the caller's signed envelopes and compare their
   `boot_id`, `manager_instance_id`, `lease_id`,
   `epoch`, `fencing_token`, `nonce`, physical
   `manifest_epoch`, and exact signed manifest SHA-256
   (`staging_authority_ingress_composition.py:75-124`).
2. Require a **pristine pair of ledgers** and initial `journal_counter=0`,
   `epoch=fencing_token=1` (`:126-143`). Existing
   separately consumed nonce, externally reserved lease, stale
   highwater, or prior paired consumption is **BLOCKED**, including
   after an object/process reconstruction. Subsequent epochs need
   a separate privileged recovery/rotation protocol; this staging
   implementation does not invent one.
3. Create private temporary adapters that satisfy the original
   `ExternalAtomicStore` and `AtomicConsumptionStore` interfaces,
   but stage their updates in memory only. Execute the unchanged
   `StagingAuthority.reserve`, the unchanged
   `ingress_attestation_contract.assess`, and the unchanged
   `StagingAuthority.consume_ingress` inside the same parent
   callback (`:145-166,173-266`).
4. Verify both staged nonce histories, current lease/manifest/epoch,
   exactly two authority state transitions and one physical consumption.
   Return the paired state for **one outer commit**. A missing/doubled
   callback, invalid ACK, clock drift or exception fails closed.
   If the backend commits then loses its acknowledgement, the
   caller receives `CompositionDenied` and retries cannot reuse the
   committed nonce.

The authority verifier and physical verifier must be different
`ExternalSignatureVerifier` objects with independent signer roles.
The mock uses distinct HMAC keys; these keys do not establish physical
signer provenance. Each published validator still performs its
own signature checks and strict HTTP/SSE/queue/GPU data checks.

The **only** successful return value from `compose` is a
synthetic `status=BLOCKED`, `production_admissible=False`,
`manager_lease_proof=None` result, with no lease ID, fencing token,
renewal capability or privileged scope. `status()` is
unconditionally BLOCKED. The parent atomic store is only an
interface. The test implementation has a local mutex and ordinary
RAM; it is NOT a durable root-owned authority.

## Red/green, source-mutant and combined CPU evidence

Local Windows Python 3.12, source copy as frozen above:

```powershell
& 'C:\Users\matte\AppData\Local\Programs\Python\Python312\python.exe' -m unittest ops.tests.test_authority_ingress_composition_cpu ops.tests.test_external_monotonic_proof_staging ops.tests.test_physical_ingress_attestation_contract_20261114 -v
& 'C:\Users\matte\AppData\Local\Programs\Python\Python312\python.exe' -m py_compile ops/staging_authority_ingress_composition.py ops/tests/test_authority_ingress_composition_cpu.py
```

**Actual outcome: 65 / 65 PASS, 0 errors, 0 failures, 0 skipped**:
20 new composition tests + 22 published external-authority tests +
23 published physical-ingress tests. Both new Python files
`py_compile` PASS. Earlier new-test green: 18/18; two additional
callback-forgery tests added and included in final 20/20.

The new tests cover a successful **synthetic** one-callback pairing,
same-lease replay and rotated lease replay, separately consumed ingress,
both separately consumed ledgers with equal highwater (still denied),
boot/manager/lease/epoch/fence/nonce drift, separately signed manifest
SHA/epoch mismatch, changed physical manifest, unsigned/incorrect-role
signatures, boolean-verifier rejection, active SSE, HTTP queues and unknown
connections, crash before parent commit, lost acknowledgement after
parent commit, exception after staging authority state but before ingress
validation, concurrent duplicate attempts (exactly one synthetic
commit), clock expiry during callback, malformed/forged parent ACK, and
a store returning ACK without calling its mutation.

The in-memory **source mutation** test in
`test_source_mutant_cross_protocol_lease_binding_is_caught` compiles a
modified copy of `_matching_context` which skips the two external
ticket lease-ID comparisons while retaining the physical manifest
comparison. Two individually correctly signed external claims then
refer to lease F, while a correctly signed physical package refers to
lease C. The real composition rejects the mismatch; the mutant
accepts a synthetic pair. The test proves the added cross-protocol
guard is necessary. Mutant exists only in test memory.

These tests model crashes and concurrency; no actual process kill,
physical SSE stream, external issuer, kernel GPU context, reboot or
hardware antirollback store was exercised. Test PASS is **not**
production feasibility proof.

## Non-negotiable blockers

1. No independently authenticated external `AtomicCompositionStore`
   implementation exists. Two separately durable stores cannot achieve
   the described atomic commit without an externally enforced shared
   transaction or recovery protocol.
2. No privileged physical source verifies complete Tabby ingress
   (including active SSE), direct/Tailscale/gateway routes, systemd/Laya
   restart interlocks, or physical two-GPU ownership. The published
   issuer/store hooks remain abstract and synthetic.
3. The current new composer intentionally supports only a pristine
   initial transaction. It refuses later epochs and any crash residue
   without externally attested, owner-reviewed recovery. It is not a
   production rotation mechanism.
4. The original manager/deploy core does not consume these composite
   receipts; no new wire protocol, lease grant, deploy hook or service
   setting was installed. Source-only positive fixtures must not be
   interpreted as a physical grant.

**Release gate: CPU composition witness PASS; real multi-ingress
authority/store and hardware owner grant absent; physical operation
remains BLOCKED / NO-GO.** Prime owns any future integration or
publication after frozen-source review.
