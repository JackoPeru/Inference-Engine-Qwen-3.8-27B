# HERMES-27 — external monotonic lease authority contract (CPU/offline staging)

**Date:** 2026-11-14. **Physical deployment verdict: BLOCKED / NO-GO.**
Owned new files only:

- `ops/manager_lease_external_authority_staging.py`
- `ops/tests/test_external_monotonic_proof_staging.py`
- `reports/LEASE_EXTERNAL_AUTHORITY_STAGING_NEXT.md` (this report)

The reported remote Git HEAD at the start of this assignment is
`fe95ae70` (prime supplied); this is **not independently verified** by
the Windows `/saas/Inference Engine Qwen 3.8 27B/hermes-27` source
copy, which has no `.git` directory. No Git, network, SSH, inference,
CUDA, NVML, live ingress, systemd, services, privileged issuer or GPUs
were accessed or modified. Original manager/deploy/runtime sources remain
outside this worker's edit scope.

## Frozen source hashes (SHA-256)

| File | SHA-256 |
| --- | --- |
| NEW `ops/manager_lease_external_authority_staging.py` | `02980FC28CAE6BE33EA5B886058FC36B1425A23FE8C52E501EF80DCAF85D3FE3` |
| NEW `ops/tests/test_external_monotonic_proof_staging.py` | `4459185BEC06C01BBDD6A1C3AB017FA89F46EFB9BC7D970985A266C70DF54028` |
| read-only `ops/production_deploy.py` | `D5F77BF852A9ABFAEB9EC674864D42F437933B2496D7C5BD8E173D87F143B36A` |
| read-only `ops/manager_lease_core.py` | `2C00C5C4BF67EB3D44216142AD48C99617725ED0F84E15A32A0620139F30A213` |
| read-only `ops/manager_lease_runtime.py` | `CBB0D8A274720C6A65159A8C0BF9756F2A3EFB0B878D9E8322CCE0637C37EAD2` |

This report's own final SHA follows after file creation in the handoff.

## New fail-closed contract

The module is stdlib-only and **has no production store, signer, service
adapter, or system-trust implementation**. It defines two interfaces:

1. `ExternalAtomicStore.atomic_update`: one externally serialized,
   independently durable read-modify-write of `LedgerState`. Commit
   must precede ACK; unavailability, corruption, rollback, or uncertain
   commit outcome must fail closed. Caller-provided `trusted=True`,
   in-memory locks and ordinary local files do not establish this
   external authority.
2. `ExternalSignatureVerifier.verify(role, canonical_message, signature)`:
   a pinned signature verifier for **three distinct roles**:
   `authority` (signed reservation/highwater), `ingress` (all-route
   zero-inflight/SSE) and `gpu-owner` (restoration). The method must
   raise on rejection and return **exactly `None`** on successful
   verification; a boolean `True` is explicitly rejected. The real
   privilege and coverage of each signer must be established
   independently; implementing this interface cannot confer trust.

`StagingAuthority.status()` **always** returns
`status=BLOCKED`, `production_admissible=False`, and
`manager_lease_proof=None`. Synthetic reservation, ingress
consumption and restoration receipts also report
`production_admissible=False`, never produce a
`hermes-manager-lease-v1` grant, and fail the unmodified
`manager_lease_core.require_ingress` validator. There is no
`allow_production` switch.

Signed claims are canonical JSON with role/domain separation, strict
field sets, bounded size, 32-hex boot/manager/lease/nonce identities,
monotonic `epoch` and `fencing_token`, and a finite ten-second
timestamp. Reservation must atomically confirm the independently stored
highwater equals the authenticated manager journal's current counter,
then increment epoch/token **exactly once** and persist the nonce before
responding. A signed `ingress-closed` proof must match the reserved
boot, manager instance, lease, epoch, token, and nonce, with
`blocked=True`, `inflight=0`, `active_sse=0`,
`unknown_connections=0`, and a route-manifest hash/epoch independently
signed by the reservation authority. Consumption is a single atomic
state transition. Restoration requires separately signed two-GPU
owner evidence and a distinct unspent nonce.

The external ledger keeps a non-pruned nonce history; at capacity
(4096 entries) or 63-bit fencing/epoch exhaustion it denies further
admission, without wrapping or deleting consumed nonces. The reservation
requires capacity for **both** reservation and restoration nonces.
New process incarnations cannot silently continue a previous process's
pending or consumed lease: they require a future explicit external
reconciliation mechanism.

## Two prior open gaps: synthetic contract coverage

**Previous XFAIL A (older valid HMAC journal after restart):**
a synthetic external ledger advances to token 2 and a new authority
instance receives a previously genuine local journal counter 1.
The reservation fails with `journal highwater mismatch` and does not
roll back the external token 2. Test-only SQLite reopen proves a
separate local database retained the last token. This **does not**
remove the original XFAIL in `manager_lease_core.py`: that core
still lacks the external highwater integration, and SQLite can itself
be copied or rolled back. Real independent antirollback storage and
privileged journal binding remain absent.

**Previous XFAIL B (signed ingress proof reused across leases):**
the exact signer-signed observation is accepted once, then rejected on
an immediate duplicate, and rejected again for a later lease even
inside the original ten-second freshness interval. Wrong boot,
manager, lease, epoch, fencing token, nonce, issuer role, route hash
or route epoch is rejected. Test-only separate HMAC keys simulate
role separation; they are **not a physically privileged issuer**.
The original `ManagerLeaseV1._require_ingress/_require_proof` still
lacks those bindings and remains unchanged.

## Red/green CPU evidence

Baseline **RED**: before creating the module, the new test import
raised `ImportError: cannot import name
'manager_lease_external_authority_staging'`. After implementation
and test fixes the final run was **22 tests PASS, 0 FAIL, 0 ERROR,
0 XFAIL/XPASS** on local Windows Python 3.12. Both new files passed
`py_compile`.

```powershell
& 'C:\Users\matte\AppData\Local\Programs\Python\Python312\python.exe' -m unittest ops.tests.test_external_monotonic_proof_staging -v
& 'C:\Users\matte\AppData\Local\Programs\Python\Python312\python.exe' -m py_compile ops/manager_lease_external_authority_staging.py ops/tests/test_external_monotonic_proof_staging.py
```

Checks cover absent issuer/store, rejection of boolean providers and
boolean verifier success, signed/tampered/unknown-role claims,
manifest-hash mismatch, active SSE/unknown routes/inflight counters,
stale/future observations, independent process identity and boot
binding, nonce replay within and across leases, synthetic partial
commit failure before and after durable update, fresh object
construction after a crash, two simultaneous consumes with exactly one
winner, corrupted/unavailable ledger, nonce-log/sequence exhaustion,
and refusal to feed synthetic receipts into the existing manager.

Two tests additionally use a **disposable local SQLite fixture** outside
the manager's journal with `BEGIN IMMEDIATE` and
`PRAGMA synchronous=FULL`; they recreate separate store connections
and authority objects from the committed database file. One confirms
spent nonces/highwater persist after a completed cycle; one confirms a
previously consumed but still in-flight lease remains fenced after
reconstruction. These are controlled reopening simulations,
**not** an actual system crash, TPM antirollback test or physical
all-route drain observation. Pure in-memory tests simulate ambiguous
commit acknowledgements. No GPUs, CUDA or real connections are used.

## Release blockers

- **External monotonic root:** an independently protected highwater
  anchor must be provisioned outside the manager journal, with durable
  cross-process atomicity, nonce history and explicit migration or
  operator recovery. SQLite is not a hardware monotonic counter, TPM,
  enclave or other antirollback authority.
- **Physical ingress issuer:** inventory and actually interlock every
  Tabby ingress path, including local, Tailscale, reverse proxies and
  existing SSE requests. A signed JSON assertion and a hash field do
  not prove physical completeness.
- **External signer separation:** pin real verifier keys to privileged
  signers for reservation, ingress and owner proof. The test uses
  independent HMAC secrets only as synthetic role fixtures, and
  `ExternalSignatureVerifier` can be implemented incorrectly by a
  caller; the unconditional production BLOCKED verdict accounts for
  this.
- **Runtime integration:** the existing production manager must be
  reviewed and integrated with the external journal counter/nonce
  authority atomically. The old two production-source XFAILs are still
  open; this new staging code is not wired into either manager or
  deploy. A crash after external reservation but before manager grant
  must require independently attested reconciliation.
- **GPU and restart interlock:** a privileged two-GPU owner source,
  PID/boot fencing and independent systemd/Laya restart prevention
  remain required before any allocation or Tabby restore.

**Final decision: staging contract tests PASS, real physical authority
absent, production BLOCKED / NO-GO.** No success receipt here can
authorize GPU use or change the user's live inference services.
