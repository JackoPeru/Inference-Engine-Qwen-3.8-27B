# HERMES-27 — physical ingress attestation contract (CPU/OFFLINE)

**Date:** 2026-11-14. **Production verdict: NO-GO / BLOCKED.**
**Ownership:** worker-20. This is a NEW, isolated, strict staging contract;
no production ingress, lease manager, networking, GPUs, SSH, services or Git
were accessed or modified.

## New artifacts and frozen source identity

| File | SHA-256 |
| --- | --- |
| `ops/ingress_attestation_contract.py` | `43E37D02A2E9FF6B21F8603EA95D617E055A73A52B0244224D054E63BF80DAB7` |
| `ops/tests/test_physical_ingress_attestation_contract_20261114.py` | `7B788FAB2462C94D2475D1E4D525590841611F09B76955F41B7E3A73B4C88768` |

Read-only existing dependencies at inspection time:

| Published source | SHA-256 |
| --- | --- |
| `ops/manager_lease_external_authority_staging.py` | `02980FC28CAE6BE33EA5B886058FC36B1425A23FE8C52E501EF80DCAF85D3FE3` |
| `ops/ingress_drain_gate_staging.py` | `30635C873310F93BBF5EA1FE33A182757B2971CFEB0590FA8CCCA31DC81B9B48` |
| `ops/gpu_owner_attestor_staging.py` | `A2D6CEAA90654CBF707F3E48CC71C4E0CD6D4FB70488A30584AA9428C824E0F3` |

The report's own SHA-256 is supplied separately at handoff because putting
it inside this report would make its hash self-referential. Prime must verify
the actual published/integrated repository provenance independently; this
worker performed no Git operation.

## Contract, data flow and strict denies

1. The authority-role signature binds a route inventory to the same boot ID,
   manager instance, lease ID, epoch, fencing token, nonce and inventory
   revision. The manifest binds the six currently known physical ingress /
   control surfaces from `REQUIRED_ROUTES`: UniNote Tailscale TCP forward
   (8001), direct Tabby loopback listener (8000), dashboard, Hub,
   Laya/systemd direct-restart control, and other local/Tailscale clients.
   Each surface has a unique listener token and pinned PID/birth tick/generation
   plus an endpoint-scope descriptor. Unknown/unlisted listeners and admission
   paths must have literal integer zero counts. No caller-set boolean can
   establish exclusive physical ingress.
2. Each of the six ingress-role signatures independently binds its exact
   route ID, listener token, PID/start ticks/generation, the SHA-256 digest of
   the authority-role manifest and current lease/boot/epoch/nonce. Every HTTP
   request, SSE stream, queued HTTP or job, upstream inflight job, partial
   request, unparsed bytes, keepalive/open connection, unknown connection, new
   admission and unfenced restart count must be **exact integer zero**.
   Missing, duplicate or unknown routes deny.
3. A distinct GPU-owner-role signature binds both pinned GPU UUID/PCI
   identities, owner PID/start ticks/boot/cgroup, exact known driver contexts,
   and zero unknown driver/device/process inventories to the same signed
   manifest and lease incarnation. GPU observations must follow drained
   route observations. The tabby cgroup path is pinned; HERMES or other
   contexts are refused. This schema is conservative and synthetic.
4. All signed observations must be at most five seconds old in both wall and
   boot-scoped monotonic time and never from the future. The manifest-to-route
   and route-to-GPU monotonic skews are limited to two seconds. Verify the
   exact canonical JSON snapshot through pinned, role-separated signature
   callbacks; accept only verifier return value `None`, never a trust boolean.
   During the external atomic operation, reject clock rollback and expiry
   of the *oldest* validated observation.
5. `AtomicConsumptionStore.atomic_update` is only an abstract interface
   requiring serializable, durably committed, anti-rollback state. The
   candidate consumes the entire lease nonce once and requires a strictly
   consecutive high-water epoch; retries, old proof reuse, skipped epochs,
   nonce replay after restart, overflow and exhausted bounded nonce history
   deny. An ACK lost after commit is ambiguous and must deny; retrying a
   consumed nonce denies as well. Missing/corrupt state denies.

The sole successful synthetic path returns:

```json
{
  "status": "BLOCKED",
  "synthetic_candidate": "SYNTHETIC_VALIDATED",
  "production_admissible": false,
  "manager_lease_proof": null,
  "reason": "no privileged physical ingress/GPU issuer or antirollback anchor"
}
```

This result cannot be passed as the existing manager's `scope=all-tabby-clients`
proof. The module does not emit or convert to `ManagerLeaseV1` proofs, does
not reserve an existing external authority lease, and never issues a
production-admissible response. The HMAC keys and ledger used in tests are
explicitly **SYNTHETIC** fixtures, with no privileged trust.

## Validation

On Windows Python 3.12, launched with `-B` (no bytecode artifacts):

```powershell
& 'C:\Users\matte\AppData\Local\Programs\Python\Python312\python.exe' -B -m unittest ops.tests.test_physical_ingress_attestation_contract_20261114 -v
```

**23/23 PASS, 0 failures, 0 errors**, elapsed 0.056 s. AST parsing of both
new Python files: **PASS**.

Negative tests include all 12 per-route nonzero counters, missing/unknown/
duplicate routes, unknown inventory/restart path, listener PID/starttime/token
drift, both GPU UUID/PID/starttime/boot/context identity, unknown GPU owner
inventory, unsigned/role-swapped signatures, wrong lease/epoch/manifest/nonce,
future/expired/contradictory monotonic observations, literal bool in integer
fields, no verifier/store, corrupt or stale store, same-epoch and rotated
epoch nonce reuse, crash with commit-but-lost-ACK, rollover/exhaustion,
and timeout/clock rollback *during* atomic mutation.

Two in-memory source mutations remove (a) the signed route nonzero queue
counter deny and (b) GPU PID birth-tick comparison. Each makes its
corresponding otherwise passing negative test fail, verifying that the tests
exercise source checks. Neither mutated source is written to disk.

## Exact physical blockers — no synthetic promotion

- There is no privileged root/kernel issuer that can discover and physically
  fence **all** Tabby listener paths (including raw UniNote Tailscale forward,
  direct localhost, dashboard/Hub, unknown local clients, Laya direct
  systemctl commands and independent systemd restart policies). Signed lists
  or `unknown_routes=0` do not prove inventory completeness.
- There is no proven external zero-HTTP/SSE/inflight/queue snapshot coupled
  to a durable *admission fence* that prevents a new request/stream or a
  listener restart between observations, proof consumption, and GPU handoff.
- There is no privileged complete CUDA/MPS/graphics/kernel/namespace
  enumerator for both GPU UUIDs with independently pinned PID start times,
  boot identity, and continuous ownership exclusion.
- There is no pinned, physically protected, role-separated
  authority/ingress/GPU signing infrastructure and no external durable
  serialized monotonic anti-rollback issuer/store that survives crashes,
  arbitrary valid-journal replay and host reboot.
- No production code consumes this new protocol or enforces its assumptions.
  The existing external-authority staging state machine and this contract
  have not been connected to a real privileged issuer. A fake
  `ExternalSignatureVerifier` or `AtomicConsumptionStore` always leaves
  production admissibility **false**.

**Integration decision:** preserve NO-GO until a physical privileged issuer,
route admission interlocks, GPU owner exclusion, externally trusted high-water
and signed source-of-truth evidence exist. No numerical, GPU, performance or
live-service claims result from these CPU-only tests.
