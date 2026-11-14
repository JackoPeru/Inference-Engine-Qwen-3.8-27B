# HERMES-27 deployment lease / external authority — CPU negative witness

**Date:** 2026-11-14. **Decision:** **NO-GO for physical cutover, GPU access, or lease admission.**
Exclusive owned additions:
`ops/tests/test_deployment_lease_atomicity_next_contract.py` and this report.
The manager/deploy/attestor/ingress source was inspected read-only; no existing
source, service, unit, Git index, model, ingress route or GPU was changed.
Source copy: Windows `/saas/Inference Engine Qwen 3.8 27B/hermes-27`,
which has **no .git metadata**. Prime reports published Ubuntu/GitHub HEAD
`6da5f5334b17e7fc839296b8c213220d7d79e480`; this local audit
does **not** independently verify that Git provenance or Ubuntu runtime.

## Frozen SHA-256 (Windows source copy)

| File | SHA-256 |
| --- | --- |
| `ops/production_deploy.py` | `FE5D5081293422F7107B2E02F63A3C95D3968C2EA0D3C02CCB913E9D9E6105F3` |
| `ops/manager_lease_core.py` | `2C00C5C4BF67EB3D44216142AD48C99617725ED0F84E15A32A0620139F30A213` |
| `ops/manager_lease_runtime.py` | `CBB0D8A274720C6A65159A8C0BF9756F2A3EFB0B878D9E8322CCE0637C37EAD2` |
| `ops/ingress_drain_gate_staging.py` | `30635C873310F93BBF5EA1FE33A182757B2971CFEB0590FA8CCCA31DC81B9B48` |
| `ops/gpu_owner_attestor_staging.py` | `A2D6CEAA90654CBF707F3E48CC71C4E0CD6D4FB70488A30584AA9428C824E0F3` |
| **NEW** `ops/tests/test_deployment_lease_atomicity_next_contract.py` | `A47817290DC483A3248D18C9711677276410BDBDAEF3C82FFC39EA84CD72F7BA` |

## Actual verification

Command (local Python 3.12):

```powershell
& 'C:\Users\matte\AppData\Local\Programs\Python\Python312\python.exe' -m unittest ops.tests.test_deployment_lease_atomicity_next_contract -v
& 'C:\Users\matte\AppData\Local\Programs\Python\Python312\python.exe' -m py_compile ops/tests/test_deployment_lease_atomicity_next_contract.py
```

**9 tests executed, 6 ordinary PASS, 3 `expectedFailure` (XFAIL),
0 unexpected FAIL, 0 ERROR, 0 XPASS.** `py_compile`: PASS.
The 3 XFAILs are **known, reproducible safety contract violations**:
they must not be counted as 9 secure-path passes. They are explicitly
marked XFAIL because modifications to lease/deploy production cores belong
to other owners and were not authorized in this worker.

For Windows-only journal tests, a constrained disposable-fixture reader
compensates for non-POSIX file modes and CRLF conversions; it rejects
missing, symlinked, non-file and oversized synthetic journals. It does
not weaken HMAC validation or proof verifier rules. On Linux the unmodified
secure journal reader is exercised. No new Linux run was performed here.

## Source-anchored counterexamples requiring external authority

**XFAIL A — replay of an *authentic old* signed journal across reboot.**
An isolated test grants lease 1, releases it, saves the HMAC-valid
`counter=1,lease=null` journal, grants and releases lease 2, then restores
the original valid bytes before reconstructing a manager with a new
instance ID and bootstrap disabled. The new process loads the earlier
journal without fault/recovery because `ManagerLeaseV1._load`
(`manager_lease_core.py:227-295`) validates HMAC and internal fields
but has **no independently persisted monotonic high-water anchor**.
The live process's `_persisted_hash` would detect replacement, but
its in-memory expectation disappears on process restart
(`_verify_store:201-225`). The assertion that restart must remain
fenced fails. This can roll back a previously issued fencing counter
and create duplicate tokens after a later acquire. A journal HMAC alone
cannot establish anti-rollback, even if the secret key remains private.

**XFAIL B — signed ingress observation reused for another lease.**
A mock issuer signs a closed, zero-inflight ingress observation. Lease
1 acquires and releases. One second later lease 2 acquires using the
*exact same issuer-signed proof*, within the configured ten-second
freshness window. `require_ingress` checks age, scope and counters
(`manager_lease_core.py:54-67`), and `_require_proof` checks the
configured verifier (`:359-377`), but no lease ID, nonce, boot ID,
epoch or one-shot consumption binds the proof. The required rejection
assertion fails. The fixture's issuer is synthetic; a future genuine
privileged issuer must implement and enforce those bindings independently.

**XFAIL C — lease guard can accept an unexpected Tabby restart.**
With a simulated supervisor `phase=hermes-watch`, active lease,
Tabby service active and manager `current_state=LLM_READY`, the
`production_deploy.lease_guard` call succeeds. Its expected state
derives from Tabby's **observed** activity
(`production_deploy.py:316-324`) and does not reject Tabby running
in the Hermes-owning phase. `LeaseRuntime.current_state` also returns
`LLM_READY` for an active Tabby service under the fence
(`manager_lease_runtime.py:98-113`). `supervise` separately calls
`service_active` and eventually rejects Tabby revival
(`production_deploy.py:744-746,756-763`), but that periodic check
is **detection after the fact**, with no atomic external prohibition
against a systemd/Laya restart. This simulation does not show that
a real systemd restart occurred or that GPUs overlapped.

## Strict passing negative gates

1. An unconfigured independent issuer rejects otherwise well-formed
   acquisition. `LeaseRuntime._trusted_ready/_configured` and
   `ManagerLeaseV1._require_proof` remain fail closed.
2. Reconstructing a manager with an *active* authenticated journal
   retains `fenced=True` and `recovery_required=True`; a new
   acquisition is denied. Distinct from replay of an already released
   older journal (XFAIL A).
3. An issuer-signed ingress observation older than ten seconds is rejected.
   Distinct from a freshly replayed signature (XFAIL B).
4. Synthetic rollback from `lease-active` with a valid-scope checker
   reporting `inflight=1` (representing an unfinished SSE) refuses
   lease release, keeps the journal, does not call the reopen hook and
   does not modify services. No active SSE was terminated.
5. A partial-scope drain observation before lease acquisition refuses
   the rollback's reopen path and preserves its recovery journal.
6. An active Tabby process while the manager still reports
   `HERMES_MAINTENANCE` causes `lease_guard` to reject the conflicting
   state. Distinct from internally consistent but unsafe `LLM_READY`
   XFAIL C.

Existing `ops/ingress_drain_gate_staging.py` is only a state-machine
simulation and explicitly emits `production_ingress_proof=None`.
`ops/gpu_owner_attestor_staging.py` always emits BLOCKED, never a
manager-compatible physical owner proof. These outputs **cannot**
authorize runtime lease admission or recovery. An `inflight=0` mock
cannot prove an untracked local/Tailscale route or real SSE stream drained.

## Integration blocker / admission requirements

Before any live HERMES cutover, the prime/physical owner must separately
produce: a protected sole-ingress route census and controlled drain
across direct, gateway and Tailscale clients including SSE; a privileged
fresh boot/epoch/lease-bound issuer and independent anti-rollback
high-water storage; prevention of every manager-external systemd/Laya
Tabby restart while Hermes may hold either card; and a privileged,
complete two-GPU context/PID/starttime attestation synchronized with
restoration and process admission. The owner must explicitly grant
physical operations. These controls are not implemented by this test
fixture and are not inferred from source inspection.

**Release gate: strict CPU rejection paths pass; three independently
reproduced gaps remain XFAIL; physical deployment and GPU operations
remain NO-GO.** The prime should rerun the focused suite on Linux after
integrating these two new files and separately reconcile source hashes
with the published Git HEAD. No unchanged full suite was repeated here.
