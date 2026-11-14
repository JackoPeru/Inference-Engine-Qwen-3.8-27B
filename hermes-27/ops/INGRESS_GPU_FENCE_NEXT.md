# HERMES-27 — ingress / GPU ownership / manager lease negative contracts

**Scope:** CPU/OFFLINE staging only. Updated 2026-11-14 after the
manager lease core/runtime hardening and explicit trust-injection migration.
**Production decision: NO-GO.**

## Exact new owned paths

- ops/tests/test_ingress_gpu_fence_contract_interop.py
- ops/INGRESS_GPU_FENCE_NEXT.md

No core/runtime/attestor/ingress production source or installed service was
modified. Tests use synthetic HTTP bytes, synthetic GPU/PID snapshots,
an injected clock, a disposable journal and two random per-test HMAC keys.
There were no network/SSH requests, root/sudo/systemctl, CUDA, NVML
hardware calls or GPU operations. The Windows workspace is a source
copy, so Git provenance and prime's integrated commit must be verified
independently by prime.

## Cross-component negative contracts

| Published implementation | Exercises | Observed gate |
| --- | --- | --- |
| ops/ingress_drain_gate_staging.py | CoverageManifest, DrainGate, journal replay, epoch, partial/unparsed HTTP, raw UniNote traffic, keepalive, bypass marker | Never emits a manager-compatible all-tabby-clients proof; unknown traffic and crash remain fenced |
| ops/gpu_owner_attestor_staging.py | Two-card UUID/PCI, NVML vs process list, cgroup, PID birth, FD completeness, snapshot race | Even consistent synthetic Tabby ownership is BLOCKED; unknown context/owner fails |
| ops/manager_lease_core.py | acquire/renew/release, ingress and GPU proof validators, fencing token, authenticated journal, expiry and recovery | Rejects untrusted shape-only proofs, deleted journal and -30s clock rollback; test-only signed fixtures allow offline state-machine branches |

The positive branch fixtures require explicitly injected, TEST-ONLY
`journal_key`, one-time `bootstrap_authorized=True` for a fresh
temporary directory, and `proof_verifier(kind, proof)`. Each test generates
independent `os.urandom(32)` journal/issuer keys and signs a synthetic
ingress/GPU dictionary with a type-bound HMAC-SHA256. This is solely a
mock issuer: **HMAC-valid test data is not independently trusted physical
GPU/ingress evidence**. Direct staging outputs, unsigned but correctly
shaped dictionaries, altered signatures, false counters and failed scope
remain rejected. All source trust gates stay enabled and unchanged.

Tested fail-closed contracts include:

1. DRAINED_CANDIDATE does not create scope=all-tabby-clients evidence.
   Missing UniNote path, unknown direct localhost client, an unparsed
   request, partial server response, or open keepalive cannot count as
   zero verified inflight.
2. A synthetic manifest with unfenced Laya restart blocks scope.
   An unknown systemd/direct restart route taints the gate. These tests
   model bypasses; they do NOT prove existing systemd/Laya permissions
   have been interlocked.
3. Unknown NVML GPU PID, incorrect UUID, unreadable/unmapped GPU device
   FD, bad cgroup, PID starttime change and a process race between
   snapshots prohibit a manager release. Even complete synthetic two-card
   Tabby evidence is refused because the owner attestor always reports
   missing independent all-context coverage.
4. Repeated identical drain calls preserve one epoch. After a process
   restart, restored and missing staging drain journals both remain
   RECOVERY_REQUIRED. Stale ingress evidence, a wrong lease token,
   incorrect manager instance ID, or another lease ID are rejected.
5. Manager expiry keeps its fence through reconstruction with an intact
   journal, and staging proof outputs cannot clear required recovery.
   After an explicitly test-signed synthetic release and reload from an
   intact journal, the new lease fencing counter increments (1 -> 2).

## Three repaired negative security assertions (G1 / G2 / G3)

The old characterization tests **passed when the core was unsafe**.
They are now **strict assertions of refusal**, against the frozen,
hardened lease core. No core logic or trust checks were relaxed.

### G1 — Journal deletion must remain fenced

Test: `test_G1_deleted_manager_journal_fences_current_and_restarted_manager`.
Acquire a synthetic HMAC-authorized lease; delete the **disposable**
journal. The current manager must become `fenced=True` and
`faulted=True`. Restart with the same test journal key, the same
authorized verifier and **bootstrap disabled**. The restarted manager
must be fenced/faulted, require recovery and reject another acquire.
Deleting a journal is never authorization to start a new epoch.

### G2 — Wallclock rollback must deny renewal

Test: `test_G2_rollback_clock_refuses_renew_and_keeps_expiry`.
Record the original signed lease's expiry and renewal token. Move
the injected wallclock back 30 seconds and call `renew` directly.
Require `LeaseRejected` (rollback), unchanged expiry and a durable
`recovery_required=True` fence surviving process reconstruction.
The core must not extend or shorten a lease across this rollback.

### G3 — Shape-only proof cannot clear the lease

Test: `test_G3_unsigned_shape_only_proofs_cannot_clear_lease`.
Acquire using a **test-only signed ingress** proof. Attempt release
with (a) an unsigned but correctly shaped ingress dictionary,
(b) unsigned but correctly shaped Tabby/GPU dictionary, and
(c) a structurally correct but forged issuer MAC. All three attempts
must raise `LeaseRejected`; the original lease must remain fenced.

The positive fixture's HMAC verifier only establishes *fixture
integrity*. It does not authenticate a live system process, actual
all-route ingress coverage or physical NVIDIA device ownership.
Production still requires an independently root-protected trust
source and a durable monotonic epoch/anti-rollback anchor; journal
HMAC alone cannot detect replay of a **previously validly signed**
journal after restart.

## Integration requirements not yet met

- Physically enforced sole ingress, including the active raw UniNote
  Tailscale TCP forward to local Tabby port 8000, dashboard, Hub and
  unknown direct local clients. No stream/keepalive/partial-byte
  tracker can cover traffic that bypasses its gateway.
- Ingress drain must atomically refuse *new* work while allowing
  existing requests and SSE streams to finish without forced
  cancellations or interruption. Keepalives, incomplete records
  and unknown connections cannot count as zero.
- External Laya direct systemctl restart of Tabby and systemd
  Restart relationships must be protected even when GPU Manager is
  dead, unresponsive or restarted.
- A privileged two-GPU attestor must resolve every CUDA, MPS, graphics,
  kernel and namespace context against stable GPU UUIDs and
  PID-starttime/cgroup identity. Deny incomplete/hidden inventories
  and enforce no new owner admission between proof and restart.
- Durable epoch/journal recovery across crashes, reboots, clock
  rollback and journal deletion must be established outside the
  in-process staging prototypes.

## Repeatable CPU-only tests

From the project root, using available local Python:

```powershell
& 'C:\Users\matte\AppData\Local\Programs\Python\Python312\python.exe' `
    -m unittest ops.tests.test_ingress_gpu_fence_contract_interop -v
& 'C:\Users\matte\AppData\Local\Programs\Python\Python312\python.exe' `
    -m py_compile ops/tests/test_ingress_gpu_fence_contract_interop.py
```

**New frozen interop test result on local Windows Python 3.12: 27/27
PASS, 0 FAIL, 0 ERROR**, including all three **strict fail-closed**
G1/G2/G3 assertions; test file `py_compile` PASS.

On Windows only, the test replaces the hardened core's *journal read
method* with a limited test-fixture adapter: native Windows file
mode reports 0666 instead of Linux 0600, and native text writes CRLF
instead of LF. The adapter rejects missing files, final-component or
parent symlinks, directories and oversized files, and normalizes CRLF
bytes for the disposable journal's SHA comparison. It does **not**
disable HMAC journal authentication, issuer verification, lease expiry
or recovery requirements. This adapter is never installed; Linux
executes the core's original secure journal-read implementation and
real private-directory mode checks. The separate Linux lease core tests
must remain a mandatory production review gate.

The **full local ops discovery** on the same Windows host was also
attempted using an in-memory, non-listening FastAPI stub because the
local Python environment lacks FastAPI. Actual result: **317 tests
executed, 11 FAIL, 28 ERROR, 0 SKIP**. These are not interop-file
failures: they originate in *other, untouched tests* that assume Linux
POSIX journal mode, fsync and newline semantics, especially
`test_manager_lease_core`, `test_manager_lease_boot_journal_regression`
and `test_manager_lease_runtime`. The stub is not real FastAPI.
Do **not** report 317/317 PASS, nor treat these platform failures as
a reason to weaken the secure lease implementation.

Earlier, prime's isolated Ubuntu CPU scratch returned 289/289 PASS
**excluding** the 27 old interop tests. That is an earlier source
snapshot, not a fresh proof of a 317/317 current Linux suite.
Prime should integrate these two reconciled files and rerun the real
full ops suite under Linux, using the actual FastAPI dependency and
unchanged hardened core/runtime, to establish the new total.

No physical GPU attestor collector, real listening socket, systemd
service, network/SSH connection, GPU, or privileged command was used
for this reconciliation. Passing only proves offline contract behavior.

**Final gate: interop source contract PASS locally; Linux integration
and physical all-ingress fencing remain production NO-GO.**
