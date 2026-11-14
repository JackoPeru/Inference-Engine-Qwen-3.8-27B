# Manager lease crash / journal hardening — offline staging only

Date: 2026-11-14. Owner: worker-3. **NOT installed. NOT production GO.**

## Audited failures and fixed state transitions

At the original `ops/manager_lease_core.py`:

1. `_load` treated a missing `lease.json` as a clean first install. After
   an active lease and journal deletion, a new process could start unfenced.
2. `renew` unconditionally assigned `expires_at_unix = clock() + 90`.
   A backwards wallclock adjustment by 30 seconds shortened the expiration.
3. `require_ingress` / `require_gpu_safe` validated structural JSON
   fields, scope and freshness, but **not the provenance** of a supplied
   dictionary. A fabricated correctly shaped proof could release the fence.

The protocol stays `hermes-manager-lease-v1`; request/ack names, fencing
tokens and GPU Manager state transitions are unchanged. Additional optional
constructor keywords are now required to **authorize** state transitions:

```python
ManagerLeaseV1(
    "/var/lib/hermes-gpu-manager/lease-v1.json",
    journal_key=privately_provisioned_bytes_32_or_longer,
    proof_verifier=trusted_proof_verifier,  # (kind, proof) -> bool
    bootstrap_authorized=False,             # only true for reviewed initial bootstrap
)
```

These examples are *interface shapes*, not instructions to install keys or
enable services. The key must be independently provisioned, protected from
the journal's writer and loaded through a trusted integration. Do not place a
literal key in source, log files, environment variables, CLI flags or the
journal directory. `proof_verifier` must validate the real root-owned
checker/signature and fresh PID/boot/GPU evidence, not merely return `True`
for a valid-looking dictionary. The offline tests use a test-only HMAC
verifier; **no real attestor is connected**.

### Guarded boot and persistence

- `journal_key=None` is **fenced/faulted** even on initial startup.
  Invalid keys shorter than 32 bytes are rejected at construction.
- Missing journal **and** directory are also fenced by default. Initial
  bootstrap requires `bootstrap_authorized=True`, a private key, and a
  genuinely new absent directory. Once the first journal is durably written
  the in-process bootstrap permission is consumed. An existing empty
  directory never counts as first install.
- The journal is authenticated with HMAC-SHA256 over stable canonical JSON.
  Loaded journal MAC, protocol version, counter, lease owner/scope/id,
  fencing token, expiration and release record must be consistent.
  A valid-looking forged JSON file without the private MAC cannot pass boot.
- Journal reads require a private regular file of size <=16 KiB, owner matching
  the manager process, no group/world permissions, no symlinks/hardlinks,
  and the same device/inode before and after reading. The parent directory
  must be private, not a symlink. Atomic writes use an exclusive private
  temp file, fsync(file), rename, fsync(parent), and no-follow semantics.
  A stale `.writing` file or unsafe hardlink fails closed rather than
  truncating an untrusted target.
- The process pins a hash of the last successfully persisted journal and
  re-verifies that the same durable contents remain present before reporting
  an unfenced state or accepting a mutation. Deletion, external edit, unsafe
  permission changes, or replacement races set a permanent local fault.
  On restart a valid **active** journal always requires recovery; missing or
  invalid journal never authorizes a new lease.
- A write failure after any in-memory state change marks the manager
  faulted/fenced and never acknowledges a lease. A following restart reads
  the previous committed active journal and requires independent recovery.

### Time and proof guards

- The last observed transition wallclock is persisted as
  `last_clock_unix`. If active-lease wallclock moves backwards by more than
  one second, `recovery_required` is durably set and no renew is admitted.
  An adjustment within the tolerance still uses
  `max(previous_expiry, now + TTL_SECONDS)`, so renewal **never shortens**
  a previously acknowledged expiration. Ordinary expiry still retains the
  fence and requires independently verified recovery.
- A restart with an active lease is always fenced/recovery-required. There
  is no attempt to use `time.monotonic()` across process restarts or reboots:
  its origin cannot be assumed persistent across boot identity changes.
- The core checks the existing structural requirements first, then calls
  `proof_verifier(kind, proof)`. The supported `kind` values are
  `ingress`, `gpu-free` and `gpu-restored`. The verifier must return the
  literal boolean `True`; absent verifier, exception, or any other result
  refuses the operation. No unchecked shape-only proof can drive acquire,
  renew, release or fenced recovery.

## CPU-only proof and reproducer

Exactly the following files changed (no changes in installed manager,
runtime adapter, ingestion, owner attestor, service units or live GPU state):

- `ops/manager_lease_core.py`
- `ops/tests/test_manager_lease_core.py`
- NEW `ops/tests/test_manager_lease_boot_journal_regression.py`
- NEW `ops/MANAGER_LEASE_CRASH_FIX.md`

From a disposable Linux scratch copy with these modules:

```bash
PYTHONPATH=/tmp/h27-worker3-lease-core-20261114 \
  /opt/hermes/runtimes/exllamav3-venv/bin/python \
  -m unittest discover -s ops/tests -p 'test_manager_lease*.py' -v
```

Final offline result: **55/55 PASS**, ~0.6 s, no skips. Python
`py_compile` also passed for all three Python files. This comprises
32 existing lease-core cases and 23 new regression cases. The new cases
cover clean authenticated bootstrap and counter recovery, unprovisioned
restart, deletion of an active or released journal, deletion of both journal
and parent directory, forged/invalid-MAC journal, wrong/missing trust key,
syntactically valid untrusted ingress evidence, -30s rollback and -0.5s
renewal, a direct -30s renew without a preceding status read, symlink/
hardlink/temp-file attacks, permissive file/directory mode, same-content
inode replacement race, verifier exceptions and truthy-but-nonboolean
responses, and simulated atomic-rename **and post-rename directory-fsync**
failures followed by restart. All inputs and credentials were synthetic/
test-only.

## Interop migration required; explicit NO-GO

**DO NOT deploy this source as a drop-in runtime replacement.**
Current `ops/manager_lease_runtime.py:39-41` constructs `ManagerLeaseV1`
without `journal_key`, `proof_verifier` or a reviewed one-time bootstrap.
With this hardening, that prototype will deliberately remain fenced. The
operator and prime must provision an independent root-protected trust key,
authenticated checker provenance, durable first-boot ceremony and a safe
migration for existing unsigned v1 journals **before** adapting runtime.
An unsigned legacy journal cannot simply be relabeled as trusted. No live
manager, credential, systemd, ingress, GPU service or runtime module was
modified by this worker.

Worker-7's unpublished
`ops/tests/test_ingress_gpu_fence_contract_interop.py` has three former
negative-characterization tests whose expectations **must be inverted**
after prime migrates their fixture to explicit test-only journal keys,
bootstrap and authenticated mock proof verification:

- G1 `test_known_manager_missing_journal_is_unfenced_on_restart_gap`:
  expect `fenced=True`, `faulted=True`, further acquire refused.
- G2 `test_known_backward_clock_renew_can_shorten_expiry_gap`:
  expect no shortened expiration, `recovery_required=True` for -30s,
  and renewal refusal.
- G3 `test_known_shape_only_proofs_can_clear_lease_without_source_provenance`:
  expect untrusted proof refused and the original lease still fenced.

Those worker-7-owned tests were **not edited or claimed passing** here.
Their other positive synthetic cases will also require explicit offline
trusted-proof fixtures; an untrusted dictionary must not be reclassified
as physical GPU evidence merely to restore a green test count.

### Residual boundaries — no false assurance

1. HMAC authenticates *contents*, not a monotonically anchored external
   epoch. A privileged attacker holding an **older valid signed journal**
   could replay it after a restart unless prime provisions an independent,
   root-protected monotonic high-water authority. Similarly, a leaked
   journal key defeats the MAC. Neither an authenticated epoch source nor
   TPM-backed anti-rollback is implemented in these files.
2. The optional bootstrap flag is caller authorization, **not proof by
   itself** that a machine has never held an earlier lease. It must be
   restricted to a reviewed one-time first-install process; never set it
   persistently in a restart unit.
3. This core still cannot prevent Laya/systemd or other unauthorized
   external processes from restarting Tabby. Full all-ingress fencing,
   GPU/PID/boot attribution, systemd restart interlocks and actual active
   lease deployment remain separate NO-GO conditions.
4. The Python RLock is intra-process only. An operator-owned singleton,
   cross-process serialization and trusted filesystem/credential isolation
   must be provided by the surrounding manager deployment.

Prime must approve any integration beyond the exact four files above.
