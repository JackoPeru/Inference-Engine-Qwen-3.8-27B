# GPU Manager lease v1: manager-side prototype (PREPARED / NO-GO)

Status **13 Nov 2026**: code is isolated under `ops/`; NOTHING has been
installed under `/opt/hermes` or systemd. No real lease, ingress drain,
watchdog change, GPU workload, or production cutover was performed. The
reviewed source is the installed read-only
`/opt/hermes/gpu-manager/manager.py`, SHA-256:

`15744e3b769c0f6cd5ed26c2b85e3f282871f4a24681abd324c8cbc329a73eef`

The installed watchdog probes every ~30 s and may restore Tabby on repeated
failures, via `restore_llm_with_retries()` and `transition_to_llm()`; that
ultimately invokes `systemctl restart hermes-tabby.service`. The manager also
has `_drive()`, mode routes, `/force`, a supervisor and Character ID reaper
paths that may restore Tabby or move GPU ownership. A valid lease must fence
**all** these manager-initiated paths, not just `llm_watchdog()`.

## Reviewable deliverables

- `manager_lease_core.py`: pure Python persistent state machine. Fenced
  exclusive lease, 90 s TTL, monotonic fencing token, idempotent acquire,
  renew and release, atomic fsync journal, crash fencing and post-expiry
  recovery. Ingress and GPU safety proofs are *inputs from a trusted local
  checker*, not assertions supplied by an HTTP caller.
- `manager_lease_runtime.py`: prototype FastAPI installation on the existing
  manager app. Four routes (`acquire`, `renew`, `release`, `recover`), existing
  full-control manager Bearer role + separate local credential + strict
  loopback peer. Uses the original `_lock` to serialize with mode changes,
  and `_submit_lock` for atomic media submission admission. The patched
  central `create_job()` repeats the fence check after earlier in-flight
  HTTP requests pass through middleware, including smart/resubmit/character
  callers.
  Disables mutation endpoints, watchdog, `_drive`, GPU systemctl/unload and
  CUDA cleanup when fenced. Automatic recovery on expiry requires proof of
  no HERMES GPU owner before restoring Tabby.
- `manager_lease_adapter.py`: command-line stdin/stdout JSON transport for
  the `production_deploy.py` contract. Opens only `127.0.0.1`; no proxy,
  redirect, remote host, secret argv or credential in stdout.
- `manager_lease_patch.py`: source-hash-pinned, offline **staging generator**.
  Expects the exact audited `manager.py`, rejects unexpected anchors or
  existing output paths, syntax-compiles the resulting source and writes
  `manager.py`, two sibling modules and a unified diff to a **new directory**.
- `manager_lease_v1_proposal.patch`: reviewable unified diff against the
  pinned manager source. It contains no credentials or model files.
- `tests/test_manager_lease_{core,runtime,patch,adapter}.py`: CPU-only mock
  tests covering restart, concurrent ownership, lease expiry, storage fault,
  stale/incomplete safety evidence, auth, active jobs, safe recovery, CLI
  redirects and source drift.

No production file is touched by running the patch generator on a **local
copy** of the audited manager:

```bash
python3 ops/manager_lease_patch.py \
  --source /tmp/hermes-gpu-manager-audited/manager.py \
  --output-dir /tmp/hermes-gpu-manager-staged-v1
```

The target must not exist. The example above is a proposed offline staging
command only; never point `--output-dir` at `/opt/hermes` or systemd. The
generator will refuse any source checksum drift before writing files.

## Proposed authenticated endpoints (NOT present on live manager)

All calls are `POST` against the manager's **existing** listener on
`127.0.0.1:8643`, no new listening port. Requests from any non-loopback peer
are rejected, even if correctly authenticated. The existing manager
`require_laya` full-control Bearer is mandatory, followed by a separate
`X-Hermes-Manager-Lease-Token` header from a private file. The weaker app
key cannot grant a lease.

| Endpoint | Action |
| --- | --- |
| `/maintenance-lease/v1/acquire` | Grants a single lease only if manager `LLM_READY`, no queued/current/training/direct/gaming work, force window inactive, Tabby online, and independent ingress proof closed with zero in-flight requests |
| `/maintenance-lease/v1/renew` | Extends the current lease's 90 s TTL only for matching ID/owner/instance/fencing token and fresh closed ingress proof |
| `/maintenance-lease/v1/release` | Accepts a verified Tabby return only when HERMES pids are absent and Tabby exclusively owns both GPUs; repeat release with the same fencing tuple is idempotent |
| `/maintenance-lease/v1/recover` | Attempts crash/expiry recovery only with independent proof that ingress remains closed and the GPU owner is known; otherwise leaves the journal fenced |

`GET /status` continues to use the existing manager/app Bearer role and would
expose `watchdog_lease_contract_version: 1`, `manager_instance_id` (changes on
each process restart) and `maintenance_lease`. If local checker configuration
is absent, the version is `null` so the client fails before cutover. During a
valid lease, the reported state follows Tabby process status: `LLM_READY`
before stop, `HERMES_MAINTENANCE` while Tabby is down. After expiry/crash,
the manager reports `HERMES_RECOVERY_REQUIRED` and suppresses **every**
automatic GPU transition until independent safe recovery succeeds.

The JSON stdin contract used by `production_deploy.py` is documented in
`ops/MANAGER_LEASE_CONTRACT.md`; the command adapter is optionally configured
as three argv arrays, e.g. `[/usr/bin/python3,
/path/to/manager_lease_adapter.py, --action, acquire]` (values are shown as
conceptual argv elements, not valid JSON). Real service credentials and
environment files have not been created.

## Required local checker contracts (UNIMPLEMENTED)

These are critical dependencies, not mock scripts to install directly.

`HERMES_MANAGER_INGRESS_CHECK` must name an absolute, private executable
whose output is generated by a **trusted ingress controller**. It must
certify **every** route to Tabby, including direct loopback requests, local
apps, Tailscale and gateway connections. A parsed example:

```json
{
  "scope": "all-tabby-clients",
  "blocked": true,
  "inflight": 0,
  "observed_at_unix": 1794610000
}
```

`HERMES_MANAGER_GPU_CHECK` must be an independent privileged CPU-side probe
of both GPU process owners and actual HERMES PIDs. Before manager-initiated
recovery/restart it must attest:

```json
{
  "scope": "all-hermes-gpus-and-pids",
  "gpu_count": 2,
  "gpu_owner": "none",
  "gpu_free": true,
  "hermes_pids": [],
  "other_gpu_processes": [],
  "observed_at_unix": 1794610000
}
```

After Tabby has been safely restored, the checker must report
`gpu_owner: "tabby"`, `tabby_owns_both_gpus: true`, and empty HERMES/unknown
process lists. The manager accepts this as ownership evidence for release or
orphan-fence recovery. It **does not require empty GPU VRAM** after Tabby is
already serving; doing so would deadlock. All proofs must be at most 10 s old.
The checkers must be installed only after independent review and must never
trust evidence sent by an arbitrary lease client.

The future manager process also requires private credentials/configuration:

| Environment variable | Meaning |
| --- | --- |
| `HERMES_MANAGER_LEASE_STORE` | Durable private journal, e.g. `/var/lib/hermes-gpu-manager/lease-v1.json` |
| `HERMES_MANAGER_LEASE_TOKEN_FILE` | Restricted second-factor token for manager-side routes and client adapter |
| `HERMES_MANAGER_INGRESS_CHECK` | Trusted all-client ingress proof executable |
| `HERMES_MANAGER_GPU_CHECK` | Trusted owner/PID/VRAM proof executable |
| `HERMES_MANAGER_API_KEY_FILE` | **Adapter-only** file with existing privileged manager Bearer (not app key) |
| `HERMES_MANAGER_LEASE_URL` | **Adapter-only**, defaults to `http://127.0.0.1:8643`; only `127.0.0.1` accepted |

All secret files require private regular-file permissions. The source patch
does not create, install, log or embed credentials; staging and tests contain
only synthetic IDs and dummy tokens.

## Failure semantics and known remaining gaps

1. An active lease suppresses manager watchdog and mode transitions. A
   manager restart with persisted outstanding lease, a corrupted journal or
   an expired lease **keeps the GPU fence**, with no automatic Tabby restart.
2. The manager may restore Tabby after expiry ONLY if independent proof
   confirms ingress still blocked, no HERMES PIDs, no other GPU owners and
   both GPUs free. It then validates Tabby ownership before clearing the
   fence. If the supervised client already restored Tabby, the manager skips
   redundant `systemctl restart` and verifies exclusive Tabby ownership.
3. A failed GPU proof, uncertain process identity, missing ingress coverage,
   disk write error, changed source checksum, missing auth or in-flight
   requests keeps the system fenced. No lease may be granted with nonzero
   media queue, direct/gaming/training work or an active `force` window.
   A media submission holding `_submit_lock` also rejects lease acquisition;
   both regular and smart media submit routes and central job creation have
   an additional fence check for submissions admitted before middleware
   activated. Direct out-of-process Character ID writers remain a separate
   integration boundary that must be audited before production.
4. The persistent journal's directory must survive reboot. **Journal
   deletion/storage loss cannot be detected from an absent file alone** and
   remains a release blocker without external durable fencing or an
   independent boot-time GPU-owner gate.
5. External `systemctl` jobs outside GPU Manager, another daemon that
   restarts Tabby and direct localhost clients bypassing ingress are outside
   the manager patch's authority. Validate those paths before any live test.
6. Manager source changes must trigger re-audit of every mode/force/job,
   supervisor/cleanup and systemctl path. The patch generator intentionally
   rejects changes to the pinned source/anchors.

The manager's existing ~30 s watchdog loop, older recovery shortcuts and
automatic switching cannot be treated as a trustworthy lease until the
staged code is reviewed, all checkers are installed, a controlled CPU-only
integration suite passes and the prime/user authorize a maintenance test.

## Offline validation

```bash
python3 -m unittest discover -s ops/tests -p 'test_manager_lease_*.py' -v
python3 -m py_compile ops/manager_lease_*.py
```

The complete `ops/tests` suite includes the earlier 40 client supervisor
tests. On **13 November 2026** the full local Python 3.13 suite completed
**113/113 PASS** (40 existing cutover/client tests + 73 new manager-side
tests, 58.507 s), and `py_compile` plus Bash syntax checking passed. The
source patch generator also successfully compiled a staged copy of the
exact SHA-pinned real manager and emitted a 7.1 KB reviewable diff.
Offline source parsing/compilation and mocked HTTP tests are not a substitute
for a real integration run. **Production status remains NO-GO.**

**13 November 2026 follow-up — whitespace/EOF and permission gate:**
The regenerated 7,083-byte patch contains zero trailing-whitespace lines
and no blank line at EOF. Against the exact SHA-pinned live-source copy,
Git apply dry-run, actual scratch apply with byte-equal output and Git
staged-diff whitespace check all return success. The manager-side runtime
now rejects group-readable lease-operator token files. Updated full
offline Python 3.13 ops suite: **115/115 PASS** (48.897 s), with Python
compilation and Bash syntax checks also passing. The prime's preceding
server-venv 113/113 test applies to the previous revision, not this
latest local follow-up. The ready-to-commit manifest, exact digests and
remaining no-go integration gates are in MANAGER_LEASE_PATCH_GATE.md.
