# GPU Manager ↔ HERMES-27 maintenance lease, proposed protocol v1

**Status: specification only; NOT deployed or supported by the present GPU
Manager.** The existing `/opt/hermes/gpu-manager/manager.py` watchdog can
restart Tabby approximately every 30 seconds. `run --authorized` must reject
the current manager before ingress or systemd changes. The 20 cutover scenarios
in `production_simulate.py` use a synthetic manager; they are **not evidence of
a real manager lease**.

An independently reviewable manager-side prototype now exists in
`manager_lease_{core,runtime,patch,adapter}.py` and the static
`manager_lease_v1_proposal.patch`, documented in `manager_lease_REVIEW.md`.
It was generated from **read-only** inspection of manager.py SHA256
`15744e3b769c0f6cd5ed26c2b85e3f282871f4a24681abd324c8cbc329a73eef`.
It is still NOT installed. The new code binds routes to the manager's existing
FastAPI app and requires the existing full-control Bearer plus a separate
loopback-only lease credential; no new listener is opened.

## Required authoritative manager API

`GET http://127.0.0.1:8643/status` continues using its existing Bearer app
credential. A future GPU Manager version must add the fields below while
retaining `desired_mode`, `current_state`, `llm_online`, `llm_serving`,
`queue_length`, `current_job`, `worker_alive_s`, `direct_active`,
`gaming_active`, `character_training`. Any absent/unknown field fails closed.

```json
{
  "watchdog_lease_contract_version": 1,
  "manager_instance_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "maintenance_lease": {
    "id": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "owner": "hermes-27",
    "status": "active",
    "scope": "tabby-restart-and-mode-transition",
    "watchdog_restart_suppressed": true,
    "fencing_token": 9,
    "expires_at_unix": 1794610000
  }
}
```

The IDs and timestamp above are illustrative, never real credentials. A
manager with no active lease must explicitly return `maintenance_lease: null`.
`manager_instance_id` must be a fresh 32-digit lower-case hexadecimal ID on
**every manager process start**. The version is exactly integer `1`. An active
lease must name its unique ID, owner and scope and attest inhibition of both
Tabby watchdog restart **and every automatic/manual GPU-mode transition**.
The positive `fencing_token` is monotonic for newly granted leases and stable
through renewals. After startup, the client demands a remaining TTL of
**25–180 seconds** at every check; its renewal interval defaults to 8 seconds.
On manager restart, changed fencing token, expired lease or missing fields,
HERMES stops and attempts a protected rollback.

### Required manager semantics

1. Grant at most **one** lease per GPU resource group, atomically under the
   same lock used by mode transitions, pending jobs and the Tabby watchdog.
   Reject overlapping media/direct/gaming/training requests or defer them
   without changing GPU ownership. The lease starts only after client ingress
   has confirmed a complete drain.
2. While a lease is active, never restart/reload Tabby, enter `MEDIA`, `DIRECT`
   or `GAMING`, run cleanup of HERMES-owned CUDA processes, or force a model
   restore. On intentional Tabby stop, report
   `current_state: "HERMES_MAINTENANCE"`; once Tabby is restored and serving,
   report `current_state: "LLM_READY"` even before lease release.
3. **Lease expiry must not restart Tabby automatically.** The manager must
   transition into a fenced recovery-required state and keep Tabby restart
   inhibited until it can establish that HERMES has stopped and relinquished
   GPU ownership. This rule also applies after manager crash/restart and
   watchdog failover. Without it, a 30-second watchdog can race an in-flight
   HERMES upload and cause OOM.

   The prototype implements this as a durable, fsynced lease journal and
   fenced `llm_watchdog`, `drive_auto_once`, transition/systemctl/unload paths,
   force/mode requests and mutating manager routes. Automatic safe recovery
   after expiry uses independent checker output. If the supervised client
   already restored Tabby, recovery verifies it owns both GPUs and clears the
   fence without restarting the process again.

   Admission is additionally serialized against the installed manager's
   separate `_submit_lock`. `create_job()` and both normal/smart submit
   critical sections have a second lease-fence check, so a request admitted
   through HTTP middleware before lease acquisition cannot enqueue work
   afterward. Independent Character ID actors and external process launchers
   still require explicit integration review.
4. Release may be accepted only for the matching lease ID/owner and fencing
   token. A repeated release of an already released matching lease must be
   idempotent. Never release another actor's lease. The status endpoint must
   reflect the released lease (`null`) before ingress is reopened.
5. Lease issuance/renewal/release must be authenticated by a local operator
   permission distinct from ordinary client traffic, audited without logging
   credentials, and never exposed to the public Internet. Manager-provided
   status must represent actual synchronized watchdog state, not an unrelated
   adapter's self-reported flag.

### Proposed manager-side HTTP API and evidence

The staged manager adds `POST /maintenance-lease/v1/acquire`, `/renew`,
`/release`, `/recover`, protected by the full-control manager role, a
second local token in `X-Hermes-Manager-Lease-Token`, and a strict loopback
source check. The first three accept the JSON payload described below;
`recover` requires local privileged safety checkers and cannot be used to
override a missing GPU ownership proof.

The manager requires private local executable proof providers named by
`HERMES_MANAGER_INGRESS_CHECK` and `HERMES_MANAGER_GPU_CHECK`, plus a durable
`HERMES_MANAGER_LEASE_STORE`. The ingress checker produces a complete
`scope=all-tabby-clients` proof with `blocked=true`, `inflight=0` and fresh
timestamp. The GPU checker must independently certify no HERMES processes
or unexpected GPU processes on **both** cards; before restarting Tabby,
both GPUs must be free (`gpu_owner="none"`, `gpu_free=true`). After Tabby
returns, an independent `gpu_owner="tabby"`,
`tabby_owns_both_gpus=true` attestation permits release. Exact JSON examples,
privileged credential paths and remaining trust boundaries are in
`manager_lease_REVIEW.md`.

## Local command adapters, not yet implemented

The following config fields are mandatory and point to **separate installed
executables**; they are deliberately missing from the current server:

```json
{
  "manager_lease_acquire_command": ["/usr/local/libexec/hermes-manager-lease-acquire"],
  "manager_lease_renew_command": ["/usr/local/libexec/hermes-manager-lease-renew"],
  "manager_lease_release_command": ["/usr/local/libexec/hermes-manager-lease-release"],
  "manager_lease_renew_every_sec": 8
}
```

Each adapter reads **exactly one JSON object on stdin** and prints a JSON
acknowledgement on stdout; no shell, API key, bearer secret or password goes
into argv, logs or the cutover journal. An example `acquire` input:

```json
{
  "protocol": "hermes-manager-lease-v1",
  "action": "acquire",
  "owner": "hermes-27",
  "lease_id": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "manager_instance_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "fencing_token": null
}
```

The acknowledgement has `protocol: "hermes-manager-lease-v1"`, matching
`action` and `lease_id`, and `ok: true`. On `renew` and `release`, the client
includes the fencing token learned from the manager's authenticated `/status`.
**The acknowledgement is never sufficient:** the supervisor independently
fetches authenticated manager status and verifies the lease ID, instance,
scope, watchdog inhibition and expiry. Adapter output must be small JSON and
must not include credentials. The adapter must implement acquire keyed by
the caller-generated lease ID and idempotent release so that a crash between
acquisition and journal update remains recoverable.

## External ingress contract (also NOT implemented on the live gateway)

Before obtaining the lease, `drain_command` must block **all** requests into
the existing Tabby endpoint, including direct loopback callers, proxies and
Tailscale clients. `drain_check_command` returns a fresh, independently
verified JSON proof:

```json
{
  "scope": "all-tabby-clients",
  "observed_at_unix": 1794610000,
  "blocked": true,
  "inflight": 0
}
```

The proof must describe all routing paths, be no more than 10 seconds old,
have nonnegative integer in-flight count, and remain blocked throughout
HERMES runtime. On unexpected reopening, the supervisor stops HERMES, runs
the drain hook again, and retains the journal and blocked ingress for manual
review. After verified Tabby recovery and lease release, the restore hook
reopens ingress and the checker must attest `blocked: false`.

## PREPARED / SIMULATED validation

```bash
python3 ops/production_simulate.py --scenario all
bash scripts/production_smoke.sh
```

`production_simulate.py` replaces **every external effect** with in-memory
functions, including service state, lease adapters, network endpoints, GPU
checks, ingress, process spawning and signal handling. It exercises clean
release, 17 injected failure conditions and 2 successful operator recovery
retries after a previously failed rollback. A `PASS (HELD_CLOSED)` outcome
means an incident deliberately retains the journal and ingress block; it is
not a successful live recovery. The real cutover remains NO-GO until the
manager, ingress and build contracts are independently implemented and
integration-tested during an explicitly approved maintenance window.
