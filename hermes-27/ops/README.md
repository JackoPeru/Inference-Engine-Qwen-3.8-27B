# HERMES-27 production maintenance supervisor

This directory contains an **opt-in maintenance cutover**; it is not a
zero-downtime replacement for TabbyAPI. The existing Tabby systemd unit and
its port 8000 remain the protected rollback target. No service is installed,
enabled, stopped or restarted by copying these files.

Read [`../reports/PRODUCTION_DEPLOYMENT.md`](../reports/PRODUCTION_DEPLOYMENT.md)
for verified host details, prerequisites, security, limitations and test evidence.

## Files

- `production_deploy.py` — `dry-run`, `preflight`, `run`, `recover`, `health`.
- `hermes-27.json.example` — checkpoint/tokenizer SHA256 and three deployment
  SHA256 pins (server binary, GGUF dump binary, tokenizer bridge), exact build
  directory, manager status/auth, ingress hooks and isolated HTTP options.
  The all-zero binary/bridge digests are **deliberately invalid placeholders**;
  replace each with the SHA256 from a reviewed and tested deployed build.
- `systemd/hermes-27-cutover.service.example` — **disabled template only**.
- `tests/test_production_deploy.py` — CPU-only simulated failure/rollback tests.
- `production_simulate.py` — 20 full-lifecycle failure-injection/recovery scenarios
  using a fake manager/ingress/systemd/GPU/process, without external calls.
- `MANAGER_LEASE_CONTRACT.md` — mandatory **proposed** watchdog lease and
  ingress contract; both interfaces are absent on the current server.
- `manager_lease_REVIEW.md` — detailed manager-side staged prototype with
  original-source SHA256 pin, installed-manager hook inventory and exact
  offline integration/recovery gaps.
- `manager_lease_{core,runtime,adapter,patch}.py`,
  `manager_lease_v1_proposal.patch` — **new, uninstalled** manager-side
  implementation prototype and reviewable source diff. No service side effects
  occur from importing these modules or viewing the generated patch.
- `../scripts/production_smoke.sh` — Python compile, unittest and Bash syntax.

## Dry-run and preflight (non-destructive)

```bash
bash scripts/production_smoke.sh
python3 ops/production_simulate.py --scenario all
python3 ops/production_deploy.py dry-run --config /etc/hermes/hermes-27.json --skip-hash
python3 ops/production_deploy.py preflight --config /etc/hermes/hermes-27.json
```

`dry-run` checks configured files and hook executables without probing the
manager or GPU. It also executes CPU-only binary compatibility probes:
`hermes_server --help`, `gguf_dump MODEL --chat-template`, and the tokenizer
bridge ready/encode handshake. The three deployed components must match
their pinned SHA256 digests; `hermes_server` and `gguf_dump` must be in the
**same configured `build_dir`**. The bridge must exist at
`build_dir/../serving/tokenizer_bridge.py`, the location derived by the real
C++ server. A descriptive GGUF dump (lines beginning `version=` / `KV-S`)
fails even when `gguf_dump` returns exit code 0 and its hash was repinned.

`--skip-hash` affects only the very large GGUF model and tokenizer hashes;
it never skips the **binary/bridge** hash or behavior checks. Full preflight
and `run` also stream the GGUF and tokenizer and reject mismatched hashes.
No deployment or service change occurs in `dry-run` or `preflight`.

## Operator-approved cutover and recovery

```bash
# Example only; disruptive and NOT executed by this worker:
python3 ops/production_deploy.py run --config /etc/hermes/hermes-27.json --authorized
# If the guardian was terminated unexpectedly and a journal remains:
python3 ops/production_deploy.py recover --config /etc/hermes/hermes-27.json --authorized
```

The supervisor writes a private journal before ingress drain, requires a fresh
all-client drain proof and authenticated GPU Manager `/status` with the **v1
maintenance lease contract**, acquires the lease, then stops Tabby while
**keeping GPU Manager active**. It renews and checks the lease throughout the
isolated Hermes run. The rollback stops HERMES, restores Tabby, verifies the
manager and releases the lease before reopening ingress. Unsafe failures
retain the journal and ingress block for manual recovery. The systemd
template adds `ExecStopPost` recovery if its process exits.

The manager currently has **no HERMES lease** and can automatically restart
Tabby; the configured lease adapters and ingress hooks do not exist either.
The current real system therefore fails the readiness gate. The simulated
tests prove software guard behavior only; see `MANAGER_LEASE_CONTRACT.md` for
the required integration and manager-side fail-closed lease-expiry semantics.
