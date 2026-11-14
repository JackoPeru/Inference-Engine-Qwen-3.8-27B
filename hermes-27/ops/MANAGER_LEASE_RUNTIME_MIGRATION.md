# Manager lease runtime: explicit trust injection (staging, CPU/offline)

Date: 2026-11-14. Worker-3. **NO PRODUCTION GO; NO installed changes.**

## Scope and existing breakage

The hardened, frozen `ops/manager_lease_core.py` requires independent
`journal_key`, `proof_verifier(kind, proof)` and one-time explicit bootstrap
authorization. Previously, `ops/manager_lease_runtime.py:LeaseRuntime.__init__`
constructed `ManagerLeaseV1` with only a store path, so it now **correctly
fails closed**. Positive runtime fixtures were still using syntactically valid
unsigned proof dictionaries, which cannot authorize physical ingress or GPU
operations under the hardened core. The prior ops suite reported 283 tests
with 4 failures and 9 errors in runtime tests (prime's earlier snapshot).
We do **not** weaken the core or treat these fixtures as real attestations.

Changes are restricted to:

- `ops/manager_lease_runtime.py`
- `ops/tests/test_manager_lease_runtime.py`
- NEW `ops/tests/test_lease_runtime_staging_migration_contract.py`
- NEW `ops/MANAGER_LEASE_RUNTIME_MIGRATION.md`

No edits to the frozen core, worker-7 interop tests, installed
`/opt/hermes/laya_bridge.py`, manager, systemd, proxy, ingress, GPU state,
other services, credentials, or CI/CMake sources.

## Exact constructor / entrypoint contract

```python
from ops.manager_lease_runtime import (
    LeaseRuntime,
    LeaseTrustInputs,
    install_manager_lease,
)

# A future separately reviewed, authorized caller may inject externally
# provisioned trust. The values below are interface placeholders only.
trust = LeaseTrustInputs(
    journal_key=privately_loaded_root_protected_key_bytes,
    proof_verifier=independent_authenticated_checker_verifier,
    bootstrap_authorized=False,
)
runtime = LeaseRuntime(manager_namespace, trust=trust)
# Or, when the external deployment has explicit prime approval:
runtime = install_manager_lease(manager_namespace, trust=trust)
```

`LeaseTrustInputs` is an immutable dataclass with secret fields hidden
from `repr`. It rejects non-bytes/short (under 32 bytes) journal keys,
missing/non-callable verifiers and non-boolean bootstrap flags. The
installer and constructor preserve **one-argument backward compatibility**:

```python
runtime = LeaseRuntime(manager_namespace)
runtime = install_manager_lease(manager_namespace)
```

Both **deliberately remain fenced**, because neither receives trusted
material. There is NO fallback to old shape-only proofs, no implicit
bootstrap and no journal signing key loaded from environment variables,
JSON requests, checker stdout, installed credentials, or source literals.
`HERMES_MANAGER_LEASE_STORE` still selects the non-secret journal path.

`_trusted_ready()` is independent of the existing `_configured()` check.
The latter now requires both explicit trust and the existing local token
and checker executable paths. `status_fields()` must not advertise
`watchdog_lease_contract_version=1` when either check fails. `operate()`
and `recover()` require both checks before touching manager locks,
checkers, or GPU transition callbacks. Even if a test monkeypatches
`_configured` to `True`, missing core trust still forces rejection.
The core separately verifies the origin of ingress/GPU proof data through
`proof_verifier(kind, proof) is True`.

### Authorized caller boundary

This migration does **not** implement or infer a production issuer. An
external authorized and root-isolated integration must prove the journal
key's independent provenance, validate checker identity, boot/PID birth
and GPU owner epoch, and verify that the complete ingress scope is
actually interlocked (including Laya/systemd/direct reaper). Only such an
integration can supply `proof_verifier`. Python accepting a callable
does not prove that it is trustworthy. The examples above are not
production-ready keys, claims, or provisioning instructions.

The original `install_manager_lease(namespace)` call in the deployed
patch/runtime remains intentionally untrusted and fenced. Do not make
`bootstrap_authorized=True` a constant in a service, env flag or generic
restart path: it is a one-time, independently reviewed first-install grant.
An existing unsigned journal cannot be silently promoted into a signed
epoch. Core HMAC authentication also does not independently prevent
replay of an older authentic journal after a restart; durable monotonic
epoch anchoring is a separate **NO-GO** requirement.

## Test-only mock boundary

`test_manager_lease_runtime.py` now creates independent per-test
`os.urandom(32)` journal and fixture-signing keys and explicitly injects
`LeaseTrustInputs(..., bootstrap_authorized=True)` for each fresh private
TemporaryDirectory. It signs *synthetic* ingress, GPU-free and GPU-restored
dictionaries with an HMAC bound to the proof type. The test's verifier
requires a matching signature; altered or untrusted values cannot be
accepted by the core. All `_proof` checkers, manager state and Tabby
restore callbacks remain **in-memory mocks**, not physical machine
observations. This allows positive state-machine testing without claiming
any live ingress/GPU proof or weakening the hardened core.

The new migration tests ensure the no-trust default is fenced even
if checker availability is mocked true; a trusted journal key alone
cannot auto-bootstrap or skip checker requirements; invalid trust
objects are rejected; key material is not included in `repr`; and
unrelated environment values cannot implicitly enable trust. The final
added test also checks `install_manager_lease(..., trust=...)` and the
default installer path in a FastAPI-style inert app, without a listener.

## Reproducible CPU-only results and remaining failures

Initial verified source in isolated Linux scratch:

```bash
cd /tmp/h27-worker3-runtime-migration-20261114
PYTHONPATH=$PWD /opt/hermes/runtimes/exllamav3-venv/bin/python \
    -m unittest discover -s ops/tests -p test_manager_lease_runtime.py -v
PYTHONPATH=$PWD /opt/hermes/runtimes/exllamav3-venv/bin/python \
    -m unittest discover -s ops/tests \
    -p test_lease_runtime_staging_migration_contract.py -v
```

Results with real FastAPI on Ubuntu: **20/20 runtime PASS**,
**6/6 new staging migration PASS** (before the final seventh installer
test was added).

Then a fresh, disposable complete `ops/` copy in
`/tmp/h27-worker3-runtime-fullops-20261114` ran:

```bash
cd /tmp/h27-worker3-runtime-fullops-20261114
PYTHONPATH=$PWD /opt/hermes/runtimes/exllamav3-venv/bin/python \
    -m unittest discover -s ops/tests -p 'test_*.py' -q
```

**Exact result at that snapshot: 316 tests; 1 FAIL + 15 ERROR.**
All 16 results are in worker-7-owned
`ops/tests/test_ingress_gpu_fence_contract_interop.py`, whose old
fixtures construct the hardened core without journal key, proof verifier,
or authorized bootstrap. In particular, G1/G2/G3 still encode prior
insecure behavior as expectations; see `ops/MANAGER_LEASE_CRASH_FIX.md`.
**Do not change** the core or worker-7 suite to force green. Prime owns
coordinating authenticated offline fixtures and inverting those
characterization assertions.

A second complete-suite pass **excluding only** that frozen interop module
reported **289/289 PASS, 0 FAIL, 0 ERROR, 0 SKIP**, using actual isolated
Linux Python/FastAPI and the hardened core. It is not represented as a
whole-suite PASS.

Afterwards, a seventh new installer test was added, without modifying
the runtime implementation. An attempted fresh Ubuntu run was blocked
by new **Tailscale SSH authentication**, so that exact final test was
instead run on local Python 3.12 with an isolated **in-memory FastAPI
stub**: **7/7 PASS**. This is a limited installer/API contract check,
**not a replacement** for running the final test with actual FastAPI.
Local `py_compile` passed for the three Python files. The last Linux
full-suite result remains the precise 316/1/15 snapshot above;
do not claim a new 317-case Linux PASS without rerunning it.

The source is frozen for prime review. The trust issuer, authenticated
GPU/ingress attestations and external monotonic journal epoch are not
provisioned in this task; production deployment therefore stays
**BLOCKED**. Prime must explicitly approve any changes outside the
four-file ownership boundary, especially live manager integration.
