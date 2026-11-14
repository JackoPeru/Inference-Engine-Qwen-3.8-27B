# HERMES-27: fail-closed Tabby restart during Hermes lease ownership

**2026-11-14. CPU simulated; release and GPU access still NO-GO.**

## Three independent deployment admission gaps

The new scoped lease investigation
`reports/DEPLOYMENT_LEASE_EXTERNAL_AUTHORITY_NEXT.md` reproduced three
counterexamples on the published staging implementation:

1. **HMAC-valid old journal replay** after manager restart can reset the
   fencing counter, because the journal has no independent monotonic external
   high-water mark.
2. **Same signed ingress observation** can be reused by two successive leases
   within the ten-second freshness period, because the proof is not bound to
   the lease ID, epoch and a consumed nonce.
3. **Unexpected Tabby restart:** during `hermes-watch`, a live Tabby service
   plus manager `LLM_READY` was accepted by `lease_guard` because its
   expected-state expression was based on *actual* Tabby activity.

All three were initially observed as **expected failures** in nine new
CPU-only tests. They were not misreported as nine PASS. Unlike the first
two, the third admits a useful local fail-closed correction without
pretending to implement a physical restart interlock.

## RED → GREEN, source-anchored

On the real published source prior to this fix, a direct CPU invocation of
`RollbackAndRestartOffline.test_OPEN_GAP_tabby_restart_while_hermes_watch_must_revoke_lease`
raised an assertion that the test's expected `DeploymentError` was **not
raised**. Observed RED receipt:

`RED_SOURCE_ANCHORED_LEASE_GUARD_ACCEPTED_TABBY_RESTART`.

The corrected `ops/production_deploy.py::lease_guard` rejects an active
`hermes-tabby.service` immediately if the recorded journal phase is
`hermes-start` or `hermes-watch`, **before** accepting a manager's
potentially consistent `LLM_READY` report. It caches the Tabby service
observation for the subsequent existing state validation. No service
operation was executed and the existing rollback path was not changed.
`draining`, `tabby-stop` and restoration paths retain their prior
state checks; the new prohibition applies only where the Hermes state
machine has already transitioned into its own GPU phase.

The original one known-gap test has become an ordinary passing test,
and another strict negative covers `hermes-start`. The original
maintenance-state test now expects the stronger early error rather than
the former later manager-state error; no assertion has been relaxed.

## Actually executed Linux CPU gates

- New lease-negative suite: **10 tests; 8 PASS, 2 expectedFailure
  (XFAIL), 0 unexpected failures/errors/XPASS**.
- Existing `ops.tests.test_production_deploy`: **20/20 PASS**.
- `ops/production_simulate.py --scenario all`: **20/20 PASS**.
- No real ingress request, GPU execution, installed service stop/restart,
  systemd mutation, model checkpoint access or cutover was involved.

The remaining **two XFAIL are security blockers**, not test progress:
no source-only replacement for an externally authenticated monotonic
counter and independent per-lease, nonce-consuming ingress issuer was
installed. Even with the local restart check, an asynchronous systemd/Laya
restart could access a GPU between polls until a separately verified
physical interlock prevents it. The safe deployment gate remains NO-GO.
