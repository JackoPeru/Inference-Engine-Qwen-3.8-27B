"""Deterministic CPU-only contract tests for an UNTRUSTED staging authority.

All three HMAC keys are test fixtures. No physical issuer, network, GPU,
service, live journal or manager runtime is accessed by this module.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, replace
import hashlib
import hmac
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from ops import manager_lease_external_authority_staging as external
from ops import manager_lease_core as existing_lease


class FakeClock:
    def __init__(self):
        self.now = 20000.0

    def __call__(self):
        return self.now


class FakeDurableAtomicStore(external.ExternalAtomicStore):
    """Independent persistent *fixture*: survives new authority instances.

    In production, this interface MUST be provided by an external privileged,
    antirollback authority. A mutex + in-memory record is NOT durable.
    """

    def __init__(self, state=None):
        self.state = state if state is not None else external.LedgerState()
        self.lock = threading.Lock()
        self.unavailable = False
        self.crash_before_commit = False
        self.crash_after_commit = False

    def atomic_update(self, mutate):
        with self.lock:
            if self.unavailable or self.crash_before_commit:
                raise OSError("synthetic unavailability before durable commit")
            following, receipt = mutate(self.state)
            self.state = following
            if self.crash_after_commit:
                self.crash_after_commit = False
                raise OSError("synthetic lost acknowledgement after durable commit")
            return receipt


class FakeDiskAtomicStore(external.ExternalAtomicStore):
    """SQLite fixture: a *separate on-disk store*, not a trusted highwater."""

    def __init__(self, filename):
        self.path = Path(filename)
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("CREATE TABLE IF NOT EXISTS ledger "
                       "(id INTEGER PRIMARY KEY CHECK (id=1), value TEXT NOT NULL)")
            db.execute("INSERT OR IGNORE INTO ledger (id,value) VALUES (1,?)",
                       (json.dumps(asdict(external.LedgerState())),))
            db.commit()

    @property
    def state(self):
        with closing(sqlite3.connect(self.path)) as db:
            value = db.execute("SELECT value FROM ledger WHERE id=1").fetchone()[0]
            decoded = json.loads(value)
            decoded["spent_nonces"] = tuple(decoded["spent_nonces"])
            return external.LedgerState(**decoded)

    def atomic_update(self, mutate):
        # Each call uses an independent database handle; BEGIN IMMEDIATE
        # serializes competing transactions and COMMIT precedes the receipt.
        # SQLite is NOT an external tamper-proof monotonic authority.
        db = sqlite3.connect(self.path, isolation_level=None, timeout=3)
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            old = self.state
            following, receipt = mutate(old)
            db.execute("UPDATE ledger SET value=? WHERE id=1",
                       (json.dumps(asdict(following), sort_keys=True),))
            db.execute("COMMIT")
            return receipt
        except Exception:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()


class FakeIndependentSigners(external.ExternalSignatureVerifier):
    def __init__(self):
        # Deliberately different keys for all three external roles.
        self.keys = {role: os.urandom(32)
                     for role in ("authority", "ingress", "gpu-owner")}

    def sign(self, role, claims):
        return {
            "claims": claims,
            "signature": hmac.new(
                self.keys[role], external.canonical_message(role, claims),
                hashlib.sha256,
            ).hexdigest(),
        }

    def verify(self, role, message, signature):
        expected = hmac.new(self.keys[role], message, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, signature):
            raise external.StagingDenied("synthetic signer role/signature mismatch")
        # Strict exception-on-failure contract: never return a trust boolean.
        return None


class ExternalAuthorityContract(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.store = FakeDurableAtomicStore()
        self.signers = FakeIndependentSigners()
        self.authority = self.new_authority()
        self.boot = "a" * 32
        self.manager = "b" * 32
        self.lease = "c" * 32
        self.next_nonce = 1

    def new_authority(self, *, store=None, verifier=None):
        return external.StagingAuthority(
            store=self.store if store is None else store,
            verifier=self.signers if verifier is None else verifier,
            clock=self.clock,
        )

    def nonce(self):
        self.next_nonce += 1
        return f"{self.next_nonce:032x}"

    def base(self, kind, *, nonce=None, boot=None, manager=None, lease=None,
             epoch=None, token=None, observed=None):
        state = self.store.state
        is_reservation = kind == "reserve"
        claims = {
            "protocol": external.PROTOCOL, "kind": kind,
            "boot_id": self.boot if boot is None else boot,
            "manager_instance_id": self.manager if manager is None else manager,
            "lease_id": self.lease if lease is None else lease,
            "epoch": ((state.epoch + 1) if is_reservation else state.epoch)
            if epoch is None else epoch,
            "fencing_token": ((state.highwater + 1) if is_reservation else state.highwater)
            if token is None else token,
            "nonce": self.nonce() if nonce is None else nonce,
            "observed_at_unix": self.clock.now if observed is None else observed,
        }
        if kind in ("reserve", "ingress-closed"):
            claims.update({"route_manifest_sha256": "e" * 64,
                           "route_manifest_epoch": 1})
        return claims

    def reserve(self, *, journal=None, role="authority", **overrides):
        claims = self.base("reserve", **overrides)
        signed = self.signers.sign(role, claims)
        if journal is None:
            journal = self.store.state.highwater
        return self.authority.reserve(signed, journal_counter=journal), signed

    def ingress(self, *, role="ingress", **overrides):
        claims = self.base("ingress-closed", nonce=self.store.state.active_nonce,
                           boot=self.store.state.boot_id,
                           manager=self.store.state.manager_instance_id,
                           lease=self.store.state.lease_id, **overrides)
        claims.update({"scope": "all-tabby-clients", "blocked": True,
                       "inflight": 0, "active_sse": 0,
                       "unknown_connections": 0,
                       "route_manifest_sha256": "e" * 64,
                       "route_manifest_epoch": 1})
        return self.signers.sign(role, claims)

    def restored(self, *, role="gpu-owner", **overrides):
        claims = self.base("gpu-restored", boot=self.store.state.boot_id,
                           manager=self.store.state.manager_instance_id,
                           lease=self.store.state.lease_id)
        claims.update({"scope": "all-hermes-gpus-and-pids", "gpu_count": 2,
                       "gpu_owner": "tabby", "tabby_owns_both_gpus": True,
                       "hermes_pids": [], "other_gpu_processes": []})
        claims.update(overrides)
        return self.signers.sign(role, claims)

    def acquire_candidate(self):
        first, ticket = self.reserve()
        self.assertFalse(first["production_admissible"])
        measured = self.authority.consume_ingress(self.ingress())
        self.assertFalse(measured["production_admissible"])
        return ticket

    def complete(self):
        self.authority.complete_restore(self.restored())

    def test_no_provider_and_boolean_provider_never_grant(self):
        missing_store = external.StagingAuthority(
            verifier=self.signers, clock=self.clock)
        missing_verifier = external.StagingAuthority(
            store=self.store, clock=self.clock)
        signature = self.signers.sign("authority", self.base("reserve"))
        for target in (missing_store, missing_verifier):
            with self.subTest(target=target):
                self.assertEqual(target.status()["status"], "BLOCKED")
                with self.assertRaises(external.StagingDenied):
                    target.reserve(signature, journal_counter=0)
        for value in (True, False, {}, object()):
            with self.subTest(untrusted_provider=type(value).__name__):
                with self.assertRaises(external.StagingDenied):
                    external.StagingAuthority(store=value, verifier=value)

    def test_positive_synthetic_cycle_never_claims_real_production_authority(self):
        self.acquire_candidate()
        self.complete()
        self.assertEqual(self.store.state.highwater, 1)
        self.assertEqual(self.store.state.epoch, 1)
        self.assertEqual(self.store.state.phase, "idle")
        self.assertEqual(self.authority.status()["status"], "BLOCKED")
        self.assertFalse(self.authority.status()["production_admissible"])

    def test_synthetic_receipts_never_become_existing_manager_ingress_proofs(self):
        reserve_receipt, _ = self.reserve()
        ingress_receipt = self.authority.consume_ingress(self.ingress())
        for candidate in (reserve_receipt, ingress_receipt, self.authority.status()):
            with self.subTest(candidate=candidate):
                self.assertFalse(candidate["production_admissible"])
                self.assertIsNone(candidate["manager_lease_proof"])
                with self.assertRaises(existing_lease.LeaseRejected):
                    existing_lease.require_ingress(candidate, now=self.clock.now)

    def test_older_valid_signed_journal_counter_cannot_reset_external_highwater(self):
        self.acquire_candidate()
        self.complete()
        self.lease = "d" * 32
        self.boot = "e" * 32
        self.manager = "f" * 32
        self.authority = self.new_authority()
        self.acquire_candidate()
        self.complete()
        self.assertEqual(self.store.state.highwater, 2)
        self.authority = self.new_authority()
        old_journal_counter = 1  # authentic former lease journal replay
        signed = self.signers.sign("authority", self.base("reserve"))
        with self.assertRaisesRegex(external.StagingDenied, "journal highwater mismatch"):
            self.authority.reserve(signed, journal_counter=old_journal_counter)
        self.assertEqual(self.store.state.highwater, 2)

    def test_reopened_disk_store_retains_nonce_consumption_and_highwater(self):
        with tempfile.TemporaryDirectory(prefix="hermes-lease-disk-witness-") as scratch:
            path = Path(scratch) / "separate-issuer-ledger.sqlite"
            self.store = FakeDiskAtomicStore(path)
            self.authority = self.new_authority()
            self.reserve()
            first = self.ingress()
            self.authority.consume_ingress(first)
            self.complete()
            old_nonce = first["claims"]["nonce"]
            # Release all SQLite handles and reconstruct both objects from
            # the same temporary DB file, simulating a manager process restart.
            self.store = FakeDiskAtomicStore(path)
            self.authority = self.new_authority()
            self.assertEqual(self.store.state.highwater, 1)
            self.assertIn(old_nonce, self.store.state.spent_nonces)
            self.lease = "d" * 32
            with self.assertRaisesRegex(external.StagingDenied, "journal highwater mismatch"):
                forged_old_head = self.signers.sign("authority", self.base("reserve"))
                self.authority.reserve(forged_old_head, journal_counter=0)
            signed_reused_nonce = self.signers.sign(
                "authority", self.base("reserve", nonce=old_nonce))
            with self.assertRaisesRegex(external.StagingDenied, "nonce"):
                self.authority.reserve(signed_reused_nonce, journal_counter=1)
            self.assertEqual(self.store.state.highwater, 1)

    def test_disk_reopen_inflight_consumed_proof_requires_external_recovery(self):
        with tempfile.TemporaryDirectory(prefix="hermes-lease-disk-crash-") as scratch:
            filename = Path(scratch) / "separate-ledger.sqlite"
            self.store = FakeDiskAtomicStore(filename)
            self.authority = self.new_authority()
            self.reserve()
            signed = self.ingress()
            self.authority.consume_ingress(signed)
            self.assertEqual(self.store.state.phase, "drain-verified")
            self.store = FakeDiskAtomicStore(filename)
            self.authority = self.new_authority()
            with self.assertRaises(external.StagingDenied):
                self.authority.consume_ingress(signed)
            with self.assertRaises(external.StagingDenied):
                self.authority.complete_restore(self.restored())
            self.assertEqual(self.store.state.phase, "drain-verified")
            self.assertEqual(self.authority.status()["status"], "BLOCKED")

    def test_fresh_signed_ingress_proof_is_single_use_across_lease_rotation(self):
        self.reserve()
        previous = self.ingress()
        self.authority.consume_ingress(previous)
        with self.assertRaises(external.StagingDenied):
            self.authority.consume_ingress(previous)
        self.complete()
        self.clock.now += 1  # same age window as the existing manager core
        self.lease = "d" * 32
        self.reserve()
        with self.assertRaises(external.StagingDenied):
            self.authority.consume_ingress(previous)
        self.authority.consume_ingress(self.ingress())

    def test_crash_after_reservation_commit_before_ack_requires_recovery(self):
        signed = self.signers.sign("authority", self.base("reserve"))
        self.store.crash_after_commit = True
        with self.assertRaises(external.StagingDenied):
            self.authority.reserve(signed, journal_counter=0)
        self.assertEqual(self.store.state.phase, "pending")
        self.assertEqual(self.store.state.highwater, 1)
        self.authority = self.new_authority()  # process incarnation changed
        with self.assertRaises(external.StagingDenied):
            self.authority.reserve(signed, journal_counter=0)
        with self.assertRaises(external.StagingDenied):
            self.authority.consume_ingress(self.ingress())
        self.assertFalse(self.authority.status()["production_admissible"])

    def test_crash_after_ingress_consumed_cannot_replay_after_restart(self):
        self.reserve()
        proof = self.ingress()
        self.store.crash_after_commit = True
        with self.assertRaises(external.StagingDenied):
            self.authority.consume_ingress(proof)
        self.assertEqual(self.store.state.phase, "drain-verified")
        self.authority = self.new_authority()
        with self.assertRaises(external.StagingDenied):
            self.authority.consume_ingress(proof)
        with self.assertRaises(external.StagingDenied):
            self.authority.complete_restore(self.restored())
        self.assertFalse(self.authority.status()["production_admissible"])

    def test_storage_failure_before_commit_does_not_advance_any_counter(self):
        signed = self.signers.sign("authority", self.base("reserve"))
        self.store.crash_before_commit = True
        with self.assertRaises(external.StagingDenied):
            self.authority.reserve(signed, journal_counter=0)
        self.assertEqual(self.store.state.highwater, 0)
        self.assertEqual(self.store.state.phase, "idle")
        self.store.crash_before_commit = False
        self.authority.reserve(signed, journal_counter=0)
        self.assertEqual(self.store.state.highwater, 1)

    def test_cross_role_signatures_and_boolean_success_are_denied(self):
        signed = self.signers.sign("ingress", self.base("reserve"))
        with self.assertRaises(external.StagingDenied):
            self.authority.reserve(signed, journal_counter=0)
        self.reserve()
        with self.assertRaises(external.StagingDenied):
            self.authority.consume_ingress(self.ingress(role="authority"))
        self.authority.consume_ingress(self.ingress())
        with self.assertRaises(external.StagingDenied):
            self.authority.complete_restore(self.restored(role="ingress"))

        class FakeBooleanVerifier(external.ExternalSignatureVerifier):
            def verify(self, role, message, signature):
                return True

        another = self.new_authority(verifier=FakeBooleanVerifier())
        with self.assertRaises(external.StagingDenied):
            another.complete_restore(self.restored())

    def test_signed_zero_inflight_requires_no_sse_or_unknown_connections(self):
        self.reserve()
        for change in ({"inflight": 1}, {"active_sse": 1},
                       {"unknown_connections": 1}, {"blocked": False},
                       {"scope": "single-gateway-only"},
                       {"route_manifest_sha256": "unknown"},
                       {"route_manifest_epoch": 0}):
            with self.subTest(change=change):
                claims = self.ingress()["claims"]
                claims.update(change)
                valid_signature = self.signers.sign("ingress", claims)
                with self.assertRaises(external.StagingDenied):
                    self.authority.consume_ingress(valid_signature)
                self.assertEqual(self.store.state.phase, "pending")

    def test_independently_signed_route_manifest_mismatch_denies_ingress(self):
        self.reserve()
        for change in ({"route_manifest_sha256": "f" * 64},
                       {"route_manifest_epoch": 2}):
            with self.subTest(change=change):
                claims = self.ingress()["claims"]
                claims.update(change)
                with self.assertRaisesRegex(
                        external.StagingDenied, "route-manifest mismatch"):
                    self.authority.consume_ingress(self.signers.sign("ingress", claims))
                self.assertEqual(self.store.state.phase, "pending")

    def test_tamper_or_untrusted_flags_cannot_authenticate(self):
        self.reserve()
        signed = self.ingress()
        tampered = {"claims": {**signed["claims"], "inflight": 1},
                    "signature": signed["signature"]}
        with self.assertRaises(external.StagingDenied):
            self.authority.consume_ingress(tampered)
        flagged = {"claims": {**signed["claims"], "privileged": True},
                   "signature": signed["signature"]}
        with self.assertRaises(external.StagingDenied):
            self.authority.consume_ingress(flagged)
        annotated = {**signed, "trusted": True}
        with self.assertRaises(external.StagingDenied):
            self.authority.consume_ingress(annotated)

    def test_boot_lease_epoch_nonce_or_fencing_mismatch_refused(self):
        self.reserve()
        for change in ({"boot_id": "d" * 32}, {"lease_id": "d" * 32},
                       {"manager_instance_id": "d" * 32}, {"epoch": 99},
                       {"fencing_token": 99}, {"nonce": "f" * 32}):
            with self.subTest(change=change):
                claims = self.ingress()["claims"]
                claims.update(change)
                with self.assertRaises(external.StagingDenied):
                    self.authority.consume_ingress(self.signers.sign("ingress", claims))

    def test_old_and_future_attestations_denied(self):
        self.reserve()
        for observed in (self.clock.now - 10.01, self.clock.now + 0.01,
                         float("nan"), float("inf")):
            with self.subTest(observed=observed):
                claims = self.ingress()["claims"]
                claims["observed_at_unix"] = observed
                # Canonical bytes reject NaN/Inf, before any signer call.
                if observed != observed or observed == float("inf"):
                    with self.assertRaises(external.StagingDenied):
                        self.signers.sign("ingress", claims)
                else:
                    with self.assertRaises(external.StagingDenied):
                        self.authority.consume_ingress(self.signers.sign("ingress", claims))

    def test_stale_restore_proof_and_incomplete_owner_refused(self):
        self.acquire_candidate()
        for change in ({"hermes_pids": [123]}, {"gpu_count": 1},
                       {"tabby_owns_both_gpus": False},
                       {"other_gpu_processes": [99]},
                       {"gpu_owner": "none"},
                       {"scope": "gpu-0-only"}):
            with self.subTest(change=change):
                with self.assertRaises(external.StagingDenied):
                    self.authority.complete_restore(self.restored(**change))
        self.assertEqual(self.store.state.phase, "drain-verified")

    def test_nonce_reuse_after_clean_epoch_is_denied_even_with_new_signature(self):
        first, ticket = self.reserve()
        old_nonce = ticket["claims"]["nonce"]
        self.authority.consume_ingress(self.ingress())
        self.complete()
        self.lease = "d" * 32
        signed = self.signers.sign("authority", self.base("reserve", nonce=old_nonce))
        with self.assertRaisesRegex(external.StagingDenied, "nonce"):
            self.authority.reserve(signed, journal_counter=1)
        self.assertEqual(self.store.state.highwater, 1)

    def test_concurrent_duplicate_ingress_proof_commits_once(self):
        self.reserve()
        proof = self.ingress()
        def consume():
            try:
                self.authority.consume_ingress(proof)
                return "accepted"
            except external.StagingDenied:
                return "rejected"
        with ThreadPoolExecutor(max_workers=2) as pool:
            result = list(pool.map(lambda _: consume(), range(2)))
        self.assertCountEqual(result, ["accepted", "rejected"])
        self.assertEqual(self.store.state.phase, "drain-verified")

    def test_counter_epoch_and_nonce_log_rollover_fail_closed(self):
        for change in (
                {"highwater": external.MAX_SEQUENCE, "epoch": external.MAX_SEQUENCE},
                {"spent_nonces": tuple(f"{n:032x}" for n in range(external.MAX_NONCES))}):
            with self.subTest(change=change):
                self.store.state = replace(external.LedgerState(), **change)
                ticket = self.signers.sign("authority", self.base("reserve"))
                with self.assertRaises(external.StagingDenied):
                    self.authority.reserve(ticket, journal_counter=self.store.state.highwater)
                self.assertEqual(self.store.state.phase, "idle")

    def test_nonce_log_nearly_exhausted_blocks_before_orphaning_lease(self):
        self.store.state = replace(
            external.LedgerState(),
            spent_nonces=tuple(f"{n:032x}" for n in range(external.MAX_NONCES - 1)))
        signed = self.signers.sign("authority", self.base("reserve"))
        with self.assertRaisesRegex(external.StagingDenied, "exhausted"):
            self.authority.reserve(signed, journal_counter=0)
        self.assertEqual(self.store.state.phase, "idle")

    def test_corrupt_external_record_and_unavailable_store_fail_closed(self):
        signed = self.signers.sign("authority", self.base("reserve"))
        self.store.state = {"trusted": True, "highwater": 0}
        with self.assertRaises(external.StagingDenied):
            self.authority.reserve(signed, journal_counter=0)
        self.store.state = external.LedgerState()
        self.store.unavailable = True
        with self.assertRaises(external.StagingDenied):
            self.authority.reserve(signed, journal_counter=0)
        self.assertEqual(self.authority.status()["status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
