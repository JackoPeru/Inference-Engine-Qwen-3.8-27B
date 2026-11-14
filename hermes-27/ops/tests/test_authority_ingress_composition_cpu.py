"""Offline compositional tests. ALL signers, ledger and hardware are synthetic.

Published staging production sources are imported, never modified. No sockets,
SSH, CUDA, GPU, Git, service or physical oracle is used.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import hmac
import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest import mock

from ops import ingress_attestation_contract as ingress
from ops import manager_lease_external_authority_staging as authority
from ops import staging_authority_ingress_composition as composition
from ops.tests import test_physical_ingress_attestation_contract_20261114 as physical_test


class MockAuthoritySigner(authority.ExternalSignatureVerifier):
    """Separate synthetic issuer keys for this contract's three roles."""

    def __init__(self):
        self.keys = {
            "authority": b"test-only external authority signer",
            "ingress": b"test-only aggregate ingress signer",
            "gpu-owner": b"test-only external gpu owner signer",
        }

    def sign(self, role, claims):
        return {
            "claims": deepcopy(claims),
            "signature": hmac.new(
                self.keys[role], authority.canonical_message(role, claims),
                hashlib.sha256,
            ).hexdigest(),
        }

    def verify(self, role, message, signature):
        expected = hmac.new(self.keys[role], message, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError("independent test signer mismatch")
        return None


class MockAtomicCompositeStore(composition.AtomicCompositionStore):
    """SERIALIZABLE fixture; no external antirollback or durable authority."""

    def __init__(self, state=None):
        self.state = state if state is not None else composition.CompositionState()
        self.lock = threading.Lock()
        self.calls = 0
        self.fail_before = False
        self.fail_after = False
        self.clock_mutation = None
        self.return_invalid = False
        self.skip_mutation = False
        self.call_twice = False

    def atomic_update(self, mutate):
        with self.lock:
            self.calls += 1
            if self.fail_before:
                raise OSError("synthetic before-commit fault")
            if self.clock_mutation is not None:
                self.clock_mutation()
            if self.skip_mutation:
                return dict(composition._BLOCKED)
            following, receipt = mutate(self.state)
            if self.call_twice:
                mutate(self.state)
            self.state = following
            if self.fail_after:
                self.fail_after = False
                raise OSError("synthetic lost acknowledgement after commit")
            return {"production_admissible": True} if self.return_invalid else receipt


class CompositionCPU(unittest.TestCase):
    def setUp(self):
        # Reuse the exact published 23-test fixture for the physical package.
        self.physical = physical_test.ContractTests(
            methodName="test_full_synthetic_candidate_stays_production_blocked")
        self.physical.setUp()
        self.physical_signer = self.physical.signer
        self.authority_signer = MockAuthoritySigner()
        self.context = self.physical.ctx
        self.store = MockAtomicCompositeStore()
        self.package = self.physical.package()
        self.manifest_sha = hashlib.sha256(
            authority.canonical_message(
                "authority", self.package["manifest"]["claims"])
        ).hexdigest()
        self.reserve = self.signed_reserve()
        self.ingress_proof = self.signed_ingress()

    def basic_claims(self, kind):
        ctx = self.context
        return {
            "protocol": authority.PROTOCOL,
            "kind": kind,
            "boot_id": ctx.boot_id,
            "manager_instance_id": ctx.manager_instance_id,
            "lease_id": ctx.lease_id,
            "epoch": ctx.epoch,
            "fencing_token": ctx.fencing_token,
            "nonce": ctx.nonce,
            "observed_at_unix": self.physical.clock.wall,
            "route_manifest_sha256": self.manifest_sha,
            "route_manifest_epoch": ctx.manifest_epoch,
        }

    def signed_reserve(self):
        return self.authority_signer.sign("authority", self.basic_claims("reserve"))

    def signed_ingress(self):
        return self.authority_signer.sign("ingress", {
            **self.basic_claims("ingress-closed"),
            "scope": "all-tabby-clients", "blocked": True,
            "inflight": 0, "active_sse": 0, "unknown_connections": 0,
        })

    def kwargs(self, **changes):
        opts = {
            "authority_reservation": self.reserve,
            "authority_ingress": self.ingress_proof,
            "physical_package": self.package,
            "context": self.context,
            "journal_counter": 0,
            "authority_verifier": self.authority_signer,
            "physical_verifier": self.physical_signer,
            "store": self.store,
            "wall_clock": self.physical.clock.wall_time,
            "monotonic_clock": self.physical.clock.mono_time,
        }
        opts.update(changes)
        return opts

    def attempt(self, **changes):
        return composition.compose(**self.kwargs(**changes))

    def assert_denied_no_commit(self, **changes):
        old = self.store.state
        with self.assertRaises(composition.CompositionDenied):
            self.attempt(**changes)
        self.assertEqual(self.store.state, old)

    def test_synthetic_pair_is_one_commit_but_still_production_blocked(self):
        answer = self.attempt()
        self.assertEqual(self.store.calls, 1)
        self.assertEqual(answer["status"], "BLOCKED")
        self.assertIs(answer["production_admissible"], False)
        self.assertIsNone(answer["manager_lease_proof"])
        state = self.store.state
        self.assertEqual(state.paired_nonce, self.context.nonce)
        self.assertEqual(state.paired_manifest, self.manifest_sha)
        self.assertEqual(state.authority_ledger.highwater, 1)
        self.assertEqual(state.ingress_ledger.highwater, 1)
        self.assertEqual(state.authority_ledger.phase, "drain-verified")
        self.assertEqual(state.ingress_ledger.spent_nonces, (self.context.nonce,))
        self.assertIs(composition.status()["production_admissible"], False)

    def test_replayed_pair_is_not_transferable_to_second_lease(self):
        self.attempt()
        prior = self.store.state
        with self.assertRaises(composition.CompositionDenied):
            self.attempt()
        self.assertEqual(self.store.state, prior)
        # Rotating the manager/lease/nonce while the previous pair is still
        # pending also requires an explicit independently attested recovery.
        self.context = replace(
            self.context, lease_id="f" * 32, nonce="e" * 32,
            manager_instance_id="1" * 32, epoch=2, fencing_token=2,
            manifest_epoch=2)
        self.physical.ctx = self.context
        self.package = self.physical.package()
        self.manifest_sha = hashlib.sha256(authority.canonical_message(
            "authority", self.package["manifest"]["claims"])).hexdigest()
        self.reserve = self.signed_reserve()
        self.ingress_proof = self.signed_ingress()
        with self.assertRaises(composition.CompositionDenied):
            self.attempt(journal_counter=1)
        self.assertEqual(self.store.state, prior)

    def test_independent_ingress_nonce_consumption_cannot_be_transferred(self):
        # Physically impossible to promote this independent acknowledgement
        # into the external reservation's separate atomic authority.
        fake_ingress = physical_test.SyntheticStore()
        self.physical.assess(self.package, store=fake_ingress)
        self.assertEqual(fake_ingress.state.spent_nonces, (self.context.nonce,))
        self.store.state = composition.CompositionState(
            ingress_ledger=fake_ingress.state)
        self.assert_denied_no_commit()

    def test_independently_consumed_both_ledgers_not_a_joint_commit(self):
        class SeparateAuthorityStore(authority.ExternalAtomicStore):
            def __init__(self):
                self.state = authority.LedgerState()

            def atomic_update(self, change):
                self.state, reply = change(self.state)
                return reply

        local_auth = SeparateAuthorityStore()
        validator = authority.StagingAuthority(
            store=local_auth, verifier=self.authority_signer,
            clock=self.physical.clock.wall_time)
        validator.reserve(self.reserve, journal_counter=0)
        validator.consume_ingress(self.ingress_proof)
        local_ingress = physical_test.SyntheticStore()
        self.physical.assess(self.package, store=local_ingress)
        self.assertEqual(local_auth.state.highwater, local_ingress.state.highwater)
        self.store.state = composition.CompositionState(
            authority_ledger=local_auth.state,
            ingress_ledger=local_ingress.state,
            paired_nonce=self.context.nonce, paired_manifest=self.manifest_sha)
        # Both separately verified signatures and consumed nonce: still NOT
        # equivalent to the composition's one atomic commit.
        self.assert_denied_no_commit()

    def test_cross_protocol_boot_lease_epoch_fence_nonce_mismatch(self):
        for key, changed in (
            ("boot_id", "e" * 32), ("manager_instance_id", "e" * 32),
            ("lease_id", "e" * 32), ("nonce", "e" * 32),
            ("epoch", 2), ("fencing_token", 2),
        ):
            with self.subTest(key=key):
                claims = self.reserve["claims"].copy()
                claims[key] = changed
                self.assert_denied_no_commit(
                    authority_reservation=self.authority_signer.sign(
                        "authority", claims))

    def test_individually_signed_manifest_mismatch_fails_without_consumption(self):
        for kind in ("reservation", "ingress"):
            for key, replacement in (
                ("route_manifest_sha256", "f" * 64),
                ("route_manifest_epoch", 2)):
                with self.subTest(kind=kind, key=key):
                    source = self.reserve if kind == "reservation" else self.ingress_proof
                    claims = source["claims"].copy()
                    claims[key] = replacement
                    revised = self.authority_signer.sign(
                        "authority" if kind == "reservation" else "ingress",
                        claims)
                    prop = ("authority_reservation" if kind == "reservation"
                            else "authority_ingress")
                    self.assert_denied_no_commit(**{prop: revised})

    def test_physical_manifest_digest_change_cannot_reuse_old_authority_ticket(self):
        changed = deepcopy(self.package)
        changed["manifest"]["claims"]["inventory_revision"] += 1
        changed["manifest"] = self.physical_signer.sign(
            changed["manifest"]["claims"], "authority")
        # External reservation and ingress aggregate remain separately
        # signed with prior digest. No ledger side effects.
        self.assert_denied_no_commit(physical_package=changed)

    def test_nonzero_sse_or_queue_or_unknown_route_fails_atomically(self):
        for field in ("sse_streams", "queued_http", "queued_jobs",
                      "unknown_connections", "unfenced_restarts"):
            with self.subTest(field=field):
                modified = deepcopy(self.package)
                modified["routes"][0]["claims"]["counters"][field] = 1
                modified["routes"][0] = self.physical_signer.sign(
                    modified["routes"][0]["claims"], "ingress")
                self.assert_denied_no_commit(physical_package=modified)

    def test_tampered_or_cross_role_signature_does_not_advance_either_ledger(self):
        bad = deepcopy(self.reserve)
        bad["signature"] = "0" * 64
        self.assert_denied_no_commit(authority_reservation=bad)
        self.assert_denied_no_commit(authority_reservation=
                                    self.authority_signer.sign(
                                        "ingress", self.reserve["claims"]))
        modified = deepcopy(self.package)
        modified["routes"][0] = self.physical_signer.sign(
            modified["routes"][0]["claims"], "authority")
        self.assert_denied_no_commit(physical_package=modified)

    def test_wrong_or_boolean_verifiers_do_not_grant(self):
        class BoolVerifier(authority.ExternalSignatureVerifier):
            def verify(self, role, message, signature):
                return True

        self.assert_denied_no_commit(authority_verifier=BoolVerifier())
        self.assert_denied_no_commit(physical_verifier=BoolVerifier())
        self.assert_denied_no_commit(physical_verifier=self.authority_signer)
        self.assert_denied_no_commit(authority_verifier=self.physical_signer)
        self.assert_denied_no_commit(physical_verifier=self.authority_signer,
                                     authority_verifier=self.authority_signer)

    def test_crash_before_final_commit_leaves_no_partial_ledger(self):
        self.store.fail_before = True
        self.assert_denied_no_commit()
        self.store.fail_before = False
        self.attempt()
        self.assertEqual(self.store.state.authority_ledger.highwater, 1)
        self.assertEqual(self.store.state.ingress_ledger.highwater, 1)

    def test_commit_then_ack_loss_is_denied_and_replay_stays_fenced(self):
        self.store.fail_after = True
        with self.assertRaises(composition.CompositionDenied):
            self.attempt()
        assert self.store.state.authority_ledger.highwater == 1
        assert self.store.state.ingress_ledger.highwater == 1
        reconstructed = MockAtomicCompositeStore(state=self.store.state)
        self.assert_denied_no_commit(store=reconstructed)
        self.assertEqual(reconstructed.state, self.store.state)

    def test_crash_after_staged_authority_mutation_rolls_back_both(self):
        modified = deepcopy(self.package)
        modified["routes"][0]["claims"]["counters"]["http_inflight"] = 1
        modified["routes"][0] = self.physical_signer.sign(
            modified["routes"][0]["claims"], "ingress")
        self.assert_denied_no_commit(physical_package=modified)
        self.assertEqual(self.store.state, composition.CompositionState())
        # Replacing the invalid physical package succeeds in the test-only
        # single transaction; prior staged authority reservation was discarded.
        self.attempt()

    def test_clock_drift_during_atomic_callback_denies_without_commit(self):
        self.store.clock_mutation = lambda: (
            setattr(self.physical.clock, "wall", self.physical.clock.wall + 9),
            setattr(self.physical.clock, "mono", self.physical.clock.mono + 9))
        self.assert_denied_no_commit()

    def test_noncanonical_receipt_does_not_grant_even_if_store_committed(self):
        self.store.return_invalid = True
        with self.assertRaises(composition.CompositionDenied):
            self.attempt()
        self.assertEqual(self.store.state.authority_ledger.highwater, 1)
        self.assertEqual(self.store.state.ingress_ledger.highwater, 1)
        with self.assertRaises(composition.CompositionDenied):
            self.attempt()

    def test_store_cannot_forge_success_without_running_atomic_callback(self):
        self.store.skip_mutation = True
        self.assert_denied_no_commit()
        self.assertEqual(self.store.state, composition.CompositionState())

    def test_double_invocation_of_callback_denied_without_commit(self):
        self.store.call_twice = True
        self.assert_denied_no_commit()
        self.assertEqual(self.store.state, composition.CompositionState())

    def test_concurrent_duplicate_composition_commits_only_once(self):
        def worker(_):
            try:
                self.attempt()
                return "one synthetic commit"
            except composition.CompositionDenied:
                return "refused"
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(worker, range(2)))
        self.assertCountEqual(outcomes, ["one synthetic commit", "refused"])
        self.assertEqual(self.store.state.authority_ledger.highwater, 1)
        self.assertEqual(self.store.state.ingress_ledger.highwater, 1)

    def test_missing_store_manifest_or_issuer_stays_blocked(self):
        self.assert_denied_no_commit(store=None)
        self.assert_denied_no_commit(authority_verifier=None)
        self.assert_denied_no_commit(physical_verifier=None)
        self.assert_denied_no_commit(physical_package={})
        self.assert_denied_no_commit(authority_reservation={"trusted": True})
        self.assertIs(composition.status()["production_admissible"], False)
        self.assertIsNone(composition.status()["manager_lease_proof"])

    def test_source_mutant_cross_protocol_lease_binding_is_caught(self):
        source = inspect.getsource(composition._matching_context)
        needle = 'for claims in (ticket["claims"], ingress_ticket["claims"], manifest):'
        self.assertEqual(source.count(needle), 1)
        mutant = source.replace(needle, 'for claims in (manifest,):', 1)
        symbols = dict(vars(composition))
        exec(compile(mutant, "<lease-crosscheck-mutant-cpu>", "exec"), symbols)
        # Both external signatures agree on the SAME WRONG lease ID, while
        # the separately signed physical manifest has the original ID.
        # Published verifiers check each protocol independently, so the
        # composite context check is the critical cross-protocol guard.
        reserve_claims = self.reserve["claims"].copy()
        ingress_claims = self.ingress_proof["claims"].copy()
        reserve_claims["lease_id"] = "f" * 32
        ingress_claims["lease_id"] = "f" * 32
        altered_reserve = self.authority_signer.sign("authority", reserve_claims)
        altered_ingress = self.authority_signer.sign("ingress", ingress_claims)
        self.assert_denied_no_commit(
            authority_reservation=altered_reserve,
            authority_ingress=altered_ingress)
        with mock.patch.object(composition, "_matching_context",
                               symbols["_matching_context"]):
            # A failure of this negative assertion is expected ONLY for this
            # in-memory mutant, demonstrating the actual lease-binding gate.
            with self.assertRaises(AssertionError):
                self.assert_denied_no_commit(
                    authority_reservation=altered_reserve,
                    authority_ingress=altered_ingress)


if __name__ == "__main__":
    unittest.main()
