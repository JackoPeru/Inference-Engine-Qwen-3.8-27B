"""Offline manager lease v1 persistence/fencing tests; no systemctl or GPU."""

from pathlib import Path
import json
import tempfile
import unittest
from unittest import mock

from ops import manager_lease_core as lease


class ManagerLeaseCoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="h27-lease-core-")
        self.addCleanup(self.tmp.cleanup)
        self.journal = Path(self.tmp.name) / "private" / "lease.json"
        self.now = 2000.0
        self.patch_dir = mock.patch.object(lease, "private_directory", return_value=True)
        self.patch_dir.start()
        self.addCleanup(self.patch_dir.stop)
        self.instance = "a" * 32
        self.manager = lease.ManagerLeaseV1(self.journal, clock=lambda: self.now,
                                           instance_id=self.instance)
        self.request = {
            "protocol": lease.PROTOCOL,
            "action": "acquire", "owner": lease.OWNER,
            "lease_id": "b" * 32, "manager_instance_id": self.instance,
            "fencing_token": None,
        }

    def idle(self):
        return {"current_state": "LLM_READY", "desired_mode": "AUTO",
                "queue_length": 0, "current_job": None, "direct_active": False,
                "gaming_active": False, "character_training": [], "force_active": False,
                "llm_online": True, "llm_serving": True,
                "worker_alive_s": 1.0, "transition_lock_held": True}

    def ingress(self):
        return {"scope": "all-tabby-clients", "blocked": True,
                "inflight": 0, "observed_at_unix": self.now}

    def gpu_before(self):
        return {"scope": "all-hermes-gpus-and-pids", "gpu_count": 2,
                "gpu_owner": "none", "gpu_free": True,
                "hermes_pids": [], "other_gpu_processes": [],
                "observed_at_unix": self.now}

    def gpu_after(self):
        return {**self.gpu_before(), "gpu_owner": "tabby",
                "gpu_free": False, "tabby_owns_both_gpus": True}

    def acquire(self):
        return self.manager.acquire(self.request, snapshot=self.idle(), ingress_proof=self.ingress())

    def holder(self, action):
        return {**self.request, "action": action,
                "fencing_token": self.manager.status()["maintenance_lease"]["fencing_token"]}

    def test_new_manager_unfenced(self):
        self.assertFalse(self.manager.fenced)
        self.assertFalse(self.manager.recovery_required)
        status = self.manager.status()
        self.assertEqual(status["watchdog_lease_contract_version"], 1)
        self.assertEqual(status["manager_instance_id"], self.instance)
        self.assertIsNone(status["maintenance_lease"])

    def test_acquire_is_persisted_before_ack(self):
        ack = self.acquire()
        self.assertTrue(ack["ok"])
        self.assertEqual(ack["lease_id"], "b" * 32)
        self.assertTrue(self.journal.exists())
        journal = json.loads(self.journal.read_text())
        self.assertEqual(journal["lease"]["fencing_token"], 1)
        self.assertTrue(self.manager.fenced)
        self.assertEqual(self.manager.status()["maintenance_lease"]["status"], "active")

    def test_acquire_replay_is_idempotent_without_new_fence(self):
        self.acquire()
        first = self.journal.read_text()
        self.assertEqual(self.acquire()["action"], "acquire")
        self.assertEqual(self.journal.read_text(), first)

    def test_acquire_replay_denies_unblocked_ingress(self):
        self.acquire()
        bad = {**self.ingress(), "blocked": False}
        with self.assertRaisesRegex(lease.LeaseRejected, "active requests"):
            self.manager.acquire(self.request, snapshot=self.idle(), ingress_proof=bad)

    def test_concurrent_different_lease_denied(self):
        self.acquire()
        with self.assertRaisesRegex(lease.LeaseRejected, "another GPU"):
            self.manager.acquire({**self.request, "lease_id": "c" * 32},
                                 snapshot=self.idle(), ingress_proof=self.ingress())

    def test_acquire_requires_manager_lock(self):
        bad = {**self.idle(), "transition_lock_held": False}
        with self.assertRaisesRegex(lease.LeaseRejected, "outside GPU transition lock"):
            self.manager.acquire(self.request, snapshot=bad, ingress_proof=self.ingress())

    def test_acquire_rejects_each_busy_or_unknown_manager_state(self):
        invalid = (
            {"current_state": "MEDIA_BUSY"}, {"desired_mode": "DIRECT"},
            {"queue_length": 1}, {"queue_length": "0"},
            {"current_job": "running"}, {"direct_active": True},
            {"gaming_active": True}, {"character_training": ["job"]},
            {"force_active": True}, {"llm_online": False},
            {"llm_serving": False}, {"worker_alive_s": 80},
            {"worker_alive_s": None}, {"transition_lock_held": False},
        )
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(lease.LeaseRejected):
                self.manager.acquire(self.request, snapshot={**self.idle(), **change},
                                     ingress_proof=self.ingress())
            self.assertFalse(self.journal.exists())

    def test_ingress_rejects_every_untrusted_proof_case(self):
        invalid = (None, {}, {"scope": "partial"}, {"blocked": False},
                   {"inflight": 1}, {"inflight": -1}, {"inflight": 0.0},
                   {"observed_at_unix": self.now - 11},
                   {"observed_at_unix": self.now + 1},
                   {"observed_at_unix": "2000"})
        for change in invalid:
            p = change if change is None or change == {} else {**self.ingress(), **change}
            with self.subTest(proof=change), self.assertRaises(lease.LeaseRejected):
                self.manager.acquire(self.request, snapshot=self.idle(), ingress_proof=p)
        self.assertFalse(self.manager.fenced)

    def test_invalid_request_rejected_before_journal_write(self):
        changes = ({"protocol": "old"}, {"owner": "other"},
                   {"lease_id": "not-hex"}, {"lease_id": "A" * 32},
                   {"action": "renew"}, {"manager_instance_id": "c" * 32},
                   {"fencing_token": True}, {"fencing_token": 9})
        for change in changes:
            with self.subTest(change=change), self.assertRaises(lease.LeaseRejected):
                self.manager.acquire({**self.request, **change}, snapshot=self.idle(),
                                     ingress_proof=self.ingress())
        self.assertFalse(self.journal.exists())

    def test_renew_extends_same_fence_without_changing_owner(self):
        self.acquire()
        self.now += 15
        reply = self.manager.renew(self.holder("renew"), ingress_proof=self.ingress())
        self.assertEqual(reply["action"], "renew")
        self.assertEqual(self.manager.status()["maintenance_lease"]["expires_at_unix"], self.now + 90)
        self.assertEqual(self.manager.status()["maintenance_lease"]["fencing_token"], 1)

    def test_renew_wrong_fence_denied(self):
        self.acquire()
        bad = {**self.holder("renew"), "fencing_token": 19}
        with self.assertRaisesRegex(lease.LeaseRejected, "stale"):
            self.manager.renew(bad, ingress_proof=self.ingress())

    def test_renew_when_ingress_reopened_denied(self):
        self.acquire()
        with self.assertRaisesRegex(lease.LeaseRejected, "active requests"):
            self.manager.renew(self.holder("renew"),
                               ingress_proof={**self.ingress(), "blocked": False})
        self.assertTrue(self.manager.fenced)

    def test_expiry_fences_manager_until_independent_recovery(self):
        self.acquire()
        req = self.holder("renew")
        self.now += 91
        self.assertTrue(self.manager.fenced)
        self.assertTrue(self.manager.recovery_required)
        self.assertEqual(self.manager.status()["maintenance_lease"]["status"], "recovery_required")
        with self.assertRaises(lease.LeaseRejected):
            self.manager.renew(req, ingress_proof=self.ingress())
        with self.assertRaises(lease.LeaseRejected):
            self.acquire()

    def test_process_restart_keeps_outstanding_fence(self):
        self.acquire()
        fresh = lease.ManagerLeaseV1(self.journal, clock=lambda: self.now,
                                    instance_id="d" * 32)
        self.assertTrue(fresh.fenced)
        self.assertTrue(fresh.recovery_required)
        self.assertEqual(fresh.status()["maintenance_lease"]["status"], "recovery_required")
        with self.assertRaises(lease.LeaseRejected):
            fresh.renew(self.holder("renew"), ingress_proof=self.ingress())

    def test_fresh_manager_without_outstanding_lease_recovers_counter(self):
        self.acquire()
        self.manager.release(self.holder("release"), ingress_proof=self.ingress(),
                             gpu_safe_proof=self.gpu_after(), tabby_serving=True)
        fresh = lease.ManagerLeaseV1(self.journal, clock=lambda: self.now,
                                    instance_id="d" * 32)
        self.assertFalse(fresh.fenced)
        req = {**self.request, "manager_instance_id": "d" * 32,
               "lease_id": "e" * 32}
        fresh.acquire(req, snapshot=self.idle(), ingress_proof=self.ingress())
        self.assertEqual(fresh.status()["maintenance_lease"]["fencing_token"], 2)

    def test_release_rejects_before_tabby_healthy(self):
        self.acquire()
        with self.assertRaisesRegex(lease.LeaseRejected, "Tabby not confirmed"):
            self.manager.release(self.holder("release"), ingress_proof=self.ingress(),
                                 gpu_safe_proof=self.gpu_after(), tabby_serving=False)
        self.assertTrue(self.manager.fenced)

    def test_release_rejects_hermes_still_allocated(self):
        self.acquire()
        bad = {**self.gpu_after(), "hermes_pids": [4242]}
        with self.assertRaisesRegex(lease.LeaseRejected, "not independently released"):
            self.manager.release(self.holder("release"), ingress_proof=self.ingress(),
                                 gpu_safe_proof=bad, tabby_serving=True)

    def test_release_rejects_gpu_falsely_claiming_free(self):
        self.acquire()
        with self.assertRaisesRegex(lease.LeaseRejected, "exclusively Tabby"):
            self.manager.release(self.holder("release"), ingress_proof=self.ingress(),
                                 gpu_safe_proof=self.gpu_before(), tabby_serving=True)

    def test_release_idempotent_after_verified_tabby_restore(self):
        self.acquire()
        request = self.holder("release")
        first = self.manager.release(request, ingress_proof=self.ingress(),
                                     gpu_safe_proof=self.gpu_after(), tabby_serving=True)
        second = self.manager.release(request, ingress_proof=self.ingress(),
                                      gpu_safe_proof=self.gpu_after(), tabby_serving=True)
        self.assertEqual(first, second)
        self.assertFalse(self.manager.fenced)
        self.assertIsNone(self.manager.status()["maintenance_lease"])

    def test_idempotent_release_wrong_owner_rejected(self):
        self.acquire()
        req = self.holder("release")
        self.manager.release(req, ingress_proof=self.ingress(),
                             gpu_safe_proof=self.gpu_after(), tabby_serving=True)
        with self.assertRaises(lease.LeaseRejected):
            self.manager.release({**req, "lease_id": "c" * 32}, ingress_proof=self.ingress(),
                                 gpu_safe_proof=self.gpu_after(), tabby_serving=True)

    def test_expired_release_must_use_post_expiry_recovery(self):
        self.acquire()
        req = self.holder("release")
        self.now += 91
        with self.assertRaises(lease.LeaseRejected):
            self.manager.release(req, ingress_proof=self.ingress(),
                                 gpu_safe_proof=self.gpu_after(), tabby_serving=True)

    def test_post_expiry_recovery_denied_with_hermes_alive(self):
        self.acquire()
        self.now += 91
        with self.assertRaises(lease.LeaseRejected):
            self.manager.assert_recovery_safe(ingress_proof=self.ingress(),
                                              gpu_safe_proof={**self.gpu_before(), "hermes_pids": [9]})
        self.assertTrue(self.manager.fenced)

    def test_post_expiry_recovery_if_tabby_already_owns_both_gpus(self):
        self.acquire()
        self.now += 91
        self.manager.assert_recovery_already_restored(
            ingress_proof=self.ingress(), gpu_safe_proof=self.gpu_after())
        self.manager.finish_recovery(ingress_proof=self.ingress(),
                                     gpu_safe_proof=self.gpu_after(), tabby_serving=True)
        self.assertFalse(self.manager.fenced)

    def test_post_expiry_already_restored_path_denies_active_hermes_pid(self):
        self.acquire()
        self.now += 91
        with self.assertRaises(lease.LeaseRejected):
            self.manager.assert_recovery_already_restored(
                ingress_proof=self.ingress(),
                gpu_safe_proof={**self.gpu_after(), "hermes_pids": [22]})

    def test_post_expiry_recovery_denied_when_ingress_unblocked(self):
        self.acquire()
        self.now += 91
        with self.assertRaises(lease.LeaseRejected):
            self.manager.assert_recovery_safe(ingress_proof={**self.ingress(), "blocked": False},
                                              gpu_safe_proof=self.gpu_before())

    def test_post_expiry_recovery_requires_gpu_free_then_tabby_owned(self):
        self.acquire()
        self.now += 91
        self.manager.assert_recovery_safe(ingress_proof=self.ingress(),
                                          gpu_safe_proof=self.gpu_before())
        with self.assertRaises(lease.LeaseRejected):
            self.manager.finish_recovery(ingress_proof=self.ingress(),
                                         gpu_safe_proof=self.gpu_before(), tabby_serving=True)
        result = self.manager.finish_recovery(ingress_proof=self.ingress(),
                                              gpu_safe_proof=self.gpu_after(), tabby_serving=True)
        self.assertTrue(result["ok"])
        self.assertFalse(self.manager.fenced)

    def test_post_expiry_recovery_denies_tabby_health_failure(self):
        self.acquire()
        self.now += 91
        with self.assertRaises(lease.LeaseRejected):
            self.manager.finish_recovery(ingress_proof=self.ingress(),
                                         gpu_safe_proof=self.gpu_after(), tabby_serving=False)
        self.assertTrue(self.manager.fenced)

    def test_corrupt_persisted_journal_never_unfences(self):
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        self.journal.write_text("{invalid JSON")
        fresh = lease.ManagerLeaseV1(self.journal, clock=lambda: self.now)
        self.assertTrue(fresh.faulted)
        self.assertTrue(fresh.fenced)
        with self.assertRaises(lease.LeaseStoreFault):
            fresh.acquire(self.request, snapshot=self.idle(), ingress_proof=self.ingress())

    def test_failed_persist_keeps_fence_and_never_acknowledges(self):
        with mock.patch.object(lease, "private_directory", return_value=False):
            with self.assertRaises(lease.LeaseStoreFault):
                self.acquire()
        self.assertTrue(self.manager.fenced)
        self.assertTrue(self.manager.faulted)

    def test_malformed_persisted_fence_never_unfences(self):
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        self.journal.write_text(json.dumps({"version": 1, "counter": -1,
                                            "recovery_required": False, "lease": None}))
        fresh = lease.ManagerLeaseV1(self.journal, clock=lambda: self.now)
        self.assertTrue(fresh.fenced)

    def test_gpu_proof_denies_active_unknown_gpu_process(self):
        proof = {**self.gpu_before(), "other_gpu_processes": ["unknown"]}
        with self.assertRaises(lease.LeaseRejected):
            lease.require_gpu_safe(proof, now=self.now)

    def test_gpu_proof_denies_stale_or_wrong_gpu_count(self):
        for proof in ({**self.gpu_before(), "observed_at_unix": self.now - 11},
                      {**self.gpu_before(), "gpu_count": 1},
                      {**self.gpu_before(), "scope": "partial"}):
            with self.subTest(proof=proof), self.assertRaises(lease.LeaseRejected):
                lease.require_gpu_safe(proof, now=self.now)


if __name__ == "__main__":
    unittest.main()
