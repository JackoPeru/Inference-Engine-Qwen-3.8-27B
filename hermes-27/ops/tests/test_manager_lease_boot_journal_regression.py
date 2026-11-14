"""Offline fail-closed crash, lease journal, clock and proof-provenance tests.

All files are disposable tempfile fixtures; no GPU, service, HTTP or network
access. Keys here are TEST-ONLY; the installed manager is not configured.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ops import manager_lease_core as lease


class BootJournalRegression(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h27-journal-regression-")
        self.addCleanup(self.temp.cleanup)
        self.journal = Path(self.temp.name) / "private" / "lease.json"
        self.now = 2000.0
        self.key = b"synthetic-journal-trust-anchor-only-for-tests-2026"
        self.proof_key = b"synthetic-independent-proof-key-only-for-tests"
        self.instance = "a" * 32
        self.request = {
            "protocol": lease.PROTOCOL, "action": "acquire",
            "owner": lease.OWNER, "lease_id": "b" * 32,
            "manager_instance_id": self.instance, "fencing_token": None,
        }
        self.manager = self.make(bootstrap=True)

    def make(self, *, instance=None, proof=True, key=None, bootstrap=False):
        return lease.ManagerLeaseV1(
            self.journal, clock=lambda: self.now,
            instance_id=instance or self.instance,
            proof_verifier=self.verify if proof else None,
            journal_key=self.key if key is None else key,
            bootstrap_authorized=bootstrap,
        )

    def sign(self, kind, proof):
        bare = {k: v for k, v in proof.items() if k != "_proof_mac"}
        encoded = (kind + "\0" + json.dumps(bare, sort_keys=True,
                    separators=(",", ":"))).encode()
        return {**bare, "_proof_mac": hmac.new(
            self.proof_key, encoded, hashlib.sha256).hexdigest()}

    def verify(self, kind, proof):
        if not isinstance(proof, dict) or not isinstance(proof.get("_proof_mac"), str):
            return False
        return hmac.compare_digest(
            self.sign(kind, proof)["_proof_mac"], proof["_proof_mac"],
        )

    def ingress(self):
        return self.sign("ingress", {
            "scope": "all-tabby-clients", "blocked": True,
            "inflight": 0, "observed_at_unix": self.now,
        })

    def gpu_restored(self):
        return self.sign("gpu-restored", {
            "scope": "all-hermes-gpus-and-pids", "gpu_count": 2,
            "gpu_owner": "tabby", "gpu_free": False,
            "tabby_owns_both_gpus": True, "hermes_pids": [],
            "other_gpu_processes": [], "observed_at_unix": self.now,
        })

    def idle(self):
        return {
            "current_state": "LLM_READY", "desired_mode": "AUTO",
            "queue_length": 0, "current_job": None, "direct_active": False,
            "gaming_active": False, "character_training": [],
            "force_active": False, "llm_online": True, "llm_serving": True,
            "worker_alive_s": 1.0, "transition_lock_held": True,
        }

    def acquire(self):
        return self.manager.acquire(
            self.request, snapshot=self.idle(), ingress_proof=self.ingress())

    def holder(self, action):
        return {**self.request, "action": action, "fencing_token":
                self.manager.status()["maintenance_lease"]["fencing_token"]}

    def release(self):
        return self.manager.release(
            self.holder("release"), ingress_proof=self.ingress(),
            gpu_safe_proof=self.gpu_restored(), tabby_serving=True)

    def test_clean_initial_boot_with_explicit_trust_keys(self):
        self.assertFalse(self.manager.fenced)
        self.acquire()
        self.assertTrue(self.manager.fenced)
        self.release()
        self.assertFalse(self.make(instance="c" * 32).fenced)

    def test_missing_active_journal_fences_current_and_restart(self):
        self.acquire()
        self.journal.unlink()
        self.assertTrue(self.manager.fenced)
        self.assertTrue(self.manager.faulted)
        with self.assertRaises(lease.LeaseStoreFault):
            self.release()
        fresh = self.make(instance="c" * 32)
        self.assertTrue(fresh.fenced)
        self.assertTrue(fresh.faulted)
        with self.assertRaises(lease.LeaseStoreFault):
            fresh.acquire(
                {**self.request, "manager_instance_id": "c" * 32},
                snapshot=self.idle(), ingress_proof=self.ingress())

    def test_missing_released_journal_does_not_reset_counter(self):
        self.acquire()
        self.release()
        self.journal.unlink()
        fresh = self.make(instance="c" * 32)
        self.assertTrue(fresh.fenced)
        self.assertTrue(fresh.recovery_required)
        self.assertTrue(fresh.faulted)

    def test_empty_preexisting_parent_requires_bootstrap_review(self):
        self.journal.parent.mkdir(mode=0o700)
        fresh = self.make(instance="c" * 32, bootstrap=True)
        self.assertTrue(fresh.fenced)
        self.assertTrue(fresh.faulted)

    def test_absent_journal_and_directory_no_implicit_bootstrap(self):
        unconfigured = self.make(instance="c" * 32)
        self.assertTrue(unconfigured.fenced)
        self.assertTrue(unconfigured.faulted)
        self.assertFalse(self.journal.parent.exists())

    def test_lost_entire_directory_after_lease_still_fences_restart(self):
        self.acquire()
        self.journal.unlink()
        self.journal.parent.rmdir()
        restarted = self.make(instance="c" * 32)
        self.assertTrue(restarted.fenced)
        self.assertTrue(restarted.faulted)
        with self.assertRaises(lease.LeaseStoreFault):
            restarted.acquire(
                {**self.request, "manager_instance_id": "c" * 32},
                snapshot=self.idle(), ingress_proof=self.ingress())

    def test_trust_key_missing_denies_even_clean_boot(self):
        with self.assertRaises(ValueError):
            self.make(key=b"")
        without = lease.ManagerLeaseV1(
            self.journal, clock=lambda: self.now,
            instance_id="c" * 32, proof_verifier=self.verify)
        self.assertTrue(without.fenced)
        self.assertTrue(without.faulted)
        with self.assertRaises(lease.LeaseStoreFault):
            without.acquire(
                {**self.request, "manager_instance_id": "c" * 32},
                snapshot=self.idle(), ingress_proof=self.ingress())

    def test_proof_verifier_missing_rejects_valid_shape(self):
        absent = self.make(proof=False, instance="c" * 32, bootstrap=True)
        with self.assertRaisesRegex(lease.LeaseRejected, "verifier not provisioned"):
            absent.acquire(
                {**self.request, "manager_instance_id": "c" * 32},
                snapshot=self.idle(), ingress_proof=self.ingress())
        self.assertFalse(self.journal.exists())

    def test_forged_untrusted_but_structurally_valid_ingress_rejected(self):
        forged = {key: value for key, value in self.ingress().items()
                  if key != "_proof_mac"}
        with self.assertRaisesRegex(lease.LeaseRejected, "authentication failed"):
            self.manager.acquire(
                self.request, snapshot=self.idle(), ingress_proof=forged)
        self.assertFalse(self.journal.exists())

    def test_tampered_authenticated_fields_detected_live_and_on_restart(self):
        self.acquire()
        changed = json.loads(self.journal.read_text())
        changed["lease"] = None
        changed["recovery_required"] = False
        # Keep the original MAC: an untrusted syntactically valid replacement
        # must not become a clean lease journal.
        self.journal.write_text(json.dumps(changed))
        self.journal.chmod(0o600)
        self.assertTrue(self.manager.fenced)
        self.assertTrue(self.manager.faulted)
        fresh = self.make(instance="c" * 32)
        self.assertTrue(fresh.faulted)
        self.assertTrue(fresh.fenced)

    def test_wrong_journal_key_rejects_valid_file(self):
        self.acquire()
        fresh = self.make(instance="c" * 32,
                          key=b"synthetic-different-journal-key-32bytes")
        self.assertTrue(fresh.faulted)
        self.assertTrue(fresh.fenced)

    def test_clock_minus_thirty_never_moves_expiry_backward(self):
        self.acquire()
        before = self.manager.status()["maintenance_lease"]["expires_at_unix"]
        self.now -= 30
        self.assertTrue(self.manager.fenced)
        self.assertTrue(self.manager.recovery_required)
        self.assertEqual(self.manager.status()["maintenance_lease"]["expires_at_unix"],
                         before)
        with self.assertRaises(lease.LeaseRejected):
            self.manager.renew(self.holder("renew"), ingress_proof=self.ingress())
        fresh = self.make(instance="c" * 32)
        self.assertTrue(fresh.recovery_required)
        self.assertTrue(fresh.fenced)

    def test_direct_renew_after_rollback_rejects_without_prior_status(self):
        self.acquire()
        before = json.loads(self.journal.read_text())["lease"]["expires_at_unix"]
        renewal = self.holder("renew")
        self.now -= 30
        with self.assertRaisesRegex(lease.LeaseRejected, "rollback"):
            self.manager.renew(renewal, ingress_proof=self.ingress())
        state = json.loads(self.journal.read_text())
        self.assertTrue(state["recovery_required"])
        self.assertEqual(state["lease"]["expires_at_unix"], before)

    def test_small_backward_adjustment_cannot_reduce_expiry(self):
        self.acquire()
        original = self.manager.status()["maintenance_lease"]["expires_at_unix"]
        self.now -= .5
        self.manager.renew(self.holder("renew"), ingress_proof=self.ingress())
        self.assertEqual(self.manager.status()["maintenance_lease"]["expires_at_unix"],
                         original)
        self.assertFalse(self.manager.recovery_required)

    def test_journal_symlink_replacement_fences_current_and_restart(self):
        self.acquire()
        external = Path(self.temp.name) / "external.json"
        external.write_bytes(self.journal.read_bytes())
        self.journal.unlink()
        self.journal.symlink_to(external)
        self.assertTrue(self.manager.fenced)
        self.assertTrue(self.make(instance="c" * 32).fenced)
        self.assertTrue(self.make(instance="d" * 32).faulted)

    def test_temp_symlink_refused_without_touching_target(self):
        self.acquire()
        external = Path(self.temp.name) / "untouched.txt"
        external.write_text("do-not-overwrite")
        temp_path = self.journal.with_name(self.journal.name + ".writing")
        temp_path.symlink_to(external)
        self.now += 5
        with self.assertRaises(lease.LeaseStoreFault):
            self.manager.renew(self.holder("renew"), ingress_proof=self.ingress())
        self.assertEqual(external.read_text(), "do-not-overwrite")
        self.assertTrue(self.manager.fenced)

    def test_preexisting_temp_hardlink_never_truncates_target(self):
        self.acquire()
        external = Path(self.temp.name) / "external-persistent.txt"
        external.write_text("unaltered")
        temp_path = self.journal.with_name(self.journal.name + ".writing")
        os.link(external, temp_path)
        self.now += 4
        with self.assertRaises(lease.LeaseStoreFault):
            self.manager.renew(self.holder("renew"), ingress_proof=self.ingress())
        self.assertEqual(external.read_text(), "unaltered")
        self.assertTrue(self.manager.fenced)

    def test_stale_crash_temp_requires_operator_review(self):
        self.acquire()
        temp_path = self.journal.with_name(self.journal.name + ".writing")
        temp_path.write_text("crashed-previous-write")
        self.now += 4
        with self.assertRaises(lease.LeaseStoreFault):
            self.manager.renew(self.holder("renew"), ingress_proof=self.ingress())
        self.assertTrue(self.manager.faulted)
        # The previous signed active journal remains authoritative on restart.
        self.assertTrue(self.make(instance="c" * 32).recovery_required)

    def test_directory_fsync_failure_after_replace_never_acknowledges(self):
        self.acquire()
        self.now += 4
        real_fsync = lease.os.fsync
        calls = {"count": 0}

        def fail_dir_fsync(fd):
            calls["count"] += 1
            if calls["count"] == 2:
                raise OSError("synthetic directory fsync failure")
            return real_fsync(fd)

        with mock.patch.object(lease.os, "fsync", side_effect=fail_dir_fsync):
            with self.assertRaises(lease.LeaseStoreFault):
                self.manager.renew(self.holder("renew"), ingress_proof=self.ingress())
        self.assertTrue(self.manager.faulted)
        self.assertTrue(self.manager.fenced)
        # The new journal may have been renamed but not durably acknowledged.
        # Whichever version survives, restart MUST require fenced recovery.
        self.assertTrue(self.make(instance="c" * 32).recovery_required)

    def test_verifier_exception_or_truthy_nonboolean_never_grants(self):
        for verifier in (
                lambda *_: 1,
                lambda *_: "trusted",
                mock.Mock(side_effect=RuntimeError("checker failed")),
        ):
            with self.subTest(verifier=verifier):
                probe = lease.ManagerLeaseV1(
                    self.journal, clock=lambda: self.now,
                    instance_id="c" * 32,
                    journal_key=self.key, proof_verifier=verifier,
                    bootstrap_authorized=True)
                with self.assertRaisesRegex(
                        lease.LeaseRejected, "authentication failed"):
                    probe.acquire(
                        {**self.request, "manager_instance_id": "c" * 32},
                        snapshot=self.idle(), ingress_proof=self.ingress())
                self.assertFalse(self.journal.exists())

    def test_permissive_file_and_directory_fail_closed(self):
        self.acquire()
        self.journal.chmod(0o644)
        self.assertTrue(self.make(instance="c" * 32).faulted)
        self.journal.chmod(0o600)
        self.journal.parent.chmod(0o777)
        self.assertTrue(self.make(instance="d" * 32).faulted)
        self.journal.parent.chmod(0o700)

    def test_secure_read_inode_race_fences_even_identical_bytes(self):
        self.acquire()
        actual_open = lease.os.open
        shifted = {"done": False}

        def swapped_open(path, flags, *args, **kwargs):
            if Path(path) == self.journal and not shifted["done"]:
                shifted["done"] = True
                copied = self.journal.with_name("swap.json")
                copied.write_bytes(self.journal.read_bytes())
                copied.chmod(0o600)
                os.replace(copied, self.journal)
            return actual_open(path, flags, *args, **kwargs)

        with mock.patch.object(lease.os, "open", side_effect=swapped_open):
            self.assertTrue(self.manager.fenced)
        self.assertTrue(shifted["done"])
        self.assertTrue(self.manager.faulted)

    def test_failed_atomic_replace_fences_and_restart_recovers(self):
        self.acquire()
        self.now += 5
        with mock.patch.object(lease.os, "replace", side_effect=OSError("synthetic-fsync-fault")):
            with self.assertRaises(lease.LeaseStoreFault):
                self.manager.renew(self.holder("renew"), ingress_proof=self.ingress())
        self.assertTrue(self.manager.faulted)
        self.assertTrue(self.manager.fenced)
        restarted = self.make(instance="c" * 32)
        self.assertTrue(restarted.fenced)
        self.assertTrue(restarted.recovery_required)


if __name__ == "__main__":
    unittest.main()
