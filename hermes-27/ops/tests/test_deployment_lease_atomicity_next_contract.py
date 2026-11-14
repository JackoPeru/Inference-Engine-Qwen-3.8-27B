"""CPU-only counterexamples for deploy/lease atomicity and external authority.

The HMAC keys and observations are synthetic. In particular, these tests do
not provide, install, or impersonate a physical ingress/GPU proof issuer.
Expected failures identify open security contracts rather than passing gates.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from ops import manager_lease_core as lease
from ops import production_deploy as deploy


class Clock:
    def __init__(self):
        self.value = 20000.0

    def __call__(self):
        return self.value


class LeaseAtomicityOffline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="hermes-lease-atomicity-next-")
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "private" / "lease.json"
        self.clock = Clock()
        self.journal_key = os.urandom(32)
        self.issuer_key = os.urandom(32)
        self.instance = "a" * 32
        if os.name == "nt":
            # Windows reports different POSIX modes and newline bytes. This
            # adapter is confined to disposable fixtures; Linux uses the real
            # hardened inode/permission checks without substitution.
            p = mock.patch.object(lease, "private_directory", return_value=True)
            p.start()
            self.addCleanup(p.stop)

            def synthetic_nt_read(manager):
                path = manager.path
                if path.is_symlink() or path.parent.is_symlink() or not path.is_file():
                    raise OSError("unsafe synthetic journal")
                data = path.read_bytes()
                if len(data) > 16384:
                    raise OSError("synthetic journal exceeds size limit")
                return data.replace(b"\r\n", b"\n")

            p = mock.patch.object(lease.ManagerLeaseV1, "_journal_bytes", synthetic_nt_read)
            p.start()
            self.addCleanup(p.stop)

    def sign(self, kind, evidence):
        body = (kind + "\0" + json.dumps(
            evidence, sort_keys=True, separators=(",", ":"))).encode("utf-8")
        return {**evidence, "_offline_mac": hmac.new(
            self.issuer_key, body, hashlib.sha256).hexdigest()}

    def verify(self, kind, evidence):
        if not isinstance(evidence, dict) or not isinstance(evidence.get("_offline_mac"), str):
            return False
        unsigned = {k: v for k, v in evidence.items() if k != "_offline_mac"}
        return hmac.compare_digest(self.sign(kind, unsigned)["_offline_mac"],
                                   evidence["_offline_mac"])

    def ingress(self):
        return self.sign("ingress", {"scope": "all-tabby-clients", "blocked": True,
                                    "inflight": 0, "observed_at_unix": self.clock.value})

    def restored_gpu(self):
        return self.sign("gpu-restored", {
            "scope": "all-hermes-gpus-and-pids", "gpu_count": 2,
            "gpu_owner": "tabby", "tabby_owns_both_gpus": True,
            "hermes_pids": [], "other_gpu_processes": [],
            "observed_at_unix": self.clock.value})

    @staticmethod
    def idle():
        return {"current_state": "LLM_READY", "desired_mode": "AUTO",
                "queue_length": 0, "current_job": None, "direct_active": False,
                "gaming_active": False, "character_training": [],
                "force_active": False, "llm_online": True, "llm_serving": True,
                "worker_alive_s": 1.0, "transition_lock_held": True}

    def new_manager(self, *, instance=None, bootstrap=False, verifier=True):
        return lease.ManagerLeaseV1(
            self.path, clock=self.clock, instance_id=instance or self.instance,
            journal_key=self.journal_key,
            proof_verifier=self.verify if verifier else None,
            bootstrap_authorized=bootstrap)

    def request(self, action, lease_id, *, instance=None, token=None):
        return {"protocol": lease.PROTOCOL, "action": action, "owner": lease.OWNER,
                "lease_id": lease_id, "manager_instance_id": instance or self.instance,
                "fencing_token": token}

    def acquire(self, manager, lease_id, *, proof=None):
        manager.acquire(self.request("acquire", lease_id),
                        snapshot=self.idle(), ingress_proof=proof or self.ingress())
        return manager.status()["maintenance_lease"]["fencing_token"]

    def release(self, manager, lease_id, token):
        manager.release(self.request("release", lease_id, token=token),
                        ingress_proof=self.ingress(), gpu_safe_proof=self.restored_gpu(),
                        tabby_serving=True)

    def test_absent_independent_proof_issuer_denies_a_fresh_lease(self):
        manager = self.new_manager(bootstrap=True, verifier=False)
        with self.assertRaisesRegex(lease.LeaseRejected, "verifier not provisioned"):
            self.acquire(manager, "b" * 32)
        self.assertFalse(self.path.exists())

    def test_active_journal_after_boot_requires_fenced_recovery(self):
        manager = self.new_manager(bootstrap=True)
        self.assertEqual(self.acquire(manager, "b" * 32), 1)
        restarted = self.new_manager(instance="d" * 32)
        self.assertTrue(restarted.fenced)
        self.assertTrue(restarted.recovery_required)
        with self.assertRaises(lease.LeaseRejected):
            restarted.acquire(self.request("acquire", "e" * 32, instance="d" * 32),
                              snapshot=self.idle(), ingress_proof=self.ingress())

    def test_expired_signed_ingress_proof_is_not_accepted(self):
        manager = self.new_manager(bootstrap=True)
        prior = self.ingress()
        self.clock.value += lease.PROOF_MAX_AGE_SECONDS + 0.1
        with self.assertRaisesRegex(lease.LeaseRejected, "scope/age/counters"):
            self.acquire(manager, "b" * 32, proof=prior)
        self.assertFalse(self.path.exists())

    @unittest.expectedFailure
    def test_OPEN_GAP_prior_valid_signed_journal_must_not_reset_epoch(self):
        """XFAIL: HMAC authenticity alone cannot detect older valid journal replay."""
        manager = self.new_manager(bootstrap=True)
        token = self.acquire(manager, "b" * 32)
        self.release(manager, "b" * 32, token)
        old_authentic_journal = self.path.read_bytes()
        token = self.acquire(manager, "c" * 32)
        self.assertEqual(token, 2)
        self.release(manager, "c" * 32, token)
        self.assertTrue(manager.fenced is False)
        self.path.write_bytes(old_authentic_journal)
        # An external authoritative monotonic high-water counter is missing;
        # this reboot currently trusts the old counter=1, unfenced journal.
        restarted = self.new_manager(instance="d" * 32)
        self.assertTrue(restarted.fenced,
                        "authentic older journal rolled counter 2 back to 1")

    @unittest.expectedFailure
    def test_OPEN_GAP_fresh_signed_proof_must_not_replay_across_leases(self):
        """XFAIL: issuer proof has no lease/epoch/nonce binding inside 10 seconds."""
        manager = self.new_manager(bootstrap=True)
        signed_before_first_lease = self.ingress()
        token = self.acquire(manager, "b" * 32, proof=signed_before_first_lease)
        self.release(manager, "b" * 32, token)
        self.clock.value += 1
        with self.assertRaises(lease.LeaseRejected,
                               msg="same signed ingress observation authorized a new epoch"):
            self.acquire(manager, "c" * 32, proof=signed_before_first_lease)


class RollbackAndRestartOffline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="hermes-rollback-atomicity-next-")
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.journal = root / "state" / "cutover.json"
        self.cfg = {"_path": str((root / "cfg.json").resolve()),
                    "state_dir": str(root / "state"),
                    "drain_check_command": ["mock-check"],
                    "drain_command": ["mock-drain"],
                    "restore_command": ["mock-open"],
                    "tabby_health_url": "http://127.0.0.1:8000/v1/models"}

    def proof(self, *, blocked=True, inflight=0, scope="all-tabby-clients"):
        return json.dumps({"scope": scope, "blocked": blocked, "inflight": inflight,
                           "observed_at_unix": time.time()})

    def assert_rollback_held(self, *, lease_pending, ingress_json):
        deploy.save_state(self.journal, {
            "config_path": self.cfg["_path"], "phase": "lease-active" if lease_pending else "draining",
            "tabby_was_active": True, "manager_was_active": True,
            "lease_pending": lease_pending, "ingress_pending": True})
        with mock.patch.object(deploy, "service_active", return_value=True), \
             mock.patch.object(deploy, "manager_status",
                               return_value={"maintenance_lease": {"id": "synthetic"}}), \
             mock.patch.object(deploy, "manager_guard", return_value={}), \
             mock.patch.object(deploy, "request_json", return_value={}), \
             mock.patch.object(deploy, "run_hook", return_value=ingress_json) as hook, \
             mock.patch.object(deploy, "lease_adapter") as adapter, \
             mock.patch.object(deploy, "change_service") as changes:
            with self.assertRaisesRegex(deploy.DeploymentError, "ROLLBACK INCOMPLETE"):
                deploy.restore(self.cfg, self.journal)
            self.assertTrue(self.journal.exists())
            self.assertTrue(all(call.args[0] != self.cfg["restore_command"]
                                for call in hook.call_args_list))
            adapter.assert_not_called()
            changes.assert_not_called()

    def test_inflight_sse_during_lease_release_keeps_ingress_closed(self):
        # SSE spans the lease rollback; proof says one active request.
        self.assert_rollback_held(lease_pending=True,
                                  ingress_json=self.proof(inflight=1))

    def test_partial_scope_drain_before_acquire_does_not_reopen(self):
        self.assert_rollback_held(lease_pending=False,
                                  ingress_json=self.proof(scope="missing-tailscale"))

    def test_manager_detects_tabby_restarted_while_reporting_maintenance(self):
        cfg = {}
        state = {"manager_instance_id": "a" * 32, "lease_id": "b" * 32,
                 "fencing_token": 5, "phase": "hermes-watch"}
        status = {"manager_instance_id": "a" * 32,
                  "watchdog_lease_contract_version": 1,
                  "desired_mode": "AUTO", "queue_length": 0, "current_job": None,
                  "worker_alive_s": 1.0, "direct_active": False,
                  "gaming_active": False, "character_training": [],
                  "current_state": "HERMES_MAINTENANCE",
                  "maintenance_lease": {"id": "b" * 32, "owner": lease.OWNER,
                      "scope": lease.SCOPE, "status": "active",
                      "watchdog_restart_suppressed": True,
                      "fencing_token": 5, "expires_at_unix": time.time() + 90}}
        with mock.patch.object(deploy, "service_active", return_value=True), \
             mock.patch.object(deploy, "manager_status", return_value=status):
            # The stronger physical-phase gate now refuses the restarted
            # Tabby before accepting even a contradictory manager snapshot.
            with self.assertRaisesRegex(deploy.DeploymentError, "Tabby restarted"):
                deploy.lease_guard(cfg, state)

    def test_tabby_restart_while_hermes_watch_is_immediately_refused(self):
        """Even a manager-consistent LLM_READY must not mask a restarted Tabby.

        This local guard is NOT an external systemd restart interlock or a
        guarantee that a process cannot allocate GPU before the next poll.
        """
        state = {"manager_instance_id": "a" * 32, "lease_id": "b" * 32,
                 "fencing_token": 5, "phase": "hermes-watch"}
        status = {"manager_instance_id": "a" * 32,
                  "watchdog_lease_contract_version": 1,
                  "desired_mode": "AUTO", "queue_length": 0, "current_job": None,
                  "worker_alive_s": 1.0, "direct_active": False,
                  "gaming_active": False, "character_training": [],
                  "current_state": "LLM_READY",
                  "maintenance_lease": {"id": "b" * 32, "owner": lease.OWNER,
                      "scope": lease.SCOPE, "status": "active",
                      "watchdog_restart_suppressed": True,
                      "fencing_token": 5, "expires_at_unix": time.time() + 90}}
        with mock.patch.object(deploy, "service_active", return_value=True), \
             mock.patch.object(deploy, "manager_status", return_value=status):
            with self.assertRaises(deploy.DeploymentError,
                                   msg="hermes-watch lease accepted after Tabby restart"):
                deploy.lease_guard({}, state)

    def test_tabby_restart_during_hermes_start_is_immediately_refused(self):
        state = {"phase": "hermes-start"}
        with mock.patch.object(deploy, "service_active", return_value=True), \
             mock.patch.object(deploy, "manager_status") as get_status:
            with self.assertRaisesRegex(deploy.DeploymentError,
                                        "Tabby restarted"):
                deploy.lease_guard({}, state)
            get_status.assert_not_called()


if __name__ == "__main__":
    unittest.main()
