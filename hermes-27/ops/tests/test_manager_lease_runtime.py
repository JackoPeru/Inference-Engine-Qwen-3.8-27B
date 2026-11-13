"""Mock-only FastAPI manager integration tests. No HTTP listeners or CUDA."""

import asyncio
from pathlib import Path
import os
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from fastapi import HTTPException
from ops import manager_lease_core as core
from ops import manager_lease_runtime as runtime_mod


class ManagerLeaseRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(prefix="h27-lease-runtime-")
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.path = self.root / "journal" / "lease-v1.json"
        self.token = self.root / "op-token.txt"
        self.token.write_text("dummy-local-test-token-64-chars-xxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        patcher = mock.patch.dict(os.environ, {
            "HERMES_MANAGER_LEASE_STORE": str(self.path),
        })
        patcher.start()
        self.addCleanup(patcher.stop)
        private = mock.patch.object(core, "private_directory", return_value=True)
        private.start()
        self.addCleanup(private.stop)
        self.now = time.time()
        self.tabby_up = True
        self.restored = False
        self.restore_calls = 0
        self.state = {"current_state": "LLM_READY", "desired_mode": "AUTO",
                      "current_job": None, "last_drive_ts": time.monotonic()}

        async def true_online():
            return self.tabby_up

        async def service_active(_unit):
            return self.tabby_up

        async def restore():
            self.restore_calls += 1
            if self.runtime.blocked_operation():
                raise AssertionError("explicit recovery task incorrectly fenced")
            self.tabby_up = True
            self.restored = True
            self.state["current_state"] = "LLM_READY"
            return True

        self.ns = {
            "_lock": asyncio.Lock(), "_submit_lock": asyncio.Lock(), "_state": self.state,
            "CONFIG": {"llm": {"service": "hermes-tabby.service"}},
            "queued_jobs": lambda: [], "_character_training_hold": lambda: False,
            "_character_store": None, "_force_llm_active": lambda: False,
            "llm_online": true_online, "llm_serving": true_online,
            "service_active": service_active, "transition_to_llm": restore,
        }
        self.runtime = runtime_mod.LeaseRuntime(self.ns)
        self.runtime.core.clock = lambda: self.now
        self.enabled = mock.patch.object(self.runtime, "_configured", return_value=True)
        self.enabled.start()
        self.addCleanup(self.enabled.stop)

        async def fake_proof(variable):
            if variable == "HERMES_MANAGER_INGRESS_CHECK":
                return self.ingress()
            if variable == "HERMES_MANAGER_GPU_CHECK":
                return self.gpu_after() if self.tabby_up else self.gpu_before()
            raise AssertionError(f"unexpected checker {variable}")

        self.runtime._proof = fake_proof

    def run_async(self, coro):
        return asyncio.run(coro)

    def ingress(self):
        return {"scope": "all-tabby-clients", "blocked": True,
                "inflight": 0, "observed_at_unix": self.now}

    def gpu_before(self):
        return {"scope": "all-hermes-gpus-and-pids", "gpu_count": 2,
                "hermes_pids": [], "other_gpu_processes": [],
                "gpu_owner": "none", "gpu_free": True,
                "observed_at_unix": self.now}

    def gpu_after(self):
        return {**self.gpu_before(), "gpu_owner": "tabby", "gpu_free": False,
                "tabby_owns_both_gpus": True}

    def request(self, action, token=None):
        return {"protocol": core.PROTOCOL, "action": action,
                "owner": core.OWNER, "lease_id": "b" * 32,
                "manager_instance_id": self.runtime.core.instance_id,
                "fencing_token": token}

    def acquire(self):
        self.run_async(self.runtime.operate("acquire", self.request("acquire")))
        return self.runtime.core.status()["maintenance_lease"]["fencing_token"]

    def test_acquire_renew_release_wired_to_core(self):
        token = self.acquire()
        self.assertTrue(self.runtime.fenced)
        ack = self.run_async(self.runtime.operate("renew", self.request("renew", token)))
        self.assertEqual(ack["action"], "renew")
        result = self.run_async(self.runtime.operate("release", self.request("release", token)))
        self.assertTrue(result["ok"])
        self.assertFalse(self.runtime.fenced)

    def test_missing_checker_config_never_claims_contract(self):
        with mock.patch.object(self.runtime, "_configured", return_value=False):
            self.assertIsNone(self.runtime.status_fields()["watchdog_lease_contract_version"])
            with self.assertRaises(core.LeaseRejected):
                self.run_async(self.runtime.operate("acquire", self.request("acquire")))
        self.assertFalse(self.runtime.fenced)

    def test_manager_busy_refuses_acquire(self):
        self.ns["queued_jobs"] = lambda: [{"job_id": "in-progress"}]
        with self.assertRaises(core.LeaseRejected):
            self.acquire()
        self.assertFalse(self.runtime.fenced)

    def test_manager_missing_submit_lock_refuses_grant(self):
        self.ns.pop("_submit_lock")
        with self.assertRaisesRegex(core.LeaseRejected, "submit lock unavailable"):
            self.acquire()
        self.assertFalse(self.runtime.fenced)

    def test_inflight_submit_lock_denies_lease_before_persist(self):
        async def held_submission():
            async with self.ns["_submit_lock"]:
                with self.assertRaisesRegex(core.LeaseRejected, "in-flight media submission"):
                    await self.runtime.operate("acquire", self.request("acquire"))
        self.run_async(held_submission())
        self.assertFalse(self.runtime.fenced)
        self.assertFalse(self.path.exists())

    def test_repeated_acquire_after_tabby_stopped_is_idempotent(self):
        first = self.acquire()
        self.tabby_up = False
        reply = self.run_async(self.runtime.operate("acquire", self.request("acquire")))
        self.assertTrue(reply["ok"])
        self.assertEqual(self.runtime.core.status()["maintenance_lease"]["fencing_token"], first)

    def test_inflight_requests_refuse_acquire(self):
        async def proof(variable):
            return {**self.ingress(), "inflight": 3}
        self.runtime._proof = proof
        with self.assertRaises(core.LeaseRejected):
            self.acquire()

    def test_renew_fails_after_unexpected_ingress_reopening(self):
        token = self.acquire()
        async def open_ingress(_variable):
            return {**self.ingress(), "blocked": False}
        self.runtime._proof = open_ingress
        with self.assertRaises(core.LeaseRejected):
            self.run_async(self.runtime.operate("renew", self.request("renew", token)))
        self.assertTrue(self.runtime.fenced)

    def test_valid_lease_status_transitions_with_real_tabby_state(self):
        self.acquire()
        self.assertEqual(self.run_async(self.runtime.current_state("ERROR")), "LLM_READY")
        self.tabby_up = False
        self.assertEqual(self.run_async(self.runtime.current_state("LLM_READY")), "HERMES_MAINTENANCE")
        self.now += 91
        self.assertEqual(self.run_async(self.runtime.current_state("LLM_READY")), "HERMES_RECOVERY_REQUIRED")

    def test_post_expiry_autorecovery_requires_independent_proofs(self):
        self.acquire()
        self.tabby_up = False
        self.now += 91
        self.assertTrue(self.runtime.fenced)
        result = self.run_async(self.runtime.heartbeat())
        self.assertTrue(result)
        self.assertEqual(self.restore_calls, 1)
        self.assertFalse(self.runtime.fenced)
        self.assertTrue(self.tabby_up)

    def test_post_expiry_with_hermes_pid_does_not_restart_tabby(self):
        self.acquire()
        self.tabby_up = False
        self.now += 91
        async def unsafe(variable):
            if variable == "HERMES_MANAGER_GPU_CHECK":
                return {**self.gpu_before(), "hermes_pids": [444]}
            return self.ingress()
        self.runtime._proof = unsafe
        self.run_async(self.runtime.heartbeat())
        self.assertEqual(self.restore_calls, 0)
        self.assertTrue(self.runtime.fenced)

    def test_post_expiry_with_tabby_previously_restored_only_clears_safe_fence(self):
        self.acquire()
        self.now += 91
        self.assertTrue(self.runtime.fenced)
        self.assertTrue(self.tabby_up)
        self.run_async(self.runtime.heartbeat())
        self.assertEqual(self.restore_calls, 0)
        self.assertFalse(self.runtime.fenced)

    def test_operator_recover_fails_closed_if_checker_unavailable(self):
        self.acquire()
        self.tabby_up = False
        self.now += 91
        async def unavailable(_variable):
            raise core.LeaseRejected("local checker unavailable")
        self.runtime._proof = unavailable
        with self.assertRaises(core.LeaseRejected):
            self.run_async(self.runtime.recover())
        self.assertEqual(self.restore_calls, 0)
        self.assertTrue(self.runtime.fenced)

    def test_blocked_operation_denies_outside_recovery_task(self):
        self.acquire()
        self.assertTrue(self.runtime.blocked_operation())
        self.now += 91
        self.tabby_up = False
        self.run_async(self.runtime.recover())
        self.assertFalse(self.runtime.blocked_operation())

    def test_reject_nonloopback_peer_even_with_correct_header(self):
        fake = SimpleNamespace(client=SimpleNamespace(host="203.0.113.2"),
                               headers={"x-hermes-manager-lease-token": self.token.read_text()})
        with self.assertRaises(HTTPException) as captured:
            self.runtime._authorize(fake)
        self.assertEqual(captured.exception.status_code, 403)

    def test_reject_unknown_peer(self):
        fake = SimpleNamespace(client=None, headers={})
        with self.assertRaises(HTTPException):
            self.runtime._authorize(fake)

    def test_reject_wrong_second_factor_even_on_loopback(self):
        fake = SimpleNamespace(client=SimpleNamespace(host="127.0.0.1"),
                               headers={"x-hermes-manager-lease-token": "wrong"})
        with mock.patch.object(self.runtime, "_check_path", return_value=self.token):
            with self.assertRaises(HTTPException) as captured:
                self.runtime._authorize(fake)
        self.assertEqual(captured.exception.status_code, 403)

    def test_allow_correct_second_factor_on_loopback(self):
        fake = SimpleNamespace(client=SimpleNamespace(host="::1"),
                               headers={"x-hermes-manager-lease-token": self.token.read_text()})
        with mock.patch.object(self.runtime, "_check_path", return_value=self.token):
            self.runtime._authorize(fake)

    def test_status_contract_requires_all_safety_adapter_paths(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(core.LeaseRejected):
                runtime_mod.LeaseRuntime._check_path("HERMES_MANAGER_GPU_CHECK", executable=True)

    def test_group_readable_operator_token_is_rejected(self):
        # Simulate Linux mode on Windows without changing any real credential.
        key = "HERMES_MANAGER_LEASE_TOKEN_FILE"
        with mock.patch.dict(os.environ, {key: str(self.token.resolve())}), \
                mock.patch.object(runtime_mod.Path, "stat",
                                  return_value=SimpleNamespace(st_mode=0o100640)):
            with self.assertRaisesRegex(core.LeaseRejected, "securely configured"):
                runtime_mod.LeaseRuntime._check_path(key, executable=False)


if __name__ == "__main__":
    unittest.main()
