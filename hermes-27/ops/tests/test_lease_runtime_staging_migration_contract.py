"""Offline-only contract for explicit, non-deployable lease runtime trust.

All state is in a TemporaryDirectory. No installed manager, GPU, service,
listener, root key, credential, or real checker is ever accessed.
"""

import asyncio
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from fastapi import FastAPI
from ops import manager_lease_core as lease
from ops import manager_lease_runtime as runtime


class RuntimeTrustMigration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h27-runtime-staging-")
        self.addCleanup(self.temp.cleanup)
        self.journal = Path(self.temp.name) / "lease-state" / "lease.json"
        patch = mock.patch.dict(os.environ, {
            "HERMES_MANAGER_LEASE_STORE": str(self.journal),
        })
        patch.start()
        self.addCleanup(patch.stop)

    @staticmethod
    def request():
        return {
            "protocol": lease.PROTOCOL, "action": "acquire",
            "owner": lease.OWNER, "lease_id": "b" * 32,
            "manager_instance_id": "a" * 32, "fencing_token": None,
        }

    @staticmethod
    def fixture_trust(*, bootstrap):
        # Staging-only random material, never a deployed key or physical
        # proof. This verifier returns False to ensure no GPU transition
        # could ever be falsely authorized in this test.
        return runtime.LeaseTrustInputs(
            journal_key=os.urandom(32),
            proof_verifier=lambda _kind, _proof: False,
            bootstrap_authorized=bootstrap,
        )

    def test_default_runtime_blocks_all_transitions_without_external_trust(self):
        instance = runtime.LeaseRuntime({})
        self.assertTrue(instance.fenced)
        self.assertTrue(instance.core.faulted)
        self.assertIsNone(instance.status_fields()["watchdog_lease_contract_version"])
        self.assertTrue(instance.status_fields()["maintenance_recovery_required"])
        self.assertFalse(self.journal.exists())
        # Even if the privileged checker availability check is incorrectly
        # mocked as True, absence of separately injected trust prevails.
        with mock.patch.object(instance, "_configured", return_value=True):
            with self.assertRaisesRegex(lease.LeaseRejected, "trusted journal"):
                asyncio.run(instance.operate("acquire", self.request()))
            with self.assertRaisesRegex(lease.LeaseRejected, "trusted journal"):
                asyncio.run(instance.recover())
            self.assertIsNone(instance.status_fields()["watchdog_lease_contract_version"])
        self.assertFalse(self.journal.exists())

    def test_caller_trust_is_not_permission_for_automatic_bootstrap(self):
        instance = runtime.LeaseRuntime(
            {}, trust=self.fixture_trust(bootstrap=False),
        )
        self.assertTrue(instance.fenced)
        self.assertTrue(instance.core.faulted)
        self.assertFalse(self.journal.parent.exists())
        self.assertIsNone(instance.status_fields()["watchdog_lease_contract_version"])

    def test_explicit_reviewed_bootstrap_does_not_bypass_checker_requirements(self):
        instance = runtime.LeaseRuntime(
            {}, trust=self.fixture_trust(bootstrap=True),
        )
        self.assertFalse(instance.core.faulted)
        self.assertFalse(instance.fenced)
        self.assertFalse(instance._configured())
        self.assertIsNone(instance.status_fields()["watchdog_lease_contract_version"])
        with self.assertRaisesRegex(lease.LeaseRejected, "checker missing"):
            asyncio.run(instance.operate("acquire", self.request()))
        self.assertFalse(self.journal.exists())

    def test_invalid_trust_rejected_before_runtime_installation(self):
        with self.assertRaisesRegex(TypeError, "LeaseTrustInputs"):
            runtime.LeaseRuntime({}, trust={"journal_key": b"bogus"})
        with self.assertRaisesRegex(ValueError, "journal trust key"):
            runtime.LeaseTrustInputs(
                journal_key=b"tiny", proof_verifier=lambda *_: True,
            )
        with self.assertRaisesRegex(ValueError, "independent proof verifier"):
            runtime.LeaseTrustInputs(
                journal_key=os.urandom(32), proof_verifier=None,
            )
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            runtime.LeaseTrustInputs(
                journal_key=os.urandom(32), proof_verifier=lambda *_: True,
                bootstrap_authorized=1,
            )
        self.assertFalse(self.journal.exists())

    def test_trust_material_never_appears_in_repr(self):
        key = os.urandom(32)
        trust = runtime.LeaseTrustInputs(
            journal_key=key, proof_verifier=lambda *_: False,
            bootstrap_authorized=False,
        )
        display = repr(trust)
        self.assertNotIn(repr(key), display)
        self.assertNotIn("journal_key=", display)
        self.assertNotIn("proof_verifier=", display)

    def test_injection_must_be_an_explicit_install_argument(self):
        # Verify the live adapter's existing one-argument call does not
        # magically acquire rights from an environment variable.
        with mock.patch.dict(os.environ, {
            "HERMES_MANAGER_LEASE_KEY": "not-a-real-key",
            "HERMES_MANAGER_BOOTSTRAP": "1",
            "HERMES_MANAGER_PROOF_VERIFIER": "true",
        }):
            instance = runtime.LeaseRuntime({})
            self.assertTrue(instance.fenced)
            self.assertTrue(instance.core.faulted)
            self.assertIsNone(instance.status_fields()["watchdog_lease_contract_version"])

    def test_staging_installer_preserves_default_fence_and_explicit_injection(self):
        for trust in (None, self.fixture_trust(bootstrap=True)):
            with self.subTest(explicit=trust is not None):
                app = FastAPI()
                namespace = {"app": app, "require_laya": lambda: None}
                instance = runtime.install_manager_lease(namespace, trust=trust)
                route_paths = [getattr(route, "path", None) for route in app.routes]
                self.assertIn("/maintenance-lease/v1/{action}", route_paths)
                self.assertIsNone(
                    instance.status_fields()["watchdog_lease_contract_version"])
                self.assertFalse(self.journal.exists())
                self.assertEqual(instance.core.faulted, trust is None)
                # A staged FastAPI registration is not a listening server.
                self.assertFalse(app.router.lifespan_context is None)


if __name__ == "__main__":
    unittest.main()
