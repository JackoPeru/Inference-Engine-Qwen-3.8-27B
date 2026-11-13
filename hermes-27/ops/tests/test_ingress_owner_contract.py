"""Offline proof-contract tests only; no sockets, credentials, NVML or GPU."""

import unittest

from ops.manager_lease_core import LeaseRejected, require_gpu_safe, require_ingress


class CompleteIngressProofTests(unittest.TestCase):
    now = 1794600000.0

    def proof(self, **changes):
        return {
            "scope": "all-tabby-clients", "blocked": True,
            "inflight": 0, "observed_at_unix": self.now, **changes,
        }

    def test_complete_fresh_and_drained_shape(self):
        require_ingress(self.proof(), now=self.now)

    def test_absent_proof_refused(self):
        with self.assertRaises(LeaseRejected):
            require_ingress(None, now=self.now)

    def test_partial_scope_refused(self):
        with self.assertRaises(LeaseRejected):
            require_ingress(self.proof(scope="dashboard-only"), now=self.now)

    def test_unknown_stream_counter_refused(self):
        with self.assertRaises(LeaseRejected):
            require_ingress(self.proof(inflight=None), now=self.now)

    def test_one_active_request_refused(self):
        with self.assertRaises(LeaseRejected):
            require_ingress(self.proof(inflight=1), now=self.now)

    def test_negative_counter_refused(self):
        with self.assertRaises(LeaseRejected):
            require_ingress(self.proof(inflight=-1), now=self.now)

    def test_reopened_proxy_refused(self):
        with self.assertRaises(LeaseRejected):
            require_ingress(self.proof(blocked=False), now=self.now)

    def test_expired_or_future_ingress_evidence_refused(self):
        for timestamp in (self.now - 11, self.now + 1):
            with self.subTest(timestamp=timestamp), self.assertRaises(LeaseRejected):
                require_ingress(self.proof(observed_at_unix=timestamp), now=self.now)

    def test_boolean_counter_refused(self):
        with self.assertRaises(LeaseRejected):
            require_ingress(self.proof(inflight=False), now=self.now)


class BothGpuOwnerProofTests(unittest.TestCase):
    now = 1794600000.0

    def proof(self, **changes):
        return {
            "scope": "all-hermes-gpus-and-pids",
            "gpu_count": 2,
            "gpu_owner": "none",
            "gpu_free": True,
            "hermes_pids": [],
            "other_gpu_processes": [],
            "observed_at_unix": self.now,
            **changes,
        }

    def test_free_both_gpus_with_no_pid_shape(self):
        require_gpu_safe(self.proof(), now=self.now)

    def test_only_one_gpu_refused(self):
        with self.assertRaises(LeaseRejected):
            require_gpu_safe(self.proof(gpu_count=1), now=self.now)

    def test_gpu_count_boolean_refused(self):
        with self.assertRaises(LeaseRejected):
            require_gpu_safe(self.proof(gpu_count=True), now=self.now)

    def test_unknown_gpu_owner_refused(self):
        with self.assertRaises(LeaseRejected):
            require_gpu_safe(self.proof(gpu_owner="unknown"), now=self.now)

    def test_missing_pid_list_refused(self):
        proof = self.proof()
        del proof["hermes_pids"]
        with self.assertRaises(LeaseRejected):
            require_gpu_safe(proof, now=self.now)

    def test_hermes_pid_still_alive_refused(self):
        with self.assertRaises(LeaseRejected):
            require_gpu_safe(self.proof(hermes_pids=[4242]), now=self.now)

    def test_other_gpu_actor_refused(self):
        with self.assertRaises(LeaseRejected):
            require_gpu_safe(self.proof(other_gpu_processes=[123]), now=self.now)

    def test_incomplete_gpu_scope_refused(self):
        with self.assertRaises(LeaseRejected):
            require_gpu_safe(self.proof(scope="gpu0-only"), now=self.now)

    def test_stale_gpu_evidence_refused(self):
        with self.assertRaises(LeaseRejected):
            require_gpu_safe(self.proof(observed_at_unix=self.now - 11), now=self.now)

    def test_cpu_only_free_assertion_inconsistent_refused(self):
        with self.assertRaises(LeaseRejected):
            require_gpu_safe(self.proof(gpu_free=False), now=self.now)

    def test_post_restore_requires_exclusive_tabby_owner(self):
        with self.assertRaises(LeaseRejected):
            require_gpu_safe(self.proof(), now=self.now, after_tabby_restore=True)
        proof = self.proof(gpu_owner="tabby", gpu_free=False, tabby_owns_both_gpus=True)
        require_gpu_safe(proof, now=self.now, after_tabby_restore=True)

    def test_fresh_shape_does_not_prove_physical_provenance(self):
        # A syntactically valid forged dict is indistinguishable here.
        # Only the FUTURE privileged checker and its protected transport
        # could make this an actual ownership attestation. This is a known
        # NO-GO limitation, not evidence of hardware safety.
        synthetic_untrusted_proof = self.proof()
        require_gpu_safe(synthetic_untrusted_proof, now=self.now)


if __name__ == "__main__":
    unittest.main()
