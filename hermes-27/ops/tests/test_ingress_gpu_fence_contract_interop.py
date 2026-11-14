"""CPU/OFFLINE cross-component negative contracts; no production code changes.

Tests exercise the staging ingress gate, staging GPU attestor, and hardened
manager lease core together. Wire traffic, PID/GPU evidence, epoch clocks,
journal keys and proof signatures are PER-TEST SYNTHETIC. The explicit
test-only HMAC issuer is NOT independently deployed/privileged attestation.
G1/G2/G3 now assert refusal rather than characterize old insecure behavior.
"""

from dataclasses import replace
import hashlib
import hmac
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ops import gpu_owner_attestor_staging as gpu
from ops import ingress_drain_gate_staging as ingress
from ops import manager_lease_core as lease


class Clock:
    def __init__(self, value: float = 2000.0):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, delta: float) -> None:
        self.value += delta


def all_routes_mock() -> ingress.CoverageManifest:
    """Synthetic flags: they do NOT authenticate or interlock actual routes."""
    return ingress.CoverageManifest(
        "synthetic-only", 1, ingress.REQUIRED_ROUTES,
        exclusive_ingress_verified=True,
        direct_local_fenced=True,
        laya_restart_fenced=True,
        trust_anchor_verified=True,
        unknown_routes=False,
    )


def card_pair() -> tuple[gpu.Card, gpu.Card]:
    return (
        gpu.Card("GPU-11111111-1111-1111-1111-111111111111",
                 "0000:03:00.0", 0, 195, 0),
        gpu.Card("GPU-22222222-2222-2222-2222-222222222222",
                 "0000:04:00.0", 1, 195, 1),
    )


def gpu_empty(clock: Clock):
    cards = tuple(gpu.CardSample(c, (), True, True, True, "disabled")
                  for c in card_pair())
    snapshot = gpu.Snapshot(
        "offline-boot", clock.value - .2, 100., cards, (),
        nvml_complete=True, proc_complete=True, root_verified=True,
    )
    next_snapshot = replace(snapshot, unix_s=clock.value, monotonic_s=100.1)
    return snapshot, next_snapshot


def gpu_tabby(clock: Clock):
    a, b = card_pair()
    birth = 3333
    owner = gpu.ProcSample(
        pid=100, start_ticks=birth, cgroup=gpu.TABBY_CGROUP, uid=1000,
        card_fds=frozenset({(195, 0), (195, 1)}),
        fds_complete=True, before_ticks=birth, after_ticks=birth,
    )
    snapshot = gpu.Snapshot(
        "offline-boot", clock.value - .2, 100.,
        (gpu.CardSample(a, (gpu.GPUProcess(100, "compute"),),
                        True, True, True, "disabled"),
         gpu.CardSample(b, (gpu.GPUProcess(100, "compute"),),
                        True, True, True, "disabled")),
        (owner,), nvml_complete=True, proc_complete=True, root_verified=True,
    )
    return snapshot, replace(snapshot, unix_s=clock.value, monotonic_s=100.1)


def admission_ready():
    """Synthetic manager idle snapshot; no real manager process is queried."""
    return {
        "current_state": "LLM_READY", "desired_mode": "AUTO",
        "queue_length": 0, "current_job": None, "direct_active": False,
        "gaming_active": False, "character_training": [],
        "force_active": False, "llm_online": True, "llm_serving": True,
        "worker_alive_s": 1.0, "transition_lock_held": True,
    }


class InteropTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.temp = tempfile.TemporaryDirectory(prefix="h27-fence-contract-offline-")
        self.addCleanup(self.temp.cleanup)
        self.journal = Path(self.temp.name) / "private" / "manager.json"
        self._journal_key = os.urandom(32)
        self._issuer_key = os.urandom(32)
        self.manager = lease.ManagerLeaseV1(
            self.journal, clock=self.clock, instance_id="a" * 32,
            journal_key=self._journal_key,
            proof_verifier=self._verify_test_only_proof,
            bootstrap_authorized=True,
        )
        self.gate = ingress.DrainGate(
            all_routes_mock(), clock=self.clock, instance_id="offline-a",
            bootstrap_for_offline_test=True,
        )
        self.lease_id = "b" * 32
        self.acquire_request = {
            "protocol": lease.PROTOCOL,
            "action": "acquire",
            "owner": lease.OWNER,
            "lease_id": self.lease_id,
            "manager_instance_id": "a" * 32,
            "fencing_token": None,
        }
        if os.name == "nt":
            # Windows cannot reproduce Linux owner/mode/atomic journal
            # semantics: NTFS presents chmod(0600) as mode 0666 and fdopen
            # text writes CRLF. This narrow test-only byte adapter permits
            # semantic HMAC/epoch tests on Windows. Linux executes the REAL
            # hardened _journal_bytes() including inode/permission checks.
            nt_private = mock.patch.object(
                lease, "private_directory", return_value=True)
            nt_private.start()
            self.addCleanup(nt_private.stop)

            def windows_synthetic_journal_read(manager):
                path = manager.path
                if (path.parent.is_symlink() or path.is_symlink() or
                        not path.is_file()):
                    raise OSError("missing or unsafe synthetic journal")
                raw = path.read_bytes()
                if len(raw) > 16384:
                    raise OSError("oversized synthetic journal")
                return raw.replace(b"\r\n", b"\n")

            nt_only = mock.patch.object(
                lease.ManagerLeaseV1, "_journal_bytes",
                windows_synthetic_journal_read)
            nt_only.start()
            self.addCleanup(nt_only.stop)

    def _sign_test_only_proof(self, kind, evidence):
        """HMAC ONLY to exercise offline core transitions, never real trust."""
        bare = {key: value for key, value in evidence.items()
                if key != "_test_only_issuer_mac"}
        body = (kind + "\0" + json.dumps(
            bare, sort_keys=True, separators=(",", ":"))).encode("utf-8")
        return {**bare, "_test_only_issuer_mac":
                hmac.new(self._issuer_key, body, hashlib.sha256).hexdigest()}

    def _verify_test_only_proof(self, kind, evidence):
        if not isinstance(evidence, dict):
            return False
        supplied = evidence.get("_test_only_issuer_mac")
        if not isinstance(supplied, str):
            return False
        expected = self._sign_test_only_proof(
            kind, evidence)["_test_only_issuer_mac"]
        return hmac.compare_digest(supplied, expected)

    def unsigned_ingress_shape(self):
        """Hostile shape-only JSON; MUST NEVER authorize a transition."""
        return {"scope": "all-tabby-clients", "blocked": True,
                "inflight": 0, "observed_at_unix": self.clock.value}

    def unsigned_gpu_shape(self):
        """Hostile shape-only JSON; MUST NEVER clear a manager fence."""
        return {
            "scope": "all-hermes-gpus-and-pids",
            "gpu_count": 2, "gpu_owner": "tabby", "gpu_free": False,
            "hermes_pids": [], "other_gpu_processes": [],
            "tabby_owns_both_gpus": True,
            "observed_at_unix": self.clock.value,
        }

    def synthetic_signed_ingress(self):
        return self._sign_test_only_proof("ingress", self.unsigned_ingress_shape())

    def synthetic_signed_gpu_restored(self):
        return self._sign_test_only_proof("gpu-restored", self.unsigned_gpu_shape())

    def _restarted_manager(self, instance_id="c" * 32):
        # Reuses only an in-memory test key, not any real root credential.
        # A restarted manager MUST NOT be permitted to bootstrap a new epoch.
        return lease.ManagerLeaseV1(
            self.journal, clock=self.clock, instance_id=instance_id,
            journal_key=self._journal_key,
            proof_verifier=self._verify_test_only_proof,
            bootstrap_authorized=False,
        )

    def acquire_with_synthetic_issuer(self):
        return self.manager.acquire(
            self.acquire_request, snapshot=admission_ready(),
            ingress_proof=self.synthetic_signed_ingress(),
        )

    def holder(self, action="renew"):
        return {
            **self.acquire_request, "action": action,
            "fencing_token": self.manager.status()["maintenance_lease"]["fencing_token"],
        }

    def test_drain_candidate_cannot_be_used_as_manager_acquire_proof(self):
        self.gate.begin_drain(key="only-offline")
        self.assertTrue(self.gate.status()["drained_candidate"])
        staging = self.gate.status()
        self.assertFalse(staging["manager_lease_admissible"])
        with self.assertRaises(lease.LeaseRejected):
            self.manager.acquire(
                self.acquire_request, snapshot=admission_ready(),
                ingress_proof=staging,
            )
        self.assertFalse(self.journal.exists())
        self.assertFalse(self.manager.fenced)

    def test_incomplete_scope_never_promotes_to_complete_lease_proof(self):
        missing = ingress.REQUIRED_ROUTES - {"uninote-tailnet-8001"}
        self.gate.install_manifest(replace(
            all_routes_mock(), covered_routes=missing,
        ))
        self.assertIn("scope-partial-routes", self.gate.status()["unknown_reasons"])
        with self.assertRaises(lease.LeaseRejected):
            self.manager.acquire(
                self.acquire_request, snapshot=admission_ready(),
                ingress_proof=self.gate.status(),
            )
        self.assertFalse(self.journal.exists())

    def test_raw_uninote_unknown_request_bytes_never_yield_zero_proof(self):
        self.assertTrue(self.gate.open_connection(
            "raw-uni", route="uninote-tailnet-8001",
        ))
        self.assertTrue(self.gate.feed_client("raw-uni", b"\x16\x03"))
        self.gate.begin_drain(key="raw")
        self.assertGreater(self.gate.status()["inflight_requests"], 0)
        self.assertFalse(self.gate.status()["drained_candidate"])
        with self.assertRaises(lease.LeaseRejected):
            lease.require_ingress(self.gate.status(), now=self.clock.value)

    def test_unparsed_server_bytes_and_keepalive_are_not_idle_proofs(self):
        self.assertTrue(self.gate.open_connection(
            "local", route="tabby-direct-loopback-8000",
        ))
        self.assertTrue(self.gate.feed_client(
            "local", b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n",
        ))
        self.assertTrue(self.gate.feed_server("local", b"HTTP/1.1 200"))
        self.gate.begin_drain(key="keepalive")
        self.assertGreater(self.gate.status()["inflight_requests"], 0)
        self.assertFalse(self.gate.status()["drained_candidate"])
        self.assertFalse(self.gate.close_connection("local"))
        self.assertEqual(self.gate.status()["phase"], "RECOVERY_REQUIRED")

    def test_unknown_direct_tabby_path_forces_fence_not_success(self):
        self.gate.observe_unknown_connection(source="new-loopback-client")
        self.assertEqual(self.gate.status()["phase"], "RECOVERY_REQUIRED")
        self.assertFalse(self.gate.open_connection(
            "second", route="tabby-direct-loopback-8000",
        ))
        self.assertFalse(self.gate.status()["manager_lease_admissible"])

    def test_laya_restart_bypass_invalidates_scope_before_drain(self):
        manifest = replace(all_routes_mock(), laya_restart_fenced=False)
        gate = ingress.DrainGate(
            manifest, clock=self.clock, bootstrap_for_offline_test=True,
        )
        gate.begin_drain(key="laya-fallback")
        self.assertFalse(gate.status()["drained_candidate"])
        self.assertIn("scope-laya-direct-restart-open",
                      gate.status()["unknown_reasons"])
        with self.assertRaises(lease.LeaseRejected):
            lease.require_ingress(gate.status(), now=self.clock.value)

    def test_external_systemd_restart_not_interlocked_by_staging_gate(self):
        # A service restart policy exists independently of in-memory
        # manager/watchdog locks. Invalidate the coverage as soon as an
        # unverified restart path is discovered. No service is touched.
        self.gate.observe_unknown_connection(
            source="unfenced-systemd-tabby-restart",
        )
        self.assertIn(
            "untracked-bypass-or-connection:unfenced-systemd-tabby-restart",
            self.gate.status()["unknown_reasons"],
        )
        self.assertFalse(self.gate.status()["manager_lease_admissible"])

    def test_bad_owner_gpu_uuid_is_blocked_and_rejected_by_lease(self):
        first, second = gpu_empty(self.clock)
        pair = list(card_pair())
        pair[0] = replace(pair[0], uuid="GPU-33333333-3333-3333-3333-333333333333")
        verdict = gpu.evaluate(
            first, second, expected=tuple(pair),
            manifest=gpu.OwnerManifest(), phase="empty-before-tabby",
            now_s=self.clock.value,
        )
        self.assertIn("uuid-pci-minor-mismatch", verdict.reasons)
        self.assertEqual(verdict.as_dict()["status"], "BLOCKED")
        with self.assertRaises(lease.LeaseRejected):
            lease.require_gpu_safe(verdict.as_dict(), now=self.clock.value)

    def test_unknown_gpu_owner_pid_is_blocked_and_cannot_release(self):
        self.acquire_with_synthetic_issuer()
        first, second = gpu_tabby(self.clock)
        extra = gpu.GPUProcess(999, "compute")
        bad_gpu = replace(
            first, cards=(
                replace(first.cards[0], owners=(extra,)),
                first.cards[1],
            ),
        )
        verdict = gpu.evaluate(
            bad_gpu, replace(bad_gpu, unix_s=second.unix_s,
                             monotonic_s=second.monotonic_s),
            expected=card_pair(), manifest=gpu.OwnerManifest(tabby={100: 3333}),
            phase="tabby-restored", now_s=self.clock.value,
        )
        self.assertIn("unknown-gpu-owner", verdict.reasons)
        with self.assertRaises(lease.LeaseRejected):
            self.manager.release(
                self.holder("release"),
                ingress_proof=self.synthetic_signed_ingress(),
                gpu_safe_proof=verdict.as_dict(), tabby_serving=True,
            )
        self.assertTrue(self.manager.fenced)

    def test_unknown_gpu_fd_or_inaccessible_proc_blocks_release(self):
        self.acquire_with_synthetic_issuer()
        first, second = gpu_tabby(self.clock)
        for modification, expected_reason in (
            ({"unknown_gpu_fd": True}, "unknown-global-drm-or-gpu-fd"),
            ({"fds_complete": False}, "proc-fd-enumeration-incomplete"),
            ({"after_ticks": 3334}, "pid-starttime-changed"),
            ({"cgroup": "/system.slice/untrusted.service"},
             "tabby-birth-or-cgroup-mismatch"),
        ):
            with self.subTest(modification=modification):
                bad = replace(first, processes=(
                    replace(first.processes[0], **modification),
                ))
                verdict = gpu.evaluate(
                    bad, replace(bad, unix_s=second.unix_s,
                                 monotonic_s=second.monotonic_s),
                    expected=card_pair(), manifest=gpu.OwnerManifest(tabby={100: 3333}),
                    phase="tabby-restored", now_s=self.clock.value,
                )
                self.assertIn(expected_reason, verdict.reasons)
                with self.assertRaises(lease.LeaseRejected):
                    self.manager.release(
                        self.holder("release"),
                        ingress_proof=self.synthetic_signed_ingress(),
                        gpu_safe_proof=verdict.as_dict(), tabby_serving=True,
                    )
                self.assertTrue(self.manager.fenced)

    def test_consistent_gpu_candidate_still_cannot_clear_manager_fence(self):
        self.acquire_with_synthetic_issuer()
        first, second = gpu_tabby(self.clock)
        verdict = gpu.evaluate(
            first, second, expected=card_pair(),
            manifest=gpu.OwnerManifest(tabby={100: 3333}),
            phase="tabby-restored", now_s=self.clock.value,
        )
        self.assertEqual(verdict.candidate, "tabby")
        self.assertEqual(verdict.reasons, (
            "independent-all-driver-contexts-coverage-unavailable",
        ))
        with self.assertRaises(lease.LeaseRejected):
            self.manager.release(
                self.holder("release"),
                ingress_proof=self.synthetic_signed_ingress(),
                gpu_safe_proof=verdict.as_dict(), tabby_serving=True,
            )
        self.assertTrue(self.manager.fenced)

    def test_staging_ingress_and_owner_combination_cannot_finish_recovery(self):
        self.acquire_with_synthetic_issuer()
        self.clock.advance(lease.TTL_SECONDS + 1)
        self.assertTrue(self.manager.fenced)
        self.assertTrue(self.manager.recovery_required)
        self.gate.begin_drain(key="expired-fence")
        first, second = gpu_empty(self.clock)
        verdict = gpu.evaluate(
            first, second, expected=card_pair(),
            manifest=gpu.OwnerManifest(), phase="empty-before-tabby",
            now_s=self.clock.value,
        )
        with self.assertRaises(lease.LeaseRejected):
            self.manager.assert_recovery_safe(
                ingress_proof=self.gate.status(),
                gpu_safe_proof=verdict.as_dict(),
            )
        with self.assertRaises(lease.LeaseRejected):
            self.manager.finish_recovery(
                ingress_proof=self.gate.status(),
                gpu_safe_proof=verdict.as_dict(), tabby_serving=True,
            )
        self.assertTrue(self.manager.fenced)

    def test_manager_renew_rejects_staging_scope_and_preserves_fence(self):
        self.acquire_with_synthetic_issuer()
        token = self.holder("renew")["fencing_token"]
        self.gate.begin_drain(key="renew-test")
        with self.assertRaises(lease.LeaseRejected):
            self.manager.renew(
                self.holder("renew"), ingress_proof=self.gate.status(),
            )
        self.assertEqual(self.holder("renew")["fencing_token"], token)
        self.assertTrue(self.manager.fenced)

    def test_manager_renew_rejects_expired_or_future_external_scope_timestamp(self):
        self.acquire_with_synthetic_issuer()
        for timestamp in (self.clock.value - 11, self.clock.value + 1):
            with self.subTest(timestamp=timestamp):
                with self.assertRaises(lease.LeaseRejected):
                    self.manager.renew(
                        self.holder("renew"),
                        ingress_proof={
                            **self.synthetic_signed_ingress(),
                            "observed_at_unix": timestamp,
                        },
                    )
                self.assertTrue(self.manager.fenced)

    def test_drain_epochs_idempotent_and_crash_resume_remain_fenced(self):
        first = self.gate.begin_drain(key="one")
        self.clock.advance(4)
        self.assertEqual(self.gate.begin_drain(key="one"), first)
        journal = self.gate.journal()
        restart = ingress.DrainGate(
            all_routes_mock(), clock=self.clock,
            instance_id="offline-b", restored=journal,
        )
        self.assertEqual(restart.resume_drain(key="one"), first)
        self.assertEqual(restart.status()["phase"], "RECOVERY_REQUIRED")
        with self.assertRaises(ingress.GateError):
            restart.begin_drain(key="two")
        self.assertFalse(restart.status()["manager_lease_admissible"])

    def test_drain_journal_missing_after_process_restart_never_reopens(self):
        self.gate.begin_drain(key="lost-journal")
        # Neither ingress drain nor the hardened manager lease may
        # interpret a missing persisted epoch as permission to reopen.
        reloaded = ingress.DrainGate(
            all_routes_mock(), clock=self.clock, instance_id="offline-b",
        )
        self.assertEqual(reloaded.status()["phase"], "RECOVERY_REQUIRED")
        self.assertFalse(reloaded.open_connection(
            "retry", route="uninote-tailnet-8001",
        ))
        self.assertFalse(reloaded.status()["manager_lease_admissible"])

    def test_drain_manifest_epoch_mismatch_remains_recovery_required(self):
        self.gate.begin_drain(key="epoch")
        old_record = self.gate.journal()
        tampered = replace(old_record, manifest_revision=old_record.manifest_revision + 1)
        restarted = ingress.DrainGate(
            all_routes_mock(), clock=self.clock,
            instance_id="offline-after-restart", restored=tampered,
        )
        self.assertEqual(restarted.status()["phase"], "RECOVERY_REQUIRED")
        self.assertIn("persisted-journal-corrupt-or-coverage-drift",
                      restarted.status()["unknown_reasons"])
        self.assertFalse(restarted.status()["manager_lease_admissible"])

    def test_unknown_ingress_after_drain_cannot_be_recovered_by_timeout(self):
        self.assertTrue(self.gate.open_connection(
            "unparsed", route="uninote-tailnet-8001",
        ))
        self.gate.feed_client("unparsed", b"GET /partial")
        self.gate.begin_drain(key="timeout", timeout_s=3)
        self.clock.advance(4)
        self.gate.tick()
        self.assertEqual(self.gate.status()["phase"], "RECOVERY_REQUIRED")
        self.assertTrue(self.gate.status()["drain_timeout"])
        self.assertFalse(self.gate.status()["manager_lease_admissible"])

    def test_manager_fencing_token_never_replaced_by_replayed_acquire(self):
        self.acquire_with_synthetic_issuer()
        old_counter = json.loads(self.journal.read_text())["counter"]
        old_token = self.holder("renew")["fencing_token"]
        self.clock.advance(1)
        self.acquire_with_synthetic_issuer()
        self.assertEqual(old_token, self.holder("renew")["fencing_token"])
        self.assertEqual(old_counter, json.loads(self.journal.read_text())["counter"])

    def test_manager_release_never_accepts_stale_lease_id_or_token(self):
        self.acquire_with_synthetic_issuer()
        original = self.holder("release")
        for changed in ({"lease_id": "c" * 32},
                        {"fencing_token": original["fencing_token"] + 1},
                        {"manager_instance_id": "c" * 32}):
            with self.subTest(changed=changed), self.assertRaises(lease.LeaseRejected):
                self.manager.release(
                    {**original, **changed},
                    ingress_proof=self.synthetic_signed_ingress(),
                    gpu_safe_proof=self.synthetic_signed_gpu_restored(),
                    tabby_serving=True,
                )
            self.assertTrue(self.manager.fenced)

    def test_manager_rejects_wrong_fencing_token_after_time_advances(self):
        self.acquire_with_synthetic_issuer()
        current = self.holder("renew")["fencing_token"]
        self.clock.advance(5)
        for token in (current - 1, current + 1):
            with self.subTest(token=token), self.assertRaises(lease.LeaseRejected):
                self.manager.renew(
                    {**self.holder("renew"), "fencing_token": token},
                    ingress_proof=self.synthetic_signed_ingress(),
                )
        self.assertTrue(self.manager.fenced)

    def test_expired_lease_remains_fenced_across_manager_and_ingress_restart(self):
        self.acquire_with_synthetic_issuer()
        self.gate.begin_drain(key="persistent")
        self.clock.advance(lease.TTL_SECONDS + 2)
        self.assertTrue(self.manager.recovery_required)
        next_manager = self._restarted_manager()
        next_ingress = ingress.DrainGate(
            all_routes_mock(), clock=self.clock,
            instance_id="offline-b", restored=self.gate.journal(),
        )
        self.assertTrue(next_manager.fenced)
        self.assertTrue(next_manager.recovery_required)
        self.assertEqual(next_ingress.status()["phase"], "RECOVERY_REQUIRED")
        self.assertFalse(next_ingress.status()["manager_lease_admissible"])
        with self.assertRaises(lease.LeaseRejected):
            next_manager.acquire(
                {**self.acquire_request, "manager_instance_id": "c" * 32,
                 "lease_id": "d" * 32},
                snapshot=admission_ready(),
                ingress_proof=self.synthetic_signed_ingress(),
            )

    def test_lease_monotonic_counter_survives_clean_verified_recovery(self):
        self.acquire_with_synthetic_issuer()
        self.assertEqual(self.holder("release")["fencing_token"], 1)
        # Explicitly HMAC-signed TEST-ONLY synthetic evidence exercises
        # durable counter transitions; no physical GPU/ingress is observed.
        self.manager.release(
            self.holder("release"),
            ingress_proof=self.synthetic_signed_ingress(),
            gpu_safe_proof=self.synthetic_signed_gpu_restored(),
            tabby_serving=True,
        )
        next_manager = self._restarted_manager()
        ack = next_manager.acquire(
            {**self.acquire_request, "manager_instance_id": "c" * 32,
             "lease_id": "d" * 32},
            snapshot=admission_ready(),
            ingress_proof=self.synthetic_signed_ingress(),
        )
        self.assertTrue(ack["ok"])
        self.assertEqual(next_manager.status()["maintenance_lease"]["fencing_token"], 2)

    def test_G1_deleted_manager_journal_fences_current_and_restarted_manager(self):
        """G1 secure contract: loss of last journal NEVER starts a new epoch."""
        self.acquire_with_synthetic_issuer()
        self.assertTrue(self.manager.fenced)
        self.journal.unlink()
        self.assertTrue(self.manager.fenced)
        self.assertTrue(self.manager.faulted)
        next_manager = self._restarted_manager()
        self.assertTrue(next_manager.fenced)
        self.assertTrue(next_manager.faulted)
        self.assertTrue(next_manager.recovery_required)
        with self.assertRaises(lease.LeaseStoreFault):
            next_manager.acquire(
                {**self.acquire_request, "manager_instance_id": "c" * 32,
                 "lease_id": "d" * 32},
                snapshot=admission_ready(),
                ingress_proof=self.synthetic_signed_ingress(),
            )
        self.assertFalse(self.journal.exists())

    def test_G2_rollback_clock_refuses_renew_and_keeps_expiry(self):
        """G2 secure contract: -30s clock drift denies renew, never shortens."""
        self.acquire_with_synthetic_issuer()
        original = self.manager.status()["maintenance_lease"]["expires_at_unix"]
        holder = self.holder("renew")
        self.clock.value -= 30
        with self.assertRaisesRegex(lease.LeaseRejected, "rollback"):
            self.manager.renew(
                holder, ingress_proof=self.synthetic_signed_ingress(),
            )
        changed = self.manager.status()["maintenance_lease"]["expires_at_unix"]
        self.assertEqual(changed, original)
        self.assertTrue(self.manager.fenced)
        self.assertTrue(self.manager.recovery_required)
        self.assertTrue(self._restarted_manager().recovery_required)

    def test_G3_unsigned_shape_only_proofs_cannot_clear_lease(self):
        """G3 secure contract: plausible JSON without issuer MAC is denied."""
        self.acquire_with_synthetic_issuer()
        self.assertTrue(self.manager.fenced)
        for ingress_proof, gpu_proof in (
                (self.unsigned_ingress_shape(),
                 self.synthetic_signed_gpu_restored()),
                (self.synthetic_signed_ingress(),
                 self.unsigned_gpu_shape()),
                ({**self.synthetic_signed_ingress(), "inflight": 0,
                  "_test_only_issuer_mac": "0" * 64},
                 self.synthetic_signed_gpu_restored())):
            with self.subTest(ingress=ingress_proof, gpu=gpu_proof):
                with self.assertRaises(lease.LeaseRejected):
                    self.manager.release(
                        self.holder("release"),
                        ingress_proof=ingress_proof,
                        gpu_safe_proof=gpu_proof,
                        tabby_serving=True,
                    )
                self.assertTrue(self.manager.fenced)
                self.assertIsNotNone(self.manager.status()["maintenance_lease"])

    def test_nvml_ownership_race_across_two_snapshots_cannot_clear_lease(self):
        self.acquire_with_synthetic_issuer()
        first, second = gpu_tabby(self.clock)
        changed = replace(second, processes=(
            replace(second.processes[0], after_ticks=3334),
        ))
        decision = gpu.evaluate(
            first, changed, expected=card_pair(),
            manifest=gpu.OwnerManifest(tabby={100: 3333}),
            phase="tabby-restored", now_s=self.clock.value,
        )
        self.assertIn("gpu-process-or-pid-changed-between-snapshots",
                      decision.reasons)
        with self.assertRaises(lease.LeaseRejected):
            self.manager.release(
                self.holder("release"),
                ingress_proof=self.synthetic_signed_ingress(),
                gpu_safe_proof=decision.as_dict(),
                tabby_serving=True,
            )
        self.assertTrue(self.manager.fenced)


if __name__ == "__main__":
    unittest.main()
