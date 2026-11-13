"""100% offline deterministic staging-attestor tests: no NVML or GPU calls."""

from dataclasses import replace
import contextlib
import ast
import io
import json
import os
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from ops import gpu_owner_attestor_staging as a
from ops.manager_lease_core import LeaseRejected, require_gpu_safe


def synthetic_stat(pid: int = 100, ticks: int = 12345, *, name: str = "qwen ) training") -> str:
    # Field 3 is state, field 22 is starttime; comm can contain both
    # spaces and closing parentheses without altering indexing.
    return f"{pid} ({name}) " + " ".join(["S"] + ["1"] * 18 + [str(ticks)]) + "\n"


class OwnerSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.card0 = a.Card("GPU-11111111-1111-1111-1111-111111111111", "0000:03:00.0", 0, 195, 0)
        self.card1 = a.Card("GPU-22222222-2222-2222-2222-222222222222", "0000:04:00.0", 1, 195, 1)
        self.expected = (self.card0, self.card1)
        self.now = 10000.0
        self.empty = a.Snapshot(
            "boot-id-123", self.now - 1, 500.0,
            (self.card(self.card0), self.card(self.card1)), (),
            nvml_complete=True, proc_complete=True, root_verified=True,
        )
        self.boot_pair = (self.empty, replace(self.empty, unix_s=self.now, monotonic_s=500.2))
        self.proc_tabby = a.ProcSample(
            100, 12345, a.TABBY_CGROUP, 1000,
            frozenset(((195, 0), (195, 1))),
            False, True, 12345, 12345,
        )
        self.tabby = replace(self.empty,
                             cards=(self.card(self.card0, (a.GPUProcess(100, "compute"),)),
                                    self.card(self.card1, (a.GPUProcess(100, "graphics"),))),
                             processes=(self.proc_tabby,))
        self.tabby_pair = (self.tabby, replace(self.tabby, unix_s=self.now, monotonic_s=500.2))

    def card(self, identity, owners=()):
        return a.CardSample(identity, tuple(owners), True, True, True, "disabled")

    def evaluate_empty(self, **kwargs):
        opts = dict(expected=self.expected, manifest=a.OwnerManifest(),
                    phase="empty-before-tabby", now_s=self.now)
        opts.update(kwargs)
        return a.evaluate(self.boot_pair[0], self.boot_pair[1], **opts)

    def evaluate_tabby(self, **kwargs):
        opts = dict(expected=self.expected, manifest=a.OwnerManifest(tabby={100: 12345}),
                    phase="tabby-restored", now_s=self.now)
        opts.update(kwargs)
        return a.evaluate(self.tabby_pair[0], self.tabby_pair[1], **opts)

    def assert_blocked(self, result, reason=None):
        self.assertIsInstance(result, a.Verdict)
        self.assertEqual(result.as_dict()["status"], "BLOCKED")
        self.assertFalse(result.as_dict()["manager_lease_admissible"])
        self.assertNotIn("gpu_free", result.as_dict())
        self.assertNotIn("gpu_owner", result.as_dict())
        self.assertNotIn("scope", result.as_dict())
        if reason:
            self.assertIn(reason, result.reasons)

    def test_empty_pair_syntactically_consistent_but_denied(self):
        result = self.evaluate_empty()
        self.assert_blocked(result, "independent-all-driver-contexts-coverage-unavailable")
        self.assertEqual(result.candidate, "none")

    def test_tabby_pair_syntactically_consistent_but_denied(self):
        result = self.evaluate_tabby()
        self.assert_blocked(result, "independent-all-driver-contexts-coverage-unavailable")
        self.assertEqual(result.candidate, "tabby")

    def test_immutable_status_always_blocked_regardless_of_shape(self):
        for pair, phase, manifest in (
            (self.boot_pair, "empty-before-tabby", a.OwnerManifest()),
            (self.tabby_pair, "tabby-restored", a.OwnerManifest(tabby={100: 12345})),
        ):
            with self.subTest(phase=phase):
                result = a.evaluate(*pair, expected=self.expected, phase=phase,
                                    manifest=manifest, now_s=self.now)
                self.assert_blocked(result)

    def test_candidate_is_rejected_by_existing_manager_lease_proof_gate(self):
        result = self.evaluate_tabby()
        with self.assertRaises(LeaseRejected):
            require_gpu_safe(result.as_dict(), now=self.now, after_tabby_restore=True)

    def test_uuid_mismatch(self):
        wrong = replace(self.card0, uuid="GPU-33333333-3333-3333-3333-333333333333")
        self.assert_blocked(self.evaluate_empty(expected=(wrong, self.card1)),
                            "uuid-pci-minor-mismatch")

    def test_pci_mismatch(self):
        wrong = replace(self.card1, pci="0000:08:00.0")
        self.assert_blocked(self.evaluate_empty(expected=(self.card0, wrong)),
                            "uuid-pci-minor-mismatch")

    def test_device_minor_mismatch(self):
        wrong = replace(self.card0, dev_minor=4)
        self.assert_blocked(self.evaluate_empty(expected=(wrong, self.card1)),
                            "expected-card-device-invalid")

    def test_duplicate_gpu_identity(self):
        self.assert_blocked(self.evaluate_empty(expected=(self.card0, self.card0)),
                            "two-distinct-expected-cards-required")

    def test_duplicate_gpu_uuid_but_different_pci(self):
        wrong = replace(self.card1, uuid=self.card0.uuid)
        self.assert_blocked(self.evaluate_empty(expected=(self.card0, wrong)),
                            "card-identity-duplicate")

    def test_unknown_mig_mode(self):
        card = replace(self.empty.cards[0], mig="unknown")
        early = replace(self.empty, cards=(card, self.empty.cards[1]))
        self.assert_blocked(a.evaluate(early, early, expected=self.expected,
                                      manifest=a.OwnerManifest(), phase="empty-before-tabby",
                                      now_s=self.now), "mig-unsupported-or-unknown")

    def test_mig_enabled(self):
        card = replace(self.empty.cards[1], mig="enabled")
        early = replace(self.empty, cards=(self.empty.cards[0], card))
        self.assert_blocked(a.evaluate(early, early, expected=self.expected,
                                      manifest=a.OwnerManifest(), phase="empty-before-tabby",
                                      now_s=self.now), "mig-unsupported-or-unknown")

    def test_one_nvml_gpu_denied(self):
        one = replace(self.empty, cards=(self.empty.cards[0],))
        self.assert_blocked(a.evaluate(one, one, expected=self.expected,
                                      manifest=a.OwnerManifest(), phase="empty-before-tabby",
                                      now_s=self.now), "nvml-card-count-not-two")

    def test_missing_compute_graphics_or_mps_enumeration_denied(self):
        for field in ("compute_complete", "graphics_complete", "mps_complete"):
            first = replace(self.empty.cards[0], **{field: False})
            sample = replace(self.empty, cards=(first, self.empty.cards[1]))
            with self.subTest(missing=field):
                result = a.evaluate(sample, sample, expected=self.expected,
                                    manifest=a.OwnerManifest(), phase="empty-before-tabby",
                                    now_s=self.now)
                self.assert_blocked(result, "nvml-compute-graphics-mps-incomplete")

    def test_incomplete_nvml_global_scan(self):
        early = replace(self.empty, nvml_complete=False)
        self.assert_blocked(a.evaluate(early, early, expected=self.expected,
                                      manifest=a.OwnerManifest(), phase="empty-before-tabby",
                                      now_s=self.now), "root-or-enumeration-not-complete")

    def test_incomplete_proc_global_scan(self):
        early = replace(self.empty, proc_complete=False)
        self.assert_blocked(a.evaluate(early, early, expected=self.expected,
                                      manifest=a.OwnerManifest(), phase="empty-before-tabby",
                                      now_s=self.now), "root-or-enumeration-not-complete")

    def test_insufficient_permissions_root_false(self):
        early = replace(self.empty, root_verified=False)
        self.assert_blocked(a.evaluate(early, early, expected=self.expected,
                                      manifest=a.OwnerManifest(), phase="empty-before-tabby",
                                      now_s=self.now), "root-or-enumeration-not-complete")

    def test_wrong_boot_id(self):
        late = replace(self.boot_pair[1], boot_id="different-boot")
        self.assert_blocked(a.evaluate(self.empty, late, expected=self.expected,
                                      manifest=a.OwnerManifest(), phase="empty-before-tabby",
                                      now_s=self.now), "boot-identity-changed")

    def test_stale_observation_time(self):
        early, late = self.boot_pair
        self.assert_blocked(a.evaluate(early, replace(late, unix_s=self.now - 11),
                                      expected=self.expected, manifest=a.OwnerManifest(),
                                      phase="empty-before-tabby", now_s=self.now),
                            "snapshot-time-invalid-or-expired")

    def test_future_observation_time(self):
        self.assert_blocked(self.evaluate_empty(now_s=self.now - 1),
                            "snapshot-time-invalid-or-expired")

    def test_changed_monotonic_duration(self):
        late = replace(self.boot_pair[1], monotonic_s=530.0)
        self.assert_blocked(a.evaluate(self.empty, late, expected=self.expected,
                                      manifest=a.OwnerManifest(), phase="empty-before-tabby",
                                      now_s=self.now), "snapshot-time-invalid-or-expired")

    def test_birth_ticks_changed_during_proc_scan(self):
        corrupt = replace(self.proc_tabby, after_ticks=12346)
        bad = replace(self.tabby, processes=(corrupt,))
        self.assert_blocked(a.evaluate(bad, bad, expected=self.expected,
                                      manifest=a.OwnerManifest(tabby={100: 12345}),
                                      phase="tabby-restored", now_s=self.now),
                            "pid-starttime-changed")

    def test_birth_ticks_mismatch_manifest(self):
        self.assert_blocked(self.evaluate_tabby(manifest=a.OwnerManifest(tabby={100: 55555})),
                            "tabby-birth-or-cgroup-mismatch")

    def test_tabby_cgroup_mismatch(self):
        rogue = replace(self.proc_tabby, cgroup="/system.slice/unrelated.service")
        bad = replace(self.tabby, processes=(rogue,))
        self.assert_blocked(a.evaluate(bad, bad, expected=self.expected,
                                      manifest=a.OwnerManifest(tabby={100: 12345}),
                                      phase="tabby-restored", now_s=self.now),
                            "tabby-birth-or-cgroup-mismatch")

    def test_missing_tabby_manifest_denied(self):
        self.assert_blocked(self.evaluate_tabby(manifest=a.OwnerManifest()),
                            "unknown-gpu-owner")

    def test_hermes_manifest_after_tabby_restore_denied(self):
        self.assert_blocked(self.evaluate_tabby(
            manifest=a.OwnerManifest(tabby={100: 12345}, hermes={999: 12345})),
            "hermes-owner-manifest-not-empty")

    def test_tabby_hermes_pid_collision_denied(self):
        self.assert_blocked(self.evaluate_tabby(
            manifest=a.OwnerManifest(tabby={100: 12345}, hermes={100: 12345})),
            "tabby-hermes-pid-collision")

    def test_owner_manifest_bool_pid_denied(self):
        self.assert_blocked(self.evaluate_tabby(
            manifest=a.OwnerManifest(tabby={True: 12345})),
            "manifest-pid-birth-malformed")

    def test_owner_manifest_birth_missing_from_proc(self):
        self.assert_blocked(self.evaluate_tabby(
            manifest=a.OwnerManifest(tabby={100: 12345, 101: 12346})),
            "manifest-pid-gone-or-reused")

    def test_unknown_nvml_owner_denied(self):
        unknown = a.GPUProcess(999, "compute")
        bad = replace(self.tabby, cards=(self.card(self.card0, (unknown,)),
                                         self.tabby.cards[1]))
        self.assert_blocked(a.evaluate(bad, bad, expected=self.expected,
                                      manifest=a.OwnerManifest(tabby={100: 12345}),
                                      phase="tabby-restored", now_s=self.now),
                            "unknown-gpu-owner")

    def test_nvml_pid_not_in_proc_denied(self):
        bad = replace(self.tabby, processes=())
        self.assert_blocked(a.evaluate(bad, bad, expected=self.expected,
                                      manifest=a.OwnerManifest(tabby={100: 12345}),
                                      phase="tabby-restored", now_s=self.now),
                            "nvml-pid-missing-from-proc")

    def test_nvml_proc_gpu_device_mismatch_denied(self):
        wrong = replace(self.proc_tabby, card_fds=frozenset(((195, 0),)))
        bad = replace(self.tabby, processes=(wrong,))
        self.assert_blocked(a.evaluate(bad, bad, expected=self.expected,
                                      manifest=a.OwnerManifest(tabby={100: 12345}),
                                      phase="tabby-restored", now_s=self.now),
                            "nvml-vs-proc-device-owner-disagreement")

    def test_unmapped_gpu_device_denied(self):
        wrong = replace(self.proc_tabby, card_fds=frozenset(((195, 0), (195, 1), (195, 4))))
        bad = replace(self.tabby, processes=(wrong,))
        self.assert_blocked(a.evaluate(bad, bad, expected=self.expected,
                                      manifest=a.OwnerManifest(tabby={100: 12345}),
                                      phase="tabby-restored", now_s=self.now),
                            "unmapped-gpu-device-fd")

    def test_unknown_nvidiactl_or_drm_fd_denied(self):
        wrong = replace(self.proc_tabby, unknown_gpu_fd=True)
        bad = replace(self.tabby, processes=(wrong,))
        self.assert_blocked(a.evaluate(bad, bad, expected=self.expected,
                                      manifest=a.OwnerManifest(tabby={100: 12345}),
                                      phase="tabby-restored", now_s=self.now),
                            "unknown-global-drm-or-gpu-fd")

    def test_fd_permissions_denied(self):
        wrong = replace(self.proc_tabby, fds_complete=False)
        bad = replace(self.tabby, processes=(wrong,))
        self.assert_blocked(a.evaluate(bad, bad, expected=self.expected,
                                      manifest=a.OwnerManifest(tabby={100: 12345}),
                                      phase="tabby-restored", now_s=self.now),
                            "proc-fd-enumeration-incomplete")

    def test_duplicate_nvml_pid_kind_denied(self):
        repeated = (a.GPUProcess(100, "compute"),) * 2
        wrong = replace(self.tabby, cards=(self.card(self.card0, repeated), self.tabby.cards[1]))
        self.assert_blocked(a.evaluate(wrong, wrong, expected=self.expected,
                                      manifest=a.OwnerManifest(tabby={100: 12345}),
                                      phase="tabby-restored", now_s=self.now),
                            "nvml-duplicate-entry")

    def test_unknown_nvml_category_denied(self):
        wrong = replace(self.tabby, cards=(self.card(self.card0, (a.GPUProcess(100, "other"),)),
                                           self.tabby.cards[1]))
        self.assert_blocked(a.evaluate(wrong, wrong, expected=self.expected,
                                      manifest=a.OwnerManifest(tabby={100: 12345}),
                                      phase="tabby-restored", now_s=self.now),
                            "nvml-invalid-owner-entry")

    def test_ownership_changes_between_snapshots_denied(self):
        self.assert_blocked(a.evaluate(self.empty, self.tabby,
                                      expected=self.expected, manifest=a.OwnerManifest(),
                                      phase="empty-before-tabby", now_s=self.now),
                            "gpu-process-or-pid-changed-between-snapshots")

    def test_tabby_missing_on_second_gpu_denied(self):
        wrong = replace(self.tabby, cards=(self.tabby.cards[0], self.card(self.card1)))
        self.assert_blocked(a.evaluate(wrong, wrong, expected=self.expected,
                                      manifest=a.OwnerManifest(tabby={100: 12345}),
                                      phase="tabby-restored", now_s=self.now),
                            "tabby-missing-from-gpu")

    def test_bad_phase_denied(self):
        self.assert_blocked(self.evaluate_empty(phase="release-now"), "phase-unknown")


class ProcAndCollectorTests(unittest.TestCase):
    def test_proc_parser_survives_complex_comm_and_trailing_newline(self):
        self.assertEqual(a.parse_start_ticks(synthetic_stat(pid=13, ticks=777), 13), 777)

    def test_proc_parser_denies_wrong_pid(self):
        with self.assertRaisesRegex(a.Blocked, "malformed"):
            a.parse_start_ticks(synthetic_stat(pid=13), 14)

    def test_proc_parser_denies_truncated_record(self):
        with self.assertRaises(a.Blocked):
            a.parse_start_ticks("3 (worker) S 1 2 3\n", 3)

    def test_proc_parser_denies_embedded_newline(self):
        with self.assertRaises(a.Blocked):
            a.parse_start_ticks("3 (worker)\nS " + "1 " * 30, 3)

    def test_cgroup_v2_parse_exact_unit(self):
        self.assertEqual(a._cgroup_v2("0::/system.slice/hermes-tabby.service\n"),
                         a.TABBY_CGROUP)

    def test_cgroup_v1_only_rejected(self):
        with self.assertRaises(a.Blocked):
            a._cgroup_v2("3:cpu:/system.slice/hermes-tabby.service\n")

    def test_multiple_cgroup_v2_records_rejected(self):
        with self.assertRaises(a.Blocked):
            a._cgroup_v2("0::/system.slice/hermes-tabby.service\n0::/other\n")

    def test_cgroup_v2_rejects_nul_or_parent_traversal(self):
        for value in ("0::/system.slice/../other\n", "0::/bad\x00unit\n"):
            with self.subTest(value=repr(value)), self.assertRaises(a.Blocked):
                a._cgroup_v2(value)

    def test_require_linux_root_refuses_nonroot(self):
        with self.assertRaisesRegex(a.Blocked, "root"):
            a.require_root(euid=lambda: 1000, platform="linux")

    def test_require_linux_root_refuses_windows_even_if_root_mock(self):
        with self.assertRaisesRegex(a.Blocked, "linux"):
            a.require_root(euid=lambda: 0, platform="win32")

    def test_require_linux_root_accepts_injected_root_linux_only(self):
        a.require_root(euid=lambda: 0, platform="linux")

    def test_root_collector_nonroot_does_not_call_nvml(self):
        fake_nvml = mock.Mock()
        card0 = a.Card("GPU-11111111-1111-1111-1111-111111111111", "0000:03:00.0", 0, 195, 0)
        card1 = a.Card("GPU-22222222-2222-2222-2222-222222222222", "0000:04:00.0", 1, 195, 1)
        collector = a.RootReadOnlyCollector((card0, card1), euid=lambda: 1000,
                                            platform="linux", nvml=fake_nvml)
        with self.assertRaises(a.Blocked):
            collector.collect_once()
        fake_nvml.nvmlInit.assert_not_called()

    def test_root_collector_nvml_failure_always_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)
            (p / "sys/kernel/random").mkdir(parents=True)
            (p / "sys/kernel/random/boot_id").write_text("fake-boot-id")
            fake_nvml = mock.Mock()
            fake_nvml.nvmlInit.side_effect = RuntimeError("GPU driver unavailable")
            cards = (a.Card("GPU-11111111-1111-1111-1111-111111111111", "0000:03:00.0", 0, 195, 0),
                     a.Card("GPU-22222222-2222-2222-2222-222222222222", "0000:04:00.0", 1, 195, 1))
            collector = a.RootReadOnlyCollector(cards, proc_root=p,
                                                euid=lambda: 0, platform="linux",
                                                nvml=fake_nvml)
            with self.assertRaisesRegex(a.Blocked, "unavailable"):
                collector.collect_once()

    def test_stage_probe_catches_readonly_collector_fault_and_denies(self):
        card0 = a.Card("GPU-11111111-1111-1111-1111-111111111111", "0000:03:00.0", 0, 195, 0)
        card1 = a.Card("GPU-22222222-2222-2222-2222-222222222222", "0000:04:00.0", 1, 195, 1)
        collector = a.RootReadOnlyCollector((card0, card1))
        with mock.patch.object(collector, "collect_once", side_effect=a.Blocked("proc-not-readable")) as mocked:
            outcome = collector.stage_probe(manifest=a.OwnerManifest(), phase="empty-before-tabby")
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(outcome.as_dict()["status"], "BLOCKED")
        self.assertIn("proc-not-readable", outcome.reasons)

    def test_nvml_collect_missing_mps_surface_denied(self):
        with tempfile.TemporaryDirectory() as root:
            fake = SimpleNamespace(nvmlDeviceGetCount=lambda: 2,
                                   nvmlDeviceGetHandleByIndex=lambda idx: idx,
                                   nvmlDeviceGetUUID=lambda idx: b"GPU-11111111-1111-1111-1111-111111111111",
                                   nvmlDeviceGetPciInfo=lambda idx: SimpleNamespace(busId=b"0000:03:00.0"),
                                   nvmlDeviceGetMinorNumber=lambda idx: 0,
                                   nvmlDeviceGetMigMode=lambda idx: (0, 0),
                                   nvmlDeviceGetComputeRunningProcesses=lambda idx: (),
                                   nvmlDeviceGetGraphicsRunningProcesses=lambda idx: ())
            card0 = a.Card("GPU-11111111-1111-1111-1111-111111111111", "0000:03:00.0", 0, 195, 0)
            card1 = a.Card("GPU-22222222-2222-2222-2222-222222222222", "0000:04:00.0", 1, 195, 1)
            collector = a.RootReadOnlyCollector((card0, card1), nvml=fake, dev_root=Path(root))
            original = Path.stat
            def char_stat(path, *args, **kwargs):
                if path.name.startswith("nvidia"):
                    return SimpleNamespace(st_mode=stat.S_IFCHR | 0o600, st_rdev=195 << 8)
                return original(path, *args, **kwargs)
            with mock.patch.object(Path, "stat", char_stat), \
                    mock.patch.object(a.os, "major", lambda dev: dev >> 8, create=True), \
                    mock.patch.object(a.os, "minor", lambda dev: dev & 0xff, create=True):
                with self.assertRaisesRegex(a.Blocked, "MPS|mps|unsupported"):
                    collector._read_nvml_cards(fake)

    def test_fake_root_proc_and_nvml_collect_still_never_authorizes(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            proc_root = base / "proc"
            dev_root = base / "dev"
            (proc_root / "sys/kernel/random").mkdir(parents=True)
            (proc_root / "sys/kernel/random/boot_id").write_text("injected-boot-id")
            (proc_root / "200/fd").mkdir(parents=True)
            (proc_root / "200/stat").write_text(synthetic_stat(200, 333))
            (proc_root / "200/cgroup").write_text("0::/system.slice/test.service\n")
            (proc_root / "200/status").write_text("Name:\ttest\nUid:\t1000\t1000\t1000\t1000\n")
            dev_root.mkdir()
            uuids = ("GPU-11111111-1111-1111-1111-111111111111",
                     "GPU-22222222-2222-2222-2222-222222222222")
            pcis = ("0000:03:00.0", "0000:04:00.0")
            cards = (
                a.Card(uuids[0], pcis[0], 0, 195, 0),
                a.Card(uuids[1], pcis[1], 1, 195, 1),
            )
            fake_nvml = SimpleNamespace(
                nvmlInit=mock.Mock(),
                nvmlShutdown=mock.Mock(),
                nvmlDeviceGetCount=lambda: 2,
                nvmlDeviceGetHandleByIndex=lambda idx: idx,
                nvmlDeviceGetUUID=lambda idx: uuids[idx],
                nvmlDeviceGetPciInfo=lambda idx: SimpleNamespace(busId=pcis[idx]),
                nvmlDeviceGetMinorNumber=lambda idx: idx,
                nvmlDeviceGetMigMode=lambda idx: (0, 0),
                nvmlDeviceGetComputeRunningProcesses=lambda idx: (),
                nvmlDeviceGetGraphicsRunningProcesses=lambda idx: (),
                nvmlDeviceGetMPSComputeRunningProcesses=lambda idx: (),
            )
            original_stat = Path.stat
            def stat_without_real_gpu(path, *args, **kwargs):
                if path.parent == dev_root and path.name in ("nvidia0", "nvidia1"):
                    minor = int(path.name[-1])
                    return SimpleNamespace(st_mode=stat.S_IFCHR | 0o600,
                                           st_rdev=(195 << 8) | minor)
                return original_stat(path, *args, **kwargs)
            collector = a.RootReadOnlyCollector(
                cards, nvml=fake_nvml, proc_root=proc_root, dev_root=dev_root,
                euid=lambda: 0, platform="linux",
                wall_clock=lambda: 1000.0, monotonic=lambda: 50.0,
            )
            with mock.patch.object(Path, "stat", stat_without_real_gpu), \
                    mock.patch.object(a.os, "major", lambda dev: dev >> 8, create=True), \
                    mock.patch.object(a.os, "minor", lambda dev: dev & 0xff, create=True):
                snapshot = collector.collect_once()
            self.assertTrue(snapshot.root_verified)
            self.assertEqual(len(snapshot.cards), 2)
            self.assertEqual(snapshot.processes[0].pid, 200)
            self.assertEqual(snapshot.processes[0].start_ticks, 333)
            self.assertEqual(snapshot.processes[0].cgroup, "/system.slice/test.service")
            fake_nvml.nvmlInit.assert_called_once_with()
            fake_nvml.nvmlShutdown.assert_called_once_with()
            late = replace(snapshot, unix_s=1000.2, monotonic_s=50.2)
            verdict = a.evaluate(snapshot, late, expected=cards,
                                 manifest=a.OwnerManifest(), phase="empty-before-tabby",
                                 now_s=1000.3)
            self.assertEqual(verdict.as_dict()["status"], "BLOCKED")
            self.assertEqual(verdict.reasons,
                             ("independent-all-driver-contexts-coverage-unavailable",))

    def test_cli_is_blocked_and_does_not_touch_nvml(self):
        stdout = io.StringIO()
        with mock.patch.object(a.RootReadOnlyCollector, "collect_once", side_effect=AssertionError("GPU touched")), \
                contextlib.redirect_stdout(stdout):
            self.assertEqual(a.main(["--staging-status"]), 2)
        data = json.loads(stdout.getvalue())
        self.assertEqual(data["status"], "BLOCKED")
        self.assertNotIn("scope", data)
        self.assertNotIn("gpu_owner", data)
        self.assertNotIn("gpu_free", data)

    def test_module_has_no_service_network_or_gpu_allocation_imports(self):
        tree = ast.parse(Path(a.__file__).read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".", 1)[0])
        self.assertFalse(names & {"subprocess", "socket", "requests", "httpx",
                                  "torch", "cupy", "numba", "ctypes"})


if __name__ == "__main__":
    unittest.main()
