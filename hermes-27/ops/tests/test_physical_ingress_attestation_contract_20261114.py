"""Deterministic SYNTHETIC-only CPU tests; no socket, GPU, SSH, Git or service."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import hmac
import inspect
import unittest
from unittest import mock

from ops import ingress_attestation_contract as contract
from ops.manager_lease_external_authority_staging import (
    ExternalSignatureVerifier, canonical_message,
)


class SyntheticVerifier(ExternalSignatureVerifier):
    """In-memory role-separated HMAC keys; NEVER privileged physical signers."""

    KEYS = {
        "authority": b"synthetic authority role test only",
        "ingress": b"synthetic ingress role test only",
        "gpu-owner": b"synthetic gpu owner role test only",
    }

    def sign(self, claims: dict, role: str) -> dict:
        return {
            "claims": deepcopy(claims),
            "signature": hmac.new(
                self.KEYS[role], canonical_message(role, claims),
                hashlib.sha256,
            ).hexdigest(),
        }

    def verify(self, role: str, message: bytes, signature: bytes) -> None:
        expected = hmac.new(self.KEYS[role], message, hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError("synthetic role/key mismatch")


class SyntheticStore(contract.AtomicConsumptionStore):
    """Disposable fake ledger: gives NO real anti-rollback guarantee."""

    def __init__(self, state=None):
        self.state = state if state is not None else contract.ConsumptionState()
        self.commit_then_fail = False

    def atomic_update(self, mutation):
        following, answer = mutation(self.state)
        self.state = following
        if self.commit_then_fail:
            raise OSError("SYNTHETIC lost acknowledgement after commit")
        return answer


class Clock:
    def __init__(self):
        self.wall = 10002.0
        self.mono = 20002.0

    def wall_time(self):
        return self.wall

    def mono_time(self):
        return self.mono


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.signer = SyntheticVerifier()
        self.store = SyntheticStore()
        self.ctx = contract.Context(
            boot_id="a" * 32,
            manager_instance_id="b" * 32,
            lease_id="c" * 32,
            epoch=1,
            fencing_token=1,
            nonce="d" * 32,
            manifest_epoch=1,
            gpu_pins=(
                contract.CardPin(
                    "GPU-11111111-1111-1111-1111-111111111111",
                    "0000:01:00.0", 101, 1001,
                ),
                contract.CardPin(
                    "GPU-22222222-2222-2222-2222-222222222222",
                    "0000:02:00.0", 101, 1001,
                ),
            ),
        )

    def common(self, kind: str, *, when=10000.0, mono=20000.0):
        return {
            "protocol": contract.PROTOCOL,
            "kind": kind,
            "boot_id": self.ctx.boot_id,
            "manager_instance_id": self.ctx.manager_instance_id,
            "lease_id": self.ctx.lease_id,
            "epoch": self.ctx.epoch,
            "fencing_token": self.ctx.fencing_token,
            "nonce": self.ctx.nonce,
            "observed_at_unix": when,
            "observed_at_monotonic": mono,
        }

    def package(self):
        routes = []
        for i, route in enumerate(sorted(contract.REQUIRED_ROUTES)):
            routes.append({
                "route_id": route,
                "endpoint": contract.ROUTE_ENDPOINTS[route],
                "listener_token": format(i + 1, "032x"),
                "listener_pid": 500 + i,
                "listener_start_ticks": 1000 + i,
                "listener_generation": 7,
            })
        manifest = {
            **self.common("manifest"),
            "manifest_epoch": self.ctx.manifest_epoch,
            "inventory_revision": 7,
            "routes": routes,
            "unknown_routes": 0,
            "unlisted_listeners": 0,
            "unfenced_admission_paths": 0,
        }
        digest = hashlib.sha256(
            canonical_message("authority", manifest),
        ).hexdigest()
        samples = []
        for route in routes:
            sample = {
                **self.common("route-drained", when=10001.0, mono=20001.0),
                **route,
                "manifest_epoch": self.ctx.manifest_epoch,
                "manifest_sha256": digest,
                "inventory_revision": 7,
                "drain_fence_epoch": self.ctx.epoch,
                "counters": {k: 0 for k in contract.ROUTE_COUNTERS},
            }
            samples.append(self.signer.sign(sample, "ingress"))
        gpu = {
            **self.common("gpu-inventory", when=10002.0, mono=20002.0),
            "manifest_epoch": self.ctx.manifest_epoch,
            "manifest_sha256": digest,
            "cards": [
                {
                    "uuid": pin.uuid, "pci": pin.pci,
                    "pid": pin.pid, "start_ticks": pin.start_ticks,
                    "boot_id": self.ctx.boot_id,
                    "cgroup": "/system.slice/hermes-tabby.service",
                    "driver_contexts": 1, "unknown_contexts": 0,
                } for pin in self.ctx.gpu_pins
            ],
            "unknown_devices": 0,
            "unmapped_driver_contexts": 0,
            "unreadable_proc_pids": 0,
        }
        return {
            "manifest": self.signer.sign(manifest, "authority"),
            "routes": samples,
            "gpu": self.signer.sign(gpu, "gpu-owner"),
        }

    def assess(self, pkg=None, **overrides):
        args = {
            "context": self.ctx,
            "verifier": self.signer,
            "store": self.store,
            "wall_clock": self.clock.wall_time,
            "monotonic_clock": self.clock.mono_time,
        }
        args.update(overrides)
        return contract.assess(pkg if pkg is not None else self.package(), **args)

    def resign(self, pkg, piece, *, index=0, role=None):
        if piece == "manifest":
            pkg["manifest"] = self.signer.sign(
                pkg["manifest"]["claims"], role or "authority",
            )
        elif piece == "gpu":
            pkg["gpu"] = self.signer.sign(pkg["gpu"]["claims"], role or "gpu-owner")
        else:
            pkg["routes"][index] = self.signer.sign(
                pkg["routes"][index]["claims"], role or "ingress",
            )

    def denied(self, package, **kwargs):
        with self.assertRaises(contract.IngressDenied):
            self.assess(package, **kwargs)
        self.assertEqual(self.store.state.highwater, 0)

    def test_full_synthetic_candidate_stays_production_blocked(self):
        result = self.assess()
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["synthetic_candidate"], "SYNTHETIC_VALIDATED")
        self.assertIs(result["production_admissible"], False)
        self.assertIsNone(result["manager_lease_proof"])
        self.assertEqual(self.store.state.highwater, 1)
        self.assertEqual(self.store.state.spent_nonces, (self.ctx.nonce,))

    def test_no_external_authority_or_ledger_is_denied(self):
        self.denied(self.package(), verifier=None)
        self.denied(self.package(), store=None)

    def test_missing_extra_and_duplicate_route_evidence(self):
        for action in ("missing", "extra", "duplicate", "unknown"):
            with self.subTest(action=action):
                pkg = self.package()
                if action == "missing":
                    pkg["routes"].pop()
                elif action == "extra":
                    pkg["routes"].append(deepcopy(pkg["routes"][0]))
                elif action == "duplicate":
                    pkg["routes"][-1] = deepcopy(pkg["routes"][0])
                else:
                    pkg["routes"][0]["claims"]["route_id"] = "unlisted-port-9000"
                    self.resign(pkg, "route", index=0)
                self.denied(pkg)

    def test_manifest_scope_unknown_listener_and_restart_are_denied(self):
        for key in ("unknown_routes", "unlisted_listeners", "unfenced_admission_paths"):
            with self.subTest(key=key):
                pkg = self.package()
                pkg["manifest"]["claims"][key] = 1
                self.resign(pkg, "manifest")
                self.denied(pkg)
        pkg = self.package()
        pkg["manifest"]["claims"]["routes"].pop()
        self.resign(pkg, "manifest")
        self.denied(pkg)

    def test_each_nonzero_http_sse_queue_keepalive_or_unknown_counter_denied(self):
        for key in contract.ROUTE_COUNTERS:
            with self.subTest(key=key):
                pkg = self.package()
                pkg["routes"][0]["claims"]["counters"][key] = 1
                self.resign(pkg, "route")
                self.denied(pkg)

    def test_bool_cannot_masquerade_as_zero_counter(self):
        pkg = self.package()
        pkg["routes"][0]["claims"]["counters"]["sse_streams"] = False
        self.resign(pkg, "route")
        self.denied(pkg)

    def test_listener_pid_start_token_and_generation_drift_denied(self):
        for key, replacement in (
            ("listener_pid", 9999), ("listener_start_ticks", 9999),
            ("listener_token", "f" * 32), ("listener_generation", 999),
            ("endpoint", "hidden-tcp-endpoint"),
        ):
            with self.subTest(key=key):
                pkg = self.package()
                pkg["routes"][0]["claims"][key] = replacement
                self.resign(pkg, "route")
                self.denied(pkg)

    def test_gpu_uuid_pid_birth_boot_and_context_denied(self):
        for field, replacement in (
            ("uuid", "GPU-33333333-3333-3333-3333-333333333333"),
            ("pci", "0000:09:00.0"),
            ("pid", 102), ("start_ticks", 1002),
            ("boot_id", "e" * 32), ("cgroup", "/system.slice/unknown.service"),
            ("driver_contexts", 2), ("unknown_contexts", 1),
        ):
            with self.subTest(field=field):
                pkg = self.package()
                pkg["gpu"]["claims"]["cards"][0][field] = replacement
                self.resign(pkg, "gpu")
                self.denied(pkg)

    def test_gpu_global_unknown_inventory_denied(self):
        for key in ("unknown_devices", "unmapped_driver_contexts", "unreadable_proc_pids"):
            with self.subTest(key=key):
                pkg = self.package()
                pkg["gpu"]["claims"][key] = 1
                self.resign(pkg, "gpu")
                self.denied(pkg)

    def test_context_epoch_nonce_and_manifest_digest_bindings(self):
        for piece, key, value in (
            ("manifest", "nonce", "e" * 32),
            ("route", "epoch", 2),
            ("route", "manifest_epoch", 2),
            ("route", "manifest_sha256", "f" * 64),
            ("gpu", "lease_id", "e" * 32),
            ("gpu", "manifest_sha256", "f" * 64),
        ):
            with self.subTest(piece=piece, key=key):
                pkg = self.package()
                envelope = pkg["routes"][0] if piece == "route" else pkg[piece]
                envelope["claims"][key] = value
                self.resign(pkg, piece)
                self.denied(pkg)

    def test_context_bool_epoch_masquerade_is_denied(self):
        pkg = self.package()
        pkg["routes"][0]["claims"]["epoch"] = True
        self.resign(pkg, "route")
        self.denied(pkg)
        for piece, key in (
            ("route", "manifest_epoch"), ("route", "drain_fence_epoch"),
            ("gpu", "manifest_epoch"),
        ):
            with self.subTest(piece=piece, key=key):
                pkg = self.package()
                envelope = pkg["routes"][0] if piece == "route" else pkg["gpu"]
                envelope["claims"][key] = True
                self.resign(pkg, piece)
                self.denied(pkg)

    def test_reject_stale_future_and_cross_boot_monotonic_observations(self):
        for piece, field, value in (
            ("route", "observed_at_unix", 9990.0),
            ("route", "observed_at_monotonic", 19990.0),
            ("route", "observed_at_monotonic", 20006.0),
            ("gpu", "observed_at_monotonic", 20000.5),
            ("gpu", "observed_at_unix", 10002.0 + 10),
        ):
            with self.subTest(piece=piece, field=field):
                pkg = self.package()
                envelope = pkg["routes"][0] if piece == "route" else pkg[piece]
                envelope["claims"][field] = value
                self.resign(pkg, piece)
                self.denied(pkg)

    def test_invalid_signature_and_role_substitution_denied(self):
        pkg = self.package()
        pkg["routes"][0]["signature"] = "0" * 64
        self.denied(pkg)
        pkg = self.package()
        self.resign(pkg, "route", role="authority")
        self.denied(pkg)

    def test_verifier_returning_bool_does_not_authorize(self):
        class BoolVerifier(ExternalSignatureVerifier):
            def verify(self, role, message, signature):
                return True
        self.denied(self.package(), verifier=BoolVerifier())

    def test_epoch_reuse_and_nonce_replay_after_process_restart(self):
        self.assess()
        recovered = SyntheticStore(self.store.state)  # same TEST state after crash
        with self.assertRaises(contract.IngressDenied):
            self.assess(store=recovered)
        self.assertEqual(recovered.state.highwater, 1)
        self.assertEqual(recovered.state.spent_nonces, (self.ctx.nonce,))
        with self.assertRaises(contract.IngressDenied):
            self.assess()

    def test_new_epoch_and_rotated_nonce_remain_synthetic_only(self):
        self.assess()
        self.ctx = replace(
            self.ctx, epoch=2, fencing_token=2,
            nonce="e" * 32, manifest_epoch=2,
            lease_id="f" * 32, manager_instance_id="1" * 32,
        )
        answer = self.assess()
        self.assertIs(answer["production_admissible"], False)
        self.assertIsNone(answer["manager_lease_proof"])
        self.assertEqual(self.store.state.highwater, 2)
        self.assertEqual(
            self.store.state.spent_nonces, ("d" * 32, "e" * 32),
        )

    def test_epoch_rotated_but_nonce_reused_is_denied(self):
        self.assess()
        self.ctx = replace(
            self.ctx, epoch=2, fencing_token=2, manifest_epoch=2,
            lease_id="f" * 32, manager_instance_id="1" * 32,
        )
        with self.assertRaises(contract.IngressDenied):
            self.assess()

    def test_expiry_or_clock_rollback_during_atomic_consume_is_denied(self):
        class DelayedStore(SyntheticStore):
            def __init__(self, clock, delta):
                super().__init__()
                self.clock = clock
                self.delta = delta

            def atomic_update(self, mutation):
                self.clock.wall += self.delta
                self.clock.mono += self.delta
                return super().atomic_update(mutation)

        for drift in (8.0, -1.0):
            with self.subTest(drift=drift):
                self.clock = Clock()
                delayed = DelayedStore(self.clock, drift)
                with self.assertRaises(contract.IngressDenied):
                    self.assess(store=delayed)
                self.assertEqual(delayed.state.highwater, 0)

    def test_missing_corrupt_rolledback_or_exhausted_store_denied(self):
        for state in (
            None,
            contract.ConsumptionState(highwater=True),
            contract.ConsumptionState(highwater=2),
            contract.ConsumptionState(highwater=1),
            contract.ConsumptionState(highwater=0, spent_nonces=(self.ctx.nonce,)),
            contract.ConsumptionState(highwater=0, spent_nonces=("e" * 32,)),
            contract.ConsumptionState(
                highwater=0,
                spent_nonces=tuple(format(i, "032x") for i in range(contract.MAX_SPENT)),
            ),
        ):
            with self.subTest(state=repr(state)[:75]):
                self.store.state = state
                with self.assertRaises(contract.IngressDenied):
                    self.assess()
        self.store.state = contract.ConsumptionState()

    def test_bounded_nonce_log_exhaustion_cannot_rollover(self):
        self.store.state = contract.ConsumptionState(
            highwater=contract.MAX_SPENT,
            spent_nonces=tuple(
                format(i, "032x") for i in range(contract.MAX_SPENT)
            ),
        )
        self.ctx = replace(
            self.ctx, epoch=contract.MAX_SPENT + 1,
            fencing_token=contract.MAX_SPENT + 1,
            manifest_epoch=contract.MAX_SPENT + 1,
            nonce="e" * 32,
        )
        with self.assertRaises(contract.IngressDenied):
            self.assess()
        self.assertEqual(self.store.state.highwater, contract.MAX_SPENT)

    def test_commit_then_crash_ack_ambiguity_denies_retry(self):
        self.store.commit_then_fail = True
        with self.assertRaises(contract.IngressDenied):
            self.assess()
        self.assertEqual(self.store.state.highwater, 1)
        self.store.commit_then_fail = False
        with self.assertRaises(contract.IngressDenied):
            self.assess()

    def test_restarted_manager_cannot_use_previous_incarnation_proof(self):
        pkg = self.package()
        newer = contract.Context(
            **{**vars(self.ctx), "manager_instance_id": "f" * 32},
        )
        self.denied(pkg, context=newer)

    def test_source_mutants_prove_nonzero_queue_and_gpu_birth_gates(self):
        # Execute in-memory source mutants only. Assert that the corresponding
        # ordinary negative test fails when a single security check is erased.
        for function, needle, replacement, test in (
            (
                contract._validate_routes,
                "or any(not _uint(value, 0, 0) for value in counters.values())):",
                "or False):",
                self._negative_queue,
            ),
            (
                contract._validate_gpu,
                'or card["start_ticks"] != pin.start_ticks',
                "or False",
                self._negative_gpu_birth,
            ),
        ):
            source = inspect.getsource(function)
            self.assertEqual(source.count(needle), 1)
            namespace = dict(vars(contract))
            exec(compile(source.replace(needle, replacement), "<cpu-mutant>", "exec"), namespace)
            with mock.patch.object(contract, function.__name__, namespace[function.__name__]):
                with self.assertRaises(AssertionError):
                    test()

    def _negative_queue(self):
        pkg = self.package()
        pkg["routes"][0]["claims"]["counters"]["queued_jobs"] = 1
        self.resign(pkg, "route")
        self.denied(pkg)

    def _negative_gpu_birth(self):
        pkg = self.package()
        pkg["gpu"]["claims"]["cards"][0]["start_ticks"] = 1002
        self.resign(pkg, "gpu")
        self.denied(pkg)


if __name__ == "__main__":
    unittest.main()
