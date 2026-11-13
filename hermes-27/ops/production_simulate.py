#!/usr/bin/env python3
"""Completely offline HERMES-27 PREPARED/SIMULATED cutover fault matrix.

No GPU, network, systemctl, subprocess, service mutation, or credentials.
All external boundaries are replaced with in-memory fixtures. This file is
test equipment, not an ingress or GPU Manager lease implementation.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import time
from unittest import mock

try:
    from . import production_deploy as deploy
except ImportError:  # executable directly as `python3 ops/production_simulate.py`
    import production_deploy as deploy


SCENARIOS = (
    "clean_shutdown", "manager_unsupported", "manager_busy", "drain_unproven",
    "lease_acquire_denied", "lease_scope_invalid", "tabby_stop_denied",
    "gpu_busy", "ambiguous_spawn", "lease_renew_denied", "lease_expired",
    "manager_restarted", "ingress_reopened", "hermes_health_failed",
    "child_stuck", "tabby_restore_failed", "lease_release_failed",
    "ingress_restore_failed", "recover_after_lease_release_failure",
    "recover_after_ingress_restore_failure",
)


class SimulatedChild:
    pid = 4242

    def __init__(self, scene):
        self.scene = scene
        self.returncode = None

    def poll(self):
        return None if self.scene.child_running else 0


class Scene:
    """An explicit in-memory manager, proxy, systemd and process model."""

    def __init__(self, fault: str, cfg: dict):
        self.fault = fault
        self.cfg = cfg
        self.events: list[str] = []
        self.ingress_blocked = False
        self.tabby_running = True
        self.manager_running = True
        self.child_running = False
        self.child_started = False
        self.instance = "a" * 32
        self.lease = None
        self.health_calls = 0
        self.renew_calls = 0
        self.ingress_reopened_once = False

    def status(self):
        status = {
            "watchdog_lease_contract_version": 1,
            "manager_instance_id": self.instance,
            "maintenance_lease": dict(self.lease) if self.lease is not None else None,
            "desired_mode": "AUTO", "current_state": "LLM_READY" if self.tabby_running else "HERMES_MAINTENANCE",
            "llm_serving": self.tabby_running, "llm_online": self.tabby_running,
            "queue_length": 0, "current_job": None, "direct_active": False,
            "gaming_active": False, "character_training": [], "worker_alive_s": 0.3,
        }
        if self.fault == "manager_unsupported":
            status.pop("watchdog_lease_contract_version")
            status.pop("maintenance_lease")
        if self.fault == "manager_busy":
            status["queue_length"] = 1
        return status

    def http(self, url, *_args):
        if url == self.cfg["manager_status_url"]:
            return self.status()
        if url == self.cfg["tabby_health_url"]:
            return {"data": [{"id": "baseline"}]}
        raise AssertionError("simulation attempted an unexpected HTTP endpoint")

    def service(self, name):
        if name == "hermes-tabby.service":
            return self.tabby_running
        if name == "hermes-gpu-manager.service":
            return self.manager_running
        raise AssertionError("unexpected simulated systemd unit")

    def change(self, action, name):
        if name != "hermes-tabby.service":
            raise AssertionError("cutover attempted to stop/restart GPU Manager")
        self.events.append(f"tabby_{action}")
        if action == "stop" and self.fault == "tabby_stop_denied":
            raise deploy.DeploymentError("injected Tabby stop failure")
        if action == "start" and self.fault == "tabby_restore_failed":
            raise deploy.DeploymentError("injected Tabby restore failure")
        self.tabby_running = action == "start"

    def ingress(self, argv, *_args):
        if argv == self.cfg["drain_command"]:
            self.events.append("ingress_drain")
            self.ingress_blocked = True
            return ""
        if argv == self.cfg["restore_command"]:
            self.events.append("ingress_restore")
            if self.fault == "ingress_restore_failed":
                raise deploy.DeploymentError("injected ingress restore failure")
            self.ingress_blocked = False
            return ""
        if argv == self.cfg["drain_check_command"]:
            if (self.fault == "ingress_reopened" and self.child_started and
                    not self.ingress_reopened_once):
                self.ingress_blocked = False
                self.ingress_reopened_once = True
            return json.dumps({
                "scope": "partial" if self.fault == "drain_unproven" else "all-tabby-clients",
                "observed_at_unix": time.time(), "blocked": self.ingress_blocked,
                "inflight": 0,
            })
        raise AssertionError("simulation attempted an unexpected ingress hook")

    def adapter(self, _cfg, action, state):
        self.events.append(f"lease_{action}")
        if action == "acquire":
            if self.fault == "lease_acquire_denied":
                raise deploy.DeploymentError("injected lease acquire denial")
            self.lease = {
                "id": state["lease_id"], "owner": deploy.LEASE_OWNER,
                "status": "active", "scope": "tabby-restart-and-mode-transition",
                "watchdog_restart_suppressed": self.fault != "lease_scope_invalid",
                "fencing_token": 9, "expires_at_unix": time.time() + 100,
            }
        elif action == "renew":
            self.renew_calls += 1
            if self.fault == "lease_renew_denied" and self.child_started:
                raise deploy.DeploymentError("injected lease renewal failure")
            if self.lease is not None:
                self.lease["expires_at_unix"] = time.time() + 100
                if self.fault == "lease_expired" and self.child_started:
                    self.lease["expires_at_unix"] = time.time() - 1
                if self.fault == "manager_restarted" and self.child_started:
                    self.instance = "b" * 32
        elif action == "release":
            if self.fault == "lease_release_failed":
                raise deploy.DeploymentError("injected lease release failure")
            self.lease = None  # idempotent recovery, including denied acquisition
        else:
            raise AssertionError("invalid lease action")

    def spawn(self, *_args, **_kwargs):
        self.events.append("hermes_spawn")
        self.child_started = True
        if self.fault == "ambiguous_spawn":
            raise OSError("simulated interrupted spawn before PID journal write")
        self.child_running = True
        return SimulatedChild(self)

    def stop_child(self, _state, _cfg):
        self.events.append("hermes_stop")
        if self.fault != "child_stuck":
            self.child_running = False

    def gpu(self, _cfg, callback):
        callback()
        if self.fault == "gpu_busy":
            raise deploy.DeploymentError("injected VRAM contention")

    def health(self, _cfg):
        self.health_calls += 1
        if self.fault == "hermes_health_failed" and self.health_calls > 1:
            raise deploy.DeploymentError("injected HTTP health failure")
        if self.health_calls > 1 and self.fault != "hermes_health_failed":
            deploy.STOP_REQUESTED = True  # normal graceful release after one heartbeat


def run_scenario(name: str) -> dict:
    """Raise AssertionError on any unsafe simulated action/rollback outcome."""
    if name not in SCENARIOS:
        raise ValueError(f"unknown simulated fault: {name}")
    with tempfile.TemporaryDirectory(prefix="hermes-cutover-sim-") as tmp:
        root = Path(tmp)
        cfg = {
            "_path": str((root / "cutover.json").resolve()),
            "state_dir": str(root / "state"),
            "model_path": str(root / "mock.gguf"),
            "tokenizer_path": str(root / "tokenizer.json"),
            "gguf_dump_path": str(root / "build" / "gguf_dump"),
            "python_path": str(root / "python"),
            "server_path": str(root / "build" / "hermes_server"),
            "model_id": "hermes-27", "port": 18127,
            "manager_status_url": "http://127.0.0.1:8643/status",
            "manager_token_file": str(root / "unused-token"),
            "tabby_health_url": "http://127.0.0.1:8000/v1/models",
            "drain_command": ["/SIMULATED/ingress-drain"],
            "drain_check_command": ["/SIMULATED/ingress-check"],
            "restore_command": ["/SIMULATED/ingress-restore"],
            "manager_lease_acquire_command": ["/SIMULATED/lease-acquire"],
            "manager_lease_renew_command": ["/SIMULATED/lease-renew"],
            "manager_lease_release_command": ["/SIMULATED/lease-release"],
            "drain_timeout_sec": 1,
            "manager_restore_timeout_sec": 1,
            "tabby_restore_timeout_sec": 1,
            "hermes_start_timeout_sec": 10,
        }
        fault = {
            "recover_after_lease_release_failure": "lease_release_failed",
            "recover_after_ingress_restore_failure": "ingress_restore_failed",
        }.get(name, name)
        scene = Scene(fault, cfg)
        _, journal = deploy.state_paths(cfg)
        saved_stop = deploy.STOP_REQUESTED
        deploy.STOP_REQUESTED = False

        def fake_keeper_check(keeper):
            try:
                keeper.refresh()  # deterministic heartbeat: no thread or clock sleeps
                deploy.lease_guard(keeper.cfg, keeper.state)
            except Exception as exc:
                keeper.failure = str(exc)
                raise

        def fake_sleep(_seconds):
            if name == "ambiguous_spawn":
                return

        error = None
        try:
            with ExitStack() as stack:
                # Every external interaction is mocked. Any new unmocked
                # external call fails the scenario rather than touching host.
                stack.enter_context(mock.patch.object(deploy, "service_active", side_effect=scene.service))
                stack.enter_context(mock.patch.object(deploy, "change_service", side_effect=scene.change))
                stack.enter_context(mock.patch.object(deploy, "request_json", side_effect=scene.http))
                stack.enter_context(mock.patch.object(deploy, "run_hook", side_effect=scene.ingress))
                stack.enter_context(mock.patch.object(deploy, "lease_adapter", side_effect=scene.adapter))
                stack.enter_context(mock.patch.object(deploy, "port_available", return_value=None))
                stack.enter_context(mock.patch.object(deploy, "free_gpu_guard", side_effect=scene.gpu))
                stack.enter_context(mock.patch.object(deploy, "check_hermes", side_effect=scene.health))
                stack.enter_context(mock.patch.object(deploy, "stop_child", side_effect=scene.stop_child))
                stack.enter_context(mock.patch.object(deploy, "process_start_ticks", return_value="111"))
                stack.enter_context(mock.patch.object(deploy, "process_running", side_effect=lambda _pid: scene.child_running))
                stack.enter_context(mock.patch.object(deploy.subprocess, "Popen", side_effect=scene.spawn))
                stack.enter_context(mock.patch.object(deploy.subprocess, "run", side_effect=AssertionError("real subprocess forbidden")))
                stack.enter_context(mock.patch.object(deploy.LeaseKeeper, "start", return_value=None))
                stack.enter_context(mock.patch.object(deploy.LeaseKeeper, "check", fake_keeper_check))
                stack.enter_context(mock.patch.object(deploy.time, "sleep", side_effect=fake_sleep))
                with redirect_stdout(io.StringIO()):
                    try:
                        deploy.supervise(cfg, journal)
                    except (deploy.DeploymentError, OSError) as exc:
                        error = str(exc)
                    if name.startswith("recover_after_"):
                        if error is None or not journal.exists() or not scene.ingress_blocked:
                            raise AssertionError("injected failure was not held closed before operator recovery")
                        scene.fault = "operator_fixed_adapter"
                        deploy.restore(cfg, journal)
        finally:
            deploy.STOP_REQUESTED = saved_stop

        safe_recovered = (scene.tabby_running and not scene.ingress_blocked and
                          scene.lease is None and not scene.child_running and not journal.exists())
        denied_before_stop = (scene.tabby_running and "tabby_stop" not in scene.events and
                              "hermes_spawn" not in scene.events)
        retained_fail_closed = (scene.ingress_blocked and journal.exists())
        if "manager_stop" in scene.events or "manager_start" in scene.events:
            raise AssertionError("simulation tried to stop GPU Manager")
        if not scene.manager_running:
            raise AssertionError("manager must remain active during the lease")
        if name == "clean_shutdown":
            if error is not None or not safe_recovered or scene.renew_calls == 0:
                raise AssertionError(f"clean cutover/restore was not proven: {error}")
            ordered = ("ingress_drain", "lease_acquire", "tabby_stop", "hermes_spawn",
                       "hermes_stop", "tabby_start", "lease_release", "ingress_restore")
            indexes = [scene.events.index(item) for item in ordered]
            if indexes != sorted(indexes):
                raise AssertionError("unsafe action ordering in successful rollback")
        elif name in ("manager_unsupported", "manager_busy"):
            if error is None or not denied_before_stop or journal.exists() or scene.ingress_blocked:
                raise AssertionError("unsupported/busy manager did not reject before drain")
        elif name.startswith("recover_after_"):
            if error is None or not safe_recovered:
                raise AssertionError("retrying recovery after fixing adapter did not restore service")
        elif name in ("drain_unproven", "lease_scope_invalid", "lease_renew_denied",
                          "lease_expired", "manager_restarted", "ingress_reopened",
                          "ambiguous_spawn", "child_stuck", "tabby_restore_failed",
                          "lease_release_failed", "ingress_restore_failed"):
            if error is None or not retained_fail_closed:
                raise AssertionError(f"{name}: unsafe state not held closed; error={error!r} "
                                     f"blocked={scene.ingress_blocked} journal={journal.exists()} "
                                     f"events={scene.events}")
            if name in ("drain_unproven", "lease_scope_invalid") and not denied_before_stop:
                raise AssertionError("bad lease/drain allowed Tabby to stop")
        else:
            # Recoverable failures: drain/lease/Tabby service remain safe.
            if error is None or not safe_recovered:
                raise AssertionError("recoverable fault did not restore Tabby/ingress")
            if name == "lease_acquire_denied" and not denied_before_stop:
                raise AssertionError("denied lease allowed Tabby to stop")
        outcome = ("DENIED_PRE_MUTATION" if name in ("manager_unsupported", "manager_busy") else
                   "COMPLETE" if safe_recovered else
                   "HELD_CLOSED" if retained_fail_closed else "DENIED_PRE_MUTATION")
        return {"scenario": name, "simulated": True,
                "result": "PASS", "rollback": outcome,
                "events": scene.events, "error_injected": name != "clean_shutdown"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    outcomes = [run_scenario(name) for name in (SCENARIOS if args.scenario == "all" else (args.scenario,))]
    if args.json:
        print(json.dumps({"simulated": True, "count": len(outcomes), "cases": outcomes}, indent=2))
    else:
        for result in outcomes:
            print(f"{result['scenario']}: PASS ({result['rollback']})")
        print(f"{len(outcomes)}/{len(outcomes)} simulated cutover scenarios PASS; no live calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
