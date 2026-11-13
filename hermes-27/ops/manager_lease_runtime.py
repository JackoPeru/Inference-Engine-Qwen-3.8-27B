"""Prototype wiring for the read-only audited Hermes GPU Manager (2026-11-13).

Copied alongside a *staged* patched manager.py by manager_lease_patch.py.
No live endpoints are installed by importing this module alone.

Safety: manager transition lock serializes all mutations; a persisted lease
fences watchdog, worker drive, force/mode/jobs and GPU-service commands. An
expired or orphaned lease stays fenced until two independent local checkers
prove ingress blocked and HERMES entirely off the GPUs. No client-provided
proof is accepted in place of those checkers.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hmac
import ipaddress
import json
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse

try:
    from .manager_lease_core import LeaseRejected, LeaseStoreFault, ManagerLeaseV1
except ImportError:  # staged next to live manager.py, import as top-level module
    from manager_lease_core import LeaseRejected, LeaseStoreFault, ManagerLeaseV1


class LeaseRuntime:
    def __init__(self, namespace: dict):
        self.ns = namespace
        self.core = ManagerLeaseV1(
            os.environ.get("HERMES_MANAGER_LEASE_STORE", "/var/lib/hermes-gpu-manager/lease-v1.json")
        )
        self._recovery_task: asyncio.Task | None = None
        self._last_recovery_try = 0.0

    @property
    def fenced(self) -> bool:
        return self.core.fenced

    def status_fields(self) -> dict:
        # Claim protocol v1 ONLY when all external attestation hooks are
        # configured. No-op/present-but-unconfigured must remain NO-GO.
        if not self._configured():
            return {"watchdog_lease_contract_version": None,
                    "manager_instance_id": self.core.instance_id,
                    "maintenance_lease": None,
                    "maintenance_recovery_required": self.core.fenced}
        return self.core.status()

    async def current_state(self, original: str) -> str:
        if not self.fenced:
            return original
        if self.core.recovery_required:
            return "HERMES_RECOVERY_REQUIRED"
        # This is the observed Tabby service state, not a optimistic
        # assumption that systemctl stop completed on the client.
        try:
            tabby_active = await asyncio.wait_for(
                self.ns["service_active"](str(self.ns["CONFIG"]["llm"]["service"])),
                timeout=3)
        except asyncio.TimeoutError:
            return "HERMES_RECOVERY_REQUIRED"
        if tabby_active:
            return "LLM_READY"
        return "HERMES_MAINTENANCE"

    def blocked_operation(self) -> bool:
        """Only the *current, authenticated recovery task* may restore GPUs."""
        if not self.fenced:
            return False
        try:
            return asyncio.current_task() is not self._recovery_task
        except RuntimeError:
            return True

    @staticmethod
    def _check_path(variable: str, *, executable: bool) -> Path:
        value = os.environ.get(variable, "")
        path = Path(value)
        disallowed = 0o022 if executable else 0o077
        if (not value or not path.is_absolute() or path.is_symlink() or
                not path.is_file() or path.stat().st_mode & disallowed or
                (executable and not os.access(path, os.X_OK))):
            raise LeaseRejected(f"privileged {variable} not securely configured")
        return path

    def _configured(self) -> bool:
        try:
            for key, executable in (("HERMES_MANAGER_LEASE_TOKEN_FILE", False),
                                    ("HERMES_MANAGER_INGRESS_CHECK", True),
                                    ("HERMES_MANAGER_GPU_CHECK", True)):
                self._check_path(key, executable=executable)
            return True
        except (LeaseRejected, OSError, ValueError):
            return False

    def _authorize(self, request: Request) -> None:
        peer = getattr(getattr(request, "client", None), "host", "")
        try:
            address = ipaddress.ip_address(peer)
            loopback = address.is_loopback or bool(getattr(address, "ipv4_mapped", None) and address.ipv4_mapped.is_loopback)
        except ValueError:
            loopback = False
        if not loopback:
            raise HTTPException(403, "maintenance lease endpoint is loopback-only")
        try:
            token_file = self._check_path("HERMES_MANAGER_LEASE_TOKEN_FILE", executable=False)
            if token_file.stat().st_size > 512:
                raise LeaseRejected("invalid lease operator key file")
            expected = token_file.read_text(encoding="utf-8").strip()
            provided = request.headers.get("x-hermes-manager-lease-token", "")
            if not 32 <= len(expected) <= 256 or not hmac.compare_digest(provided, expected):
                raise LeaseRejected("operator lease token invalid")
        except (LeaseRejected, OSError, UnicodeError) as exc:
            raise HTTPException(403, "maintenance lease authentication unavailable or invalid") from None

    @staticmethod
    def _read_proof(variable: str) -> dict:
        executable = LeaseRuntime._check_path(variable, executable=True)
        try:
            result = subprocess.run([str(executable)], check=True, timeout=2,
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            if len(result.stdout) > 8192:
                raise LeaseRejected("local safety proof exceeds limit")
            value = json.loads(result.stdout)
            if not isinstance(value, dict):
                raise ValueError("safety proof must be JSON object")
            return value
        except (OSError, subprocess.SubprocessError, ValueError):
            raise LeaseRejected("independent local safety proof unavailable") from None

    async def _proof(self, variable: str) -> dict:
        # The local checker can perform short I/O; do not block the event loop
        # holding the manager lock while it executes.
        return await asyncio.to_thread(self._read_proof, variable)

    async def _idle_snapshot(self) -> dict:
        state = self.ns["_state"]
        active_training = self.ns["_character_training_hold"]()
        store = self.ns.get("_character_store")
        if store is not None:
            active_training = active_training or bool(store.active_training_jobs())
        try:
            online = await asyncio.wait_for(self.ns["llm_online"](), timeout=1)
            serving = await asyncio.wait_for(self.ns["llm_serving"](), timeout=2)
        except asyncio.TimeoutError:
            raise LeaseRejected("Tabby readiness timed out during lease admission") from None
        return {
            "current_state": state["current_state"],
            "desired_mode": state["desired_mode"],
            "queue_length": len(self.ns["queued_jobs"]()),
            "current_job": state.get("current_job"),
            "direct_active": state["current_state"] == "DIRECT",
            "gaming_active": state["current_state"] == "GAMING",
            "character_training": ["active"] if active_training else [],
            "force_active": self.ns["_force_llm_active"](),
            "llm_online": online,
            "llm_serving": serving,
            "worker_alive_s": time.monotonic() - float(state.get("last_drive_ts") or 0),
            "transition_lock_held": self.ns["_lock"].locked(),
        }

    @asynccontextmanager
    async def _guarded_lock(self):
        # Client adapter has a hard 8s HTTP timeout. Never queue a lease
        # acquisition behind a multi-minute media transition and then grant
        # it after the original supervisor gave up.
        lock = self.ns["_lock"]
        try:
            await asyncio.wait_for(lock.acquire(), timeout=1)
        except asyncio.TimeoutError:
            raise LeaseRejected("GPU Manager transition in progress: lease admission denied") from None
        try:
            yield
        finally:
            lock.release()

    @asynccontextmanager
    async def _guarded_submit_lock(self):
        """Serialize lease grant with both normal and smart media submits."""
        lock = self.ns.get("_submit_lock")
        if lock is None:
            raise LeaseRejected("manager media submit lock unavailable")
        try:
            await asyncio.wait_for(lock.acquire(), timeout=1)
        except asyncio.TimeoutError:
            raise LeaseRejected("in-flight media submission: lease denied") from None
        try:
            yield
        finally:
            lock.release()

    async def operate(self, action: str, body: dict) -> dict:
        if action not in ("acquire", "renew", "release"):
            raise LeaseRejected("unsupported lease operation")
        if not self._configured():
            raise LeaseRejected("lease integration unavailable: missing local checker or token")
        async with self._guarded_lock():
            if action == "acquire":
                # _submit_lock is separate from _lock in the installed
                # manager.py. Hold both while examining the backlog and
                # persisting the grant. In addition, patched create_job()
                # itself rejects already-admitted requests once fenced.
                async with self._guarded_submit_lock():
                    proof = await self._proof("HERMES_MANAGER_INGRESS_CHECK")
                    present = self.core.status()["maintenance_lease"]
                    repeated = (isinstance(present, dict) and
                                present.get("id") == body.get("lease_id") and
                                present.get("status") == "active")
                    # Repeated acquire after Tabby stopped cannot require
                    # llm_serving again. The core still checks holder ID,
                    # current manager instance and fresh closed ingress.
                    idle = {} if repeated else await self._idle_snapshot()
                    return self.core.acquire(body, snapshot=idle,
                                             ingress_proof=proof)
            proof = await self._proof("HERMES_MANAGER_INGRESS_CHECK")
            if action == "renew":
                return self.core.renew(body, ingress_proof=proof)
            # Release occurs after the supervised Tabby restore; independent
            # GPU ownership checker must certify no HERMES CUDA/PID remains.
            gpu = await self._proof("HERMES_MANAGER_GPU_CHECK")
            try:
                serving = await asyncio.wait_for(self.ns["llm_serving"](), timeout=2)
            except asyncio.TimeoutError:
                raise LeaseRejected("Tabby release readiness timed out") from None
            return self.core.release(body, ingress_proof=proof,
                                     gpu_safe_proof=gpu, tabby_serving=serving)

    async def recover(self) -> dict:
        if not self._configured():
            raise LeaseRejected("recovery requires independent local proof executables")
        async with self._guarded_lock():
            ingress = await self._proof("HERMES_MANAGER_INGRESS_CHECK")
            gpu = await self._proof("HERMES_MANAGER_GPU_CHECK")
            if await self.ns["llm_serving"]():
                # Client may have already restored Tabby after lease expiry;
                # do not restart it just to clear the orphaned fence.
                self.core.assert_recovery_already_restored(
                    ingress_proof=ingress, gpu_safe_proof=gpu)
            else:
                self.core.assert_recovery_safe(ingress_proof=ingress,
                                               gpu_safe_proof=gpu)
                self._recovery_task = asyncio.current_task()
                try:
                    # Under the original _lock; worker/force requests stay fenced.
                    ok = await self.ns["transition_to_llm"]()
                finally:
                    self._recovery_task = None
                if not ok or not await self.ns["llm_serving"]():
                    raise LeaseRejected("Tabby restore failed; GPU fence remains active")
            after_ingress = await self._proof("HERMES_MANAGER_INGRESS_CHECK")
            after_gpu = await self._proof("HERMES_MANAGER_GPU_CHECK")
            result = self.core.finish_recovery(
                ingress_proof=after_ingress, gpu_safe_proof=after_gpu,
                tabby_serving=True)
            return result

    async def heartbeat(self) -> bool:
        """Called before original drive_auto_once side effects."""
        if not self.fenced:
            return False
        if self.core.recovery_required:
            now = time.monotonic()
            if now - self._last_recovery_try >= 15:
                self._last_recovery_try = now
                try:
                    await self.recover()
                except Exception:
                    # No fallback restart! The next tick may retry, but must
                    # never un-fence on expired lease or checker outage.
                    pass
        return True


def install_manager_lease(namespace: dict) -> LeaseRuntime:
    """Register authenticated endpoints on existing manager FastAPI app."""
    runtime = LeaseRuntime(namespace)
    app = namespace["app"]
    require_laya = namespace["require_laya"]

    @app.middleware("http")
    async def maintenance_fence(request: Request, call_next):
        if (runtime.fenced and request.method.upper() not in ("GET", "HEAD", "OPTIONS") and
                not request.url.path.startswith("/maintenance-lease/v1/")):
            return JSONResponse({"detail": "GPU Manager maintenance fence active"}, status_code=423)
        return await call_next(request)

    @app.post("/maintenance-lease/v1/{action}")
    async def maintenance_lease_operation(action: str, request: Request,
                                          _: None = Depends(require_laya)) -> dict:
        runtime._authorize(request)
        try:
            raw = await request.body()
            if len(raw) > 4096:
                raise LeaseRejected("lease request body too large")
            body = json.loads(raw)
            if action == "recover":
                return await runtime.recover()
            return await runtime.operate(action, body)
        except (LeaseRejected, LeaseStoreFault, ValueError):
            raise HTTPException(409, "maintenance lease operation refused by safety gate") from None

    return runtime
