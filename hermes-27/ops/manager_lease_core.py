"""Proposed GPU Manager maintenance lease v1; stdlib-only, no side effects on import.

This is the manager-side *state machine*, not a deployed manager integration.
Callers must serialize state transitions with the GPU Manager's existing
asyncio `_lock` and verify independent local ingress/GPU safety proofs.

Persisted state is authoritative after a crash: an outstanding lease remains
fenced. Expiry NEVER authorizes an automatic Tabby restart by itself.
"""

from __future__ import annotations

import json
import hashlib
import hmac
import math
import os
from pathlib import Path
import re
import secrets
import stat
import threading
import time
from typing import Any, Callable


PROTOCOL = "hermes-manager-lease-v1"
OWNER = "hermes-27"
SCOPE = "tabby-restart-and-mode-transition"
TTL_SECONDS = 90
MIN_TTL_SECONDS = 25
MAX_TTL_SECONDS = 180
PROOF_MAX_AGE_SECONDS = 10
CLOCK_ROLLBACK_TOLERANCE_SECONDS = 1.0
IDENTIFIER = re.compile(r"[0-9a-f]{32}\Z")


class LeaseRejected(RuntimeError):
    """Expected protocol conflict; messages contain no secrets."""


class LeaseStoreFault(LeaseRejected):
    """Persistent fencing evidence is unavailable/corrupt: deny all transitions."""


def valid_id(value: object) -> bool:
    return isinstance(value, str) and bool(IDENTIFIER.fullmatch(value))


def private_directory(path: Path) -> bool:
    return (path.stat().st_mode & 0o022) == 0


def require_ingress(proof: object, *, now: float, closed: bool = True) -> dict:
    """Accept ONLY independently measured complete-client drain evidence."""
    if not isinstance(proof, dict):
        raise LeaseRejected("ingress checker returned no trustworthy proof")
    measured = proof.get("observed_at_unix")
    if (proof.get("scope") != "all-tabby-clients" or
            type(measured) not in (int, float) or
            not 0 <= now - measured <= PROOF_MAX_AGE_SECONDS or
            type(proof.get("blocked")) is not bool or
            type(proof.get("inflight")) is not int or proof["inflight"] < 0):
        raise LeaseRejected("ingress evidence scope/age/counters invalid")
    if closed and (proof["blocked"] is not True or proof["inflight"] != 0):
        raise LeaseRejected("ingress not drained: active requests remain")
    return proof


def require_idle(snapshot: object) -> None:
    """Recheck inside manager transition lock immediately before acquire."""
    if not isinstance(snapshot, dict):
        raise LeaseRejected("manager idle state unknown")
    if snapshot.get("current_state") != "LLM_READY":
        raise LeaseRejected("manager not stable in LLM_READY")
    if snapshot.get("desired_mode") not in ("AUTO", "LLM"):
        raise LeaseRejected("manager mode incompatible with lease")
    if (type(snapshot.get("queue_length")) is not int or
            snapshot["queue_length"] != 0 or snapshot.get("current_job") is not None):
        raise LeaseRejected("media queue/current job not empty")
    if (snapshot.get("direct_active") is not False or
            snapshot.get("gaming_active") is not False or
            snapshot.get("character_training") != []):
        raise LeaseRejected("GPU owned by direct/gaming/training")
    if snapshot.get("force_active") is not False:
        raise LeaseRejected("force LLM transition still active")
    if (snapshot.get("llm_online") is not True or
            snapshot.get("llm_serving") is not True):
        raise LeaseRejected("Tabby readiness not independently confirmed")
    if type(snapshot.get("worker_alive_s")) not in (int, float) or not 0 <= snapshot["worker_alive_s"] <= 30:
        raise LeaseRejected("manager worker heartbeat stale")
    if snapshot.get("transition_lock_held") is not True:
        raise LeaseRejected("lease grant outside GPU transition lock")


def require_gpu_safe(proof: object, *, now: float, after_tabby_restore: bool = False) -> None:
    """Independent *local privileged* GPU ownership attestation for recovery.

    Never accept user-supplied JSON in place of a local checker. The runtime
    must call a preconfigured absolute executable and parse its output.
    """
    if not isinstance(proof, dict):
        raise LeaseRejected("GPU ownership proof unavailable")
    observed = proof.get("observed_at_unix")
    if (proof.get("scope") != "all-hermes-gpus-and-pids" or
            type(observed) not in (int, float) or
            not 0 <= now - observed <= PROOF_MAX_AGE_SECONDS or
            proof.get("hermes_pids") != [] or
            proof.get("other_gpu_processes") != [] or
            type(proof.get("gpu_count")) is not int or proof["gpu_count"] != 2):
        raise LeaseRejected("HERMES/GPU ownership not independently released")
    if after_tabby_restore:
        if proof.get("gpu_owner") != "tabby" or proof.get("tabby_owns_both_gpus") is not True:
            raise LeaseRejected("GPU ownership after restore is not exclusively Tabby")
    elif proof.get("gpu_owner") != "none" or proof.get("gpu_free") is not True:
        raise LeaseRejected("GPU VRAM not free before safe post-expiry recovery")


class ManagerLeaseV1:
    """Persistent fencing protocol, independent of FastAPI and CUDA.

    Threads may read `fenced` concurrently; all mutating operations must
    additionally run while holding the manager's `_lock`. An internal RLock
    protects synchronous watchdog reads and atomic journal replacement.
    """

    def __init__(self, store: str | Path, *, clock: Callable[[], float] = time.time,
                 instance_id: str | None = None,
                 proof_verifier: Callable[[str, dict], bool] | None = None,
                 journal_key: bytes | None = None,
                 bootstrap_authorized: bool = False):
        self.path = Path(store)
        self.clock = clock
        self.instance_id = instance_id or secrets.token_hex(16)
        # The caller must provision a trusted, independent root-owned proof
        # verifier. A syntactically plausible client-supplied JSON dict is
        # NEVER an authorization to stop/restart GPU services by itself.
        self.proof_verifier = proof_verifier
        if journal_key is not None and (
                not isinstance(journal_key, bytes) or len(journal_key) < 32):
            raise ValueError("journal authentication key must be >=32 private bytes")
        # Provisioned OUTSIDE the journal directory (e.g. root-only secret).
        # Without this independent trust anchor neither a new lease nor a
        # syntactically valid replacement journal can be trusted.
        self._journal_key = journal_key
        if type(bootstrap_authorized) is not bool:
            raise ValueError("bootstrap authorization must be an explicit boolean")
        # No implicit bootstrap, even when an attacker removes the entire
        # lease directory. Provision this ONLY for a reviewed one-time first
        # install, never permanently in a service restart environment.
        self._bootstrap_authorized = bootstrap_authorized
        if not valid_id(self.instance_id):
            raise ValueError("invalid manager instance identifier")
        self._mutex = threading.RLock()
        self._blocked = False
        self._fault: str | None = None
        self._counter = 0
        self._lease: dict | None = None
        self._last_release: dict | None = None
        self._recovery_required = False
        self._persisted_hash: str | None = None
        self._last_clock_unix: float | None = None
        self._load()

    def _journal_bytes(self) -> bytes:
        """Read the same private, regular inode; reject links and unsafe modes.

        O_NOFOLLOW protects the final component on Linux. lstat/fstat inode
        comparison catches replacement races even on platforms without it.
        """
        parent = self.path.parent
        if (parent.is_symlink() or not parent.is_dir() or
                not private_directory(parent)):
            raise OSError("unsafe lease journal directory")
        before = self.path.lstat()
        if (not stat.S_ISREG(before.st_mode) or before.st_mode & 0o077 or
                before.st_nlink != 1):
            raise OSError("unsafe lease journal file permissions/type")
        if hasattr(os, "geteuid") and before.st_uid != os.geteuid():
            raise OSError("lease journal owned by another account")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(self.path, flags)
        try:
            opened = os.fstat(fd)
            if ((before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino) or
                    not stat.S_ISREG(opened.st_mode) or opened.st_mode & 0o077 or
                    opened.st_nlink != 1 or
                    opened.st_size > 16384):
                raise OSError("lease journal changed during secure read")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                data = stream.read(16385)
            if len(data) > 16384:
                raise OSError("lease journal oversized")
            after = self.path.lstat()
            if (after.st_dev, after.st_ino) != (opened.st_dev, opened.st_ino):
                raise OSError("lease journal replaced during secure read")
            return data
        finally:
            os.close(fd)

    def _verify_store(self) -> None:
        if self._journal_key is None:
            self._blocked = True
            self._fault = "JournalAuthUnconfigured"
            raise LeaseStoreFault("durable lease journal trust anchor not provisioned")
        if self._persisted_hash is None:
            # Only the very first bootstrap, before any durable write,
            # intentionally has no journal hash.
            if not self._bootstrap_authorized:
                self._blocked = True
                self._fault = "BootstrapNotAuthorized"
                raise LeaseStoreFault("initial lease epoch needs explicit authorization")
            if self.path.is_symlink() or self.path.exists() or self.path.parent.exists():
                self._blocked = True
                self._fault = "MissingBootstrapEvidence"
                raise LeaseStoreFault("journal appeared during unprovisioned bootstrap")
            return
        try:
            actual = hashlib.sha256(self._journal_bytes()).hexdigest()
            if actual != self._persisted_hash:
                raise OSError("lease journal changed outside lease state machine")
        except (OSError, ValueError) as exc:
            self._blocked = True
            self._fault = type(exc).__name__
            raise LeaseStoreFault("lease fencing journal missing or tampered") from None

    def _load(self) -> None:
        if self._journal_key is None:
            self._blocked = True
            self._fault = "JournalAuthUnconfigured"
            return
        if not self.path.is_symlink() and not self.path.exists() and not self.path.parent.exists():
            # First bootstrap only. A missing journal inside an EXISTING
            # storage directory might be deleted durable fencing evidence.
            # Complete deletion of BOTH directory and journal cannot be
            # detected without an independently provisioned external anchor.
            if not self._bootstrap_authorized:
                self._blocked = True
                self._fault = "BootstrapNotAuthorized"
            return
        try:
            raw = self._journal_bytes()
            saved = json.loads(raw.decode("utf-8"))
            journal_mac = saved.get("journal_mac") if isinstance(saved, dict) else None
            if not isinstance(journal_mac, str):
                raise ValueError("lease journal missing independent authentication")
            authenticated = {key: value for key, value in saved.items()
                             if key != "journal_mac"}
            expected_mac = hmac.new(
                self._journal_key,
                json.dumps(authenticated, sort_keys=True, separators=(",", ":")).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            if not hmac.compare_digest(expected_mac, journal_mac):
                raise ValueError("lease journal authentication mismatch")
            if (not isinstance(saved, dict) or saved.get("version") != 1 or
                    type(saved.get("counter")) is not int or saved["counter"] < 0 or
                    not isinstance(saved.get("recovery_required"), bool)):
                raise ValueError("invalid lease journal header")
            lease = saved.get("lease")
            if lease is not None and (not isinstance(lease, dict) or
                                      not valid_id(lease.get("id")) or
                                      not valid_id(lease.get("manager_instance_id")) or
                                      lease.get("owner") != OWNER or
                                      lease.get("scope") != SCOPE or
                                      type(lease.get("fencing_token")) is not int or
                                      not 1 <= lease["fencing_token"] <= saved["counter"] or
                                      type(lease.get("expires_at_unix")) not in (int, float) or
                                      not math.isfinite(lease["expires_at_unix"])):
                raise ValueError("invalid persisted lease record")
            last = saved.get("last_release")
            if last is not None and (not isinstance(last, dict) or
                                     not valid_id(last.get("id")) or
                                     not valid_id(last.get("manager_instance_id")) or
                                     type(last.get("fencing_token")) is not int or
                                     not 1 <= last["fencing_token"] <= saved["counter"]):
                raise ValueError("invalid last-release record")
            highwater = saved.get("last_clock_unix")
            if (highwater is not None and
                    (type(highwater) not in (int, float) or not math.isfinite(highwater))):
                raise ValueError("invalid persisted wallclock watermark")
            self._counter = saved["counter"]
            self._lease = lease
            self._last_release = last
            self._last_clock_unix = highwater
            self._persisted_hash = hashlib.sha256(raw).hexdigest()
            self._recovery_required = saved["recovery_required"] or lease is not None
            # If the process restarted with a live or expired lease, never
            # allow the old watchdog to reallocate GPUs before fenced recovery.
            if lease is not None and not saved["recovery_required"]:
                self._write()
        except (OSError, ValueError, TypeError, UnicodeError, LeaseStoreFault) as exc:
            self._blocked = True
            self._fault = type(exc).__name__

    def _write(self) -> None:
        payload = {"version": 1, "counter": self._counter,
                   "lease": self._lease, "last_release": self._last_release,
                   "recovery_required": self._recovery_required,
                   "last_clock_unix": self._last_clock_unix}
        try:
            self._verify_store()  # never overwrite missing/tampered fencing
            payload["journal_mac"] = hmac.new(
                self._journal_key,
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if self.path.parent.is_symlink() or self.path.is_symlink():
                raise OSError("lease journal symlink forbidden")
            # Parent directory is operator-owned and private; the write is
            # durable prior to any acknowledgement to a lease client.
            if not private_directory(self.path.parent):
                raise OSError("lease journal directory is group/world writable")
            temp = self.path.with_name(self.path.name + ".writing")
            if temp.is_symlink():
                raise OSError("lease temp symlink forbidden")
            fd = os.open(
                temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                getattr(os, "O_NOFOLLOW", 0), 0o600,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    raw = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
                    handle.write(raw.decode("utf-8"))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, self.path)
                if hasattr(os, "O_DIRECTORY"):
                    dfd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(dfd)
                    finally:
                        os.close(dfd)
                self._persisted_hash = hashlib.sha256(raw).hexdigest()
                self._bootstrap_authorized = False
            finally:
                temp.unlink(missing_ok=True)
        except (OSError, LeaseStoreFault) as exc:
            self._blocked = True
            self._fault = type(exc).__name__
            raise LeaseStoreFault("cannot durably persist GPU fencing state") from None

    def _check_clock_rollback(self) -> None:
        if self._lease is None or self._recovery_required or self._last_clock_unix is None:
            return
        now = self.clock()
        if not math.isfinite(now):
            self._blocked = True
            self._fault = "InvalidClock"
            raise LeaseStoreFault("lease wallclock unavailable")
        if now < self._last_clock_unix - CLOCK_ROLLBACK_TOLERANCE_SECONDS:
            # Never extend or shorten an active lease against a clock that
            # stepped backwards: require independent, fenced GPU recovery.
            self._recovery_required = True
            self._write()
            raise LeaseRejected("wallclock rollback: fenced recovery required")

    def _require_proof(self, kind: str, proof: dict) -> None:
        # Structural JSON checks cannot establish proof provenance.
        # No trusted verifier provisioned -> NO transition/recovery grants.
        if self.proof_verifier is None:
            raise LeaseRejected("independent proof verifier not provisioned")
        try:
            trusted = self.proof_verifier(kind, proof)
        except Exception:
            trusted = False
        if trusted is not True:
            raise LeaseRejected("independent proof authentication failed")

    def _require_ingress(self, proof: dict) -> None:
        require_ingress(proof, now=self.clock())
        self._require_proof("ingress", proof)

    def _require_gpu(self, proof: dict, *, restored: bool = False) -> None:
        require_gpu_safe(proof, now=self.clock(), after_tabby_restore=restored)
        self._require_proof("gpu-restored" if restored else "gpu-free", proof)

    def _expire_if_needed(self) -> None:
        if (self._lease is not None and not self._recovery_required and
                self.clock() >= self._lease["expires_at_unix"]):
            self._recovery_required = True
            self._write()

    @property
    def fenced(self) -> bool:
        with self._mutex:
            try:
                self._verify_store()
                self._check_clock_rollback()
                self._expire_if_needed()
            except (LeaseStoreFault, LeaseRejected):
                pass
            return self._blocked or self._recovery_required or self._lease is not None

    @property
    def recovery_required(self) -> bool:
        with self._mutex:
            try:
                self._verify_store()
                self._check_clock_rollback()
                self._expire_if_needed()
            except (LeaseStoreFault, LeaseRejected):
                pass
            return self._blocked or self._recovery_required

    @property
    def faulted(self) -> bool:
        return self._blocked

    def status(self) -> dict:
        with self._mutex:
            _ = self.fenced
            lease = None
            if self._lease is not None:
                lease = {key: self._lease[key] for key in (
                    "id", "owner", "scope", "fencing_token", "expires_at_unix")}
                lease.update({"status": "recovery_required" if self._recovery_required else "active",
                              "watchdog_restart_suppressed": True})
            return {"watchdog_lease_contract_version": 1,
                    "manager_instance_id": self.instance_id,
                    "maintenance_lease": lease,
                    "maintenance_recovery_required": self._recovery_required or self._blocked}

    def _require_healthy(self) -> None:
        if self._blocked:
            raise LeaseStoreFault("lease fencing journal unavailable: service remains fenced")
        self._verify_store()
        self._check_clock_rollback()

    def acquire(self, request: dict, *, snapshot: dict, ingress_proof: dict) -> dict:
        with self._mutex:
            self._require_healthy()
            self._expire_if_needed()
            if self._recovery_required:
                raise LeaseRejected("GPU recovery required before any new lease")
            self._validate_request(request, "acquire", fence_required=False)
            if self._lease is not None:
                # Repeat an acquire with the identical ID may acknowledge the
                # original grant; the fencing token/instance never changes.
                if self._lease["id"] == request["lease_id"]:
                    self._require_ingress(ingress_proof)
                    return self._ack("acquire")
                raise LeaseRejected("another GPU maintenance lease holds the fencing token")
            require_idle(snapshot)
            self._require_ingress(ingress_proof)
            self._counter += 1
            now = self.clock()
            if not math.isfinite(now):
                raise LeaseRejected("lease clock unavailable")
            self._last_clock_unix = max(self._last_clock_unix or now, now)
            self._lease = {"id": request["lease_id"], "owner": OWNER, "scope": SCOPE,
                           "manager_instance_id": self.instance_id,
                           "fencing_token": self._counter,
                           "expires_at_unix": now + TTL_SECONDS}
            self._write()
            return self._ack("acquire")

    def renew(self, request: dict, *, ingress_proof: dict) -> dict:
        with self._mutex:
            self._require_healthy()
            self._expire_if_needed()
            self._validate_request(request, "renew")
            self._require_holder(request)
            if self._recovery_required:
                raise LeaseRejected("expired/restarted lease: recovery required")
            self._require_ingress(ingress_proof)
            now = self.clock()
            if not math.isfinite(now):
                raise LeaseRejected("lease clock unavailable")
            # A minor permitted wallclock adjustment must never DECREASE a
            # previously issued lease expiry. Larger rollback is fenced in
            # _require_healthy() and demands independently verified recovery.
            self._lease["expires_at_unix"] = max(
                self._lease["expires_at_unix"], now + TTL_SECONDS)
            self._last_clock_unix = max(self._last_clock_unix or now, now)
            self._write()
            return self._ack("renew")

    def release(self, request: dict, *, ingress_proof: dict,
                gpu_safe_proof: dict, tabby_serving: bool) -> dict:
        with self._mutex:
            self._require_healthy()
            self._expire_if_needed()
            self._validate_request(request, "release")
            if self._lease is None and self._last_release is not None:
                last = self._last_release
                if (request["lease_id"] == last["id"] and
                        request["fencing_token"] == last["fencing_token"] and
                        request["manager_instance_id"] == last["manager_instance_id"]):
                    return self._ack("release", request_id=last["id"])
            self._require_holder(request)
            if self._recovery_required:
                raise LeaseRejected("expired lease cannot use ordinary release")
            self._require_ingress(ingress_proof)
            self._require_gpu(gpu_safe_proof, restored=True)
            if tabby_serving is not True:
                raise LeaseRejected("Tabby not confirmed serving before lease release")
            self._last_release = {"id": self._lease["id"],
                                  "fencing_token": self._lease["fencing_token"],
                                  "manager_instance_id": self.instance_id}
            self._lease = None
            self._write()
            return self._ack("release", request_id=self._last_release["id"])

    def assert_recovery_safe(self, *, ingress_proof: dict, gpu_safe_proof: dict) -> None:
        with self._mutex:
            self._require_healthy()
            if not self.fenced:
                raise LeaseRejected("no recovery fence to clear")
            self._require_ingress(ingress_proof)
            self._require_gpu(gpu_safe_proof)

    def assert_recovery_already_restored(self, *, ingress_proof: dict,
                                         gpu_safe_proof: dict) -> None:
        """Tabby may already be independently restored while the fence remains.

        Never force-stop a healthy Tabby: permit clearing the orphaned fence
        only after independent proof Hermes is gone and Tabby owns both GPUs.
        """
        with self._mutex:
            self._require_healthy()
            if not self.fenced:
                raise LeaseRejected("no recovery fence to clear")
            self._require_ingress(ingress_proof)
            self._require_gpu(gpu_safe_proof, restored=True)

    def finish_recovery(self, *, ingress_proof: dict, gpu_safe_proof: dict,
                        tabby_serving: bool) -> dict:
        with self._mutex:
            self._require_healthy()
            if not self.fenced:
                raise LeaseRejected("no recovery fence to clear")
            self._require_ingress(ingress_proof)
            self._require_gpu(gpu_safe_proof, restored=True)
            if tabby_serving is not True:
                raise LeaseRejected("Tabby still unhealthy after recovery")
            if self._lease is not None:
                self._last_release = {"id": self._lease["id"],
                                      "fencing_token": self._lease["fencing_token"],
                                      "manager_instance_id": self._lease["manager_instance_id"]}
            self._lease = None
            self._recovery_required = False
            self._write()
            return {"ok": True, "protocol": PROTOCOL, "action": "recover"}

    def _ack(self, action: str, *, request_id: str | None = None) -> dict:
        lease_id = request_id or (self._lease or self._last_release or {}).get("id")
        return {"ok": True, "protocol": PROTOCOL, "action": action,
                "lease_id": lease_id}

    def _validate_request(self, request: dict, action: str, *, fence_required: bool = True) -> None:
        if (not isinstance(request, dict) or request.get("protocol") != PROTOCOL or
                request.get("action") != action or request.get("owner") != OWNER or
                not valid_id(request.get("lease_id")) or
                request.get("manager_instance_id") != self.instance_id):
            raise LeaseRejected("lease request version/owner/instance/id invalid")
        fencing = request.get("fencing_token")
        if fence_required:
            if type(fencing) is not int or fencing < 1:
                raise LeaseRejected("missing positive fencing token")
        elif fencing is not None:
            raise LeaseRejected("acquire must use null fencing token")

    def _require_holder(self, request: dict) -> None:
        lease = self._lease
        if (lease is None or lease.get("id") != request["lease_id"] or
                lease.get("manager_instance_id") != request["manager_instance_id"] or
                lease.get("fencing_token") != request["fencing_token"]):
            raise LeaseRejected("stale/non-owner lease fencing token")
