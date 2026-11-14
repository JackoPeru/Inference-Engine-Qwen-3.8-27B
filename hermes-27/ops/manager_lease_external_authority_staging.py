"""OFFLINE/STAGING ONLY: external lease high-water and one-shot proof contract.

No root issuer, signed physical ingress collector, monotonic hardware store,
or GPU owner verifier exists here. This module never provides an admissible
ManagerLeaseV1 proof, nor a production lease grant. All outcomes, including
synthetic success, preserve production_admissible=False.

An independent authority would need to guarantee all of the following:
  * Each atomic_update is serialized ACROSS processes, durably committed before
    its acknowledgement, tamper-resistant and independently anti-rollback.
  * Signatures for "authority", "ingress", and "gpu-owner" bind distinct,
    privileged issuers to current boot/process/lease/epoch/fencing/nonce.
  * Coverage of ALL physical ingress, SSE and GPU owners is real and stays
    fenced throughout the authority's handoff. A signed dict alone cannot
    establish any of these facts.

The abstract interfaces are intentionally NOT implementations of these
guarantees. Calling them with a mock object never makes production admissible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
import json
import math
import re
import secrets
import time
from typing import Callable


PROTOCOL = "hermes-external-lease-authority-STAGING-v1"
DOMAIN = b"hermes-27/lease-external-authority-staging/v1\0"
MAX_SEQUENCE = (1 << 63) - 1
MAX_NONCES = 4096  # Exhaustion fences; never silently discard spent nonces.
MAX_AGE_SECONDS = 10.0
HEX_ID = re.compile(r"[0-9a-f]{32}\Z")
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
HEX_SIGNATURE = re.compile(r"(?:[0-9a-f]{2}){32,256}\Z")
ROLES = frozenset({"authority", "ingress", "gpu-owner"})
BASE_FIELDS = frozenset({
    "protocol", "kind", "boot_id", "manager_instance_id", "lease_id",
    "epoch", "fencing_token", "nonce", "observed_at_unix",
})
RESERVE_FIELDS = BASE_FIELDS | {
    "route_manifest_sha256", "route_manifest_epoch",
}
INGRESS_FIELDS = RESERVE_FIELDS | {
    "scope", "blocked", "inflight", "active_sse", "unknown_connections",
}
RESTORED_FIELDS = BASE_FIELDS | {
    "scope", "gpu_count", "gpu_owner", "tabby_owns_both_gpus",
    "hermes_pids", "other_gpu_processes",
}


class StagingDenied(RuntimeError):
    """Fail closed; never log secrets or unsigned input as trusted evidence."""


@dataclass(frozen=True)
class LedgerState:
    """Immutable *interface* state to be stored by an external authority.

    'spent_nonces' is never pruned: a full store requires operator-mediated,
    independently authenticated rotation. A mock cannot be promoted into a
    trusted external store by returning this dataclass.
    """

    highwater: int = 0
    epoch: int = 0
    phase: str = "idle"
    boot_id: str | None = None
    manager_instance_id: str | None = None
    lease_id: str | None = None
    active_nonce: str | None = None
    holder_session: str | None = None
    route_manifest_sha256: str | None = None
    route_manifest_epoch: int | None = None
    spent_nonces: tuple[str, ...] = ()


class ExternalAtomicStore(ABC):
    """Privileged external compare/update contract; no implementation here."""

    @abstractmethod
    def atomic_update(
        self,
        mutate: Callable[[LedgerState], tuple[LedgerState, dict]],
    ) -> dict:
        """Durably commit a SERIALIZABLE read-modify-write before returning.

        Reject missing/corrupt/rolled-back storage; transaction faults may
        occur before or after a commit and must propagate to the caller.
        A local file, mutex or in-process bool does NOT satisfy this contract.
        """
        raise NotImplementedError


class ExternalSignatureVerifier(ABC):
    """Pinned role-separated public verification; no signer implementation."""

    @abstractmethod
    def verify(self, role: str, message: bytes, signature: bytes) -> None:
        """Raise on any error. Successful verification returns EXACTLY None.

        The caller must independently establish that the real signer is
        privileged to observe all ingress/GPU routes and that its key is
        protected. Merely implementing this method is not authentication.
        """
        raise NotImplementedError


def canonical_message(role: str, claims: dict) -> bytes:
    """Domain-separated, unambiguous signature bytes for role-pinned issuers."""
    if role not in ROLES or type(claims) is not dict:
        raise StagingDenied("invalid signature role or claims")
    try:
        encoded = json.dumps(
            claims, sort_keys=True, ensure_ascii=True, allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        raise StagingDenied("claims cannot be canonically encoded") from None
    if len(encoded) > 4096:
        raise StagingDenied("signed claims exceed bounded size")
    return DOMAIN + role.encode("ascii") + b"\0" + encoded


def _id(value: object) -> bool:
    return type(value) is str and HEX_ID.fullmatch(value) is not None


def _uint(value: object, *, low: int = 0, high: int = MAX_SEQUENCE) -> bool:
    return type(value) is int and low <= value <= high


def _validate_ledger(state: LedgerState) -> None:
    if type(state) is not LedgerState:
        raise StagingDenied("external ledger unavailable or structurally corrupt")
    if (not _uint(state.highwater) or not _uint(state.epoch)
            or state.epoch != state.highwater
            or state.phase not in ("idle", "pending", "drain-verified")
            or type(state.spent_nonces) is not tuple
            or len(state.spent_nonces) > MAX_NONCES
            or any(not _id(nonce) for nonce in state.spent_nonces)
            or len(set(state.spent_nonces)) != len(state.spent_nonces)):
        raise StagingDenied("external ledger corrupt: sequence/phase/nonces")
    if state.phase == "idle":
        if any(value is not None for value in (
                state.boot_id, state.manager_instance_id, state.lease_id,
                state.active_nonce, state.holder_session,
                state.route_manifest_sha256, state.route_manifest_epoch)):
            raise StagingDenied("external ledger idle with residual ownership")
    elif (not all(_id(value) for value in (
            state.boot_id, state.manager_instance_id, state.lease_id,
            state.active_nonce, state.holder_session))
            or state.active_nonce not in state.spent_nonces
            or type(state.route_manifest_sha256) is not str
            or HEX_SHA256.fullmatch(state.route_manifest_sha256) is None
            or not _uint(state.route_manifest_epoch, low=1)
            or state.highwater == 0 or state.epoch == 0):
        raise StagingDenied("external ledger active context incomplete")


class StagingAuthority:
    """Verifiable synthetic transition model; ALWAYS returns BLOCKED for prod.

    A new instance generates a new process incarnation. In-flight reservations
    from a crashed prior process cannot silently resume or be completed.
    An operator-reviewed *external* recovery mechanism is still missing.
    """

    def __init__(
        self,
        *,
        store: ExternalAtomicStore | None = None,
        verifier: ExternalSignatureVerifier | None = None,
        clock: Callable[[], float] = time.time,
    ):
        if store is not None and not isinstance(store, ExternalAtomicStore):
            raise StagingDenied("external atomic store interface not supplied")
        if verifier is not None and not isinstance(verifier, ExternalSignatureVerifier):
            raise StagingDenied("external signature verifier interface not supplied")
        if not callable(clock):
            raise StagingDenied("clock unavailable")
        self._store = store
        self._verifier = verifier
        self._clock = clock
        self._session = secrets.token_hex(16)

    @staticmethod
    def status() -> dict:
        """No trust-by-boolean switch, even if every synthetic check passes."""
        return {
            "status": "BLOCKED",
            "production_admissible": False,
            "manager_lease_proof": None,
            "reason": "no privileged independent issuer/antirollback store",
        }

    def _now(self) -> float:
        try:
            observed = self._clock()
        except Exception:
            raise StagingDenied("clock unavailable") from None
        if type(observed) not in (float, int) or not math.isfinite(observed):
            raise StagingDenied("clock invalid")
        return observed

    def _check_age(self, claims: dict) -> None:
        observed = claims["observed_at_unix"]
        if (type(observed) not in (float, int) or not math.isfinite(observed)
                or not 0 <= self._now() - observed <= MAX_AGE_SECONDS):
            raise StagingDenied("external observation expired or from future")

    def _validate_claims(self, claims: dict, kind: str) -> None:
        allowed = (INGRESS_FIELDS if kind == "ingress-closed" else
                   RESTORED_FIELDS if kind == "gpu-restored" else
                   RESERVE_FIELDS)
        if (set(claims) != allowed or claims.get("protocol") != PROTOCOL
                or claims.get("kind") != kind
                or not all(_id(claims.get(key)) for key in (
                    "boot_id", "manager_instance_id", "lease_id", "nonce"))
                or not _uint(claims.get("epoch"), low=1)
                or not _uint(claims.get("fencing_token"), low=1)):
            raise StagingDenied("external observation identity/epoch/scope invalid")
        self._check_age(claims)
        if kind in ("reserve", "ingress-closed"):
            if (type(claims["route_manifest_sha256"]) is not str or
                    HEX_SHA256.fullmatch(claims["route_manifest_sha256"]) is None or
                    not _uint(claims["route_manifest_epoch"], low=1)):
                raise StagingDenied("signed route-manifest identity invalid")
        if kind == "ingress-closed":
            if (claims["scope"] != "all-tabby-clients" or
                    claims["blocked"] is not True or
                    not all(_uint(claims.get(key), high=0) for key in
                            ("inflight", "active_sse", "unknown_connections"))):
                raise StagingDenied("signed ingress evidence not completely drained")
        if kind == "gpu-restored":
            if (claims["scope"] != "all-hermes-gpus-and-pids"
                    or type(claims["gpu_count"]) is not int
                    or claims["gpu_count"] != 2
                    or claims["gpu_owner"] != "tabby"
                    or claims["tabby_owns_both_gpus"] is not True
                    or claims["hermes_pids"] != []
                    or claims["other_gpu_processes"] != []):
                raise StagingDenied("synthetic GPU restored evidence incomplete")

    def _verify(self, envelope: object, *, role: str, kind: str) -> dict:
        if self._verifier is None:
            raise StagingDenied("independent external verifier unconfigured")
        if (type(envelope) is not dict or set(envelope) != {"claims", "signature"}
                or type(envelope.get("claims")) is not dict
                or type(envelope.get("signature")) is not str
                or HEX_SIGNATURE.fullmatch(envelope["signature"]) is None):
            raise StagingDenied("signed envelope malformed or untrusted")
        claims = envelope["claims"]
        self._validate_claims(claims, kind)
        message = canonical_message(role, claims)
        try:
            result = self._verifier.verify(
                role, message, bytes.fromhex(envelope["signature"]))
        except Exception:
            raise StagingDenied("independent signer verification failed") from None
        if result is not None:
            raise StagingDenied("signature verifier must not return a trust boolean")
        return claims

    def _atomic(self, transition: Callable[[LedgerState], tuple[LedgerState, dict]]) -> dict:
        if self._store is None:
            raise StagingDenied("independent external monotonic store unconfigured")

        def validated(state: LedgerState) -> tuple[LedgerState, dict]:
            _validate_ledger(state)
            following, result = transition(state)
            _validate_ledger(following)
            if (type(result) is not dict or
                    result.get("production_admissible") is not False):
                raise StagingDenied("synthetic transition returned unsafe result")
            return following, result

        try:
            answer = self._store.atomic_update(validated)
        except StagingDenied:
            raise
        except Exception:
            # A backend may have committed immediately before losing its
            # acknowledgement. Deny; do not infer rollback or retry success.
            raise StagingDenied("external atomic transition unavailable or ambiguous") from None
        if (type(answer) is not dict or answer.get("production_admissible") is not False):
            raise StagingDenied("external backend returned untrusted acknowledgement")
        return answer

    @staticmethod
    def _candidate(phase: str, state: LedgerState) -> dict:
        return {"synthetic_phase": phase, "candidate_epoch": state.epoch,
                "candidate_fencing_token": state.highwater,
                "production_admissible": False, "manager_lease_proof": None}

    @staticmethod
    def _require_context(state: LedgerState, claims: dict) -> None:
        if any((state.boot_id != claims["boot_id"],
                state.manager_instance_id != claims["manager_instance_id"],
                state.lease_id != claims["lease_id"],
                state.epoch != claims["epoch"],
                state.highwater != claims["fencing_token"])):
            raise StagingDenied("proof boot/manager/lease/epoch/fencing mismatch")

    def reserve(self, authority_ticket: object, *, journal_counter: int) -> dict:
        """Synthetic signed high-water reservation BEFORE any manager acquire.

        Rejects previously valid, older journals on a restarted manager.
        Trust in the true source of journal_counter is NOT implemented here.
        """
        claims = self._verify(authority_ticket, role="authority", kind="reserve")
        if not _uint(journal_counter):
            raise StagingDenied("manager journal counter invalid")

        def transition(state: LedgerState):
            self._check_age(claims)  # repeat under the external serialization
            if state.phase != "idle":
                raise StagingDenied("external authority still fenced or recovery needed")
            if journal_counter != state.highwater:
                raise StagingDenied("journal highwater mismatch: replay or divergent authority")
            if (state.highwater == MAX_SEQUENCE or state.epoch == MAX_SEQUENCE
                    or len(state.spent_nonces) + 2 > MAX_NONCES):
                raise StagingDenied("external sequence/nonce log exhausted; no rollover")
            if (claims["epoch"] != state.epoch + 1 or
                    claims["fencing_token"] != state.highwater + 1):
                raise StagingDenied("external epoch and fencing token must increment exactly once")
            if claims["nonce"] in state.spent_nonces:
                raise StagingDenied("external reservation nonce already spent")
            following = replace(
                state, highwater=state.highwater + 1, epoch=state.epoch + 1,
                phase="pending", boot_id=claims["boot_id"],
                manager_instance_id=claims["manager_instance_id"],
                lease_id=claims["lease_id"], active_nonce=claims["nonce"],
                holder_session=self._session,
                route_manifest_sha256=claims["route_manifest_sha256"],
                route_manifest_epoch=claims["route_manifest_epoch"],
                spent_nonces=state.spent_nonces + (claims["nonce"],))
            return following, self._candidate("pending", following)

        return self._atomic(transition)

    def consume_ingress(self, signed_ingress: object) -> dict:
        """Consume one signed all-route zero-inflight proof exactly once."""
        claims = self._verify(signed_ingress, role="ingress", kind="ingress-closed")

        def transition(state: LedgerState):
            self._check_age(claims)
            if state.phase != "pending" or state.holder_session != self._session:
                raise StagingDenied("ingress claim already consumed or manager restarted")
            self._require_context(state, claims)
            if state.active_nonce != claims["nonce"]:
                raise StagingDenied("ingress claim lacks the current one-shot nonce")
            if (state.route_manifest_sha256 != claims["route_manifest_sha256"]
                    or state.route_manifest_epoch != claims["route_manifest_epoch"]):
                raise StagingDenied("issuer and ingress signed route-manifest mismatch")
            following = replace(state, phase="drain-verified")
            return following, self._candidate("drain-verified", following)

        return self._atomic(transition)

    def complete_restore(self, signed_gpu_restore: object) -> dict:
        """Synthetic closure requires separate GPU-role evidence and nonce."""
        claims = self._verify(
            signed_gpu_restore, role="gpu-owner", kind="gpu-restored")

        def transition(state: LedgerState):
            self._check_age(claims)
            if state.phase != "drain-verified" or state.holder_session != self._session:
                raise StagingDenied("restore requires current unexpired fenced incarnation")
            self._require_context(state, claims)
            if claims["nonce"] in state.spent_nonces:
                raise StagingDenied("GPU restoration nonce already spent")
            if len(state.spent_nonces) >= MAX_NONCES:
                raise StagingDenied("external nonce store exhausted")
            following = replace(
                state, phase="idle", boot_id=None, manager_instance_id=None,
                lease_id=None, active_nonce=None, holder_session=None,
                route_manifest_sha256=None, route_manifest_epoch=None,
                spent_nonces=state.spent_nonces + (claims["nonce"],))
            return following, self._candidate("idle", following)

        return self._atomic(transition)
