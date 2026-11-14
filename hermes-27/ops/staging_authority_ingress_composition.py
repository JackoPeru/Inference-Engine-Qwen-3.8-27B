"""CPU/OFFLINE composition of two INDEPENDENT staging lease/ingress contracts.

Both published protocols consume nonce/counter state. An ACK from either
independent ledger cannot be transferred into the other one; two separately
committed updates do not form an atomic proof. This experiment stages *both*
existing state-machine mutations inside ONE injected parent transaction,
committing them together or denying the entire operation. It deliberately
supports only an initially pristine pair of ledgers: later epochs require an
independently attested external recovery/rotation contract that does not exist.

No privileged issuer, physical drain, GPU owner attestor, external highwater
store, or service interlock is implemented. Every response, including a
synthetically committed pair, is BLOCKED for production and carries no manager
lease proof. Never integrate this module as a live verifier or lease provider.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
from typing import Callable

from ops import ingress_attestation_contract as ingress
from ops import manager_lease_external_authority_staging as authority


class CompositionDenied(RuntimeError):
    """A rejected or ambiguous offline transaction, never an authorization."""


@dataclass(frozen=True)
class CompositionState:
    authority_ledger: authority.LedgerState = field(
        default_factory=authority.LedgerState)
    ingress_ledger: ingress.ConsumptionState = field(
        default_factory=ingress.ConsumptionState)
    paired_nonce: str | None = None
    paired_manifest: str | None = None


class AtomicCompositionStore(ABC):
    @abstractmethod
    def atomic_update(
        self, mutation: Callable[[CompositionState], tuple[CompositionState, dict]],
    ) -> dict:
        """Commit both ledgers as ONE indivisible durable, externally anchored
        transaction before ACK; reject ambiguous commit, replay and rollback.

        No implementation or authority bootstrap is supplied here. A mock
        implementing the signature does not create a physically trusted store.
        """
        raise NotImplementedError


_BLOCKED = {
    "status": "BLOCKED",
    "production_admissible": False,
    "manager_lease_proof": None,
    "synthetic_observation": "ONE_OFFLINE_TRANSACTION_ONLY",
}


def status() -> dict:
    """Unconditional production NO-GO, including on synthetic commit."""
    return {
        "status": "BLOCKED",
        "production_admissible": False,
        "manager_lease_proof": None,
        "reason": "no privileged issuer, physical owner or external atomic store",
    }


def _matching_context(
    ticket: object, ingress_ticket: object, package: object,
    context: ingress.Context,
) -> str:
    """Strict cross-protocol binding before staged signatures are verified.

    Signatures are verified by the existing implementations *inside* the
    composite atomic callback, against these exact deep-copied snapshots.
    """
    if (type(context) is not ingress.Context or
            type(ticket) is not dict or type(ingress_ticket) is not dict or
            type(package) is not dict or set(package) != {"manifest", "routes", "gpu"}):
        raise CompositionDenied("composition context or signed package missing")
    for name, envelope in (("reserve", ticket),
                           ("ingress", ingress_ticket),
                           ("manifest", package["manifest"])):
        if (type(envelope) is not dict or
                set(envelope) != {"claims", "signature"} or
                type(envelope.get("claims")) is not dict):
            raise CompositionDenied(f"composition {name} envelope incomplete")
    manifest = package["manifest"]["claims"]
    try:
        digest = hashlib.sha256(
            authority.canonical_message("authority", manifest)
        ).hexdigest()
    except Exception:
        raise CompositionDenied("manifest digest cannot be reconstructed") from None
    required = {
        "boot_id": context.boot_id,
        "manager_instance_id": context.manager_instance_id,
        "lease_id": context.lease_id,
        "epoch": context.epoch,
        "fencing_token": context.fencing_token,
        "nonce": context.nonce,
    }
    for claims in (ticket["claims"], ingress_ticket["claims"], manifest):
        if any(type(claims.get(k)) is not type(v) or claims.get(k) != v
               for k, v in required.items()):
            raise CompositionDenied("cross-protocol boot/lease/epoch/fence/nonce mismatch")
    for claims in (ticket["claims"], ingress_ticket["claims"]):
        if (type(claims.get("route_manifest_sha256")) is not str or
                claims["route_manifest_sha256"] != digest or
                type(claims.get("route_manifest_epoch")) is not int or
                claims["route_manifest_epoch"] != context.manifest_epoch):
            raise CompositionDenied("cross-protocol signed manifest binding mismatch")
    if (type(manifest.get("manifest_epoch")) is not int or
            manifest["manifest_epoch"] != context.manifest_epoch):
        raise CompositionDenied("physical manifest epoch mismatch")
    return digest


def _initial_pair(state: object, *, journal_counter: int,
                  context: ingress.Context) -> None:
    """No proof transfer from independently consumed/preexisting ledgers."""
    if type(state) is not CompositionState:
        raise CompositionDenied("composite atomic ledger unavailable or corrupt")
    try:
        authority._validate_ledger(state.authority_ledger)
        ingress._validate_store(state.ingress_ledger)
    except Exception:
        raise CompositionDenied("one or both underlying ledger records corrupt") from None
    # No independent history reconciliation/rotation exists. In particular,
    # counters that separately reached 1 cannot be considered a paired commit.
    if (state != CompositionState() or type(journal_counter) is not int or
            journal_counter != 0 or context.epoch != 1 or
            context.fencing_token != 1):
        raise CompositionDenied(
            "non-pristine or independently consumed ledgers require external recovery")


class _StagedAuthorityStore(authority.ExternalAtomicStore):
    def __init__(self, state: authority.LedgerState):
        self.state = state
        self.calls = 0

    def atomic_update(self, mutation):
        if self.calls >= 2:
            raise CompositionDenied("unexpected authority ledger transition count")
        new, receipt = mutation(self.state)
        self.state = new
        self.calls += 1
        return receipt


class _StagedIngressStore(ingress.AtomicConsumptionStore):
    def __init__(self, state: ingress.ConsumptionState):
        self.state = state
        self.calls = 0

    def atomic_update(self, mutation):
        if self.calls:
            raise CompositionDenied("ingress nonce already consumed")
        new, receipt = mutation(self.state)
        self.state = new
        self.calls += 1
        return receipt


def compose(
    *,
    authority_reservation: object,
    authority_ingress: object,
    physical_package: object,
    context: ingress.Context,
    journal_counter: int,
    authority_verifier: authority.ExternalSignatureVerifier | None,
    physical_verifier: authority.ExternalSignatureVerifier | None,
    store: AtomicCompositionStore | None,
    wall_clock: Callable[[], float],
    monotonic_clock: Callable[[], float],
) -> dict:
    """One synthetic transaction or denial. NEVER returns a production grant.

    Both published state machines operate over temporary adapters whose
    updates have no independent durability. The parent store commits their
    resulting pair exactly once, so failures before parent commit leave the
    previous pair untouched. A lost ACK after commit remains an error.
    """
    if (type(context) is not ingress.Context or
            not isinstance(authority_verifier, authority.ExternalSignatureVerifier)
            or not isinstance(physical_verifier, authority.ExternalSignatureVerifier)
            or authority_verifier is physical_verifier
            or not isinstance(store, AtomicCompositionStore)
            or not callable(wall_clock) or not callable(monotonic_clock)):
        raise CompositionDenied("separate external verifiers or atomic store unavailable")

    # Freeze mutable caller structures. Only these byte-for-byte snapshots are
    # passed to both cryptographic/structural validators.
    try:
        reservation, signed_ingress, package = deepcopy((
            authority_reservation, authority_ingress, physical_package))
    except Exception:
        raise CompositionDenied("signed evidence cannot be snapshotted") from None
    digest = _matching_context(reservation, signed_ingress, package, context)
    invoked = [0]

    def mutation(old: CompositionState) -> tuple[CompositionState, dict]:
        invoked[0] += 1
        if invoked[0] != 1:
            raise CompositionDenied("atomic callback executed more than once")
        _initial_pair(old, journal_counter=journal_counter, context=context)
        staged_auth = _StagedAuthorityStore(old.authority_ledger)
        staged_ingress = _StagedIngressStore(old.ingress_ledger)
        validator = authority.StagingAuthority(
            store=staged_auth, verifier=authority_verifier, clock=wall_clock)
        # Reserve before ingress attestations; this is staged, not durable,
        # until the single final parent-store atomic commit.
        reserved = validator.reserve(reservation, journal_counter=journal_counter)
        measured = ingress.assess(
            package, context=context, verifier=physical_verifier,
            store=staged_ingress, wall_clock=wall_clock,
            monotonic_clock=monotonic_clock)
        consumed = validator.consume_ingress(signed_ingress)
        if (staged_auth.calls != 2 or staged_ingress.calls != 1 or
                any(type(result) is not dict or
                    result.get("production_admissible") is not False or
                    result.get("manager_lease_proof") is not None
                    for result in (reserved, measured, consumed)) or
                staged_auth.state.phase != "drain-verified" or
                staged_auth.state.active_nonce != context.nonce or
                staged_ingress.state.spent_nonces != (context.nonce,) or
                staged_auth.state.spent_nonces != (context.nonce,) or
                staged_auth.state.route_manifest_sha256 != digest or
                staged_auth.state.route_manifest_epoch != context.manifest_epoch or
                staged_ingress.state.highwater != staged_auth.state.highwater or
                staged_auth.state.highwater != context.fencing_token):
            raise CompositionDenied("staged ledgers diverged or returned unsafe ACK")
        paired = CompositionState(
            authority_ledger=staged_auth.state,
            ingress_ledger=staged_ingress.state,
            paired_nonce=context.nonce,
            paired_manifest=digest,
        )
        return paired, dict(_BLOCKED)

    try:
        ack = store.atomic_update(mutation)
    except Exception:
        # The underlying external store may have committed before the ACK
        # failed. Never assume rollback, retry, or promote either nonce.
        raise CompositionDenied("atomic composition denied or commit ambiguous") from None
    if invoked[0] != 1 or type(ack) is not dict or ack != _BLOCKED:
        raise CompositionDenied("composition store returned noncanonical ACK")
    return dict(_BLOCKED)
