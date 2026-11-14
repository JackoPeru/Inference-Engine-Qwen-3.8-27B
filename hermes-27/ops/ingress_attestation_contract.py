"""CPU/OFFLINE contract for a hypothetical privileged physical ingress issuer.

This is a structural and cryptographic *staging* witness, not a collector,
gateway, lease grant, authority bootstrap, or physical trust anchor. The
caller-supplied verifier/store might be mocks. Every returned result has
production_admissible=False and manager_lease_proof=None, unconditionally.

An external issuer would have to prove exclusive control of every Tabby
listener and restart path, kernel/driver GPU owner coverage, independently
pinned signing keys, and a durable, anti-rollback atomic consumption store.
No boolean, manifest, signature, or mocked atomic store creates that issuer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Callable

from ops.ingress_drain_gate_staging import REQUIRED_ROUTES
from ops.manager_lease_external_authority_staging import (
    ExternalSignatureVerifier,
    canonical_message,
)


PROTOCOL = "hermes-physical-ingress-STAGING-v1"
MAX_AGE_SECONDS = 5.0
MAX_SNAPSHOT_SKEW = 2.0
MAX_SEQUENCE = (1 << 63) - 1
MAX_SPENT = 4096
HEX32 = re.compile(r"[0-9a-f]{32}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
SIG = re.compile(r"(?:[0-9a-f]{2}){32,256}\Z")
GPU_UUID = re.compile(
    r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
)
PCI = re.compile(r"[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]\Z")

# These are the six actual ingress/control surfaces in the published staging
# gate. Endpoints are scope descriptors, not discovered socket bindings.
ROUTE_ENDPOINTS = {
    "uninote-tailnet-8001": "tailscale-tcp-forward-8001",
    "tabby-direct-loopback-8000": "tabby-loopback-tcp-8000",
    "hermes-dashboard": "dashboard-upstream-to-tabby",
    "hermes-hub": "hub-upstream-to-tabby",
    "laya-manager-and-direct-restart": "laya-systemd-direct-control",
    "other-local-or-tailscale-tabby-clients": "other-local-tailnet-tabby",
}
if frozenset(ROUTE_ENDPOINTS) != REQUIRED_ROUTES:
    raise RuntimeError("published ingress route inventory drift; fail closed")

ROUTE_COUNTERS = frozenset({
    "http_inflight", "sse_streams", "queued_http", "queued_jobs",
    "upstream_inflight", "partial_http", "unparsed_bytes",
    "keepalive_connections", "open_connections", "unknown_connections",
    "new_admissions_since_fence", "unfenced_restarts",
})
CONTEXT_FIELDS = frozenset({
    "protocol", "kind", "boot_id", "manager_instance_id", "lease_id",
    "epoch", "fencing_token", "nonce", "observed_at_unix",
    "observed_at_monotonic",
})
MANIFEST_FIELDS = CONTEXT_FIELDS | {
    "manifest_epoch", "inventory_revision", "routes",
    "unknown_routes", "unlisted_listeners", "unfenced_admission_paths",
}
ROUTE_FIELDS = CONTEXT_FIELDS | {
    "manifest_epoch", "manifest_sha256", "inventory_revision",
    "route_id", "endpoint", "listener_token", "listener_pid",
    "listener_start_ticks", "listener_generation", "drain_fence_epoch",
    "counters",
}
GPU_FIELDS = CONTEXT_FIELDS | {
    "manifest_epoch", "manifest_sha256", "cards",
    "unknown_devices", "unmapped_driver_contexts", "unreadable_proc_pids",
}
MANIFEST_ROUTE_FIELDS = frozenset({
    "route_id", "endpoint", "listener_token", "listener_pid",
    "listener_start_ticks", "listener_generation",
})
GPU_CARD_FIELDS = frozenset({
    "uuid", "pci", "pid", "start_ticks", "boot_id", "cgroup",
    "driver_contexts", "unknown_contexts",
})


class IngressDenied(RuntimeError):
    """Fail closed; never echo an untrusted proof or signer material."""


def _uint(value: object, minimum: int = 0, maximum: int = MAX_SEQUENCE) -> bool:
    return type(value) is int and minimum <= value <= maximum


def _hexid(value: object) -> bool:
    return type(value) is str and HEX32.fullmatch(value) is not None


@dataclass(frozen=True)
class CardPin:
    uuid: str
    pci: str
    pid: int
    start_ticks: int


@dataclass(frozen=True)
class Context:
    boot_id: str
    manager_instance_id: str
    lease_id: str
    epoch: int
    fencing_token: int
    nonce: str
    manifest_epoch: int
    gpu_pins: tuple[CardPin, CardPin]


@dataclass(frozen=True)
class ConsumptionState:
    """Interface state; only a privileged external store could make it durable."""

    highwater: int = 0
    spent_nonces: tuple[str, ...] = ()


class AtomicConsumptionStore(ABC):
    @abstractmethod
    def atomic_update(
        self,
        mutation: Callable[[ConsumptionState], tuple[ConsumptionState, dict]],
    ) -> dict:
        """Serialize, durably commit and anti-rollback before acknowledging.

        On uncertain commit or crash, raise; the caller must deny and cannot
        assume rollback. No local file/mutex/dict qualifies for production.
        """
        raise NotImplementedError


def _context(ctx: object) -> None:
    if type(ctx) is not Context or not all(_hexid(v) for v in (
        ctx.boot_id, ctx.manager_instance_id, ctx.lease_id, ctx.nonce,
    )):
        raise IngressDenied("invalid manager/boot/lease/nonce identity")
    if not all(_uint(n, 1) for n in (
        ctx.epoch, ctx.fencing_token, ctx.manifest_epoch,
    )) or ctx.epoch != ctx.fencing_token:
        raise IngressDenied("lease epoch/fence invalid or divergent")
    if type(ctx.gpu_pins) is not tuple or len(ctx.gpu_pins) != 2:
        raise IngressDenied("exactly two independently pinned GPUs required")
    for pin in ctx.gpu_pins:
        if (type(pin) is not CardPin or type(pin.uuid) is not str
                or GPU_UUID.fullmatch(pin.uuid) is None
                or type(pin.pci) is not str or PCI.fullmatch(pin.pci) is None
                or not _uint(pin.pid, 1) or not _uint(pin.start_ticks, 1)):
            raise IngressDenied("GPU identity/PID/start-time pin invalid")
    if (ctx.gpu_pins[0].uuid == ctx.gpu_pins[1].uuid
            or ctx.gpu_pins[0].pci == ctx.gpu_pins[1].pci):
        raise IngressDenied("GPU UUID/PCI identities must be distinct")


def _now(clock: Callable[[], float]) -> float:
    try:
        value = clock()
    except Exception:
        raise IngressDenied("trusted clock unavailable") from None
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise IngressDenied("trusted clock invalid")
    return float(value)


def _check_common(claims: dict, ctx: Context, kind: str,
                  now_wall: float, now_mono: float) -> None:
    if (claims.get("protocol") != PROTOCOL or claims.get("kind") != kind
            or any(type(claims.get(k)) is not type(getattr(ctx, k))
                   or claims.get(k) != getattr(ctx, k) for k in (
                "boot_id", "manager_instance_id", "lease_id", "nonce",
                "epoch", "fencing_token",
            ))):
        raise IngressDenied("signed context/boot/lease/nonce mismatch")
    for field, now in (
        ("observed_at_unix", now_wall),
        ("observed_at_monotonic", now_mono),
    ):
        observed = claims.get(field)
        if (type(observed) not in (float, int) or not math.isfinite(observed)
                or not 0 <= now - observed <= MAX_AGE_SECONDS):
            raise IngressDenied("signed observation stale or from future")


def _verify(envelope: object, role: str, kind: str, fields: frozenset[str],
            ctx: Context, verifier: ExternalSignatureVerifier,
            now_wall: float, now_mono: float) -> dict:
    if (type(envelope) is not dict or set(envelope) != {"claims", "signature"}
            or type(envelope.get("claims")) is not dict
            or type(envelope.get("signature")) is not str
            or SIG.fullmatch(envelope["signature"]) is None):
        raise IngressDenied("signed evidence envelope invalid")
    # Validate the exact JSON snapshot that was signed, not mutable nested
    # dictionaries a caller could change between the signature and audit.
    try:
        message = canonical_message(role, envelope["claims"])
        claims = json.loads(message.rsplit(b"\x00", 1)[1].decode("utf-8"))
        if type(claims) is not dict or set(claims) != fields:
            raise IngressDenied("signed evidence schema incomplete or extended")
        _check_common(claims, ctx, kind, now_wall, now_mono)
        result = verifier.verify(
            role, message, bytes.fromhex(envelope["signature"]),
        )
    except IngressDenied:
        raise
    except Exception:
        raise IngressDenied("independent role-pinned signature rejected") from None
    if result is not None:
        raise IngressDenied("verifier trust boolean is not an attestation")
    return claims


def _validate_manifest(claims: dict, ctx: Context) -> dict[str, dict]:
    if (not _uint(claims["manifest_epoch"], 1)
            or claims["manifest_epoch"] != ctx.manifest_epoch
            or not _uint(claims["inventory_revision"], 1)
            or not all(_uint(claims[k], 0, 0) for k in (
                "unknown_routes", "unlisted_listeners", "unfenced_admission_paths",
            ))):
        raise IngressDenied("route inventory incomplete or stale")
    records = claims["routes"]
    if type(records) is not list or len(records) != len(REQUIRED_ROUTES):
        raise IngressDenied("all physical ingress/control surfaces required")
    found: dict[str, dict] = {}
    tokens: set[str] = set()
    for route in records:
        if (type(route) is not dict or set(route) != MANIFEST_ROUTE_FIELDS
                or type(route.get("route_id")) is not str
                or route["route_id"] not in ROUTE_ENDPOINTS
                or route["route_id"] in found
                or route["endpoint"] != ROUTE_ENDPOINTS[route["route_id"]]
                or not _hexid(route["listener_token"])
                or route["listener_token"] in tokens
                or not _uint(route["listener_pid"], 1)
                or not _uint(route["listener_start_ticks"], 1)
                or not _uint(route["listener_generation"], 1)):
            raise IngressDenied("unknown, duplicate or unpinned ingress route")
        found[route["route_id"]] = route
        tokens.add(route["listener_token"])
    if set(found) != REQUIRED_ROUTES:
        raise IngressDenied("missing physical ingress route")
    return found


def _validate_routes(envelopes: object, manifest: dict[str, dict],
                     manifest_claims: dict, digest: str, ctx: Context,
                     verifier: ExternalSignatureVerifier, wall: float,
                     mono: float) -> tuple[float, float, float]:
    if type(envelopes) is not list or len(envelopes) != len(REQUIRED_ROUTES):
        raise IngressDenied("all route proof envelopes required")
    observed: set[str] = set()
    latest_mono = manifest_claims["observed_at_monotonic"]
    earliest_wall = manifest_claims["observed_at_unix"]
    earliest_mono = manifest_claims["observed_at_monotonic"]
    for envelope in envelopes:
        claims = _verify(envelope, "ingress", "route-drained",
                         ROUTE_FIELDS, ctx, verifier, wall, mono)
        route_id = claims["route_id"]
        if (type(route_id) is not str or route_id not in manifest
                or route_id in observed):
            raise IngressDenied("unknown or duplicated signed route evidence")
        record = manifest[route_id]
        for field in MANIFEST_ROUTE_FIELDS - {"route_id"}:
            if claims[field] != record[field]:
                raise IngressDenied("listener restarted or route binding drifted")
        if (claims["manifest_sha256"] != digest
                or not _uint(claims["manifest_epoch"], 1)
                or claims["manifest_epoch"] != ctx.manifest_epoch
                or not _uint(claims["inventory_revision"], 1)
                or claims["inventory_revision"] != manifest_claims["inventory_revision"]
                or not _uint(claims["drain_fence_epoch"], 1)
                or claims["drain_fence_epoch"] != ctx.epoch):
            raise IngressDenied("route proof from wrong inventory or fence")
        counters = claims["counters"]
        if (type(counters) is not dict or set(counters) != ROUTE_COUNTERS
                or any(not _uint(value, 0, 0) for value in counters.values())):
            raise IngressDenied("route has HTTP/SSE/queue/unknown/inflight traffic")
        observed_at = claims["observed_at_monotonic"]
        if not 0 <= observed_at - manifest_claims["observed_at_monotonic"] <= MAX_SNAPSHOT_SKEW:
            raise IngressDenied("route observation predates fence or is skewed")
        latest_mono = max(latest_mono, observed_at)
        earliest_wall = min(earliest_wall, claims["observed_at_unix"])
        earliest_mono = min(earliest_mono, observed_at)
        observed.add(route_id)
    if observed != REQUIRED_ROUTES:
        raise IngressDenied("one or more signed ingress routes absent")
    return latest_mono, earliest_wall, earliest_mono


def _validate_gpu(claims: dict, ctx: Context, digest: str,
                  latest_route_mono: float) -> None:
    if (claims["manifest_sha256"] != digest
            or not _uint(claims["manifest_epoch"], 1)
            or claims["manifest_epoch"] != ctx.manifest_epoch
            or not all(_uint(claims[field], 0, 0) for field in (
                "unknown_devices", "unmapped_driver_contexts",
                "unreadable_proc_pids",
            ))):
        raise IngressDenied("GPU unknown device/context or wrong manifest")
    if not 0 <= claims["observed_at_monotonic"] - latest_route_mono <= MAX_SNAPSHOT_SKEW:
        raise IngressDenied("GPU observation predates drain or is skewed")
    cards = claims["cards"]
    if type(cards) is not list or len(cards) != 2:
        raise IngressDenied("both pinned GPU owners required")
    seen: set[str] = set()
    pins = {p.uuid: p for p in ctx.gpu_pins}
    for card in cards:
        if type(card) is not dict or set(card) != GPU_CARD_FIELDS:
            raise IngressDenied("GPU owner record malformed")
        uuid = card["uuid"]
        if type(uuid) is not str or uuid not in pins or uuid in seen:
            raise IngressDenied("unrecognized or repeated GPU UUID")
        pin = pins[uuid]
        if (card["pci"] != pin.pci or card["pid"] != pin.pid
                or type(card["pid"]) is not int
                or card["start_ticks"] != pin.start_ticks
                or type(card["start_ticks"]) is not int
                or card["boot_id"] != ctx.boot_id
                or card["cgroup"] != "/system.slice/hermes-tabby.service"
                or not _uint(card["driver_contexts"], 1, 1)
                or not _uint(card["unknown_contexts"], 0, 0)):
            raise IngressDenied("GPU PID birth/boot/owner/context mismatch")
        seen.add(uuid)
    if seen != set(pins):
        raise IngressDenied("two pinned GPU UUIDs not fully observed")


def _validate_store(state: object) -> None:
    if (type(state) is not ConsumptionState
            or not _uint(state.highwater)
            or type(state.spent_nonces) is not tuple
            or len(state.spent_nonces) > MAX_SPENT
            or state.highwater != len(state.spent_nonces)
            or any(not _hexid(n) for n in state.spent_nonces)
            or len(set(state.spent_nonces)) != len(state.spent_nonces)):
        raise IngressDenied("external consumption ledger missing or corrupt")


def assess(
    package: object, *, context: Context,
    verifier: ExternalSignatureVerifier | None,
    store: AtomicConsumptionStore | None,
    wall_clock: Callable[[], float],
    monotonic_clock: Callable[[], float],
) -> dict:
    """Validate signed synthetic evidence and atomically consume its nonce.

    Always yields a BLOCKED production result. Requires external abstract
    interfaces to be explicitly injected; this module provides no issuers.
    Repeated/crashed consumption is denied if the caller's store truly retains
    its atomic high-water. An untrusted mock cannot prove that persistence.
    """
    _context(context)
    if (not isinstance(verifier, ExternalSignatureVerifier)
            or not isinstance(store, AtomicConsumptionStore)):
        raise IngressDenied("privileged verifier or atomic ledger unavailable")
    if type(package) is not dict or set(package) != {"manifest", "routes", "gpu"}:
        raise IngressDenied("physical attestation package incomplete")
    wall, mono = _now(wall_clock), _now(monotonic_clock)
    manifest_claims = _verify(
        package["manifest"], "authority", "manifest", MANIFEST_FIELDS,
        context, verifier, wall, mono,
    )
    manifest = _validate_manifest(manifest_claims, context)
    digest = hashlib.sha256(
        canonical_message("authority", manifest_claims),
    ).hexdigest()
    latest, oldest_wall, oldest_mono = _validate_routes(
        package["routes"], manifest, manifest_claims, digest,
        context, verifier, wall, mono,
    )
    gpu_claims = _verify(
        package["gpu"], "gpu-owner", "gpu-inventory", GPU_FIELDS,
        context, verifier, wall, mono,
    )
    _validate_gpu(gpu_claims, context, digest, latest)
    oldest_wall = min(oldest_wall, gpu_claims["observed_at_unix"])
    oldest_mono = min(oldest_mono, gpu_claims["observed_at_monotonic"])
    result = {
        "status": "BLOCKED",
        "synthetic_candidate": "SYNTHETIC_VALIDATED",
        "production_admissible": False,
        "manager_lease_proof": None,
        "reason": "no privileged physical ingress/GPU issuer or antirollback anchor",
    }

    def mutation(state: ConsumptionState) -> tuple[ConsumptionState, dict]:
        _validate_store(state)
        # Strict consecutive epochs, including after a new process or reboot.
        # Never rollover, prune spent nonces or permit old-epoch replay.
        if (state.highwater == MAX_SEQUENCE
                or context.epoch != state.highwater + 1
                or context.nonce in state.spent_nonces
                or len(state.spent_nonces) >= MAX_SPENT):
            raise IngressDenied("epoch/nonce replay, rollover or ledger exhausted")
        # A proof may expire during serialization; do not re-read mutable
        # caller claims, use the timestamps captured after signed checks.
        new_wall, new_mono = _now(wall_clock), _now(monotonic_clock)
        if (not 0 <= new_wall - oldest_wall <= MAX_AGE_SECONDS
                or not 0 <= new_mono - oldest_mono <= MAX_AGE_SECONDS
                or new_wall < wall or new_mono < mono):
            raise IngressDenied("proof expired or clock rolled back during consume")
        return ConsumptionState(
            highwater=context.epoch,
            spent_nonces=state.spent_nonces + (context.nonce,),
        ), result

    try:
        acknowledgement = store.atomic_update(mutation)
    except IngressDenied:
        raise
    except Exception:
        # The external transaction may have committed before losing its ACK.
        # A retry cannot presume rollback; the nonce then remains spent.
        raise IngressDenied("atomic proof consumption unavailable or ambiguous") from None
    if type(acknowledgement) is not dict or acknowledgement != result:
        raise IngressDenied("untrusted atomic ledger acknowledgement")
    return result.copy()
