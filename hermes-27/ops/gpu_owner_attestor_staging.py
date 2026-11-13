#!/usr/bin/env python3
"""STAGING ONLY: read-only NVML /proc GPU-owner evidence, never a lease grant.

All successful-looking NVML and proc observations remain BLOCKED: those
interfaces do not independently prove that every GPU context (MPS, graphics,
driver, vanished processes, namespaces) has been discovered. Do not install
this as the manager's GPU_CHECK hook. Tests inject all inputs, never touch GPUs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
from typing import Callable, Mapping


MAX_AGE_S = 10.0
MAX_SCAN_S = 2.0
UUID_RE = re.compile(r"GPU-[a-fA-F0-9-]{16,80}\Z")
PCI_RE = re.compile(r"[a-fA-F0-9]{4,8}:[a-fA-F0-9]{2}:[a-fA-F0-9]{2}\.[0-7]\Z")
KINDS = frozenset(("compute", "graphics", "mps"))
TABBY_CGROUP = "/system.slice/hermes-tabby.service"
HERMES_CGROUP = "/system.slice/hermes-27.service"


class Blocked(RuntimeError):
    """Missing or contradictory physical evidence. No source strings/secrets."""


@dataclass(frozen=True, order=True)
class Card:
    uuid: str
    pci: str
    minor: int
    dev_major: int
    dev_minor: int


@dataclass(frozen=True, order=True)
class GPUProcess:
    pid: int
    kind: str


@dataclass(frozen=True)
class CardSample:
    identity: Card
    owners: tuple[GPUProcess, ...] = ()
    compute_complete: bool = False
    graphics_complete: bool = False
    mps_complete: bool = False
    mig: str = "unknown"


@dataclass(frozen=True)
class ProcSample:
    pid: int
    start_ticks: int
    cgroup: str
    uid: int
    # Actual st_rdev major/minor values, not names from cmdline.
    card_fds: frozenset[tuple[int, int]] = frozenset()
    unknown_gpu_fd: bool = False
    fds_complete: bool = False
    before_ticks: int = 0
    after_ticks: int = 0


@dataclass(frozen=True)
class Snapshot:
    boot_id: str
    unix_s: float
    monotonic_s: float
    cards: tuple[CardSample, ...]
    processes: tuple[ProcSample, ...]
    nvml_complete: bool = False
    proc_complete: bool = False
    root_verified: bool = False


@dataclass(frozen=True)
class OwnerManifest:
    # Must originate from a future privileged owner authority. The caller
    # cannot prove this merely by setting these fields.
    tabby: Mapping[int, int] = field(default_factory=dict)
    hermes: Mapping[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Verdict:
    reasons: tuple[str, ...]
    candidate: str | None = None

    def as_dict(self) -> dict:
        # Deliberately never output scope=all-hermes-gpus-and-pids,
        # gpu_free=true or gpu_owner=tabby accepted by ManagerLeaseV1.
        return {"status": "BLOCKED", "manager_lease_admissible": False,
                "reasons": list(self.reasons), "staging_candidate_owner": self.candidate}


def _uint(value: object, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def parse_start_ticks(raw: str, pid: int) -> int:
    """Correct Linux stat field 22 parsing when comm contains spaces/parentheses."""
    if not isinstance(raw, str):
        raise Blocked("proc-stat-malformed")
    raw = raw.rstrip("\n")
    if "\n" in raw or "\x00" in raw:
        raise Blocked("proc-stat-malformed")
    left, right = raw.find("("), raw.rfind(")")
    if left < 1 or right <= left:
        raise Blocked("proc-stat-malformed")
    try:
        fields = raw[right + 1:].split()
        if int(raw[:left].strip()) != pid or len(fields) < 20:
            raise ValueError("pid or field count")
        ticks = int(fields[19])  # fields[0]=state, fields[19]=starttime
        if ticks < 1:
            raise ValueError("invalid birth")
        return ticks
    except ValueError:
        raise Blocked("proc-stat-malformed") from None


def require_root(*, euid: Callable[[], int] | None = None,
                 platform: str | None = None) -> None:
    target = sys.platform if platform is None else platform
    uid = (os.geteuid() if hasattr(os, "geteuid") else -1) if euid is None else euid()
    if target != "linux" or uid != 0:
        raise Blocked("linux-root-required-no-sudo-fallback")


def _cgroup_v2(text: str) -> str:
    paths = [cols[2] for row in text.splitlines()
             if len(cols := row.split(":", 2)) == 3 and cols[:2] == ["0", ""]]
    if (len(paths) != 1 or not paths[0].startswith("/") or
            "\x00" in paths[0] or ".." in paths[0].split("/")):
        raise Blocked("cgroup-v2-missing-or-untrusted")
    return paths[0]


def evaluate(first: Snapshot, second: Snapshot, *,
             expected: tuple[Card, Card],
             manifest: OwnerManifest, phase: str, now_s: float) -> Verdict:
    """Cross-check injected observations; always fail closed for production.

    Strictly distinguishes structural consistency from independent context
    completeness. A malicious or incomplete source can forge both snapshots;
    this staging module has NO authenticated kernel all-contexts authority.
    """
    errors: list[str] = []

    def deny(msg: str) -> None:
        if msg not in errors:
            errors.append(msg)

    if phase not in ("empty-before-tabby", "tabby-restored"):
        deny("phase-unknown")
    if len(expected) != 2 or expected[0] == expected[1]:
        deny("two-distinct-expected-cards-required")
    for card in expected:
        if (not isinstance(card, Card) or not UUID_RE.fullmatch(card.uuid)
                or not PCI_RE.fullmatch(card.pci)):
            deny("expected-card-identity-not-pinned")
        elif not (_uint(card.minor) and _uint(card.dev_major, 1)
                  and _uint(card.dev_minor) and card.minor == card.dev_minor):
            deny("expected-card-device-invalid")
    if len(expected) == 2 and all(isinstance(c, Card) for c in expected):
        if (expected[0].uuid == expected[1].uuid
                or expected[0].pci == expected[1].pci
                or (expected[0].dev_major, expected[0].dev_minor)
                == (expected[1].dev_major, expected[1].dev_minor)):
            deny("card-identity-duplicate")

    if not isinstance(manifest, OwnerManifest):
        deny("manifest-untrusted")
        manifest = OwnerManifest()
    for mapping in (manifest.tabby, manifest.hermes):
        if not isinstance(mapping, Mapping) or any(
                not _uint(pid, 2) or not _uint(ticks, 1)
                for pid, ticks in mapping.items()):
            deny("manifest-pid-birth-malformed")
    if set(manifest.tabby) & set(manifest.hermes):
        deny("tabby-hermes-pid-collision")

    if not first.boot_id or first.boot_id != second.boot_id:
        deny("boot-identity-changed")
    if (type(now_s) not in (int, float)
            or not 0 <= now_s - second.unix_s <= MAX_AGE_S
            or not 0 <= second.unix_s - first.unix_s <= MAX_SCAN_S
            or not 0 <= second.monotonic_s - first.monotonic_s <= MAX_SCAN_S):
        deny("snapshot-time-invalid-or-expired")

    cardset = set(expected) if all(isinstance(c, Card) for c in expected) else set()
    devmap = {(c.dev_major, c.dev_minor): c for c in cardset}
    for snap in (first, second):
        if not (snap.root_verified is True and snap.nvml_complete is True
                and snap.proc_complete is True):
            deny("root-or-enumeration-not-complete")
        if len(snap.cards) != 2:
            deny("nvml-card-count-not-two")
        if set(c.identity for c in snap.cards) != cardset:
            deny("uuid-pci-minor-mismatch")
        nvml: dict[Card, set[int]] = {}
        for card in snap.cards:
            if not (card.compute_complete is True and card.graphics_complete is True
                    and card.mps_complete is True):
                deny("nvml-compute-graphics-mps-incomplete")
            if card.mig != "disabled":
                deny("mig-unsupported-or-unknown")
            if len(set(card.owners)) != len(card.owners):
                deny("nvml-duplicate-entry")
            for owner in card.owners:
                if not (_uint(owner.pid, 2) and owner.kind in KINDS):
                    deny("nvml-invalid-owner-entry")
            nvml[card.identity] = {p.pid for p in card.owners}
        procmap: dict[int, ProcSample] = {}
        fdmap: dict[Card, set[int]] = {c: set() for c in cardset}
        for proc in snap.processes:
            if not _uint(proc.pid, 1) or proc.pid in procmap:
                deny("proc-pid-duplicate-or-invalid")
            procmap[proc.pid] = proc
            if (not _uint(proc.start_ticks, 1)
                    or proc.before_ticks != proc.start_ticks
                    or proc.after_ticks != proc.start_ticks):
                deny("pid-starttime-changed")
            if (not isinstance(proc.cgroup, str) or not proc.cgroup.startswith("/")
                    or not _uint(proc.uid)):
                deny("proc-uid-cgroup-unreadable")
            if proc.fds_complete is not True:
                deny("proc-fd-enumeration-incomplete")
            if proc.unknown_gpu_fd is True:
                deny("unknown-global-drm-or-gpu-fd")
            for dev in proc.card_fds:
                card = devmap.get(dev)
                if card is None:
                    deny("unmapped-gpu-device-fd")
                else:
                    fdmap[card].add(proc.pid)
        for card in cardset:
            if nvml.get(card, set()) != fdmap.get(card, set()):
                deny("nvml-vs-proc-device-owner-disagreement")
        active = set().union(*nvml.values()) if nvml else set()
        for pid in active:
            if pid not in manifest.tabby and pid not in manifest.hermes:
                deny("unknown-gpu-owner")
            proc = procmap.get(pid)
            if proc is None:
                deny("nvml-pid-missing-from-proc")
                continue
            if pid in manifest.tabby:
                if proc.start_ticks != manifest.tabby[pid] or proc.cgroup != TABBY_CGROUP:
                    deny("tabby-birth-or-cgroup-mismatch")
            elif pid in manifest.hermes:
                if proc.start_ticks != manifest.hermes[pid] or proc.cgroup != HERMES_CGROUP:
                    deny("hermes-birth-or-cgroup-mismatch")
            else:
                deny("unknown-gpu-owner")
        for pid, birth in {**manifest.hermes, **manifest.tabby}.items():
            if pid not in procmap or procmap[pid].start_ticks != birth:
                deny("manifest-pid-gone-or-reused")
        if phase == "empty-before-tabby":
            if active or any(p.card_fds for p in snap.processes):
                deny("gpu-process-still-present")
            if manifest.tabby:
                deny("tabby-owner-manifest-not-empty")
        elif phase == "tabby-restored":
            if manifest.hermes:
                deny("hermes-owner-manifest-not-empty")
            if not manifest.tabby:
                deny("tabby-owner-manifest-missing")
            for card in cardset:
                if not nvml.get(card):
                    deny("tabby-missing-from-gpu")
                elif not nvml[card] <= set(manifest.tabby):
                    deny("non-tabby-gpu-process")

    if first.cards != second.cards or first.processes != second.processes:
        deny("gpu-process-or-pid-changed-between-snapshots")
    # This is an unconditional release blocker, even when snapshots match:
    # NVML+proc can miss kernel contexts, MPS, namespace and transient owners.
    deny("independent-all-driver-contexts-coverage-unavailable")
    candidate = None
    if len(errors) == 1 and phase in ("empty-before-tabby", "tabby-restored"):
        candidate = "none" if phase == "empty-before-tabby" else "tabby"
    return Verdict(tuple(errors), candidate)


def _read_proc(pid: int, root: Path, cards: tuple[Card, Card]) -> ProcSample:
    """Metadata only. Any permission/race or unmappable GPU FD blocks proof."""
    directory = root / str(pid)
    try:
        first = parse_start_ticks((directory / "stat").read_text(), pid)
        cgroup = _cgroup_v2((directory / "cgroup").read_text())
        status_text = (directory / "status").read_text()
        match = re.search(r"(?m)^Uid:\s+(\d+)\s+\d+\s+\d+\s+\d+\s*$", status_text)
        if match is None:
            raise Blocked("proc-uid-unavailable")
        uid = int(match.group(1))
        devices = {(c.dev_major, c.dev_minor) for c in cards}
        found: set[tuple[int, int]] = set()
        unknown = False
        with os.scandir(directory / "fd") as entries:
            for entry in entries:
                if not entry.name.isdecimal():
                    continue
                fd = directory / "fd" / entry.name
                # Never open descriptors or GPU devices. stat follows the
                # existing process FD without creating a new GPU context.
                target = os.readlink(fd)
                metadata = os.stat(fd)
                if target.startswith(("/dev/nvidia", "/dev/dri/", "/dev/vfio/")):
                    if not stat.S_ISCHR(metadata.st_mode):
                        unknown = True
                        continue
                    if target.startswith("/dev/nvidia") and target[11:].isdigit():
                        device = (os.major(metadata.st_rdev), os.minor(metadata.st_rdev))
                        if device not in devices:
                            unknown = True
                        else:
                            found.add(device)
                    else:
                        # nvidiactl, nvidia-uvm, nvidia-caps, DRM and VFIO
                        # can cover multiple cards without per-card evidence.
                        unknown = True
        second = parse_start_ticks((directory / "stat").read_text(), pid)
        if first != second:
            raise Blocked("pid-birth-changed-during-proc-scan")
        return ProcSample(pid, first, cgroup, uid, frozenset(found), unknown,
                          True, first, second)
    except Blocked:
        raise
    except (OSError, UnicodeError, ValueError, TypeError):
        raise Blocked("proc-fd-permission-pid-race-or-incomplete-scan") from None


def _read_all_proc(root: Path, cards: tuple[Card, Card]) -> tuple[ProcSample, ...]:
    try:
        with os.scandir(root) as entries:
            pids = sorted(int(entry.name) for entry in entries if entry.name.isdecimal())
        if not pids:
            raise Blocked("proc-process-list-empty")
        # A process disappearing between the listing and read is a race,
        # never treated as negative evidence of GPU ownership.
        return tuple(_read_proc(pid, root, cards) for pid in pids)
    except Blocked:
        raise
    except (OSError, UnicodeError, ValueError):
        raise Blocked("proc-global-enumeration-unavailable") from None


class RootReadOnlyCollector:
    """Opt-in root/Linux evidence gatherer; does not authorize release.

    Caller injects stable GPU UUID/PCI/character-device identities. NVML is
    dynamically imported only inside collect_once, never on module import
    or during offline tests. A driver missing an MPS query or MIG status
    is not silently accepted. No shell, sudo, network or CUDA allocation.
    """

    def __init__(
        self, expected: tuple[Card, Card], *,
        nvml: object | None = None,
        proc_root: Path = Path("/proc"),
        dev_root: Path = Path("/dev"),
        wall_clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        euid: Callable[[], int] | None = None,
        platform: str | None = None,
    ):
        self.expected = expected
        self.nvml = nvml
        self.proc_root = proc_root
        self.dev_root = dev_root
        self.wall_clock = wall_clock
        self.monotonic = monotonic
        self.euid = euid
        self.platform = platform

    def _read_nvml_cards(self, api: object) -> tuple[CardSample, ...]:
        try:
            if api.nvmlDeviceGetCount() != 2:
                raise Blocked("nvml-card-count-unknown-or-not-two")
            samples = []
            for index in range(2):
                handle = api.nvmlDeviceGetHandleByIndex(index)
                uuid = api.nvmlDeviceGetUUID(handle)
                pci = api.nvmlDeviceGetPciInfo(handle).busId
                uuid = uuid.decode("ascii") if isinstance(uuid, bytes) else str(uuid)
                pci = (pci.decode("ascii") if isinstance(pci, bytes) else str(pci)).lower()
                minor = api.nvmlDeviceGetMinorNumber(handle)
                if not _uint(minor):
                    raise Blocked("nvml-device-minor-not-valid")
                node = self.dev_root / f"nvidia{minor}"
                info = node.stat()
                if not stat.S_ISCHR(info.st_mode):
                    raise Blocked("gpu-node-not-character-device")
                identity = Card(uuid, pci, minor, os.major(info.st_rdev), os.minor(info.st_rdev))
                if identity not in self.expected:
                    raise Blocked("nvml-uuid-pci-or-device-identity-changed")
                mode, _pending = api.nvmlDeviceGetMigMode(handle)
                if mode != 0:
                    raise Blocked("mig-enabled-or-unsupported")
                owners = []
                for kind, method in (
                    ("compute", "nvmlDeviceGetComputeRunningProcesses"),
                    ("graphics", "nvmlDeviceGetGraphicsRunningProcesses"),
                    ("mps", "nvmlDeviceGetMPSComputeRunningProcesses"),
                ):
                    getter = getattr(api, method, None)
                    if not callable(getter):
                        raise Blocked("nvml-compute-graphics-mps-query-unsupported")
                    for process in getter(handle):
                        if not _uint(process.pid, 2):
                            raise Blocked("nvml-owner-pid-invalid")
                        owners.append(GPUProcess(process.pid, kind))
                samples.append(CardSample(identity, tuple(sorted(owners)),
                                          True, True, True, "disabled"))
            return tuple(sorted(samples, key=lambda item: item.identity.uuid))
        except Blocked:
            raise
        except Exception:
            raise Blocked("nvml-hardware-enumeration-incomplete") from None

    def collect_once(self) -> Snapshot:
        require_root(euid=self.euid, platform=self.platform)
        try:
            boot_id = (self.proc_root / "sys/kernel/random/boot_id").read_text().strip()
            if not boot_id:
                raise Blocked("proc-boot-identity-not-readable")
            if self.nvml is not None:
                api = self.nvml
            else:
                import pynvml as api  # type: ignore[import-not-found]
            # NVML is a read-only driver query here. No CUDA context creation.
            api.nvmlInit()
            try:
                cards = self._read_nvml_cards(api)
                proc = _read_all_proc(self.proc_root, self.expected)
            finally:
                api.nvmlShutdown()
            return Snapshot(
                boot_id, self.wall_clock(), self.monotonic(), cards, proc,
                nvml_complete=True, proc_complete=True, root_verified=True,
            )
        except Blocked:
            raise
        except Exception:
            raise Blocked("root-readonly-nvml-or-proc-collection-unavailable") from None

    def stage_probe(self, *, manifest: OwnerManifest, phase: str) -> Verdict:
        """Two sequential read-only collections; report BLOCKED on any fault.

        This method is intentionally library-only (not callable by CLI).
        It cannot authorize the manager even when all injected mocks agree.
        """
        try:
            first = self.collect_once()
            second = self.collect_once()
            return evaluate(first, second, expected=self.expected,
                            manifest=manifest, phase=phase, now_s=self.wall_clock())
        except Blocked as exc:
            return Verdict((str(exc), "independent-all-driver-contexts-coverage-unavailable"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-status", action="store_true",
                        help="Always return BLOCKED; does not inspect hardware")
    parser.parse_args(argv)
    # Deliberately no CLI option to run collection, grant permissions, or
    # return a manager-compatible owner proof. Future independent kernel
    # all-context attestation would require a new, separately reviewed API.
    decision = Verdict(("independent-all-driver-contexts-coverage-unavailable",))
    print(json.dumps(decision.as_dict(), sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
