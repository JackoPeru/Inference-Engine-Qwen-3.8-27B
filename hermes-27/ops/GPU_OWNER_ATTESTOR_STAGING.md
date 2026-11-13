# GPU owner attestor v0 — strictly isolated staging

**Status: BLOCKED — not a trusted manager lease provider.**
Created 2026-11-13 as a CPU/offline-only follow-up to
INGRESS_GPU_OWNER_FEASIBILITY_2026-11-13.md.
No installed files, systemd units, service code, GPU runtime, credentials,
NVML hardware state, HTTP endpoints or actual server GPUs were changed or
queried in this task. Worker-4 retained exclusive access to the two GPUs.

Only the following **new** paths are owned by this deliverable:

- ops/gpu_owner_attestor_staging.py
- ops/tests/test_gpu_owner_attestor_staging.py
- ops/GPU_OWNER_ATTESTOR_STAGING.md

## What the implementation actually does

The single Python module has two deliberately separate parts:

1. A real **read-only collection class** (RootReadOnlyCollector) which may
   be examined in staging or called with injected sources. On a future
   authorized Linux setup it would require effective UID 0, query the
   installed pynvml binding without importing CUDA or running inference,
   and read kernel /proc metadata. It never calls sudo, subprocess,
   systemctl, shell commands, the network, CUDA context creation or GPU
   memory allocation. **This path was NOT executed against a real GPU
   in this work.** There is no CLI argument that invokes it.
2. A strict **offline snapshot comparator** (evaluate) for two injected
   samples. This validates evidence format, identities and detected races.
   Regardless of inputs, including two fully consistent synthetic snapshots,
   it returns status BLOCKED with no manager-compatible successful
   gpu_owner, gpu_free or scope claim. ManagerLeaseV1.require_gpu_safe
   therefore rejects every output.

RootReadOnlyCollector.stage_probe coordinates two consecutive read-only
observations through the same evaluator and converts collector failures
into a BLOCKED verdict. This Python method is not exposed through the CLI.

The CLI is intentionally limited to a harmless status print:

    python3 ops/gpu_owner_attestor_staging.py --staging-status

It prints JSON with manager_lease_admissible=false and returns exit status
**2**. Do **NOT** configure this script as HERMES_MANAGER_GPU_CHECK.
It is a review prototype, not a deployable fail-open shortcut.

## Structural evidence and required admission checks

The collector pins exactly **two** GPUs to caller-supplied stable physical
identities: GPU UUID, PCI bus/device/function address and character device
major/minor. It reads NVML index-to-UUID-to-PCI-to-device mapping, requires
an accessible character device, and rejects unknown MIG mode or missing
compute, graphics or MPS process enumeration interfaces. A consumer-driver
NVML library with no MPS API or unsupported MIG query is simply **BLOCKED**,
not silently promoted to complete coverage.

For /proc, every listed numeric PID is read, including per-PID starttime
(kernel stat field 22, not PID alone), effective cgroup v2 path, UID and
each accessible file descriptor's metadata. The implementation:

- parses process names containing parentheses correctly;
- denies missing /proc entries, denied permissions, inaccessible file
  descriptors, changed starttime or a process that exits during scanning;
- compares file descriptor character-device major/minor with both pinned
  GPU device identities, and records ambiguous NVIDIA global, DRM, VFIO,
  UVM or nvidia-caps descriptors as **unknown** (never guessed);
- compares GPU owner PIDs visible from NVML with the PIDs carrying pinned
  per-card file descriptors and denies any difference;
- requires an independently trusted PID+birthtime manifest and exact
  Tabby/HERMES service cgroup for all recognized GPU owners; no cmdline
  name matching is treated as identity;
- rejects unrecognized owners, missing Tabby ownership on either card after
  restoration, still-running HERMES processes, mismatched GPU device
  identity, stale snapshots and changes between two complete observations;
- validates that both snapshots have a stable boot identity and are at
  most 2 seconds apart, and that the latest is no more than 10 seconds old.

Checks apply to phase empty-before-tabby or tabby-restored. For the
pre-restore phase all cards must have no GPU process owners. For the
post-restore phase only specifically attested Tabby owner PIDs must appear
and the two cards must both show Tabby ownership. The owner manifest is
**injectable data for offline verification**, not an independently
authenticated service. This is another reason no release is permitted.

## Why the result must always remain BLOCKED

Even when both snapshots agree perfectly, NVML queries and /proc FD
metadata do not independently certify that **all** GPU contexts on both
physical cards have been exposed. In particular:

- GPU drivers and installed pynvml bindings may omit MPS, graphics,
  transient, privileged or namespace-hidden contexts;
- /dev/nvidiactl, /dev/nvidia-uvm, nvidia-caps, DRM render nodes and
  container/VFIO interfaces cannot be mapped to a card merely from a path;
- no independent hardware/kernel context-enumeration authority is wired
  in; a user-supplied JSON snapshot can be syntactically consistent but
  is not necessarily genuine;
- PID+birthtime+unit cgroup inputs are not sourced from a trusted owner
  supervisor, and the current system's GPU Manager and Tabby run under
  the same UID;
- a new process may acquire the GPU **after** the last read even if two
  snapshots were identical. No admission lock or cross-process fencing
  is installed to prevent that race;
- a restarted manager, missing journal, external Laya direct Tabby restart,
  systemd auto-restarts and direct local/Tailscale ingress remain outside
  this module's authority;
- NVML used-memory percentage or a zero-looking utilization sample is
  **never** accepted as proof of zero owners or free physical GPU memory.

Accordingly, the module always adds the terminal blocker
independent-all-driver-contexts-coverage-unavailable. Its
staging_candidate_owner is **diagnostic only**. A candidate of none or
tabby is NOT a reliable hardware attestation and is not a lease grant.
The prototype does not print the protocol-valid fields used by
manager_lease_core.require_gpu_safe, by design.

## What a real implementable attestor still requires

1. An approved, separately privileged, root-owned kernel/driver attestation
   helper that can prove a complete inventory of compute, graphics, MPS,
   MIG, device-file and hidden/container GPU contexts for **both stable
   UUIDs**. If the NVIDIA driver cannot provide such a completeness
   guarantee, no success proof can be issued.
2. Privileged, immutable PID-birthtime and systemd cgroup evidence bound
   to a root-owned supervisor admission journal; all descendants,
   containers and process namespaces must be included. The helper,
   manifest and its parent directories must not be replaceable by the
   regular service UID.
3. An atomic ingress/ownership fence that prevents new HERMES, Tabby, Laya,
   Character ID, ComfyUI or external processes from acquiring either card
   between process enumeration and manager lease release. Prevent the
   independent laya_bridge.py sudo restart fallback.
4. A durable external recovery authority spanning manager restart, host
   reboot and journal deletion. Expiry **cannot** authorize Tabby restart
   while HERMES ownership is unknown.
5. Explicitly authorized acceptance tests on the two physical GPUs under
   exclusive maintenance control. Use independent driver ownership checks
   to exercise process births/reuse, partial permissions, MPS/graphics,
   GPU hot-swap, race windows, cancellation and power loss. These were
   NOT run by this staging worker.

## Offline deterministic test contract

    python3 -m unittest discover -s ops/tests -p test_gpu_owner_attestor_staging.py -v
    python3 -m py_compile ops/gpu_owner_attestor_staging.py ops/tests/test_gpu_owner_attestor_staging.py

All tests inject Card, CardSample, ProcSample, Snapshot and OwnerManifest
fixtures. A synthetic collector test uses a **temporary fake /proc tree**
and a fake NVML object, not real GPU hardware. The tests also ensure the
CLI cannot invoke the collector and that even a perfectly shaped response
is rejected by the manager's actual require_gpu_safe JSON gate.

**Validation in this task:** 57/57 new synthetic/mock tests PASS on
local Windows Python 3.13; source and test py_compile PASS. No physical
NVML hardware call, CUDA work, subprocess, systemctl or root escalation.
The complete existing + new offline ops suite also passed **193/193 tests**
(63.589 s). Expected errors in previously existing negative deployment
fixtures do not represent live service changes or failed new tests.

**Final decision: STAGING IMPLEMENTED, PHYSICAL OWNERSHIP ATTESTATION
BLOCKED, PRODUCTION NO-GO.**
