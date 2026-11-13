# HERMES-27 — feasibility of complete ingress fencing and two-GPU ownership proof

**Audit date:** 2026-11-13 (read-only socket snapshot at 22:43:11 UTC).
**Decision: PRODUCTION NO-GO.** A software implementation is feasible only
after consolidating or independently fencing **every** Tabby entry path,
removing manager-external restart bypasses, and introducing an independently
trusted two-GPU/PID owner attestor with durable fencing. Neither a complete
ingress drain nor a trustworthy owner proof exists on the observed server.

**Scope and noninterference:** SSH read-only source inspection, limited
systemctl show/list (no mutations), ss -lntp, /proc PID metadata, unit
directive classification, direct filesystem existence checks. No HTTP
requests, no credentials printed, no secrets read, no port opening, no
sudo/systemctl changes, no nvidia-smi/NVML queries, no CUDA/GPU workloads,
no modifications under /opt/hermes. The worker-4 GPU reservation was
preserved. Routes inferred from source are not asserted to have been called.

The installed, unpatched source is
/opt/hermes/gpu-manager/manager.py, SHA256:

    15744e3b769c0f6cd5ed26c2b85e3f282871f4a24681abd324c8cbc329a73eef

The existing ops manager_lease_v1_proposal.patch and the proposed contract
are **not deployed**. Searching the installed manager.py found no
watchdog_lease_contract_version, maintenance_lease or lease-v1 endpoint.

## 1. Actual listener and owner snapshot

The following are socket binds, **not verified external reachability**.
Firewall, Tailscale Serve/Funnel and upstream routing policies were not
read or tested. PIDs are transient snapshots, not trustworthy process
identity or GPU ownership evidence.

| Observed listener | PID / process and service identity | What is actually established |
| --- | --- | --- |
| 127.0.0.1:8000 | python PID 1122655; cgroup hermes-tabby.service | Local TabbyAPI OpenAI-compatible endpoint; direct local connections possible |
| Tailscale 100.94.223.14:8001 | python3 PID 3929; user unit uninote-llama-proxy.service | **Confirmed raw TCP forward to 127.0.0.1:8000**, not just a hypothetical gateway |
| 0.0.0.0:8643 | python PID 1122148; hermes-gpu-manager.service | Manager API is bound to all IPv4 interfaces; its protected routes require app/manager Bearer per source |
| 0.0.0.0:8645 | python3 PID 1122111; hermes-laya-bridge.service | Independent Laya bridge API and external manager/Tabby restart fallback |
| 0.0.0.0:9119 | hermes PID 1122656; hermes-dashboard.service | Dashboard process listening on all IPv4 interfaces; its complete Tabby routing is **unverified** |
| 0.0.0.0:8642, :8644 and [::]:8644 | hermes PID 3748492; user hermes-hub.service | Independent Hub listeners; providers/configured upstream may be dynamic, coverage **unverified** |
| 0.0.0.0:8188 | python PID 1398122; hermes-comfyui.service | Separate media service/socket; potential competing GPU allocation |
| 127.0.0.1:8650 plus Tailscale-interface :8650 | uvicorn PID 1122268 for loopback; Tailscale socket owner not shown | Hermes Studio and possible tailnet forward; exact upstream/route policy **unverified** |
| 0.0.0.0:8030 | python PID 1122267; hermes-studigram.service | Additional exposed app, not shown to forward to Tabby |
| 0.0.0.0:9377 and [::]:9377 | No PID resolved from unprivileged ss | Unknown bound service; not attributed to Tabby |

One particularly important contradiction to a “loopback only = private”
assumption: Tabby is local-only, but an active TCP passthrough on Tailscale
reaches it. No network exposure has been added by this audit.

## 2. Proven direct Tabby ingress bypass

The inspected file
/home/matteo/.local/share/uninote-llama-proxy/proxy.py
contains literal LH=100.94.223.14, LP=8001, TH=127.0.0.1, TP=8000
at lines 3–6. Its handle function at lines 23–30 opens the local target
and pipes raw bytes in **both directions**. Lines 32–37 start the server.
There is no observed HTTP request counter, per-request drain gate,
authenticated maintenance epoch or proof of zero in-flight SSE/stream
requests in this tiny proxy.

The unit uninote-llama-proxy.service is active with Restart=on-failure and
RestartUSec=5s. Calling it a “llama” proxy does not change its actual
observed target: current source routes to Tabby port 8000. Consequently,
blocking only dashboard/Hub requests cannot meet “all-tabby-clients”.

Other processes on the server may reach 127.0.0.1:8000 directly. ss -lntp
does not enumerate all present/future clients or count in-flight logical
HTTP requests. Keep-alive connections, streaming SSE and client retries
further defeat a simple TCP LISTEN/ESTABLISHED count as proof of drain.
No full-client inventory or enforced sole ingress controller was observed.

## 3. Actual manager and systemd restart/ownership paths

The installed manager.py was inspected without executing its APIs.
Exact static call-site evidence:

| Source and line(s) | Trigger and effect | Lease relevance |
| --- | --- | --- |
| manager.py:2979–3024 | llm_watchdog probes approximately every 30 seconds; force window causes immediate restore, otherwise four consecutive failed backend checks precede restore | Can restart Tabby when no manager-side lease fence is installed |
| manager.py:2198–2223 then 2156–2181 | restore_llm_with_retries -> transition_to_llm -> systemctl restart of configured LLM service, after cleanup/GPU checks | Shared mutation point; races with an independent HERMES allocation |
| manager.py:3052–3171, 3174–3233 | _drive AUTO/ERROR/forced transitions and media-queue completion/failure repeatedly invoke restore_llm_with_retries | Watchdog-only patch would miss these |
| manager.py:1758–1813, 1998, 2959 | Supervisor action restore_llm can invoke the same restore function from the worker | Separate decision logic, still inside manager process |
| manager.py:3236–3344 and 3096 | reconcile_boot may restore Tabby on boot/restart if not loaded | Crash/reboot fence must be durable before this path runs |
| manager.py:4593–4641 | Dead Character ID training reaper calls systemctl start hermes-tabby.service at line 4639 after orphan detection | In-manager direct start bypass of llm_watchdog |
| manager.py:3701–3735, 3804–3826 | Mode endpoints and /force can request switching/restore; /system/reboot route at 3887–3913 | Need authenticated admission and mode fencing |
| manager.py:1417–1458 | cleanup_stray_cuda may issue sudo kill -9 for matching unowned CUDA processes | Incorrect PID/context evidence could destroy HERMES; keep disabled while leased |
| manager.py:1561 and 3918–4040 | Job creation, normal and smart submissions use separate _submit_lock | Race unless serialized with lease and rechecked at actual job insertion |

Source inspection confirms GET /status, POST /mode/*, /force, /jobs/*,
/supervisor/ask and /system/reboot routes in the installed manager. These
are **source-declared** paths, not HTTP availability/permission tests.
Existing manager.py:848–881 only collects GPU index/name/memory/utilization
and compute-process PID/used-memory from nvidia-smi. It does **not**
constitute a complete two-GPU, graphics/MPS and PID-start-time owner proof.

**Critical independent Laya bypass:** /opt/hermes/gpu-manager/laya_bridge.py
line 487–504 (restart_manager) calls sudo -n systemctl restart of the
manager service after manager unavailability. The force_mode path at
548–586 retries the manager after HTTP connection failure/5xx; if the
requested mode is LLM and the manager remains unavailable, lines 577–585
attempt sudo -n systemctl restart of hermes-tabby.service **directly**.
The bridge is another process/unit. An HTTP 423 response from a healthy
fenced manager returns a rejected 4xx without initiating this recovery,
but outage/5xx/timeouts still permit the independent restart attempt.
Whether sudo permission ultimately allows the restart was **not tested**.

The bridge exposes source-level POST /laya and /laya/jobs routes
(laya_bridge.py:860–877) and has its own service listener 0.0.0.0:8645.
It must participate in the same durable fencing authority; simply changing
manager.py cannot remove this bypass.

### Systemd observations (read-only show and unit parsing)

| Unit (active unless marked) | Restart policy | Relevant relationships |
| --- | --- | --- |
| hermes-gpu-manager.service | on-failure, 5 s | User=matteo; Wants=network-online.target; no observed PartOf/BindsTo Tabby |
| hermes-tabby.service | on-failure, 5 s | User=matteo; Wants fan/network units; independent service restart policy; an ExecStartPre and shm-cleanup drop-in exist |
| hermes-dashboard.service | always, 10 s | User=matteo; After=hermes-tabby.service is **ordering only**, not Requires; restarting gateway can reopen ingress |
| hermes-laya-bridge.service | on-failure, 5 s | User=matteo; independent restart fallback described above |
| hermes-comfyui.service | on-failure, 5 s | User=matteo; independent media GPU process, active listener 8188 |
| user hermes-hub.service | on-failure, 5 s | Active user unit; multiple configuration drop-ins, not read for credentials |
| user uninote-llama-proxy.service | on-failure, 5 s | Active user unit; transparent Tailscale-to-Tabby TCP proxy |
| hermes-llama.service | on-failure, 10 s; **inactive** | Requires tailscaled, not active at socket snapshot |
| hermes-kokoro-tts.service | on-failure, 5 s; **inactive** | Potential separate GPU consumer when started |

The Tabby unit is not automatically restarted solely because an operator
uses systemctl stop; Restart=on-failure is instead a safety concern for
unexpected exits and independent start/restart triggers. The dashboard's
After dependency does not control Tabby ownership. Inspection found no
systemd *.service or *.timer literal reference to hermes-tabby.service
other than the dashboard unit in the inspected system/user service files;
this **does not** rule out variables, shell scripts, sudo rules or remote
callers. In particular, the literal Laya fallback was observed in Python.

User timers hermes-hub-linux-update.timer and
hermes-hub-shadow-sync.timer were active in the snapshot. They can affect
gateway state but were **not proved** to restart Tabby. Their executable
paths/configuration require a separate controlled audit before treating
the Hub as a trusted ingress choke point.

All inspected systemd GPU/manager/Tabby/dashboard/Laya units run under
User=matteo, Group=matteo. This shared UID weakens the trust boundary of
a manager-owned evidence file, checker and fencing journal if any of those
are writable by that UID. No sudoers or secret-bearing unit environment
files were printed or evaluated.

## 4. Checkers, installed lease and trusted fencing: observed ABSENT

Read-only existence probes reported:

- /usr/local/libexec/hermes-ingress-drain: absent
- /usr/local/libexec/hermes-ingress-drain-status: absent
- /usr/local/libexec/hermes-ingress-restore: absent
- /usr/local/libexec/hermes-manager-lease-acquire: absent
- /usr/local/libexec/hermes-manager-lease-renew: absent
- /usr/local/libexec/hermes-manager-lease-release: absent
- /var/lib/hermes-gpu-manager: directory present
- /var/lib/hermes-gpu-manager/lease-v1.json: absent

The presence of a directory does not prove protected persistence, correct
ownership or a deployed manager lease. This is why the proposed
manager_lease_runtime.py and manager_lease_core.py have mock-only trust
contracts and why current production_deploy.py must fail before mutation.

The generic existing ingress proof fields (scope, blocked, inflight,
timestamp) are *shape checks*. They cannot tell whether a lying or stale
local checker forgot the independent Tailscale TCP proxy. Similarly,
gpu_owner=none with empty PID arrays cannot prove absence of real GPU
contexts without an authenticated, privileged hardware/PID source.

## 5. Is a complete ingress drain safely implementable?

**In principle yes, on a restructured gateway; not safely on the
currently observed topology.** A proof acceptable for production needs
ALL the following, within one coordinated maintenance/lease authority:

1. A definitive service/client/route inventory including direct localhost
   Tabby callers, UniNote's raw TCP proxy, dashboard, Hermes Hub, Studio,
   Tailscale, reconnecting clients and local automation. Unknown path =
   refuse lease; never silently assume a listener is unrelated.
2. One enforceable entrance to Tabby (ideally a private Unix socket or
   isolated network namespace and root-owned ingress broker). The broker
   must prevent direct 127.0.0.1:8000 bypass, including other processes
   of the same user; simply pausing dashboard is insufficient.
3. Under a single admission lock/epoch, reject new requests and retries
   *before* starting drain. Count request execution, SSE streams, WebSockets,
   HTTP keepalive/pipelining and in-flight backend work, not just TCP
   connections. Wait for a bounded verified zero, or abort without stopping
   Tabby. Unresponsive clients/streams are NOT proof of zero.
4. Make UniNote and any other forwarding service obey the fence and prevent
   automatic restarts from reopening it. Manager, Laya and other callers
   must be prevented from bypassing the broker; a 423-only API change is
   insufficient when Laya can restart the backend on manager failure.
5. Sign or attest a short-lived (<=10 s) proof bound to one durable fencing
   epoch and exact route set; reevaluate while HERMES runs and immediately
   fail closed when the gateway restarts, proof expires, or routing changes.
   Merely returning JSON with blocked=true is not independent evidence.
6. Hold ingress closed during all rollback stages. Reopen only after HERMES
   is verified absent, Tabby exclusively owns the GPUs, lease release is
   attested and the original service answers normally.

Without these conditions, a client can start a Tabby request on loopback or
the transparent tailnet proxy while HERMES is loading the same GPUs.
**Today: complete ingress drain UNATTESTABLE / NO-GO.**

## 6. Is a trusted two-GPU PID owner attestor safely implementable?

**Architecturally possible with a separately privileged helper; not
currently proven or installable from the existing weak GPU snapshot.**
Required properties:

1. Root-owned, non-replaceable local attestor or narrowly privileged broker
   separate from the User=matteo GPU Manager. Verify executable owner,
   all parent directory permissions, absence of symlinks, trusted IPC peer
   identity and exact command path. Do not trust client-supplied JSON.
   Prefer a credential-gated Unix socket with SO_PEERCRED over passing
   privileged shell commands.
2. Enumerate both physical GPUs by stable UUID/PCI identifier and use a
   driver-level GPU process/context query (NVML and, where necessary,
   driver/device-fd evidence) that includes compute **and** graphics/MPS/
   containers. Index 0/1 alone and memory-utilization percentages cannot
   prove exclusivity. No driver/GPU enumeration was run during this audit.
3. Resolve every owner PID to immutable birth/start-time evidence,
   process tree, systemd cgroup/unit, namespace and GPU device/context;
   defend against PID reuse, process exit/fork races, stale snapshots and
   permissions that hide processes. Unknown, inaccessible or conflicting
   owner => **deny release and retain ingress fence**.
4. Before a post-expiry Tabby restart prove every HERMES child and CUDA
   context gone, no unknown GPU users and both cards free. After Tabby
   starts prove only the expected Tabby service cgroup/children own both
   cards, while Hermes remains absent. A single nvidia-smi query is
   weaker than this contract.
5. Bind the attestation to a fresh (<10 s), monotonic fencing/manager
   epoch and repeat immediately before a dangerous transition. Keep
   owner checks and lease decision serialized with manager/service
   admission to reduce TOCTOU. If the checker cannot acquire privileges
   or prove any necessary field, **fail closed**.
6. Persist the fence across process restart/host reboot in a journal that
   an unrelated same-UID process cannot delete. A deleted journal currently
   looks like first launch to the prototype. Without independent boot-time
   owner verification and external authoritative fencing, NO-GO.

The installed manager's nvidia-smi compute PID/VRAM snapshots plus its
command-line/cgroup-based orphan cleanup (manager.py:848–906,
1417–1458) are **not** a trustworthy grant/release attestor; they omit
immutable PID birth identity, graphics/MPS accounting and provenance
for both physical devices.

## 7. Conclusive release gates and bounded next experiments

| Mandatory gate | Observed evidence | Current result |
| --- | --- | --- |
| Versioned manager lease installed and independently authenticated | Installed manager.py unchanged, no lease-v1 source/endpoint | **BLOCKED** |
| No manager-external Tabby restart | Laya bridge direct sudo systemctl restart fallback; independent systemd policies | **BLOCKED** |
| All-client ingress admission and streaming drain | Transparent tailnet TCP proxy + direct localhost; no trustworthy route/stream counter | **BLOCKED** |
| Complete drain hook code and trusted provenance | Example executables absent | **BLOCKED** |
| Both-GPU, PID-birth-time and context-owner attestation | Current GPU status is utilization plus compute PID query only; no privileged helper | **BLOCKED** |
| Durable, out-of-process fence after crash/journal loss | Proposed journal absent; manager restarts on failure | **BLOCKED** |
| Firewall / tailnet exposure and reverse-proxy path coverage | Binds enumerated, reachability/ACLs not audited | **UNKNOWN — DENY** |
| Offline JSON proof rejection and basic shape rules | 21 mock-only tests in new test_ingress_owner_contract.py | **PASS, not physical acceptance** |

Next engineering tasks must be isolated and authorized independently:
first map every source of traffic to Tabby and design one enforced ingress
broker; separately identify and interlock Laya's sudo fallback and any
external systemd/start permission; then build/test a root-owned
both-GPU/PID attestation broker using synthetic fixtures before requesting
exclusive GPU/device validation. Never make production_deploy.py continue
by substituting mock proof providers, temporary allowlists or a successful
HTTP health response for hardware/ingress ownership.

**Verdict:** trusted complete ingress and two-GPU owner checks are
potentially implementable, but the *current* architecture has confirmed
independent bypasses and no trustworthy proof origin. **NO-GO**, even
though the prior manager-side source patch and existing ops test suite
pass offline. No live cutover or GPU validation was attempted.
