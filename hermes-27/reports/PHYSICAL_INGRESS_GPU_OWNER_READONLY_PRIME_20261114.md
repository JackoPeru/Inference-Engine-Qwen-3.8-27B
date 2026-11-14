# HERMES-27: physical ingress and GPU-owner read-only recheck — 2026-11-14

**Decision: NO-GO for GPU acquisition, live model traces, and deployment.**
This document is a point-in-time observation, **not** an all-ingress drain certificate, owner grant, or authorization to change any service.

## Evidence collected

Read-only SSH to existing Ubuntu server, observed `2026-11-14T10:36:33Z`. Commands: `date -u`, `nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits`, `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv,noheader,nounits`, `systemctl show hermes-tabby.service hermes-gpu-manager.service -p Id -p ActiveState -p SubState -p ExecMainPID -p ActiveEnterTimestamp -p InactiveEnterTimestamp`, `ss -H -lntp`, `tailscale ip -4`, and read-only `fuser` of the NVIDIA device nodes. No model, CUDA kernel, HTTP request, socket probe, process kill, service action, Tailscale configuration action, or lease write.

- GPU0: 21 MiB used, 0% utilization; GPU1: 26 MiB used, 0% utilization.
- `nvidia-smi --query-compute-apps` returned no compute app rows. `fuser` returned no process IDs to the unprivileged read. Neither is proof that all contexts and hardware ownership are controlled.
- `hermes-tabby.service`: inactive/dead since 2026-11-14 09:07:57 UTC; `hermes-gpu-manager.service`: inactive/dead since 09:07:51 UTC. Their historical ExecMainPID values are **not** current GPU-owner grants.
- The host still has listening sockets bound to all IPv4 interfaces on ports 8030, 8645, 9119, 9377 (among other system listeners); separate sockets on the Tailscale interface include 8650, 6080, 8001 and other dynamic ports; localhost listeners include 18010, 18020, 4199, 8021, 8650, 11435. Bindings alone do not prove actual traffic, routing to HERMES, or in-flight work. Exact private interface addresses and ephemeral operator-identifying data are intentionally omitted from this public report.
- A Tailscale IPv4 address was returned, showing connectivity configuration remains present. No ACL/grant or L7 routing/in-flight census was obtained.

## Mandatory unanswered questions

1. Which ingress routes (direct addresses, reverse proxies, internal gateways, Tailscale, localhost) can still issue inference or reconnect? No signed/canonical inventory, route shutdown/drain acknowledgements, or authenticated zero-inflight SSE ledger was produced.
2. Who owns both GPU device contexts, with PID+start-time+UUID/cgroup mapping, lease epoch and external attested authority? Low utilization, empty compute-app output, and inactive systemd units are not evidence of exclusive allocation.
3. Is a persistent, anti-rollback journal tied to an **external** authority and current process identity, including boot/crash fencing and independent watchdog? No physical end-to-end exercise or authoritative state receipt was produced.
4. Can cutover be rolled back automatically without ending unrelated live requests, and can restarted/stray agents be fenced before GPU allocation? Not tested under ownership-controlled staging.

## Safe next gate

Perform a separately authorized and reversible ingress-drain demonstration with system-level ownership evidence, including zero active SSE and queue/inflight requests, direct/Tailscale/gateway census, GPU PID/start-time inspection, monotonic signed lease and external revocation/boot/crash checks. Obtain a time-bounded owner grant for **both** devices and independent rollback readiness before running either llama.cpp or HERMES. Until then, perform only CPU/offline tests and compile-only CUDA checks. A successful source audit, low VRAM utilization, and systemd inactivity do **not** change this NO-GO gate.
