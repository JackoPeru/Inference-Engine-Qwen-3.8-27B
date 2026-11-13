# HERMES-27 — HTTP/1.1 ingress drain gate, isolated staging

**Verdict: CPU/OFFLINE prototype implemented; ingress authority NOT established;
PRODUCTION NO-GO.** Date: 2026-11-14, based on the 2026-11-13 read-only
audit in ops/INGRESS_GPU_OWNER_FEASIBILITY_2026-11-13.md.

Only these **new** files belong to this task:

- ops/ingress_drain_gate_staging.py — in-memory HTTP/1.1 admission parser,
  request/SSE tracking, synthetic scope manifest, monotonic drain state.
- ops/tests/test_ingress_drain_gate_staging.py — deterministic CPU simulations.
- ops/INGRESS_DRAIN_STAGING.md — this contract, limitations and integration plan.

There is no socket listener, HTTP client, proxy, GPU call, credential read,
sudo, systemctl, service invocation, network connection, production edit,
or durable filesystem write in this implementation. No live requests were
interrupted or inspected. This file **must not** be deployed as the
manager's trusted all-tabby-clients ingress checker.

## Physical reason for the prototype

The audited server has a **real independent ingress bypass**:

- TabbyAPI listens at 127.0.0.1:8000.
- The active user service uninote-llama-proxy.service binds the Tailscale
  address 100.94.223.14:8001 and proxies raw TCP in both directions to
  127.0.0.1:8000; its inspected proxy.py:23–35 has no HTTP request or
  streaming drain counter. Restart=on-failure, RestartUSec=5s.
- The same host has dashboard/Hub/Studio and possible direct localhost
  Tabby callers. The set of all localhost clients and external forwarding
  paths is not independently attested.
- laya_bridge.py:487–504 and 548–586 can attempt to restart GPU Manager,
  then directly restart Tabby on LLM fallback after manager timeout/5xx.
  That restart is outside an in-process manager watchdog lease.

A dashboard-only drain, a raw TCP socket count, an idle GPU utilization
percentage, or a simple manager/status response cannot certify that all
Tabby requests have finished. The existing manager lease v1 admission
requires independent, fresh, authenticated proof of an **all-tabby-clients**
scope with blocked ingress and inflight=0. That proof provider is still
UNIMPLEMENTED. The synthetic scope manifest here is not that provider.

## State machine and protocol handling

All state changes occur under one local lock; the injected monotonic clock
makes every timeout deterministic. The public operations have these effects:

| Operation | Staging semantics |
| --- | --- |
| DrainGate(..., bootstrap_for_offline_test=True) | Opens only a synthetic test instance. Without the flag, missing journal starts RECOVERY_REQUIRED and closed. |
| open_connection(id, route) | Accepts only while OPEN, with unique ID and listed route. New connections during drain are denied without touching existing connections. |
| feed_client(id, bytes) | Parses strict HTTP/1.1 request line/headers, bounded Content-Length and bodies, fragmented headers, multiple pipelined requests. Tracks partial pre-drain requests to completion. |
| feed_server(id, bytes) | Parses status, Content-Length or bounded chunked framing, interim 100/103 responses and Content-Type text/event-stream. SSE remains inflight until the framed terminal chunk/length. |
| begin_drain(key, timeout) | Atomically closes new admissions, increments an epoch and retains the active connection/request registry. Replaying the same hashed key is idempotent without resetting timeout or counters. |
| resume_drain(key) | Checks the same key and returns the previous epoch. It cannot remove a crash fence or recover lost connections. |
| close_connection(id, client_eof, server_eof) | Records both-direction EOF; no zero candidate if active work, unparsed bytes, or partial wire framing remains. A half-closed socket still blocks. |
| tick() | Enforces the monotonic deadline. Expiry enters RECOVERY_REQUIRED and **does not cancel** any existing request. |
| abort_without_cancellation() | Fences further admission and leaves tracked requests untouched; no auto-reopen. |
| journal() / restored=... | Models durable metadata. A restored journal **always** enters RECOVERY_REQUIRED because pre-crash sockets/streams cannot be reconstructed. The module never persists it itself. |
| status() | Bounded, watchdog-friendly in-memory summary: epoch, inflight requests, active SSE, unclosed keepalives, unknown reasons, rejects, timeout, diagnostic scope/drain candidate. No request bodies, keys or credentials. |

Only strict HTTP/1.1 messages with unambiguous framing are interpreted.
The experiment deliberately **fails closed** on:

- HTTP/2 prior-knowledge preface, HTTP/2 request/response versions, h2c,
  WebSocket upgrade, CONNECT tunnels and other unknown connection modes;
- invalid, folded or duplicate headers; invalid or conflicting
  Content-Length/Transfer-Encoding; unknown transfer coding;
- unsupported chunked **client** bodies, Expect: 100-continue requests,
  chunk extensions, non-empty trailers, undelimited responses and responses
  with no matching request;
- incomplete/unknown bytes at either direction, hidden/inaccessible routes,
  unknown connection registration, reused socket IDs, exceeded bounds;
- live requests, partial headers/bodies, active SSE and **idle HTTP/1.1
  keep-alive sockets**. An idle keep-alive may show zero parsed active
  requests but still cannot produce a drained candidate;
- unilateral FIN or a peer disconnect while a request or SSE stream is
  incomplete, as this could truncate an active response;
- token/key mismatch, journal loss or corruption, process restart,
  unverified bootstrap, manifest/revision changes or any missed deadline.

Unlike a TCP connection counter, the parser counts HTTP **request
admissions**, partial headers, pipelined requests, full responses and SSE
lifetimes. Bytes are inspected in bounded buffers; request bodies are
consumed by length and are **not stored** in status or logs. There is no
fallback that treats unknown bytes as a successful zero count.

The parser has intentionally limited protocol coverage. For example, many
normal streaming HTTP/1.1 responses are close-delimited, and production
proxies often support chunked request bodies or WebSockets. These remain
BLOCKED until reviewed framing/transport adapters can prove exact
completion, including client disconnect and downstream write events.
Unsupported traffic must never be silently considered drained.

## Coverage and trust are not the same as parsing

REQUIRED_ROUTES is a **minimum hardcoded review list**, not a complete
inventory validated on the machine:

1. uninote-tailnet-8001
2. tabby-direct-loopback-8000
3. hermes-dashboard
4. hermes-hub
5. laya-manager-and-direct-restart
6. other-local-or-tailscale-tabby-clients

CoverageManifest rejects missing/extra paths, unknown routes, unverified
exclusive ingress, direct localhost bypass, unfenced Laya restart, missing
root issuer evidence and malformed or changed revisions. The default
manifest is intentionally incomplete. Even when a **synthetic test
manifest** marks every boolean true, the result is at most
DRAINED_CANDIDATE, **never a lease attestation**. Status always contains:

    manager_lease_admissible: false
    production_ingress_proof: null

It intentionally has **no** successful scope=all-tabby-clients,
blocked=true/inflight=0/observed_at_unix JSON contract. The existing
manager_lease_core.require_ingress therefore rejects it. A mock's
trust_anchor_verified boolean is only test input; no verified root
attestation, signature, process admission barrier or physically enforced
sole ingress exists here.

Idempotency metadata is stored only as a SHA-256 of the drain key in the
model journal; raw keys are absent from watchdog status. Epochs increase
when a drain starts, and repeated calls with the same key do not increment
or extend the deadline. A service restart with a prior journal retains
its epoch but **cannot re-open ingress**, even if previous counters
would have reached zero. A missing journal starts fenced, not fresh/OPEN.
The bootstrap-for-offline-test switch is test-only and cannot safely
replace independently authorized first-boot initialization.

The module is deliberately not connected to a persistence backend.
Host crash persistence, fsync, journal deletion detection, owner ACLs
and signer/session fencing must be supplied by a separate privileged
control plane; an in-memory simulation cannot prove those guarantees.

## Safe integration architecture (future, not deployed)

**Phase A — authoritative route inventory and sole ingress enforcement.**
Before any service change, independently enumerate all Tabby clients and
port forwards, including loopback, UniNote Tailscale proxy, dashboard,
Hermes Hub, Studio, user processes, network namespaces and any direct
scripts. Design an approved, root-owned ingress controller with a private
Tabby-only backend endpoint inaccessible to bypassing same-UID clients.
An HTTP-aware, L7 stream-accounting proxy must sit on **every** route.
Without a root-enforced network namespace, Unix socket authorization or
equivalent enforced boundary, a new direct connection to
127.0.0.1:8000 defeats the claimed coverage. Do not change that binding
during this stage or treat the route list as physically complete.

**Phase B — preserve UniNote clients and existing requests.** The existing
UniNote proxy is a raw bidirectional TCP forward. It cannot simply return
a truthful L7 inflight count from a socket count. In an approved future
maintenance cutover, use a reviewed HTTP/1.1+SSE aware adapter in front of
its forwarding service or route it via the sole ingress controller.
On drain, atomically refuse **new** requests and new sockets (e.g. a
temporary Retry-After response at the actual gateway, not implemented
here); let previously admitted requests and streams complete normally.
Send Connection: close on their final response and let idle keep-alive
sockets close gracefully **after** all accepted work has completed.
Do not terminate SSE to fabricate an inflight=0 counter; if it does not
finish within the allowed deadline, enter RECOVERY_REQUIRED and abort
the cutover. Validate HTTP/2, WebSockets, chunked request bodies and
intermediary retries before routing those protocols through the adapter.

**Phase C — interlock Laya and systemd independently of GPU Manager.**
The Laya bridge's direct sudo systemctl restart hermes-tabby.service
fallback can run when the manager is absent or returns 5xx. It must obey
a root-owned durable fence *even if manager.py is crashed or killed*;
merely returning HTTP 423 from a healthy manager is insufficient.
Independently audit systemd Restart/on-failure paths, unit aliases,
Laya's privilege grant, direct Character ID/service starters and other
recovery agents. Do not restart, stop, edit or mask any unit as part of
this staging task.

**Phase D — independent root proof and synchronized lease.** A trusted
broker must prove complete route coverage and atomic closed ingress for
the current fencing epoch, count active HTTP requests/SSE/keepalive
and unparsed bytes, and bind a fresh <=10 s proof to its own root-owned
identity. The manager lease must verify **that** proof, not this
in-memory model, and independently prove both physical GPUs free
of HERMES/unknown owners before any Tabby restart. Keep ingress closed
through recovery, release and validation; reopen only on explicit
independently authorized handback.

**Phase E — crash and cancellation safety.** Persist fence and route
epoch in a protected, independently durable journal before a drain ack.
After a restart all old streams are UNKNOWN; do not infer zero from empty
in-memory registries. Preserve existing downstream work, do not issue
forced cancellations to satisfy the counter, and preserve the original
Tabby service if drain is timed out or incomplete.

Because these boundaries and the independent GPU-owner attestor remain
unimplemented, **a production deployment is not approved**.

## Deterministic offline test and acceptance criteria

Run from the project root using an already available Python interpreter:

    python -m unittest discover -s ops/tests -p 'test_ingress_drain_gate_staging.py' -v
    python -m py_compile ops/ingress_drain_gate_staging.py ops/tests/test_ingress_drain_gate_staging.py

Tests inject a fake monotonic clock, synthetic routes, request and SSE
wire bytes, journal snapshots and manifest flags. No external request,
real port, systemd unit, GPU, credential or persistent file is accessed.
They verify normal framed traffic and explicit negative cases: partial
coverage, unknown/upgrade/HTTP2 requests, SSE not completed, persistent
keepalive, fragmented messages, pipelining, body lengths, timeout,
idempotent replay, manifest drift and crash/restart fencing. A separate
test checks rejection by the actual existing require_ingress contract.

**Observed offline results:** 67/67 new focused tests PASS; full existing
plus new ops suite **260/260 PASS** (47.197 s) using local Python 3.13.
Both new Python files passed py_compile. These are unit tests of
in-memory inputs; the results do **not** establish trusted physical
ingress coverage, actual transport reliability or a deployed gateway.

**Acceptance here means only a CPU staging prototype, not all-client
drain coverage or safe production cutover. Production remains NO-GO.**
