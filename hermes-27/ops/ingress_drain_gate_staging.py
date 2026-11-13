"""Pure in-memory HTTP/1.1 ingress/drain STAGING experiment.

Not a proxy, auth provider or all-tabby-clients attestor. No sockets,
network, subprocess, credentials, systemctl or production side effects.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
import re
import secrets
import threading
import time
from typing import Callable, Deque, Mapping


REQUIRED_ROUTES = frozenset({
    "uninote-tailnet-8001",
    "tabby-direct-loopback-8000",
    "hermes-dashboard",
    "hermes-hub",
    "laya-manager-and-direct-restart",
    "other-local-or-tailscale-tabby-clients",
})
MAX_HEADER = 32 * 1024
MAX_CHUNK_LINE = 1024
MAX_BUFFERED_FEED = 1024 * 1024
MAX_CONTENT_LENGTH = 1 << 40
MAX_CONNECTIONS = 1024
MAX_REQUESTS_PER_CONN = 256
MAX_DRAIN_SECONDS = 3600.0
_TOKEN = re.compile(rb"[!#$%&'*+.^_|~0-9A-Za-z-]+\Z")
_REQUEST_LINE = re.compile(rb"([A-Z]{1,24}) ([^\x00-\x20\x7f]{1,8192}) HTTP/1\.1\Z")
_STATUS_LINE = re.compile(rb"HTTP/1\.1 ([1-5][0-9][0-9])(?: [^\r\n]{0,120})?\Z")


class GateError(RuntimeError):
    """Rejected or ambiguous staging operation; live ingress stays fenced."""


class Phase(str, Enum):
    OPEN = "OPEN"
    DRAINING = "DRAINING"
    DRAINED_CANDIDATE = "DRAINED_CANDIDATE"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


@dataclass(frozen=True)
class CoverageManifest:
    """Synthetic scope input, NOT a signature or trusted root attestation."""

    manifest_id: str
    revision: int
    covered_routes: frozenset[str]
    exclusive_ingress_verified: bool = False
    direct_local_fenced: bool = False
    laya_restart_fenced: bool = False
    trust_anchor_verified: bool = False
    unknown_routes: bool = True

    def problems(self) -> tuple[str, ...]:
        failures: list[str] = []
        if (not isinstance(self.manifest_id, str)
                or not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", self.manifest_id)
                or type(self.revision) is not int or self.revision < 1):
            failures.append("scope-manifest-identity-invalid")
        if not isinstance(self.covered_routes, frozenset):
            failures.append("scope-manifest-routes-invalid")
        else:
            if REQUIRED_ROUTES - self.covered_routes:
                failures.append("scope-partial-routes")
            if self.covered_routes - REQUIRED_ROUTES:
                failures.append("scope-unknown-routes")
        if self.unknown_routes is not False:
            failures.append("scope-uninventoried-ingress")
        if self.exclusive_ingress_verified is not True:
            failures.append("scope-sole-ingress-not-enforced")
        if self.direct_local_fenced is not True:
            failures.append("scope-tabby-local-bypass-open")
        if self.laya_restart_fenced is not True:
            failures.append("scope-laya-direct-restart-open")
        if self.trust_anchor_verified is not True:
            failures.append("scope-root-issuer-not-attested")
        return tuple(failures)


@dataclass(frozen=True)
class DrainJournal:
    """Persistable journal shape only; this module never writes to disk."""

    version: int
    instance_id: str
    epoch: int
    manifest_id: str
    manifest_revision: int
    drain_key_hash: str | None
    fenced: bool
    phase: str


@dataclass
class Request:
    sequence: int
    method: str
    body_left: int = 0
    body_done: bool = True
    response_started: bool = False
    response_done: bool = False
    sse: bool = False
    response_mode: str = ""
    response_left: int = 0
    chunk_left: int = 0
    close_after: bool = False


@dataclass
class Connection:
    identity: str
    route: str
    epoch_opened: int
    requests: Deque[Request] = field(default_factory=deque)
    request_bytes: bytearray = field(default_factory=bytearray)
    response_bytes: bytearray = field(default_factory=bytearray)
    request_body: Request | None = None
    response_body: Request | None = None
    partial_request_epoch: int | None = None
    response_state: str = "HEADER"
    next_seq: int = 1
    closed: bool = False
    server_eof: bool = False
    client_eof: bool = False
    close_after_response: bool = False


def _headers(head: bytes, *, request: bool) -> tuple[bytes, dict[bytes, bytes]]:
    """Strict HTTP/1.1 framing; differing parser interpretations are faults."""
    if not head.endswith(b"\r\n\r\n") or len(head) > MAX_HEADER:
        raise GateError("unparseable-http1-header")
    rows = head[:-4].split(b"\r\n")
    if not rows or not rows[0]:
        raise GateError("missing-http1-start-line")
    first = rows[0]
    if request:
        if first.startswith(b"PRI * HTTP/2.0"):
            raise GateError("http2-prior-knowledge-not-supported")
        if _REQUEST_LINE.fullmatch(first) is None:
            raise GateError("unknown-or-non-http11-request")
    elif _STATUS_LINE.fullmatch(first) is None:
        raise GateError("unknown-or-non-http11-response")
    parsed: dict[bytes, bytes] = {}
    for row in rows[1:]:
        if not row or row[:1] in (b" ", b"\t") or b":" not in row:
            raise GateError("ambiguous-or-folded-header")
        key, value = row.split(b":", 1)
        key = key.lower()
        if _TOKEN.fullmatch(key) is None or b"\x00" in value:
            raise GateError("invalid-http1-header")
        if key in parsed:
            raise GateError("duplicated-http1-header")
        parsed[key] = value.strip().lower()
    connection_tokens = {item.strip() for item in parsed.get(b"connection", b"").split(b",")}
    if (b"upgrade" in parsed or b"http2-settings" in parsed or
            connection_tokens & {b"upgrade", b"http2-settings"}):
        raise GateError("upgrade-websocket-or-h2c-not-supported")
    if b"trailer" in parsed:
        raise GateError("trailer-header-not-supported")
    if b"transfer-encoding" in parsed and b"content-length" in parsed:
        raise GateError("conflicting-transfer-framing")
    return first, parsed


def _length(headers: Mapping[bytes, bytes]) -> int | None:
    value = headers.get(b"content-length")
    if value is None:
        return None
    if re.fullmatch(rb"[0-9]{1,15}", value) is None:
        raise GateError("content-length-invalid")
    number = int(value)
    if number > MAX_CONTENT_LENGTH:
        raise GateError("content-length-exceeds-safe-bound")
    return number


def _chunk_encoding(headers: Mapping[bytes, bytes]) -> bool:
    value = headers.get(b"transfer-encoding")
    if value is None:
        return False
    if value.strip() != b"chunked":
        raise GateError("unsupported-transfer-coding")
    return True


def _key_digest(value: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 128 or "\x00" in value:
        raise GateError("drain-idempotency-key-invalid")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class DrainGate:
    """Serialized local drain state; all success is diagnostic, not authority.

    A restored journal always yields RECOVERY_REQUIRED: previously tracked
    sockets and requests cannot be reconstructed after a manager/proxy crash.
    Missing journal also fences by default. The offline-test bootstrap flag
    is NOT suitable for any production adapter.
    """

    def __init__(
        self, manifest: CoverageManifest, *,
        clock: Callable[[], float] = time.monotonic,
        instance_id: str | None = None,
        bootstrap_for_offline_test: bool = False,
        restored: DrainJournal | None = None,
    ):
        self._mutex = threading.RLock()
        self._clock = clock
        self.instance_id = instance_id or secrets.token_hex(16)
        self.manifest = manifest
        self._frozen_manifest = (manifest.manifest_id, manifest.revision)
        self.epoch = 0
        self._key_hash: str | None = None
        self._deadline: float | None = None
        self._connections: dict[str, Connection] = {}
        self._unknown: set[str] = set()
        self._rejected_connections = 0
        self._rejected_requests = 0
        self._timed_out = False
        self.phase = Phase.RECOVERY_REQUIRED
        if restored is not None:
            self._load_journal(restored)
        elif bootstrap_for_offline_test:
            self.phase = Phase.OPEN
        else:
            self._unknown.add("unrecoverable-missing-journal-or-unverified-bootstrap")

    def _load_journal(self, data: DrainJournal) -> None:
        if (not isinstance(data, DrainJournal) or data.version != 1
                or type(data.epoch) is not int or data.epoch < 0
                or type(data.instance_id) is not str or not data.instance_id
                or data.fenced is not True
                or (data.drain_key_hash is not None
                    and re.fullmatch(r"[0-9a-f]{64}", data.drain_key_hash) is None)
                or data.manifest_id != self.manifest.manifest_id
                or data.manifest_revision != self.manifest.revision):
            self._unknown.add("persisted-journal-corrupt-or-coverage-drift")
        else:
            self.epoch = data.epoch
            self._key_hash = data.drain_key_hash
        self._unknown.add("unreconciled-connections-after-process-restart")
        self.phase = Phase.RECOVERY_REQUIRED

    def journal(self) -> DrainJournal:
        with self._mutex:
            return DrainJournal(
                version=1, instance_id=self.instance_id, epoch=self.epoch,
                manifest_id=self._frozen_manifest[0],
                manifest_revision=self._frozen_manifest[1],
                drain_key_hash=self._key_hash,
                fenced=self.phase != Phase.OPEN, phase=self.phase.value,
            )

    def begin_drain(self, *, key: str, timeout_s: float = 300.0) -> int:
        """Repeated key never resets accepted traffic, timeout or epoch."""
        digest = _key_digest(key)
        with self._mutex:
            if self._key_hash is not None:
                if not hmac.compare_digest(self._key_hash, digest):
                    raise GateError("different-drain-key-while-fenced")
                return self.epoch
            if self.phase != Phase.OPEN:
                raise GateError("admission-not-open-fenced-or-recovery-required")
            if (type(timeout_s) not in (int, float) or
                    not 0 < timeout_s <= MAX_DRAIN_SECONDS):
                raise GateError("drain-timeout-invalid")
            self.epoch += 1
            self._key_hash = digest
            self.phase = Phase.DRAINING
            self._deadline = self._clock() + timeout_s
            self._refresh_locked()
            return self.epoch

    def resume_drain(self, *, key: str) -> int:
        """Idempotent replay after crash remains fenced; never opens ingress."""
        digest = _key_digest(key)
        with self._mutex:
            if self._key_hash is None or not hmac.compare_digest(self._key_hash, digest):
                raise GateError("drain-resume-key-not-matched")
            return self.epoch

    def install_manifest(self, replacement: CoverageManifest) -> None:
        with self._mutex:
            if replacement != self.manifest:
                self._unknown.add("coverage-configuration-changed")
                self.phase = Phase.RECOVERY_REQUIRED
            self.manifest = replacement

    def observe_unknown_connection(self, *, source: str = "unmapped") -> None:
        with self._mutex:
            self._taint("untracked-bypass-or-connection:" + source[:64])

    def open_connection(self, identity: str, *, route: str) -> bool:
        with self._mutex:
            if not isinstance(identity, str) or not identity or len(identity) > 128:
                self._taint("connection-id-invalid")
                return False
            if not isinstance(route, str) or route not in REQUIRED_ROUTES:
                self._taint("unlisted-ingress-route")
                return False
            if identity in self._connections:
                self._taint("connection-id-reused")
                return False
            if self.phase != Phase.OPEN:
                self._rejected_connections += 1
                return False
            if len(self._connections) >= MAX_CONNECTIONS:
                self._taint("connection-capacity-unknown")
                return False
            self._connections[identity] = Connection(identity, route, self.epoch)
            return True

    def _taint(self, reason: str) -> None:
        self._unknown.add(reason)
        self.phase = Phase.RECOVERY_REQUIRED

    def _refresh_locked(self) -> None:
        if self.phase not in (Phase.DRAINING, Phase.DRAINED_CANDIDATE):
            return
        if self._deadline is not None and self._clock() >= self._deadline:
            self._timed_out = True
            self._taint("drain-deadline-expired-no-cancellation")
            return
        unresolved = any(self._connection_busy(c) for c in self._connections.values())
        if (not unresolved and not self._unknown
                and not self.manifest.problems()
                and self._frozen_manifest == (self.manifest.manifest_id, self.manifest.revision)):
            self.phase = Phase.DRAINED_CANDIDATE
        else:
            self.phase = Phase.DRAINING

    @staticmethod
    def _connection_busy(conn: Connection) -> bool:
        # An idle HTTP/1.1 keep-alive is NOT zero proof.
        return (not conn.closed or bool(conn.requests)
                or bool(conn.request_bytes) or bool(conn.response_bytes)
                or conn.request_body is not None or conn.response_body is not None
                or conn.partial_request_epoch is not None)

    def tick(self) -> None:
        """Monotonic timeout; never cancels or terminates real work."""
        with self._mutex:
            self._refresh_locked()

    def status(self) -> dict:
        with self._mutex:
            self._refresh_locked()
            queued = sum(len(c.requests) for c in self._connections.values())
            partial = sum(c.partial_request_epoch is not None for c in self._connections.values())
            streams = sum(req.sse and not req.response_done for c in self._connections.values()
                          for req in c.requests)
            scope_errors = self.manifest.problems()
            reasons = sorted(self._unknown | set(scope_errors))
            return {
                "phase": self.phase.value,
                "epoch": self.epoch,
                "instance_id": self.instance_id,
                "open_connections": sum(not c.closed for c in self._connections.values()),
                "tracked_connections": len(self._connections),
                "keepalive_or_unclosed_connections": sum(
                    not c.closed and not c.requests and c.partial_request_epoch is None
                    for c in self._connections.values()
                ),
                "inflight_requests": queued + partial,
                "active_sse": streams,
                "unknown_reasons": reasons,
                "denied_new_connections": self._rejected_connections,
                "denied_new_requests": self._rejected_requests,
                "drain_timeout": self._timed_out,
                "scope_manifest_revision": self.manifest.revision,
                "scope_candidate_complete": not scope_errors,
                "drained_candidate": self.phase == Phase.DRAINED_CANDIDATE,
                "manager_lease_admissible": False,
                "production_ingress_proof": None,
            }

    def abort_without_cancellation(self) -> None:
        with self._mutex:
            self._taint("drain-aborted-no-automatic-ingress-reopen")

    def _connection(self, identity: str) -> Connection:
        conn = self._connections.get(identity)
        if conn is None or conn.closed:
            self._taint("untracked-or-closed-socket-data")
            raise GateError("socket-not-tracked")
        return conn

    @staticmethod
    def _append_bounded(target: bytearray, data: bytes) -> None:
        if not isinstance(data, bytes) or len(data) > MAX_BUFFERED_FEED:
            raise GateError("unbounded-or-nonbyte-http-data")
        if len(target) + len(data) > MAX_BUFFERED_FEED:
            raise GateError("http-buffer-unsafe-bound")
        target.extend(data)

    def feed_client(self, identity: str, data: bytes) -> bool:
        """Parse client HTTP/1.1 requests; new post-drain requests are denied.

        Bytes of a request header or body already started before the drain
        are allowed to complete. No socket is closed, cancelled or written.
        """
        with self._mutex:
            try:
                conn = self._connection(identity)
                self._append_bounded(conn.request_bytes, data)
                while conn.request_bytes:
                    body = conn.request_body
                    if body is not None:
                        consumed = min(body.body_left, len(conn.request_bytes))
                        del conn.request_bytes[:consumed]
                        body.body_left -= consumed
                        if body.body_left:
                            break
                        body.body_done = True
                        conn.request_body = None
                        while (conn.requests and conn.requests[0].body_done
                               and conn.requests[0].response_done):
                            conn.requests.popleft()
                        continue
                    if conn.partial_request_epoch is None:
                        if self.phase != Phase.OPEN:
                            self._rejected_requests += 1
                            raise GateError("new-http-request-after-admission-closed")
                        conn.partial_request_epoch = self.epoch
                    if len(conn.request_bytes) > MAX_HEADER and b"\r\n\r\n" not in conn.request_bytes[:MAX_HEADER]:
                        raise GateError("request-head-exceeds-bound")
                    boundary = conn.request_bytes.find(b"\r\n\r\n")
                    if boundary < 0:
                        # Partial request line or headers cannot count as zero
                        # inflight even if no complete request exists yet.
                        break
                    end = boundary + 4
                    if end > MAX_HEADER:
                        raise GateError("request-head-exceeds-bound")
                    head = bytes(conn.request_bytes[:end])
                    first, headers = _headers(head, request=True)
                    method = _REQUEST_LINE.fullmatch(first).group(1).decode("ascii")
                    if method in ("CONNECT", "TRACE"):
                        raise GateError("tunnel-or-untracked-method")
                    if b"expect" in headers:
                        # Expect/100-continue needs additional future
                        # protocol and cancellation handling; deny safely.
                        raise GateError("expect-continue-not-supported")
                    if _chunk_encoding(headers):
                        raise GateError("chunked-client-body-unsupported")
                    length = _length(headers) or 0
                    if len(conn.requests) >= MAX_REQUESTS_PER_CONN:
                        raise GateError("request-queue-capacity-exceeded")
                    del conn.request_bytes[:end]
                    request = Request(conn.next_seq, method, length, length == 0)
                    conn.next_seq += 1
                    conn.requests.append(request)
                    conn.partial_request_epoch = None
                    if length:
                        conn.request_body = request
                self._refresh_locked()
                return True
            except GateError as exc:
                self._taint(str(exc))
                return False

    @staticmethod
    def _pending_response(conn: Connection) -> Request | None:
        for req in conn.requests:
            if not req.response_started:
                return req
        return None

    @staticmethod
    def _finish_response(conn: Connection, request: Request) -> None:
        request.response_done = True
        conn.response_body = None
        conn.response_state = "HEADER"
        while conn.requests and conn.requests[0].body_done and conn.requests[0].response_done:
            conn.requests.popleft()

    def _response_head(self, conn: Connection, head: bytes) -> None:
        first, headers = _headers(head, request=False)
        status = int(_STATUS_LINE.fullmatch(first).group(1))
        request = self._pending_response(conn)
        if request is None:
            raise GateError("unexpected-response-without-request")
        if status == 101:
            raise GateError("websocket-or-protocol-upgrade-response")
        if 100 <= status < 200:
            # 100 Continue and 103 Early Hints do not finish a request.
            if b"transfer-encoding" in headers or b"content-length" in headers:
                raise GateError("framed-informational-response-unsupported")
            return
        request.response_started = True
        media_type = headers.get(b"content-type", b"").split(b";", 1)[0].strip()
        request.sse = media_type == b"text/event-stream"
        request.close_after = b"close" in [
            item.strip() for item in headers.get(b"connection", b"").split(b",")
        ]
        if request.close_after:
            conn.close_after_response = True
        chunked = _chunk_encoding(headers)
        length = _length(headers)
        if request.method == "HEAD" or status in (204, 304):
            if chunked or length not in (None, 0):
                raise GateError("unexpected-framing-on-bodyless-response")
            self._finish_response(conn, request)
        elif chunked:
            request.response_mode = "chunked"
            conn.response_body = request
            conn.response_state = "CHUNK_SIZE"
        elif length is not None:
            request.response_mode = "length"
            request.response_left = length
            if length == 0:
                self._finish_response(conn, request)
            else:
                conn.response_body = request
                conn.response_state = "LENGTH"
        else:
            # Valid HTTP/1.1 close-delimited streaming is NOT safely
            # enumerable as fully delivered from these staged callbacks.
            # Includes ordinary unframed SSE, even if the server EOF arrives.
            conn.response_body = request
            conn.response_state = "INDETERMINATE"
            raise GateError("close-delimited-response-not-attestable")

    def _response_step(self, conn: Connection) -> bool:
        """Consume one bounded parsing step, returning False if more needed."""
        buf = conn.response_bytes
        state = conn.response_state
        if state == "HEADER":
            boundary = buf.find(b"\r\n\r\n")
            if boundary < 0:
                if len(buf) > MAX_HEADER:
                    raise GateError("response-head-exceeds-bound")
                return False
            end = boundary + 4
            if end > MAX_HEADER:
                raise GateError("response-head-exceeds-bound")
            head = bytes(buf[:end])
            del buf[:end]
            self._response_head(conn, head)
            return True
        req = conn.response_body
        if req is None:
            raise GateError("response-parser-internal-state-invalid")
        if state == "LENGTH":
            count = min(req.response_left, len(buf))
            del buf[:count]
            req.response_left -= count
            if req.response_left == 0:
                self._finish_response(conn, req)
                return True
            return False
        if state == "CHUNK_SIZE":
            line_end = buf.find(b"\r\n")
            if line_end < 0:
                if len(buf) > MAX_CHUNK_LINE:
                    raise GateError("chunk-size-line-exceeds-bound")
                return False
            if line_end > MAX_CHUNK_LINE:
                raise GateError("chunk-size-line-exceeds-bound")
            line = bytes(buf[:line_end])
            del buf[:line_end + 2]
            # Extensions can alter semantics and may be parsed differently
            # by the upstream; conservatively refuse them in staging.
            if re.fullmatch(rb"[0-9a-fA-F]{1,16}", line) is None:
                raise GateError("ambiguous-or-extended-chunk-framing")
            size = int(line, 16)
            if size > MAX_CONTENT_LENGTH:
                raise GateError("chunk-exceeds-safe-bound")
            if size == 0:
                conn.response_state = "CHUNK_TRAILER"
            else:
                req.chunk_left = size
                conn.response_state = "CHUNK_DATA"
            return True
        if state == "CHUNK_DATA":
            count = min(req.chunk_left, len(buf))
            del buf[:count]
            req.chunk_left -= count
            if req.chunk_left == 0:
                conn.response_state = "CHUNK_CRLF"
                return True
            return False
        if state == "CHUNK_CRLF":
            if len(buf) < 2:
                return False
            if buf[:2] != b"\r\n":
                raise GateError("chunk-data-terminator-invalid")
            del buf[:2]
            conn.response_state = "CHUNK_SIZE"
            return True
        if state == "CHUNK_TRAILER":
            if len(buf) >= 2 and buf[:2] == b"\r\n":
                del buf[:2]
                self._finish_response(conn, req)
                return True
            boundary = buf.find(b"\r\n\r\n")
            if boundary < 0:
                if len(buf) > MAX_HEADER:
                    raise GateError("chunk-trailers-exceed-bound")
                return False
            if boundary + 4 > MAX_HEADER:
                raise GateError("chunk-trailers-exceed-bound")
            # Any non-empty trailers could contain framing/protocol
            # surprises. Unknown until reviewed and independently verified.
            raise GateError("chunk-trailers-not-supported")
        raise GateError("unframed-or-unknown-response-state")

    def feed_server(self, identity: str, data: bytes) -> bool:
        """Parse actual HTTP/1.1 response framing and tracked SSE streams."""
        with self._mutex:
            try:
                conn = self._connection(identity)
                self._append_bounded(conn.response_bytes, data)
                while conn.response_bytes:
                    if not self._response_step(conn):
                        break
                self._refresh_locked()
                return True
            except GateError as exc:
                self._taint(str(exc))
                return False

    def close_connection(self, identity: str, *, client_eof: bool = True,
                         server_eof: bool = True) -> bool:
        """Only explicit BOTH-direction EOF qualifies as a closed socket.

        No I/O and no cancellation. Closing with any active, unframed or
        unknown bytes is a permanent fail-closed event.
        """
        with self._mutex:
            try:
                conn = self._connection(identity)
                # Independent FIN events accumulate. An adapter may
                # observe client EOF and server EOF in separate callbacks.
                conn.client_eof = conn.client_eof or bool(client_eof)
                conn.server_eof = conn.server_eof or bool(server_eof)
                if not (conn.client_eof and conn.server_eof):
                    self._refresh_locked()
                    return False
                conn.closed = True
                if (conn.requests or conn.request_bytes or conn.response_bytes
                        or conn.request_body is not None or conn.response_body is not None
                        or conn.partial_request_epoch is not None
                        or conn.response_state != "HEADER"):
                    raise GateError("socket-closed-with-inflight-or-unparsed-http")
                self._refresh_locked()
                return True
            except GateError as exc:
                self._taint(str(exc))
                return False
