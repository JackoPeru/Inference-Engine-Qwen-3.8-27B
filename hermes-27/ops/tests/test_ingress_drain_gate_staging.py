"""Pure offline state-machine tests. No sockets, GPUs, services or network."""

from dataclasses import replace
import ast
import unittest
from unittest import mock
from pathlib import Path

from ops import ingress_drain_gate_staging as s
from ops.manager_lease_core import LeaseRejected, require_ingress


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def complete_manifest():
    """Deliberately synthetic: bool flags are NOT physical trust evidence."""
    return s.CoverageManifest(
        manifest_id="offline-synthetic",
        revision=1,
        covered_routes=s.REQUIRED_ROUTES,
        exclusive_ingress_verified=True,
        direct_local_fenced=True,
        laya_restart_fenced=True,
        trust_anchor_verified=True,
        unknown_routes=False,
    )


def req(path="/v1/chat/completions", *, body=b""):
    return (f"POST {path} HTTP/1.1\r\nHost: tabby\r\n"
            f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)


def get(path="/v1/models"):
    return f"GET {path} HTTP/1.1\r\nHost: tabby\r\n\r\n".encode("ascii")


def response(data=b"OK"):
    return b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data


def stream_chunk(data):
    return f"{len(data):X}\r\n".encode() + data + b"\r\n"


class GateTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.gate = s.DrainGate(complete_manifest(), clock=self.clock,
                                instance_id="test-process-1", bootstrap_for_offline_test=True)

    def open(self, name="c1", route="uninote-tailnet-8001"):
        self.assertTrue(self.gate.open_connection(name, route=route))
        return name

    def start(self, key="supervisor-request-1", timeout=30):
        return self.gate.begin_drain(key=key, timeout_s=timeout)

    def assert_no_lease_proof(self, gate=None):
        result = (gate or self.gate).status()
        self.assertFalse(result["manager_lease_admissible"])
        self.assertIsNone(result["production_ingress_proof"])
        self.assertNotIn("scope", result)
        self.assertNotIn("blocked", result)
        self.assertNotIn("inflight", result)
        with self.assertRaises(LeaseRejected):
            require_ingress(result, now=1000)

    def test_new_gate_defaults_to_fenced_if_no_restored_journal(self):
        gate = s.DrainGate(complete_manifest(), clock=self.clock)
        self.assertEqual(gate.status()["phase"], "RECOVERY_REQUIRED")
        self.assertFalse(gate.open_connection("x", route="uninote-tailnet-8001"))
        with self.assertRaises(s.GateError):
            gate.begin_drain(key="a")
        self.assert_no_lease_proof(gate)

    def test_complete_synthetic_drained_candidate_is_never_real_attestation(self):
        self.open()
        self.assertTrue(self.gate.feed_client("c1", get()))
        self.assertTrue(self.gate.feed_server("c1", response()))
        self.assertEqual(self.start(), 1)
        status = self.gate.status()
        self.assertEqual(status["inflight_requests"], 0)
        self.assertEqual(status["open_connections"], 1)
        self.assertFalse(status["drained_candidate"])
        self.assertTrue(self.gate.close_connection("c1"))
        self.assertTrue(self.gate.status()["drained_candidate"])
        self.assert_no_lease_proof()

    def test_no_zero_proof_on_idle_keepalive(self):
        self.open()
        self.start()
        status = self.gate.status()
        self.assertEqual(status["inflight_requests"], 0)
        self.assertEqual(status["open_connections"], 1)
        self.assertFalse(status["drained_candidate"])
        self.assert_no_lease_proof()

    def test_keepalive_after_finished_request_still_blocked(self):
        self.open()
        self.assertTrue(self.gate.feed_client("c1", get()))
        self.assertTrue(self.gate.feed_server("c1", response()))
        self.start()
        self.assertFalse(self.gate.status()["drained_candidate"])
        self.assertTrue(self.gate.close_connection("c1"))
        self.assertTrue(self.gate.status()["drained_candidate"])

    def test_inflight_body_bytes_are_not_zero(self):
        self.open()
        self.assertTrue(self.gate.feed_client("c1", req(body=b"abcde")[:-3]))
        self.start()
        self.assertEqual(self.gate.status()["inflight_requests"], 1)
        self.assertTrue(self.gate.feed_client("c1", b"cde"))
        self.assertEqual(self.gate.status()["inflight_requests"], 1)
        self.assertTrue(self.gate.feed_server("c1", response()))
        self.assertEqual(self.gate.status()["inflight_requests"], 0)
        self.assertFalse(self.gate.status()["drained_candidate"])
        self.assertTrue(self.gate.close_connection("c1"))
        self.assertTrue(self.gate.status()["drained_candidate"])

    def test_fragmented_headers_pre_drain_complete_without_interrupting(self):
        self.open()
        self.assertTrue(self.gate.feed_client("c1", b"GET /v1/mod"))
        self.start()
        self.assertEqual(self.gate.status()["inflight_requests"], 1)
        self.assertTrue(self.gate.feed_client("c1", b"els HTTP/1.1\r\nHost: t\r\n\r\n"))
        self.assertTrue(self.gate.feed_server("c1", response()))
        self.assertTrue(self.gate.close_connection("c1"))
        self.assertTrue(self.gate.status()["drained_candidate"])

    def test_new_http_request_on_keepalive_after_drain_is_refused_and_taints(self):
        self.open()
        self.assertTrue(self.gate.feed_client("c1", get()))
        self.start()
        self.assertTrue(self.gate.feed_server("c1", response()))
        self.assertFalse(self.gate.feed_client("c1", get("/again")))
        self.assertEqual(self.gate.status()["denied_new_requests"], 1)
        self.assertEqual(self.gate.status()["phase"], "RECOVERY_REQUIRED")
        self.assertFalse(self.gate.close_connection("c1"))
        self.assert_no_lease_proof()

    def test_refuses_new_connection_during_drain_without_interrupting_existing(self):
        self.open()
        self.assertTrue(self.gate.feed_client("c1", get()))
        self.start()
        self.assertFalse(self.gate.open_connection("new", route="uninote-tailnet-8001"))
        self.assertEqual(self.gate.status()["denied_new_connections"], 1)
        self.assertTrue(self.gate.feed_server("c1", response()))
        self.assertTrue(self.gate.close_connection("c1"))
        self.assertTrue(self.gate.status()["drained_candidate"])

    def test_sse_chunked_remains_inflight_until_terminating_chunk(self):
        self.open()
        self.assertTrue(self.gate.feed_client("c1", get("/v1/chat/completions?stream=1")))
        head = (b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Transfer-Encoding: chunked\r\n\r\n")
        self.assertTrue(self.gate.feed_server("c1", head + stream_chunk(b"data: hi\n\n")))
        self.start()
        self.assertEqual(self.gate.status()["active_sse"], 1)
        self.assertEqual(self.gate.status()["inflight_requests"], 1)
        self.assertFalse(self.gate.status()["drained_candidate"])
        self.assertTrue(self.gate.feed_server("c1", stream_chunk(b"data: bye\n\n")))
        self.assertEqual(self.gate.status()["active_sse"], 1)
        self.assertTrue(self.gate.feed_server("c1", b"0\r\n\r\n"))
        self.assertEqual(self.gate.status()["active_sse"], 0)
        self.assertEqual(self.gate.status()["inflight_requests"], 0)
        self.assertFalse(self.gate.status()["drained_candidate"])
        self.assertTrue(self.gate.close_connection("c1"))
        self.assertTrue(self.gate.status()["drained_candidate"])

    def test_fragmented_sse_response_and_chunk_framing(self):
        self.open()
        self.assertTrue(self.gate.feed_client("c1", get()))
        head = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
        chunks = head + stream_chunk(b"data: x\n\n") + b"0\r\n\r\n"
        for b in chunks:
            self.assertTrue(self.gate.feed_server("c1", bytes([b])))
        self.assertEqual(self.gate.status()["active_sse"], 0)
        self.start()
        self.assertFalse(self.gate.status()["drained_candidate"])
        self.assertTrue(self.gate.close_connection("c1"))
        self.assertTrue(self.gate.status()["drained_candidate"])

    def test_indeterminate_sse_without_length_or_chunked_is_permanent_unknown(self):
        self.open()
        self.assertTrue(self.gate.feed_client("c1", get()))
        self.assertFalse(self.gate.feed_server(
            "c1", b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n"
        ))
        self.assertEqual(self.gate.status()["active_sse"], 1)
        self.assertEqual(self.gate.status()["phase"], "RECOVERY_REQUIRED")
        self.assert_no_lease_proof()

    def test_fin_during_active_sse_is_not_proof_of_normal_completion(self):
        self.open()
        self.assertTrue(self.gate.feed_client("c1", get()))
        self.assertTrue(self.gate.feed_server(
            "c1", b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                  b"Transfer-Encoding: chunked\r\n\r\n" + stream_chunk(b"data: hi\n\n")
        ))
        self.start()
        self.assertFalse(self.gate.close_connection("c1"))
        self.assertEqual(self.gate.status()["phase"], "RECOVERY_REQUIRED")
        self.assert_no_lease_proof()

    def test_half_close_is_not_full_socket_disappearance(self):
        self.open()
        self.start()
        self.assertFalse(self.gate.close_connection("c1", client_eof=True, server_eof=False))
        self.assertEqual(self.gate.status()["open_connections"], 1)
        self.assertFalse(self.gate.status()["drained_candidate"])
        self.assertTrue(self.gate.close_connection("c1", client_eof=False, server_eof=True))
        self.assertEqual(self.gate.status()["open_connections"], 0)
        self.assertTrue(self.gate.status()["drained_candidate"])

    def test_pipelined_requests_before_drain_are_accounted_individually(self):
        self.open()
        self.assertTrue(self.gate.feed_client("c1", get("/1") + get("/2")))
        self.start()
        self.assertEqual(self.gate.status()["inflight_requests"], 2)
        self.assertTrue(self.gate.feed_server("c1", response(b"a")))
        self.assertEqual(self.gate.status()["inflight_requests"], 1)
        self.assertTrue(self.gate.feed_server("c1", response(b"b")))
        self.assertEqual(self.gate.status()["inflight_requests"], 0)
        self.assertTrue(self.gate.close_connection("c1"))
        self.assertTrue(self.gate.status()["drained_candidate"])

    def test_pipelined_second_request_started_after_drain_is_unknown(self):
        self.open()
        self.assertTrue(self.gate.feed_client("c1", get("/1")))
        self.start()
        self.assertFalse(self.gate.feed_client("c1", get("/2")))
        self.assertEqual(self.gate.status()["phase"], "RECOVERY_REQUIRED")

    def test_partial_unparsed_client_bytes_not_zero(self):
        self.open()
        self.assertTrue(self.gate.feed_client("c1", b"P"))
        self.start()
        self.assertEqual(self.gate.status()["inflight_requests"], 1)
        self.assertFalse(self.gate.status()["drained_candidate"])

    def test_partial_unparsed_server_bytes_not_zero(self):
        self.open()
        self.assertTrue(self.gate.feed_client("c1", get()))
        self.assertTrue(self.gate.feed_server("c1", b"HTTP/1.1 200"))
        self.start()
        self.assertEqual(self.gate.status()["inflight_requests"], 1)
        self.assertFalse(self.gate.status()["drained_candidate"])

    def test_replay_drain_key_is_idempotent_with_no_deadline_extension(self):
        self.open()
        first = self.start(key="same", timeout=20)
        self.clock.advance(15)
        self.assertEqual(first, self.gate.begin_drain(key="same", timeout_s=200))
        self.assertEqual(first, self.gate.resume_drain(key="same"))
        self.clock.advance(6)
        self.gate.tick()
        self.assertTrue(self.gate.status()["drain_timeout"])
        self.assertEqual(self.gate.status()["phase"], "RECOVERY_REQUIRED")
        self.assertEqual(self.gate.status()["epoch"], 1)

    def test_new_drain_key_during_fence_refused(self):
        self.start(key="original")
        with self.assertRaises(s.GateError):
            self.gate.begin_drain(key="different")
        with self.assertRaises(s.GateError):
            self.gate.resume_drain(key="different")
        self.assertEqual(self.gate.status()["epoch"], 1)

    def test_crash_restart_with_active_stream_does_not_claim_zero(self):
        self.open()
        self.assertTrue(self.gate.feed_client("c1", get()))
        self.start(key="drain-key")
        journal = self.gate.journal()
        self.assertTrue(journal.fenced)
        restored = s.DrainGate(complete_manifest(), clock=self.clock,
                               instance_id="new-process", restored=journal)
        self.assertEqual(restored.status()["phase"], "RECOVERY_REQUIRED")
        self.assertEqual(restored.status()["epoch"], 1)
        self.assertEqual(restored.resume_drain(key="drain-key"), 1)
        self.assertFalse(restored.open_connection("new", route="uninote-tailnet-8001"))
        self.assert_no_lease_proof(restored)

    def test_restoring_corrupt_journal_fail_closed(self):
        bad = s.DrainJournal(1, "old", 12, "wrong-manifest", 1, None, True, "DRAINING")
        restored = s.DrainGate(complete_manifest(), clock=self.clock, restored=bad)
        self.assertEqual(restored.status()["phase"], "RECOVERY_REQUIRED")
        self.assertIn("persisted-journal-corrupt-or-coverage-drift",
                      restored.status()["unknown_reasons"])
        self.assertFalse(restored.open_connection("x", route="uninote-tailnet-8001"))

    def test_restoring_unfenced_open_journal_denied(self):
        bad = self.gate.journal()
        self.assertFalse(bad.fenced)
        restored = s.DrainGate(complete_manifest(), clock=self.clock, restored=bad)
        self.assertEqual(restored.status()["phase"], "RECOVERY_REQUIRED")
        self.assert_no_lease_proof(restored)

    def test_abort_does_not_cancel_existing_request_or_reopen(self):
        self.open()
        self.assertTrue(self.gate.feed_client("c1", get()))
        self.start()
        self.gate.abort_without_cancellation()
        self.assertEqual(self.gate.status()["inflight_requests"], 1)
        self.assertTrue(self.gate.feed_server("c1", response()))
        self.assertEqual(self.gate.status()["inflight_requests"], 0)
        self.assertEqual(self.gate.status()["phase"], "RECOVERY_REQUIRED")
        self.assertFalse(self.gate.open_connection("c2", route="uninote-tailnet-8001"))

    def test_timeout_does_not_cancel_active_response(self):
        self.open()
        self.assertTrue(self.gate.feed_client("c1", get()))
        self.start(timeout=5)
        self.clock.advance(5.1)
        self.gate.tick()
        self.assertTrue(self.gate.status()["drain_timeout"])
        self.assertEqual(self.gate.status()["inflight_requests"], 1)
        self.assertTrue(self.gate.feed_server("c1", response()))
        self.assertEqual(self.gate.status()["inflight_requests"], 0)
        self.assert_no_lease_proof()

    def test_new_untracked_route_fences_even_before_drain(self):
        self.assertFalse(self.gate.open_connection("x", route="unmodeled-new-proxy"))
        self.assertIn("unlisted-ingress-route", self.gate.status()["unknown_reasons"])
        self.assertEqual(self.gate.status()["phase"], "RECOVERY_REQUIRED")

    def test_unattributed_network_connection_is_permanently_unknown(self):
        self.gate.observe_unknown_connection(source="new-tailscale-forward")
        self.assertIn("untracked-bypass-or-connection:new-tailscale-forward",
                      self.gate.status()["unknown_reasons"])
        self.assert_no_lease_proof()

    def test_reused_socket_identity_fences_even_if_old_connection_closed(self):
        self.open()
        self.assertTrue(self.gate.close_connection("c1"))
        self.assertFalse(self.gate.open_connection("c1", route="uninote-tailnet-8001"))
        self.assertEqual(self.gate.status()["phase"], "RECOVERY_REQUIRED")

    def test_connection_cap_is_bounded_and_fail_closed(self):
        with mock.patch.object(s, "MAX_CONNECTIONS", 2):
            self.open("a")
            self.open("b")
            self.assertFalse(self.gate.open_connection("c", route="uninote-tailnet-8001"))
        self.assertEqual(self.gate.status()["phase"], "RECOVERY_REQUIRED")

    def test_changed_manifest_during_drain_invalidates_prior_scope(self):
        self.start()
        self.gate.install_manifest(replace(complete_manifest(), revision=2))
        self.assertEqual(self.gate.status()["phase"], "RECOVERY_REQUIRED")
        self.assertIn("coverage-configuration-changed", self.gate.status()["unknown_reasons"])
        self.assert_no_lease_proof()

    def test_watchdog_status_contains_no_key_or_raw_request_content(self):
        self.open()
        self.assertTrue(self.gate.feed_client("c1", req("/private/user_prompt", body=b"private")))
        self.start(key="do-not-expose-this-key")
        status = str(self.gate.status())
        self.assertNotIn("do-not-expose", status)
        self.assertNotIn("/private", status)
        self.assertNotIn("private", status)
        self.assertEqual(self.gate.status()["epoch"], 1)
        self.assertEqual(self.gate.status()["inflight_requests"], 1)


class CoverageAndProtocolRejections(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.gate = s.DrainGate(complete_manifest(), clock=self.clock,
                                bootstrap_for_offline_test=True,
                                instance_id="protocol-suite")

    def open(self):
        self.assertTrue(self.gate.open_connection("p", route="uninote-tailnet-8001"))

    def taint_client(self, raw: bytes):
        self.open()
        self.assertFalse(self.gate.feed_client("p", raw))
        self.assertEqual(self.gate.status()["phase"], "RECOVERY_REQUIRED")
        self.assertFalse(self.gate.status()["drained_candidate"])
        self.assertFalse(self.gate.status()["manager_lease_admissible"])

    def taint_server(self, raw: bytes):
        self.open()
        self.assertTrue(self.gate.feed_client("p", get()))
        self.assertFalse(self.gate.feed_server("p", raw))
        self.assertEqual(self.gate.status()["phase"], "RECOVERY_REQUIRED")
        self.assertFalse(self.gate.status()["drained_candidate"])

    def test_partial_coverage_never_all_client_attestation(self):
        incomplete = replace(complete_manifest(),
                             covered_routes=frozenset({"uninote-tailnet-8001"}))
        gate = s.DrainGate(incomplete, clock=self.clock,
                           bootstrap_for_offline_test=True)
        gate.begin_drain(key="x")
        data = gate.status()
        self.assertFalse(data["scope_candidate_complete"])
        self.assertIn("scope-partial-routes", data["unknown_reasons"])
        self.assertFalse(data["drained_candidate"])
        self.assertIsNone(data["production_ingress_proof"])
        with self.assertRaises(LeaseRejected):
            require_ingress(data, now=1000)

    def test_laya_restart_bypass_denies_coverage(self):
        incomplete = replace(complete_manifest(), laya_restart_fenced=False)
        self.assertIn("scope-laya-direct-restart-open", incomplete.problems())
        gate = s.DrainGate(incomplete, clock=self.clock, bootstrap_for_offline_test=True)
        gate.begin_drain(key="x")
        self.assertFalse(gate.status()["drained_candidate"])

    def test_direct_local_tabby_bypass_denies_coverage(self):
        incomplete = replace(complete_manifest(), direct_local_fenced=False)
        self.assertIn("scope-tabby-local-bypass-open", incomplete.problems())

    def test_unverified_source_claim_denies_coverage(self):
        incomplete = replace(complete_manifest(), trust_anchor_verified=False)
        self.assertIn("scope-root-issuer-not-attested", incomplete.problems())

    def test_uninventoried_additional_tailscale_forward_denies_coverage(self):
        incomplete = replace(complete_manifest(), unknown_routes=True)
        self.assertIn("scope-uninventoried-ingress", incomplete.problems())

    def test_sole_ingress_not_enforced_denies_coverage(self):
        incomplete = replace(complete_manifest(), exclusive_ingress_verified=False)
        self.assertIn("scope-sole-ingress-not-enforced", incomplete.problems())

    def test_manifest_extra_route_is_not_silently_accepted(self):
        incomplete = replace(complete_manifest(),
                             covered_routes=s.REQUIRED_ROUTES | {"unreviewed-new-route"})
        self.assertIn("scope-unknown-routes", incomplete.problems())

    def test_coverage_bad_manifest_revision_denied(self):
        incomplete = replace(complete_manifest(), revision=0)
        self.assertIn("scope-manifest-identity-invalid", incomplete.problems())

    def test_http2_prior_knowledge_refused(self):
        self.taint_client(b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n")

    def test_http2_request_version_refused(self):
        self.taint_client(b"GET / HTTP/2.0\r\nHost: x\r\n\r\n")

    def test_tls_handshake_is_unknown_binary_bytes(self):
        self.open()
        self.assertTrue(self.gate.feed_client("p", b"\x16\x03\x01"))
        self.gate.begin_drain(key="x")
        self.assertEqual(self.gate.status()["inflight_requests"], 1)
        self.assertFalse(self.gate.close_connection("p"))
        self.assertEqual(self.gate.status()["phase"], "RECOVERY_REQUIRED")

    def test_websocket_upgrade_request_refused(self):
        self.taint_client(
            b"GET /ws HTTP/1.1\r\nHost: x\r\nConnection: keep-alive,upgrade\r\n"
            b"Upgrade: websocket\r\n\r\n")

    def test_h2c_upgrade_without_upgrade_header_refused(self):
        self.taint_client(
            b"GET / HTTP/1.1\r\nHost: x\r\nConnection: http2-settings\r\n"
            b"HTTP2-Settings: AAMAAABkAARAAAAAAAIAAAAA\r\n\r\n")

    def test_connect_tunnel_refused(self):
        self.taint_client(b"CONNECT target:8000 HTTP/1.1\r\nHost: x\r\n\r\n")

    def test_chunked_request_body_refused_until_adapter_supports_it(self):
        self.taint_client(
            b"POST / HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n"
            b"3\r\nabc\r\n0\r\n\r\n")

    def test_expect_continue_refused_until_adapter_supports_it(self):
        self.taint_client(
            b"POST / HTTP/1.1\r\nHost: x\r\nExpect: 100-continue\r\n"
            b"Content-Length: 3\r\n\r\n")

    def test_duplicate_content_length_refused(self):
        self.taint_client(
            b"POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n"
            b"Content-Length: 3\r\n\r\n")

    def test_transfer_encoding_and_content_length_conflict_refused(self):
        self.taint_client(
            b"POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n")

    def test_invalid_content_length_refused(self):
        self.taint_client(
            b"POST / HTTP/1.1\r\nHost: x\r\nContent-Length: -1\r\n\r\n")

    def test_header_obs_fold_refused(self):
        self.taint_client(
            b"GET / HTTP/1.1\r\nHost: x\r\n  X-Bad: fold\r\n\r\n")

    def test_header_trailers_refused(self):
        self.taint_client(
            b"POST / HTTP/1.1\r\nHost: x\r\nTrailer: checksum\r\n\r\n")

    def test_oversized_request_header_refused(self):
        self.taint_client(b"GET / HTTP/1.1\r\nX: " + b"A" * s.MAX_HEADER + b"\r\n\r\n")

    def test_chunked_sse_with_trailers_refused(self):
        self.taint_server(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n"
            b"Content-Type: text/event-stream\r\n\r\n"
            b"1\r\nx\r\n0\r\nDigest: unknown\r\n\r\n")

    def test_corrupt_chunk_size_refused(self):
        self.taint_server(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
            b"z\r\n")

    def test_corrupt_chunk_terminator_refused(self):
        self.taint_server(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
            b"3\r\nabcXX")

    def test_chunk_extension_not_silently_trusted(self):
        self.taint_server(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
            b"3;foo=bar\r\nabc\r\n0\r\n\r\n")

    def test_response_without_explicit_framing_refused(self):
        self.taint_server(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nx")

    def test_http2_response_refused(self):
        self.taint_server(b"HTTP/2.0 200 OK\r\nContent-Length: 1\r\n\r\nx")

    def test_upgrade_response_101_refused(self):
        self.taint_server(b"HTTP/1.1 101 Switching Protocols\r\n\r\n")

    def test_late_response_without_registered_request_fences(self):
        self.open()
        self.assertFalse(self.gate.feed_server("p", response()))
        self.assertIn("unexpected-response-without-request", self.gate.status()["unknown_reasons"])

    def test_100_continue_followed_by_final_response_accounted(self):
        self.open()
        self.assertTrue(self.gate.feed_client("p", req(body=b"x")))
        self.assertTrue(self.gate.feed_server(
            "p", b"HTTP/1.1 100 Continue\r\n\r\n" + response(b"x")))
        self.gate.begin_drain(key="k")
        self.assertTrue(self.gate.close_connection("p"))
        self.assertTrue(self.gate.status()["drained_candidate"])

    def test_sse_content_length_still_requires_full_body_and_close(self):
        self.open()
        self.assertTrue(self.gate.feed_client("p", get("/sse")))
        self.assertTrue(self.gate.feed_server(
            "p", b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                 b"Content-Length: 6\r\n\r\ndata:"))
        self.gate.begin_drain(key="k")
        self.assertEqual(self.gate.status()["active_sse"], 1)
        self.assertTrue(self.gate.feed_server("p", b" "))
        self.assertEqual(self.gate.status()["active_sse"], 0)
        self.assertFalse(self.gate.status()["drained_candidate"])
        self.assertTrue(self.gate.close_connection("p"))
        self.assertTrue(self.gate.status()["drained_candidate"])

    def test_post_body_response_before_body_finished_keeps_request_active(self):
        self.open()
        self.assertTrue(self.gate.feed_client("p", req(body=b"1234")[:-2]))
        self.assertTrue(self.gate.feed_server("p", response(b"no")))
        self.gate.begin_drain(key="k")
        self.assertEqual(self.gate.status()["inflight_requests"], 1)
        self.assertTrue(self.gate.feed_client("p", b"34"))
        self.assertEqual(self.gate.status()["inflight_requests"], 0)
        self.assertTrue(self.gate.close_connection("p"))
        self.assertTrue(self.gate.status()["drained_candidate"])

    def test_full_chunked_response_parser_does_not_parse_payload_as_headers(self):
        self.open()
        self.assertTrue(self.gate.feed_client("p", get()))
        body = b"HTTP/2.0 200 OK\r\n\r\nnot actually a response"
        self.assertTrue(self.gate.feed_server(
            "p", b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
                 + stream_chunk(body) + b"0\r\n\r\n"))
        self.gate.begin_drain(key="k")
        self.assertTrue(self.gate.close_connection("p"))
        self.assertTrue(self.gate.status()["drained_candidate"])

    def test_unknown_response_data_on_idle_keepalive_denied(self):
        self.open()
        self.assertTrue(self.gate.feed_client("p", get()))
        self.assertTrue(self.gate.feed_server("p", response()))
        self.assertTrue(self.gate.feed_server("p", b"stray"))
        self.gate.begin_drain(key="k")
        self.assertFalse(self.gate.status()["drained_candidate"])
        self.assertFalse(self.gate.close_connection("p"))
        self.assertEqual(self.gate.status()["phase"], "RECOVERY_REQUIRED")

    def test_no_service_or_network_imports_in_staging_module(self):
        tree = ast.parse(Path(s.__file__).read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(entry.name.split(".")[0] for entry in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        self.assertFalse(names & {
            "socket", "subprocess", "requests", "httpx", "aiohttp",
            "ctypes", "pynvml", "torch", "cupy", "psutil",
        })

    def test_untrusted_synthetic_manifest_is_not_an_all_client_proof(self):
        self.gate.begin_drain(key="k")
        status = self.gate.status()
        self.assertTrue(status["scope_candidate_complete"])
        self.assertTrue(status["drained_candidate"])
        self.assertFalse(status["manager_lease_admissible"])
        self.assertNotIn("scope", status)
        with self.assertRaises(LeaseRejected):
            require_ingress(status, now=100)


if __name__ == "__main__":
    unittest.main()
