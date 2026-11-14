#!/usr/bin/env python3
"""CPU/OFFLINE source-contract and virtual event scheduler checks.

References actual HERMES-27 code; deliberately does not build or invoke
CUDA, run HTTP, modify source, or estimate real hardware speedups.

Run: python3 -B -m unittest -v distributed/tests/test_throughput_stream_event_source.py
"""
from __future__ import annotations

import heapq
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


def source(relpath: str) -> str:
    path = (ROOT / relpath).resolve()
    if not path.is_relative_to(ROOT):
        raise AssertionError("invalid source location")
    return path.read_text(encoding="utf-8")


def section(text: str, begin: str, end: str) -> str:
    """Fail closed on absent/duplicated/reordered implementation anchors."""
    if text.count(begin) != 1:
        raise AssertionError(f"expected exactly one start anchor: {begin}")
    start = text.index(begin)
    pos = text.find(end, start + len(begin))
    if pos < 0:
        raise AssertionError(f"missing following end anchor: {end}")
    return text[start:pos]


def ordered(text: str, *anchors: str) -> None:
    last = -1
    for anchor in anchors:
        index = text.find(anchor, last + 1)
        if index < 0:
            raise AssertionError(f"missing or out-of-order source anchor: {anchor}")
        last = index


def wave_times(ready: dict[str, int], callback_cost: dict[str, int],
               admission: tuple[str, ...]) -> dict[str, int]:
    """Optimistic abstract model of the current join-then-sequential-commit.

    'ready' is the instant a mock GPU1 result becomes complete; this is NOT a
    measurement of any actual CUDA kernel/transfer. All work in one wave.
    """
    barrier = max(ready.values())
    output = {}
    time_now = barrier
    for job in admission:
        time_now += callback_cost.get(job, 0)
        output[job] = time_now
    return output


def ready_times(ready: dict[str, int], callback_cost: dict[str, int]) -> dict[str, int]:
    """One purely logical event-ready delivery queue (no inferred GPU gain)."""
    events = [(when, request_id) for request_id, when in ready.items()]
    heapq.heapify(events)
    output = {}
    callback_available = 0
    while events:
        when, job = heapq.heappop(events)
        callback_available = max(callback_available, when) + callback_cost.get(job, 0)
        output[job] = callback_available
    return output


class EventLease:
    """A proposed request-local event lifecycle, independent of CUDA code.

    READY means GPU1 has completed and transferred the logits; no new token
    may be launched until the owner commits/cancels and clears the lease.
    """

    def __init__(self):
        self.state: dict[str, dict] = {}
        self.log: list[tuple[str, str, int | None]] = []

    def admit(self, job: str) -> None:
        if job in self.state:
            raise AssertionError("duplicate request ID")
        self.state[job] = {
            "phase": "idle", "next_pos": 0, "inflight_pos": None,
            "cancel": False, "closed": False, "output": [],
        }

    def launch(self, job: str, position: int) -> None:
        s = self.state[job]
        if s["phase"] != "idle" or s["cancel"] or s["closed"]:
            raise AssertionError("state already in flight, cancelled or closed")
        if position != s["next_pos"]:
            raise AssertionError("duplicate / skipped absolute token position")
        s["phase"] = "gpu0"
        s["inflight_pos"] = position
        self.log.append((job, "launch_gpu0", position))

    def gpu0_done(self, job: str) -> None:
        s = self.state[job]
        if s["phase"] != "gpu0":
            raise AssertionError("GPU0 event not owned or already consumed")
        s["phase"] = "gpu1"
        self.log.append((job, "launch_gpu1", s["inflight_pos"]))

    def gpu1_done(self, job: str) -> None:
        s = self.state[job]
        if s["phase"] != "gpu1":
            raise AssertionError("GPU1 event not owned or already consumed")
        s["phase"] = "ready"
        self.log.append((job, "logits_ready", s["inflight_pos"]))
        if s["cancel"]:
            self.close(job)

    def commit(self, job: str, token: int) -> None:
        s = self.state[job]
        if s["phase"] != "ready" or s["cancel"] or s["closed"]:
            raise AssertionError("unready or cancelled GPU logits cannot emit token")
        if token < 0:
            raise AssertionError("invalid token")
        s["next_pos"] += 1
        s["output"].append(token)
        self.log.append((job, "token", s["inflight_pos"]))
        s["phase"] = "idle"
        s["inflight_pos"] = None

    def cancel(self, job: str) -> None:
        s = self.state[job]
        s["cancel"] = True
        # Cannot close buffers while GPU0/GPU1 has any outstanding command.
        if s["phase"] in ("idle", "ready"):
            self.close(job)

    def close(self, job: str) -> None:
        s = self.state[job]
        if s["phase"] not in ("idle", "ready") or s["closed"]:
            raise AssertionError("closing an in-flight or double-freed CUDA state")
        s["phase"] = "terminal"
        s["closed"] = True
        self.log.append((job, "closed", None))


class SourceContract(unittest.TestCase):
    def test_current_host_wave_join_precedes_every_token_commit(self):
        cpp = source("distributed/pp2_batch.cu")
        impl = section(cpp, "std::size_t ContinuousScheduler::tick_parallel(",
                       "void ContinuousScheduler::shutdown()")
        ordered(impl,
                "workers.emplace_back([this, &wave, i]()",
                "backend_.step(work.job->state, work.token, work.position,",
                "for (auto& worker : workers) if (worker.joinable()) worker.join();",
                "for (auto& work : wave)",
                "emit_event(work.job, EventKind::token, best_id);")
        self.assertEqual(impl.count("for (auto& worker : workers) if (worker.joinable()) worker.join();"), 1)
        self.assertNotIn("cudaEventQuery(", impl)
        self.assertNotIn("cudaStreamWaitEvent(", impl)
        self.assertIn("std::lock_guard<std::mutex> pump(pump_mutex_);", cpp)

    def test_real_async_pp2_already_private_streams_but_host_waits_each_stage(self):
        cpp = source("distributed/pp2_fwd.cu")
        async_impl = section(cpp, "static std::vector<float> decode_step_async(",
                             "std::vector<float> decode_step(")
        ordered(async_impl,
                "std::unique_lock<std::mutex> guard(ctx->step_mutex,std::try_to_lock)",
                "cudaMemcpyAsync(ds.idsbuf,ctx->host_token,",
                "cudaEventRecord(ctx->ready0,ctx->stream0)",
                "cudaEventSynchronize(ctx->ready0)",
                "cudaMemcpyAsync(ds.x1buf,ctx->host_hidden,",
                "cudaEventRecord(ctx->ready1,ctx->stream1)",
                "cudaEventSynchronize(ctx->ready1)",
                "std::memcpy(logits.data(),ctx->host_logits,")
        self.assertNotIn("cudaDeviceSynchronize()", async_impl)
        self.assertNotIn("g_xfer_mu", async_impl)
        self.assertIn("cudaStreamNonBlocking", cpp)
        self.assertIn("if(ds.async_ctx) return decode_step_async(tok,ds,pos,err);", cpp)

    def test_real_prefill_has_global_device_barriers_but_parallel_is_excluded(self):
        cpp = source("distributed/pp2_fwd.cu")
        gdn = section(cpp, "static bool run_gdn_layer(", "static bool run_full_layer(")
        full = section(cpp, "static bool run_full_layer(", "static std::vector<float> prefill_impl(")
        self.assertIn("cudaDeviceSynchronize()", gdn)
        self.assertIn("cudaDeviceSynchronize()", full)
        self.assertIn("cudaMemcpy(qq.data(),d_qq,", full)
        self.assertIn("cudaMemcpy(d_G,G.data(),", full)
        server = source("serving/server.cpp")
        cli = section(server, "int main(int argc, char** argv)", "return 0;")
        ordered(cli,
                "if (cfg.parallel_steps) {",
                "if (!cfg.continuous_scheduler || cfg.batched_prefill ||",
                "if (cfg.chunked_prefill) {",
                "if (cfg.continuous_scheduler || cfg.active != 1 ||")
        self.assertIn("HERMES_PP2_ASYNC_STREAMS", cli)
        self.assertIn("HERMES_PP2_CHUNKED_PREFILL", cli)
        self.assertIn("prefill_chunks_into_decode_state", server)
        # Current configuration BLOCKS the dangerous combination; source
        # alone does not prove a concurrent prefill race is currently live.

    def test_real_http_callback_is_synchronous_tokenizer_bridge_and_soak_is_not_sse(self):
        server = source("serving/server.cpp")
        event = section(server, "void continuous_event(", "void run_continuous()")
        ordered(event,
                "if (event.kind == pp2batch::EventKind::token)",
                "tokenizer_.stream_push(job->id, event.token_id)",
                "stream_text(job);")
        self.assertIn("continuous_->tick();", server)
        ipc = section(server, "json call(const json& request)", "\n  Config cfg_;")
        self.assertIn("std::lock_guard<std::mutex> lock(mutex_);", ipc)
        self.assertIn("return json::parse(read_line());", ipc)
        soak = source("scripts/scratch_http_soak.py")
        self.assertIn('"stream": False', soak)
        self.assertIn("check_deltas(previous, settled, clients", soak)
        self.assertIn("wait_for_drain(http, config.drain_timeout", soak)

    def test_sse16_existing_real_ab_has_parity_but_no_clear_advantage(self):
        data = json.loads(source("benchmarks/HTTP_SSE16_PARALLEL_LEGACY_AB_2026-11-13.json"))
        self.assertEqual(data["status"], "PASS_SCOPED_SSE_PARITY")
        self.assertIs(data["output_hashes_identical"], True)
        self.assertEqual(data["request_count_per_variant"], 30)
        by_n = {record["clients"]: record for record in data["data"]}
        self.assertEqual(set(by_n), {1, 2, 4, 8})
        self.assertLess(abs(by_n[4]["relative_parallel_percent"]), 1.0)
        self.assertLess(abs(by_n[8]["relative_parallel_percent"]), 1.0)
        self.assertAlmostEqual(by_n[4]["parallel_tok_per_wall_sec"], 4.871126)
        self.assertAlmostEqual(by_n[8]["legacy_tok_per_wall_sec"], 4.858091)

    def test_fail_closed_if_an_expected_source_anchor_disappears(self):
        with self.assertRaisesRegex(AssertionError, "start anchor"):
            section("void B() {}", "void A()", "void B()")
        with self.assertRaisesRegex(AssertionError, "out-of-order"):
            ordered("A then B", "B", "A")


class DeterministicVirtualEventContract(unittest.TestCase):
    def test_one_straggler_blocks_current_wave_but_not_independent_ready_queue(self):
        ready = {"A": 40, "B": 4, "C": 5, "D": 6}  # MOCK timestamps only
        callbacks = {job: 0 for job in ready}
        current = wave_times(ready, callbacks, ("A", "B", "C", "D"))
        proposed = ready_times(ready, callbacks)
        self.assertEqual(current, {job: 40 for job in ready})
        self.assertEqual(proposed, ready)
        self.assertEqual(set(current), set(proposed))
        # These are logical ordering opportunities, NOT measured milliseconds
        # or claimed GPU/HTTP throughput improvement.

    def test_slow_first_callback_blocks_current_wave_fast_sibling(self):
        ready = {"A": 4, "B": 4}
        current = wave_times(ready, {"A": 20, "B": 0}, ("A", "B"))
        self.assertEqual(current, {"A": 24, "B": 24})
        self.assertEqual(ready_times({"B": 4}, {"B": 0}), {"B": 4})
        # Offloading an IPC call must not reorder tokens within the same job
        # or let the single shared tokenizer pipe be accessed concurrently.

    def test_inflight_duplicate_position_refused_without_double_owner(self):
        pump = EventLease()
        pump.admit("A")
        pump.launch("A", 0)
        with self.assertRaisesRegex(AssertionError, "already in flight"):
            pump.launch("A", 0)
        pump.gpu0_done("A")
        pump.gpu1_done("A")
        pump.commit("A", 2)
        self.assertEqual(pump.state["A"]["next_pos"], 1)
        with self.assertRaisesRegex(AssertionError, "duplicate / skipped"):
            pump.launch("A", 0)
        pump.launch("A", 1)
        pump.gpu0_done("A")
        pump.gpu1_done("A")
        pump.commit("A", 2)
        self.assertEqual(pump.state["A"]["output"], [2, 2])
        pump.close("A")
        with self.assertRaisesRegex(AssertionError, "already in flight"):
            pump.launch("A", 2)

    def test_cancel_in_gpu0_or_gpu1_fences_free_until_logit_event(self):
        for stage in ("gpu0", "gpu1"):
            with self.subTest(stage=stage):
                pump = EventLease()
                pump.admit("A")
                pump.launch("A", 0)
                if stage == "gpu1":
                    pump.gpu0_done("A")
                pump.cancel("A")
                self.assertFalse(pump.state["A"]["closed"])
                with self.assertRaisesRegex(AssertionError, "closing an in-flight"):
                    pump.close("A")
                if stage == "gpu0":
                    pump.gpu0_done("A")
                pump.gpu1_done("A")
                self.assertTrue(pump.state["A"]["closed"])
                self.assertEqual(pump.state["A"]["output"], [])
                with self.assertRaisesRegex(AssertionError, "unready or cancelled"):
                    pump.commit("A", 2)
                self.assertEqual(sum(ev[1] == "closed" for ev in pump.log), 1)

    def test_unordered_cross_user_completion_preserves_each_user_state_order(self):
        pump = EventLease()
        for job in ("A", "B", "C"):
            pump.admit(job)
            pump.launch(job, 0)
        pump.gpu0_done("B")
        pump.gpu1_done("B")
        pump.commit("B", 4)
        pump.gpu0_done("C")
        pump.gpu1_done("C")
        pump.commit("C", 3)
        pump.gpu0_done("A")
        pump.gpu1_done("A")
        pump.commit("A", 2)
        self.assertEqual([entry for entry in pump.log if entry[1] == "token"],
                         [("B", "token", 0), ("C", "token", 0), ("A", "token", 0)])
        for job in ("A", "B", "C"):
            self.assertEqual(pump.state[job]["next_pos"], 1)
            pump.close(job)
            self.assertTrue(pump.state[job]["closed"])


if __name__ == "__main__":
    unittest.main()
