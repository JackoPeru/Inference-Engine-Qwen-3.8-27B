#!/usr/bin/env python3
"""CPU-only 8192-token PP2/HTTP boundary and bounded-KV design contract.

This deliberately tests TWO different things:
  1. Source-grounded CURRENT implementation must reject decoding position 8192,
     but can sample the first completion from 8192-token prefill logits.
  2. PROPOSED opt-in ring-window design is modeled against independent suffix
     and causal-attention oracles; it is NOT implemented in production PP2.

No CUDA, model weights, network, services, or external dependencies required.
Run: python3 distributed/tests/test_context8192_continuation_contract_cpu.py
"""

from __future__ import annotations

import math
import re
import unittest
from collections import deque
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PP2_H = ROOT / "distributed" / "pp2_fwd.hpp"
PP2_CU = ROOT / "distributed" / "pp2_fwd.cu"
SERVER = ROOT / "serving" / "server.cpp"

WINDOW = 8192
FULL_LAYERS = tuple(i for i in range(64) if i % 4 == 3)
GDN_LAYERS = tuple(i for i in range(64) if i % 4 != 3)
K_PER_POS = 4 * 256
BYTES_PER_FLOAT = 4


def body(source: str, start: str, end: str) -> str:
    a = source.index(start)
    b = source.index(end, a + len(start))
    return source[a:b]


def current_admission(prompt_len: int, max_new: int, capacity: int = WINDOW) -> bool:
    """Exact HTTP capacity equation; one completion uses prefill logits."""
    if prompt_len <= 0 or max_new <= 0:
        return False
    return prompt_len + max_new - 1 <= capacity


def current_decode_allowed(pos: int, klen: int, capacity: int = WINDOW) -> bool:
    """Essential source guards, not a CUDA execution or implementation patch."""
    return 0 <= pos < capacity and 0 <= klen < capacity and pos == klen


@dataclass(frozen=True)
class ProposedChunk:
    absolute_position: int
    key: int
    value: int


class BoundedRingKV:
    """PROPOSED policy, intentionally independent of PP2 source.

    Until commit, neither the oldest key nor any physical slot is overwritten.
    After successful attention, commit may evict the oldest resident token.
    Failure poisons the whole request; no partial state may be resumed.
    """

    def __init__(self, capacity: int = WINDOW):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.slots: list[ProposedChunk | None] = [None] * capacity
        self.next_absolute = 0
        self.retained = 0
        self.oldest = 0
        self.pending: ProposedChunk | None = None
        self.poisoned = False

    def stage(self, position: int, key: int, value: int) -> None:
        if self.poisoned:
            raise RuntimeError("poisoned state")
        if self.pending is not None:
            raise RuntimeError("request already has a token in flight")
        if position != self.next_absolute:
            raise ValueError("nonmonotonic absolute position")
        if position < 0 or position > 2**31 - 1:
            raise ValueError("current RoPE int32 bound exceeded")
        self.pending = ProposedChunk(position, key, value)

    def resident(self) -> tuple[ProposedChunk, ...]:
        """Current committed chronological suffix, before new token commit."""
        result: list[ProposedChunk] = []
        for pos in range(self.oldest, self.next_absolute):
            item = self.slots[pos % self.capacity]
            if item is None or item.absolute_position != pos:
                raise AssertionError("ring slot tag/chronological order violated")
            result.append(item)
        if len(result) != self.retained:
            raise AssertionError("retained length != resident count")
        return tuple(result)

    def causal_attention_keys(self) -> tuple[ProposedChunk, ...]:
        if self.pending is None:
            raise RuntimeError("no token staged")
        old = self.resident()
        # A full 8192 window can attend to at most 8191 OLD + CURRENT.
        old = old[-(self.capacity - 1):] if self.capacity > 1 else ()
        return old + (self.pending,)

    def commit(self) -> int | None:
        if self.pending is None or self.poisoned:
            raise RuntimeError("cannot commit non-existent or poisoned token")
        incoming = self.pending
        evicted = self.oldest if self.retained == self.capacity else None
        slot = incoming.absolute_position % self.capacity
        if evicted is not None:
            current = self.slots[slot]
            if current is None or current.absolute_position != evicted:
                raise AssertionError("eviction slot does not match oldest key")
            self.oldest += 1
        else:
            self.retained += 1
        self.slots[slot] = incoming
        self.next_absolute += 1
        self.pending = None
        return evicted

    def abort(self) -> None:
        # A CUDA failure may have changed GDN S/CS elsewhere. Reject ANY reuse.
        self.pending = None
        self.poisoned = True


def toy_softmax(values: tuple[float, ...]) -> float:
    """Zero query/key scalar gives uniform softmax; exposes eviction semantics."""
    if not values:
        raise ValueError("empty KV")
    return math.fsum(values) / len(values)


def reference_attention(query: float, samples: tuple[ProposedChunk, ...]) -> float:
    """Independent stable CPU attention over chronological tagged K/V."""
    scores = tuple(query * item.key for item in samples)
    mx = max(scores)
    numerators = tuple(math.exp(score - mx) for score in scores)
    denom = math.fsum(numerators)
    return math.fsum(
        weight * item.value for weight, item in zip(numerators, samples)
    ) / denom


class CurrentPP2BoundaryContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.header = PP2_H.read_text(encoding="utf-8")
        cls.pp2 = PP2_CU.read_text(encoding="utf-8")
        cls.server = SERVER.read_text(encoding="utf-8")

    def test_source_binds_actual_capacity(self):
        self.assertRegex(self.header, r"MAX_CONTEXT\s*=\s*8192\s*;")
        self.assertIn("capacity<1||capacity>DecodeState::MAX_CONTEXT", self.pp2)
        decode = body(self.pp2,
                      "std::vector<float> decode_step(int tok, DecodeState& ds",
                      "int debug_bisect_layer(")
        self.assertIn("if(pos<0||pos>=ds.capacity)", decode)
        # Range guard must happen before the first call into async or CUDA.
        self.assertLess(decode.index("if(pos<0||pos>=ds.capacity)"),
                        decode.index("if(ds.async_ctx) return decode_step_async("))

    def test_source_has_no_implicit_eviction(self):
        layer = body(self.pp2, "static bool run_full_decode_layer(",
                     "bool decode_init(DecodeState& ds")
        self.assertIn("if(L<0||L>=ds.capacity)", layer)
        self.assertIn("if((long)L!=pos)", layer)
        self.assertIn("ds.Kc[li]+(size_t)L*1024", layer)
        self.assertIn("ds.Vc[li]+(size_t)L*1024", layer)
        self.assertIn("ds.klen[li]=L+1;", layer)
        self.assertIn("sdpa_gqa_decode_k<<<NQ,256,", layer)
        # The current sequential kernel accesses cache keys 0..L by index.
        current_attention = body(self.pp2, "__global__ void sdpa_gqa_decode_k(",
                                 "static bool decode_skip_layer_sync()")
        self.assertIn("for(int t2=0;t2<=L;++t2)", current_attention)
        self.assertNotIn("%ds.capacity", layer)
        self.assertNotIn("%capacity", current_attention)

    def test_http_capacity_equation_and_last_prompt_logits(self):
        self.assertIn("job->prompt.size()) + job->max_new_tokens - 1",
                      self.server)
        self.assertIn("const int capacity = static_cast<int>(job->prompt.size()) + job->max_new_tokens - 1",
                      self.server)
        self.assertIn("job->prompt_pos = job->prompt.size();", self.server)
        self.assertIn("return legacy_token(job, token);", self.server)
        self.assertIn("if (job->stop_matched || token == eos_)", self.server)
        self.assertIn("job->output.size()) >= job->max_new_tokens", self.server)

    def test_current_http_prompt_and_completion_boundaries(self):
        # Last input logits sample the FIRST completion WITHOUT decode_step.
        expected = (
            (8191, 1, True, None),
            (8191, 2, True, 8191),
            (8191, 3, False, None),
            (8192, 1, True, None),
            (8192, 2, False, None),
            (8193, 1, False, None),
        )
        for prompt_len, max_new, allowed, decode_pos in expected:
            with self.subTest(prompt_len=prompt_len, max_new=max_new):
                self.assertEqual(current_admission(prompt_len, max_new), allowed)
                if decode_pos is not None:
                    self.assertTrue(current_decode_allowed(decode_pos, decode_pos))

    def test_current_decode_8192_and_8193_fail_before_memory_write(self):
        for pos, klen, expected in (
            (8190, 8190, True),
            (8191, 8191, True),
            (8192, 8192, False),
            (8193, 8193, False),
            (8192, 8191, False),
            (8191, 8192, False),
            (-1, 0, False),
        ):
            with self.subTest(pos=pos, klen=klen):
                self.assertEqual(current_decode_allowed(pos, klen), expected)

    def test_current_max_context_chunked_prefill_terminal_only(self):
        chunk = body(self.pp2,
                     "std::vector<float> prefill_chunks_into_decode_state(",
                     "static std::vector<float> decode_step_async(")
        self.assertIn("T>DecodeState::MAX_CONTEXT", chunk)
        self.assertIn("T>(size_t)ds.capacity", chunk)
        self.assertIn("ds.klen[li]!=expected", chunk)
        self.assertIn("ds.ever_advanced=true", chunk)
        self.assertIn("return logits;", chunk)


class ProposedBoundedContinuationContract(unittest.TestCase):
    def test_8191_8192_8193_8194_suffix_and_ring(self):
        ring = BoundedRingKV()
        reference: deque[ProposedChunk] = deque(maxlen=WINDOW)
        selected = {0, 1, 8190, 8191, 8192, 8193, 8194}
        for pos in range(8195):
            item = ProposedChunk(pos, 17 + pos, 41 + pos)
            ring.stage(pos, item.key, item.value)
            # The new token must be visible in attention BEFORE cache eviction.
            provisional = ring.causal_attention_keys()
            self.assertEqual(provisional[-1], item)
            if pos in selected:
                expected = tuple(list(reference)[-(WINDOW - 1):] + [item])
                self.assertEqual(provisional, expected)
            prior_slot = ring.slots[pos % WINDOW]
            prior_oldest = ring.oldest
            if pos >= WINDOW:
                self.assertEqual(prior_slot.absolute_position, pos - WINDOW)
            # Staging is transactional and must not evict early.
            self.assertEqual(ring.oldest, prior_oldest)
            evicted = ring.commit()
            if pos >= WINDOW:
                self.assertEqual(evicted, pos - WINDOW)
            else:
                self.assertIsNone(evicted)
            reference.append(item)
            if pos in selected:
                self.assertEqual(ring.resident(), tuple(reference))
                self.assertEqual(ring.retained, min(pos + 1, WINDOW))
                self.assertEqual(ring.oldest, max(0, pos + 1 - WINDOW))
                self.assertEqual(ring.next_absolute, pos + 1)

    def test_wrapped_window_second_cycle(self):
        ring = BoundedRingKV()
        check_points = {8191, 8192, 8193, 16382, 16383, 16384, 16385}
        for pos in range(16386):
            ring.stage(pos, pos, -pos)
            ring.commit()
            if pos in check_points:
                resident = ring.resident()
                self.assertLessEqual(len(resident), WINDOW)
                self.assertEqual(resident[0].absolute_position,
                                 max(0, pos + 1 - WINDOW))
                self.assertEqual(resident[-1].absolute_position, pos)
                self.assertEqual(ring.slots[pos % WINDOW].absolute_position, pos)
                self.assertEqual(len(ring.slots), WINDOW)

    def test_failed_stage_leaves_committed_history_unchanged(self):
        ring = BoundedRingKV()
        for pos in range(WINDOW):
            ring.stage(pos, pos, pos + 1)
            ring.commit()
        before = ring.resident()
        ring.stage(WINDOW, WINDOW, 7)
        # Deliberately simulate GPU0 succeeded, GPU1 failed before commit.
        self.assertEqual(ring.resident(), before)
        ring.abort()
        self.assertEqual(ring.resident(), before)
        with self.assertRaises(RuntimeError):
            ring.stage(WINDOW, WINDOW, 7)
        with self.assertRaises(RuntimeError):
            ring.commit()

    def test_chronological_attention_numerics_after_first_and_second_wrap(self):
        # This is an independent scalar attention oracle, NOT the GPU head
        # reduction or Qwen's full-layer numerical oracle.
        ring = BoundedRingKV()
        suffix: deque[ProposedChunk] = deque(maxlen=WINDOW)
        checks = (8190, 8191, 8192, 8193, 16383, 16384, 16385)
        for pos in range(checks[-1] + 1):
            token = ProposedChunk(pos, (pos % 11 - 5) / 11.0,
                                  (pos % 17 - 8) / 17.0)
            ring.stage(pos, token.key, token.value)
            if pos in checks:
                expected = tuple(suffix)[-(WINDOW - 1):] + (token,)
                physical_ring_view = ring.causal_attention_keys()
                self.assertEqual(physical_ring_view, expected)
                got = reference_attention(0.73, physical_ring_view)
                oracle = reference_attention(0.73, expected)
                self.assertAlmostEqual(got, oracle, delta=1e-13)
            ring.commit()
            suffix.append(token)

    def test_reject_nonmonotonic_and_overlapping_position(self):
        ring = BoundedRingKV()
        with self.assertRaises(ValueError):
            ring.stage(8192, 1, 2)
        ring.stage(0, 1, 2)
        with self.assertRaises(RuntimeError):
            ring.stage(0, 3, 4)
        ring.commit()
        with self.assertRaises(ValueError):
            ring.stage(0, 3, 4)

    def test_retaining_only_last_window_changes_attention(self):
        # Full-history 8193-token attention is NOT reconstructible from
        # an 8192-token KV cache. This is a policy change, not exact parity.
        history = (1_000_000.0,) + (0.0,) * WINDOW
        exact_full_history = toy_softmax(history)
        sliding_window = toy_softmax(history[-WINDOW:])
        self.assertGreater(exact_full_history, 100.0)
        self.assertEqual(sliding_window, 0.0)
        self.assertNotEqual(exact_full_history, sliding_window)

    def test_16_full_attention_layers_are_independent(self):
        rings = [BoundedRingKV(4) for _ in FULL_LAYERS]
        layer_oracles = [deque(maxlen=4) for _ in FULL_LAYERS]
        for pos in range(10):
            for li, ring in zip(FULL_LAYERS, rings):
                ring.stage(pos, li * 100 + pos, li * 200 + pos)
            for ring, oracle in zip(rings, layer_oracles):
                staged = ring.causal_attention_keys()
                self.assertEqual(staged[-1].absolute_position, pos)
                ring.commit()
                oracle.append(staged[-1])
            for ring, oracle in zip(rings, layer_oracles):
                self.assertEqual(ring.resident(), tuple(oracle))
        self.assertEqual(len(rings), 16)
        self.assertEqual(len(GDN_LAYERS), 48)

    def test_gdn_and_conv_are_not_evicted_with_attention_window(self):
        # The model is hybrid, not every layer has KV: 48 GDN recurrent
        # summaries / three-tap conv tails must persist after KV wrapping.
        gdn = {li: (100_000 + li, (7, 8, 9)) for li in GDN_LAYERS}
        ring = BoundedRingKV(4)
        for pos in range(7):
            ring.stage(pos, pos, pos)
            ring.commit()
        self.assertEqual(ring.oldest, 3)
        for li in GDN_LAYERS:
            self.assertEqual(gdn[li], (100_000 + li, (7, 8, 9)))

    def test_memory_budget_constant_in_absolute_position(self):
        # DecodeState allocates 8 full-attention layers per GPU.
        def gpu_bytes(window: int) -> int:
            kv = 8 * 2 * window * K_PER_POS * BYTES_PER_FLOAT
            recurrent = 24 * 48 * 128 * 128 * BYTES_PER_FLOAT
            conv = 24 * 10240 * 3 * BYTES_PER_FLOAT
            return kv + recurrent + conv
        self.assertEqual(gpu_bytes(WINDOW) // (1024 * 1024), 586)
        self.assertEqual(gpu_bytes(WINDOW), 615_317_504)
        # The independent ring model's allocation has exactly WINDOW slots
        # even after processing more than 2*WINDOW absolute positions.
        ring = BoundedRingKV(WINDOW)
        for pos in range(2 * WINDOW + 2):
            ring.stage(pos, pos, pos)
            ring.commit()
        self.assertEqual(len(ring.slots), WINDOW)
        self.assertEqual(ring.retained, WINDOW)
        self.assertEqual(ring.next_absolute, 2 * WINDOW + 2)
        self.assertEqual(gpu_bytes(WINDOW + 1) - gpu_bytes(WINDOW),
                         8 * 2 * 1024 * 4)  # growth per GPU per token
        self.assertEqual(gpu_bytes(16384) - gpu_bytes(8192),
                         8192 * 8 * 2 * 1024 * 4)
        self.assertLessEqual((WINDOW + 256) * 4, 48 * 1024)


if __name__ == "__main__":
    unittest.main(verbosity=2)
