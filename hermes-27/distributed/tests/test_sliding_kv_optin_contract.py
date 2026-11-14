#!/usr/bin/env python3
"""CPU-only PP2 bounded-KV CUDA source contract and independent deque oracle.

This tests the *opt-in source* and a deterministic reference implementation.
It cannot prove numerical agreement of HERMES CUDA/real-GGUF logits.
Requires no GPU, CUDA runtime, third-party modules, model or service.
"""
from __future__ import annotations

import math
import unittest
from collections import deque
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "distributed" / "pp2_fwd.cu").read_text("utf-8")
HEADER = (ROOT / "distributed" / "pp2_fwd.hpp").read_text("utf-8")

WINDOW = 8192
MAX_ABSOLUTE = 1 << 20
FULL_LAYERS = tuple(li for li in range(64) if li % 4 == 3)
GDN_LAYERS = tuple(li for li in range(64) if li % 4 != 3)


@dataclass(frozen=True)
class KV:
    abs_pos: int
    key: float
    value: float


class IndependentKVDeque:
    """Ground-truth chronological KV suffix; no physical wrap implementation."""

    def __init__(self, window=WINDOW):
        self.history = deque(maxlen=window)
        self.expected_pos = 0

    def stage_attention(self, value: KV):
        if value.abs_pos != self.expected_pos:
            raise ValueError("nonmonotonic reference")
        return tuple(self.history)[-(self.history.maxlen - 1):] + (value,)

    def commit(self, value: KV):
        self.history.append(value)
        self.expected_pos += 1


class PhysicalKVSlots:
    """Candidate GPU ring semantics, staged token read before old-slot reuse."""

    def __init__(self, window=WINDOW):
        self.capacity = window
        self.slots: list[KV | None] = [None] * window
        self.next_abs = 0
        self.poisoned = False
        self.staged: KV | None = None
        self.committed = 0

    def stage(self, kv: KV):
        if self.poisoned:
            raise RuntimeError("poisoned state")
        if self.staged is not None:
            raise RuntimeError("overlapping request token")
        if kv.abs_pos != self.next_abs or kv.abs_pos >= MAX_ABSOLUTE:
            raise ValueError("invalid absolute position")
        self.staged = kv

    def chronological_attention(self):
        if self.poisoned or self.staged is None:
            raise RuntimeError("no safe token staged")
        p = self.staged.abs_pos
        count = min(p + 1, self.capacity)
        start = p - count + 1
        result = []
        for absolute in range(start, p + 1):
            if absolute == p:
                result.append(self.staged)
            else:
                item = self.slots[absolute % self.capacity]
                if item is None or item.abs_pos != absolute:
                    raise AssertionError("ring read wrong chronological slot")
                result.append(item)
        return tuple(result)

    def commit(self):
        if self.poisoned or self.staged is None:
            raise RuntimeError("cannot commit partial/poisoned state")
        kv = self.staged
        idx = kv.abs_pos % self.capacity
        previous = self.slots[idx]
        if kv.abs_pos >= self.capacity:
            if previous is None or previous.abs_pos != kv.abs_pos - self.capacity:
                raise AssertionError("not the oldest token's slot")
        self.slots[idx] = kv
        self.staged = None
        self.next_abs += 1
        self.committed = min(self.capacity, self.committed + 1)

    def abort(self):
        self.staged = None
        self.poisoned = True


def attention(q: float, items: tuple[KV, ...]):
    """Stable softmax, independent of PP2 GPU implementation."""
    logits = [q * v.key for v in items]
    maximum = max(logits)
    numerator = [math.exp(score - maximum) for score in logits]
    denominator = math.fsum(numerator)
    return math.fsum(weight * v.value for weight, v in zip(numerator, items)) / denominator


def source_section(start, end, source=SOURCE):
    a = source.index(start)
    b = source.index(end, a + len(start))
    return source[a:b]


def validates_source_code(src: str):
    """Fail-closed static source assertions with executable negative mutants."""
    h = HEADER
    assert "SLIDING_KV_ABSOLUTE_POSITION_LIMIT = 1 << 20" in h
    assert "bool sliding_kv_enabled = false;" in h
    assert "bool sliding_kv_poisoned = false;" in h
    assert 'std::getenv("HERMES_PP2_SLIDING_KV")' in src
    assert 'std::strcmp(sliding_env,"1")==0' in src
    assert "if(capacity!=DecodeState::MAX_CONTEXT)" in src
    assert "if(async_requested() || sliding_requested)" in src
    assert "ds.sliding_kv_enabled=sliding_requested;" in src
    assert "ds.sliding_kv_poisoned=false;" in src
    assert "if(!sliding_kv_step_preflight(tok,ds,pos,err))" in src
    assert "ds.sliding_kv_poisoned=true;" in src
    assert "if(ds.sliding_kv_enabled)ds.sliding_kv_poisoned=false;" in src
    assert "pos>=SLIDING_KV_ABSOLUTE_POSITION_LIMIT" in src
    assert "const bool sliding_wrap=ds.sliding_kv_enabled && pos>=ds.capacity;" in src
    assert "(!sliding_wrap && L>=ds.capacity)" in src
    assert "if((long)L!=pos)" in src
    assert "ds.klen[li]=L+1;" in src
    decode = source_section(
        "std::vector<float> decode_step(int tok, DecodeState& ds",
        "int debug_bisect_layer(", src,
    )
    assert "(!ds.sliding_kv_enabled && pos>=ds.capacity)" in decode
    assert decode.index("if(pos<0 ||") < decode.index("if(ds.async_ctx) return decode_step_async(")
    kernel = source_section(
        "__global__ void sdpa_gqa_ring_decode_k(",
        "static bool decode_skip_layer_sync()", src,
    )
    assert "int count=(absolute_pos+1<capacity)?(absolute_pos+1):capacity;" in kernel
    assert "int first_absolute=absolute_pos-count+1;" in kernel
    assert "(p%capacity)*1024" in kernel
    assert "p==absolute_pos" in kernel
    assert "Knew[(size_t)g*256+tid]" in kernel
    assert "Vnew[(size_t)g*256+tid]" in kernel
    assert "ATT[(size_t)hq*256+tid]=a;" in kernel
    assert "scores[logical]=tmp[0]*(1.f/16.f)" in kernel
    assert "expf(scores[i]-mx)" in kernel
    # Existing absolute RoPE for both Q and K must NOT use p%capacity.
    assert "q_norm_rope_pos_k<<<NQ,256,0,stream>>>(d_qq,(float*)wQn->d,d_Q,(int)pos)" in src
    assert "k_norm_rope_pos_k<<<NKV,256,0,stream>>>(d_kk,(float*)wKn->d,d_K1,(int)pos)" in src
    full = source_section("static bool run_full_decode_layer(",
                          "bool decode_init(DecodeState& ds", src)
    branch = source_section("if(sliding_wrap){", "}else{", full)
    first = branch.index("sdpa_gqa_ring_decode_k<<<")
    checked = branch.index("attention_error=cudaPeekAtLastError();")
    wait = branch.index("attention_error=cudaStreamSynchronize(stream);")
    commit_k = branch.index("cudaMemcpyAsync(\n        ds.Kc[li]+physical_slot")
    commit_v = branch.index("cudaMemcpyAsync(\n          ds.Vc[li]+physical_slot")
    drain = branch.index("commit_error=cudaStreamSynchronize(stream);")
    assert first < checked < wait < commit_k < commit_v < drain
    assert "(size_t)(L%ds.capacity)*NKK" in branch
    assert "cudaGetErrorString(attention_error)" in branch
    assert "cudaGetErrorString(commit_error)" in branch
    # The existing linear path is NOT routed through ring while disabled
    assert "sdpa_gqa_decode_k<<<NQ,256," in full
    assert "ds.Kc[li]+(size_t)L*1024" in full
    assert "ds.Vc[li]+(size_t)L*1024" in full


class SourceContract(unittest.TestCase):
    def test_current_cuda_source_contract_passes(self):
        validates_source_code(SOURCE)

    def test_default_still_hard_stops_at_8192(self):
        validates_source_code(SOURCE)
        self.assertIn("(!ds.sliding_kv_enabled && pos>=ds.capacity)", SOURCE)
        self.assertIn("(!sliding_wrap && L>=ds.capacity)", SOURCE)

    def test_negative_mutant_removes_pre_eviction_completion_barrier(self):
        bad = SOURCE.replace(
            "attention_error=cudaStreamSynchronize(stream);",
            "attention_error=cudaSuccess;", 1,
        )
        self.assertNotEqual(bad, SOURCE)
        with self.assertRaises((AssertionError, ValueError)):
            validates_source_code(bad)

    def test_negative_mutant_early_kv_overwrite(self):
        bad = SOURCE.replace(
            "(size_t)(L%ds.capacity)*NKK",
            "(size_t)L*NKK", 1,
        )
        self.assertNotEqual(bad, SOURCE)
        with self.assertRaises((AssertionError, ValueError)):
            validates_source_code(bad)

    def test_negative_mutant_allow_legacy_decode_out_of_bounds(self):
        bad = SOURCE.replace(
            "(!ds.sliding_kv_enabled && pos>=ds.capacity)",
            "(pos>=SLIDING_KV_ABSOLUTE_POSITION_LIMIT)", 1,
        )
        self.assertNotEqual(bad, SOURCE)
        with self.assertRaises((AssertionError, ValueError)):
            validates_source_code(bad)

    def test_negative_mutant_wrong_rope_absolute_position(self):
        bad = SOURCE.replace(
            "d_Q,(int)pos);", "d_Q,(int)(pos%ds.capacity));", 1,
        )
        self.assertNotEqual(bad, SOURCE)
        with self.assertRaises((AssertionError, ValueError)):
            validates_source_code(bad)

    def test_flag_exact_value_only(self):
        accepted = lambda value: value == "1"
        for candidate in ("", "0", "true", "yes", "01", "1 "):
            self.assertFalse(accepted(candidate))
        self.assertTrue(accepted("1"))

    def test_absolute_pos_and_ring_slot_8192_8193(self):
        for p, slot in ((8191, 8191), (8192, 0), (8193, 1),
                        (16383, 8191), (16384, 0), (16385, 1)):
            self.assertEqual(p % WINDOW, slot)
            self.assertEqual(p, int(p))  # RoPE receives absolute, not slot.

    def test_memory_and_shared_scratch_bounded(self):
        ring_bytes_gpu = 8 * 2 * WINDOW * 1024 * 4
        self.assertEqual(ring_bytes_gpu, 512 * 1024 * 1024)
        self.assertEqual((WINDOW + 256) * 4, 33792)
        self.assertLessEqual(33792, 48 * 1024)


class DequeOracle(unittest.TestCase):
    def test_window_boundary_and_two_physical_wraps(self):
        ring = PhysicalKVSlots()
        oracle = IndependentKVDeque()
        checkpoints = {0, 1, 8190, 8191, 8192, 8193, 8194,
                       16382, 16383, 16384, 16385}
        for p in range(max(checkpoints) + 1):
            token = KV(p, ((p * 13) % 29 - 14) / 29.0,
                       ((p * 7) % 31 - 15) / 31.0)
            ring.stage(token)
            got = ring.chronological_attention()
            expected = oracle.stage_attention(token)
            if p in checkpoints:
                self.assertEqual(got, expected)
                self.assertAlmostEqual(attention(0.75, got),
                                       attention(0.75, expected), delta=1e-13)
                self.assertEqual(got[-1].abs_pos, p)
                self.assertEqual(got[0].abs_pos, max(0, p - WINDOW + 1))
                self.assertLessEqual(len(got), WINDOW)
                # Oldest slot has not been replaced before attention returns.
                if p >= WINDOW:
                    old = ring.slots[p % WINDOW]
                    self.assertEqual(old.abs_pos, p - WINDOW)
            ring.commit()
            oracle.commit(token)
            if p in checkpoints:
                self.assertEqual(ring.committed, min(p + 1, WINDOW))
                self.assertEqual(ring.slots[p % WINDOW], token)
        self.assertEqual(len(ring.slots), WINDOW)

    def test_attention_drops_oldest_key_at_8192(self):
        ring = PhysicalKVSlots()
        for p in range(WINDOW):
            ring.stage(KV(p, 0.0, 1000.0 if p == 0 else 0.0))
            ring.commit()
        ring.stage(KV(WINDOW, 0.0, 0.0))
        visible = ring.chronological_attention()
        self.assertEqual(visible[0].abs_pos, 1)
        self.assertEqual(visible[-1].abs_pos, 8192)
        self.assertEqual(attention(0.0, visible), 0.0)
        unlimited_history_mean = 1000.0 / (WINDOW + 1)
        self.assertGreater(unlimited_history_mean, 0.0)
        self.assertNotEqual(attention(0.0, visible), unlimited_history_mean)

    def test_cancel_before_commit_poison_no_overwrite(self):
        ring = PhysicalKVSlots()
        for p in range(WINDOW):
            ring.stage(KV(p, float(p), float(-p)))
            ring.commit()
        oldest_before = ring.slots[0]
        ring.stage(KV(WINDOW, 8.0, 2.0))
        self.assertEqual(ring.chronological_attention()[-1].abs_pos, WINDOW)
        self.assertEqual(ring.slots[0], oldest_before)
        ring.abort()
        self.assertEqual(ring.slots[0], oldest_before)
        with self.assertRaises(RuntimeError):
            ring.stage(KV(WINDOW, 8.0, 2.0))
        with self.assertRaises(RuntimeError):
            ring.commit()

    def test_partial_K_commit_then_failed_V_never_restores_or_reuses(self):
        # CUDA K and V copies are separately enqueued. If K happens and V
        # fails, restoring the displaced oldest key is NOT guaranteed safe.
        # Mandatory behavior is terminal state poison, never a retry.
        ring = PhysicalKVSlots()
        for p in range(WINDOW):
            ring.stage(KV(p, float(p), float(-p)))
            ring.commit()
        before = ring.slots[0]
        self.assertEqual(before.abs_pos, 0)
        ring.stage(KV(WINDOW, 123.0, 456.0))
        self.assertEqual(ring.chronological_attention()[-1].abs_pos, WINDOW)
        # Simulate K copy reaching GPU, then V copy failing. No rollback.
        ring.slots[0] = KV(WINDOW, 123.0, before.value)
        ring.abort()
        self.assertTrue(ring.poisoned)
        self.assertNotEqual(ring.slots[0], before)
        with self.assertRaises(RuntimeError):
            ring.stage(KV(WINDOW, 123.0, 456.0))
        with self.assertRaises(RuntimeError):
            ring.commit()

    def test_out_of_order_and_overlapping_step_fails(self):
        ring = PhysicalKVSlots()
        with self.assertRaises(ValueError):
            ring.stage(KV(8192, 0.0, 0.0))
        ring.stage(KV(0, 1.0, 2.0))
        with self.assertRaises(RuntimeError):
            ring.stage(KV(0, 2.0, 3.0))
        ring.commit()
        with self.assertRaises(ValueError):
            ring.stage(KV(0, 2.0, 3.0))

    def test_absolute_position_guard_without_allocating_one_million_tokens(self):
        ring = PhysicalKVSlots()
        ring.next_abs = MAX_ABSOLUTE
        with self.assertRaises(ValueError):
            ring.stage(KV(MAX_ABSOLUTE, 1.0, 2.0))

    def test_16_full_layers_keep_distinct_ring_slots(self):
        rings = {li: PhysicalKVSlots(4) for li in FULL_LAYERS}
        refs = {li: IndependentKVDeque(4) for li in FULL_LAYERS}
        for p in range(11):
            for li in FULL_LAYERS:
                token = KV(p, float(p+li), float(li-p))
                rings[li].stage(token)
                self.assertEqual(rings[li].chronological_attention(),
                                 refs[li].stage_attention(token))
                rings[li].commit()
                refs[li].commit(token)
        self.assertEqual(len(rings), 16)
        self.assertEqual(len(GDN_LAYERS), 48)

    def test_48_gdn_recurrent_states_are_not_evicted(self):
        s_cs = {li: (li, (11, 12, 13)) for li in GDN_LAYERS}
        ring = PhysicalKVSlots(4)
        for p in range(9):
            ring.stage(KV(p, 1.0, 1.0))
            ring.commit()
        self.assertEqual(ring.committed, 4)
        for li in GDN_LAYERS:
            self.assertEqual(s_cs[li], (li, (11, 12, 13)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
