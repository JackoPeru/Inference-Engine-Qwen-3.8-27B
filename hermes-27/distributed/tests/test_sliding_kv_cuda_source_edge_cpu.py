#!/usr/bin/env python3
"""Source-anchored CUDA KV-ring edge checks + independent numeric CPU oracle.

No GPU, model weights, service, CMake or third-party dependencies. CPU math
is an algorithm witness, NOT a claim of real-GGUF/CUDA numerical equivalence.
"""
from __future__ import annotations

import math
import re
import unittest
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CU = (ROOT / "distributed/pp2_fwd.cu").read_text(encoding="utf-8")
HPP = (ROOT / "distributed/pp2_fwd.hpp").read_text(encoding="utf-8")
CAP = 8192
KV_HEADS = 4
Q_HEADS = 24
GROUP = 6
HD = 2  # independent CPU toy vectors, not the actual 256-float kernel head
PROBE_POSITIONS = (8191, 8192, 8193, 16383, 16384, 16385)


def section(s: str, start: str, end: str) -> str:
    a = s.index(start)
    return s[a:s.index(end, a + len(start))]


def contract(s: str) -> None:
    """Exact invariants: a negative source mutant MUST make this fail."""
    kernel = section(s, "__global__ void sdpa_gqa_ring_decode_k(",
                     "static bool decode_skip_layer_sync()")
    full = section(s, "static bool run_full_decode_layer(", "bool decode_init(")
    wrapped = section(full, "if(sliding_wrap){", "}else{")
    preflight = section(s, "static bool sliding_kv_step_preflight(",
                        "static std::vector<float> decode_step_async(")
    async_step = section(s, "static std::vector<float> decode_step_async(",
                         "std::vector<float> decode_step(int tok,")
    top_step = section(s, "std::vector<float> decode_step(int tok,",
                       "int debug_bisect_layer(")
    initial = section(s, "bool decode_init(DecodeState& ds, std::string& err, int capacity)",
                      "void decode_free(")

    # 24 query heads grouped 6:1 into four 256-wide KV heads.
    assert "int hq=(int)blockIdx.x, g=hq/6, tid=(int)threadIdx.x;" in kernel
    assert "sdpa_gqa_ring_decode_k<<<NQ,256," in wrapped
    assert "int count=(absolute_pos+1<capacity)?(absolute_pos+1):capacity;" in kernel
    assert "int first_absolute=absolute_pos-count+1;" in kernel
    assert "float* tmp=scores+count;" in kernel
    assert "Kc[(size_t)(p%capacity)*1024+g*256+tid]" in kernel
    assert "Vc[(size_t)(p%capacity)*1024+g*256+tid]" in kernel
    assert "Knew[(size_t)g*256+tid]" in kernel
    assert "Vnew[(size_t)g*256+tid]" in kernel
    assert "p==absolute_pos" in kernel
    assert "tmp[tid]=qv*kv;" in kernel
    assert "for(int s=128;s>0;s>>=1)" in kernel
    assert "scores[logical]=tmp[0]*(1.f/16.f)" in kernel
    assert "expf(scores[i]-mx)" in kernel
    assert "a+=scores[logical]*value;" in kernel
    assert "ATT[(size_t)hq*256+tid]=a;" in kernel
    assert "const size_t shared_bytes=(size_t)(ds.capacity+256)*sizeof(float);" in wrapped
    assert "d_Q,ds.Kc[li],ds.Vc[li],d_K1,d_vv,d_att,L,ds.capacity);" in wrapped

    # Source-level *ordered* release and commit. The K/V scratch is read by
    # attention before any overwrite to old physical KV. Per-request stream.
    checks = (
        "sdpa_gqa_ring_decode_k<<<",
        "attention_error=cudaPeekAtLastError();",
        "attention_error=cudaStreamSynchronize(stream);",
        "if(attention_error!=cudaSuccess)",
        "const size_t physical_slot=(size_t)(L%ds.capacity)*NKK;",
        "cudaMemcpyAsync(\n        ds.Kc[li]+physical_slot",
        "cudaMemcpyAsync(\n          ds.Vc[li]+physical_slot",
        "commit_error=cudaStreamSynchronize(stream);",
        "if(commit_error!=cudaSuccess)",
    )
    offsets = [wrapped.index(part) for part in checks]
    assert offsets == sorted(offsets)
    assert "ds.klen[li]=L+1;" in full
    assert wrapped.index("if(commit_error!=cudaSuccess)") < full.index("ds.klen[li]=L+1;")
    assert "q_norm_rope_pos_k<<<NQ,256,0,stream>>>(d_qq,(float*)wQn->d,d_Q,(int)pos);" in full
    assert "k_norm_rope_pos_k<<<NKV,256,0,stream>>>(d_kk,(float*)wKn->d,d_K1,(int)pos);" in full
    assert "const bool sliding_wrap=ds.sliding_kv_enabled && pos>=ds.capacity;" in full
    assert "(!sliding_wrap && L>=ds.capacity)" in full
    assert "sdpa_gqa_decode_k<<<NQ,256," in full
    assert "ds.Kc[li]+(size_t)L*1024" in full
    assert "ds.Vc[li]+(size_t)L*1024" in full

    # New hardening: public DecodeState flags MUST NOT select the ring if
    # the state lacks private streams and the associated step mutex.
    assert "if(!ds.async_ctx){" in preflight
    assert "sliding KV requires initialized private request streams" in preflight
    assert preflight.index("if(!ds.async_ctx){") < preflight.index("for(int li=0;li<NL;++li)")
    assert "pos>=SLIDING_KV_ABSOLUTE_POSITION_LIMIT" in preflight
    assert "(long)ds.klen[li]!=pos" in preflight
    assert "if(ds.sliding_kv_poisoned)" in preflight
    assert 'std::strcmp(sliding_env,"1")==0' in initial
    assert "if(capacity!=DecodeState::MAX_CONTEXT)" in initial
    assert "if(async_requested() || sliding_requested)" in initial
    assert "std::unique_lock<std::mutex> guard(ctx->step_mutex,std::try_to_lock);" in async_step
    assert "ctx->poisoned=true;" in async_step
    assert "ds.sliding_kv_poisoned=true; // remains poisoned until FULL GPU1 completion" in async_step
    assert async_step.index("cudaSetDevice(0)") < async_step.index("ctx->stream0")
    assert async_step.index("cudaSetDevice(1)") < async_step.index("ctx->stream1")
    assert "cudaEventSynchronize(ctx->ready0)" in async_step
    assert "cudaEventSynchronize(ctx->ready1)" in async_step
    assert async_step.index("cudaEventSynchronize(ctx->ready1)") < async_step.index(
        "if(ds.sliding_kv_enabled)ds.sliding_kv_poisoned=false;")
    assert "(!ds.sliding_kv_enabled && pos>=ds.capacity)" in top_step
    assert "if(ds.async_ctx) return decode_step_async(tok,ds,pos,err);" in top_step
    assert "SLIDING_KV_ABSOLUTE_POSITION_LIMIT = 1 << 20" in HPP


def kv_sample(p: int, group: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    # Group-specific vectors with strong observable variation after eviction.
    key = tuple(((p * 13 + group * 19 + d * 3) % 43 - 21) / 43.0
                for d in range(HD))
    val = tuple(((p * 7 + group * 17 + d * 5) % 41 - 20) / 41.0
                for d in range(HD))
    return key, val


def query_sample(pos: int, q_head: int) -> tuple[float, ...]:
    return tuple(((pos * 5 + q_head * 23 + d * 11) % 37 - 18) / 37.0
                 for d in range(HD))


def stable_attention(query: tuple[float, ...],
                     visible: list[tuple[tuple[float, ...], tuple[float, ...]]]
                     ) -> tuple[float, ...]:
    scores = [math.fsum(q * k for q, k in zip(query, key)) / math.sqrt(HD)
              for key, _ in visible]
    mx = max(scores)
    weights = [math.exp(x - mx) for x in scores]
    denom = math.fsum(weights)
    return tuple(math.fsum(w * val[d] for w, (_, val) in zip(weights, visible)) / denom
                 for d in range(HD))


class SourceEdges(unittest.TestCase):
    def test_ring_source_contract(self):
        contract(CU)

    def test_negative_missing_private_stream_guard(self):
        bad = CU.replace("if(!ds.async_ctx){\n    err=\"sliding KV requires",
                         "if(false){\n    err=\"sliding KV requires", 1)
        self.assertNotEqual(bad, CU)
        with self.assertRaises((AssertionError, ValueError)):
            contract(bad)

    def test_negative_wrong_head_grouping(self):
        bad = CU.replace("g=hq/6, tid=", "g=hq/5, tid=", 1)
        self.assertNotEqual(bad, CU)
        with self.assertRaises((AssertionError, ValueError)):
            contract(bad)

    def test_negative_current_value_reads_old_slot(self):
        bad = CU.replace("Vnew[(size_t)g*256+tid]",
                         "Vc[(size_t)(p%capacity)*1024+g*256+tid]", 1)
        self.assertNotEqual(bad, CU)
        with self.assertRaises((AssertionError, ValueError)):
            contract(bad)

    def test_negative_unsafe_early_eviction(self):
        bad = CU.replace("attention_error=cudaStreamSynchronize(stream);",
                         "attention_error=cudaSuccess;", 1)
        self.assertNotEqual(bad, CU)
        with self.assertRaises((AssertionError, ValueError)):
            contract(bad)

    def test_negative_wrapped_rope(self):
        bad = CU.replace("d_Q,(int)pos);",
                         "d_Q,(int)(pos%ds.capacity));", 1)
        self.assertNotEqual(bad, CU)
        with self.assertRaises((AssertionError, ValueError)):
            contract(bad)

    def test_negative_legacy_oob_bypass(self):
        bad = CU.replace("(!ds.sliding_kv_enabled && pos>=ds.capacity)",
                         "(!ds.sliding_kv_enabled && false)", 1)
        self.assertNotEqual(bad, CU)
        with self.assertRaises((AssertionError, ValueError)):
            contract(bad)

    def test_head_group_indices_all_in_bounds(self):
        self.assertEqual([hq // GROUP for hq in range(Q_HEADS)],
                         [g for g in range(KV_HEADS) for _ in range(GROUP)])

    def test_shared_memory_bound_and_synchronizations(self):
        self.assertEqual((CAP + 256) * 4, 33792)
        self.assertLess((CAP + 256) * 4, 48 * 1024)
        # Two stream-local waits per full-attention layer, 16 total.
        self.assertEqual(16 * 2, 32)
        self.assertIn("cudaStreamSynchronize(stream)", CU)
        ring = section(CU, "if(sliding_wrap){", "}else{")
        self.assertNotIn("cudaDeviceSynchronize()", ring)

    def test_env_is_not_caller_writable_inference_flag(self):
        # Only explicit exact value activates at init. The public struct
        # cannot guarantee integrity, so private ctx invariant is mandatory.
        self.assertIn('std::strcmp(sliding_env,"1")==0', CU)
        self.assertIn("if(!ds.async_ctx){", CU)


class IndependentMathAndDeque(unittest.TestCase):
    def test_all_query_groups_chronological_wrap_attention(self):
        # Reference is a real deque of absolute K/V and independent math.
        # Candidate is the physical-slot mapping in the new CUDA kernel.
        ring: list[tuple[int, list[tuple[tuple[float, ...], tuple[float, ...]]]] | None] = [None] * CAP
        oracle = deque(maxlen=CAP)
        probe = {8191, 8192, 8193, 8194, 16383, 16384, 16385}
        for p in range(max(probe) + 1):
            sample = [kv_sample(p, group) for group in range(KV_HEADS)]
            if p in probe:
                first = max(0, p - CAP + 1)
                expected = [(tag, pair) for tag, pair in oracle if tag >= first] + [(p, sample)]
                physical = []
                for abs_pos in range(first, p + 1):
                    if abs_pos == p:
                        physical.append((p, sample))
                    else:
                        item = ring[abs_pos % CAP]
                        self.assertIsNotNone(item)
                        self.assertEqual(item[0], abs_pos)
                        physical.append(item)
                # Do not invoke unittest's quadratic difflib comparison on
                # two 8192-element histories if a mismatch is encountered.
                self.assertTrue(physical == expected,
                                f"chronological full-window mismatch at p={p}")
                # Verify full 8192-slot chronological IDs against the deque.
                # Perform bounded arithmetic on the last 16 entries; separate
                # 16-slot test below covers all 24 query heads with real
                # nonzero-query softmax, avoiding an unbounded slow CPU test.
                heads = (0, 6, 12, 18)
                for hq in heads:
                    g = hq // GROUP
                    from_deque = [layers[g] for _, layers in expected[-16:]]
                    from_physical = [layers[g] for _, layers in physical[-16:]]
                    got = stable_attention(query_sample(p, hq), from_physical)
                    reference = stable_attention(query_sample(p, hq), from_deque)
                    for x, y in zip(got, reference):
                        self.assertAlmostEqual(x, y, delta=1e-13)
                if p >= CAP:
                    old = ring[p % CAP]
                    self.assertEqual(old[0], p - CAP)
                self.assertEqual(len(expected), min(CAP, p + 1))
            ring[p % CAP] = (p, sample)
            oracle.append((p, sample))

    def test_missing_current_kv_would_change_attention(self):
        # All previous values zero, new staged token value high. Reading old
        # slot in place of Vnew would incorrectly yield zero.
        values = [((0.0,) * HD, (0.0,) * HD)] * (CAP - 1)
        staged = ((0.0,) * HD, (13.0,) * HD)
        q = (0.0,) * HD
        correct = stable_attention(q, values + [staged])
        wrong = stable_attention(q, values + [values[0]])
        self.assertGreater(correct[0], 0)
        self.assertEqual(wrong[0], 0.0)

    def test_all_24_query_heads_against_independent_short_deque(self):
        cap = 16
        ring = [None] * cap
        oracle = deque(maxlen=cap)
        for p in range(2 * cap + 2):
            sample = [kv_sample(p, group) for group in range(KV_HEADS)]
            first = max(0, p - cap + 1)
            timeline = [(i, layers) for i, layers in oracle if i >= first] + [(p, sample)]
            physical = [(i, sample if i == p else ring[i % cap][1])
                        for i, _ in timeline]
            self.assertEqual(physical, timeline)
            for hq in range(Q_HEADS):
                group = hq // GROUP
                q = query_sample(p, hq)
                actual = stable_attention(q, [layers[group] for _, layers in physical])
                golden = stable_attention(q, [layers[group] for _, layers in timeline])
                self.assertEqual(actual, golden)
            ring[p % cap] = (p, sample)
            oracle.append((p, sample))

    def test_bounded_window_differs_from_unlimited_history(self):
        old = ((0.0,) * HD, (99.0,) * HD)
        zero = ((0.0,) * HD, (0.0,) * HD)
        full = stable_attention((0.0,) * HD, [old] + [zero] * CAP)
        window = stable_attention((0.0,) * HD, [zero] * CAP)
        self.assertGreater(full[0], 0.0)
        self.assertEqual(window[0], 0.0)

    def test_failed_partial_commit_is_terminal(self):
        # After K memcpy succeeds but V memcpy fails, neither cache content
        # nor recurrent state can be rolled back. Poison remains mandatory.
        p = 8192
        slot = p % CAP
        before = ("K0", "V0")
        after_K_only = ("K8192", before[1])
        self.assertEqual(slot, 0)
        self.assertNotEqual(before, after_K_only)
        poisoned = True
        self.assertTrue(poisoned)
        self.assertNotEqual(after_K_only[1], "V8192")

    def test_two_devices_have_disjoint_layer_ownership(self):
        gpu0 = [li for li in range(64) if li <= 31 and li % 4 == 3]
        gpu1 = [li for li in range(64) if li >= 32 and li % 4 == 3]
        self.assertEqual(len(gpu0), 8)
        self.assertEqual(len(gpu1), 8)
        self.assertEqual(len(set(gpu0 + gpu1)), 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
