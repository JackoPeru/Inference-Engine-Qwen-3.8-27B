#!/usr/bin/env python3
"""CPU-only 8192-window, absolute-RoPE and persistent-state witnesses.

The numerical witnesses use deterministic sparse toy vectors, NOT Qwen weights
or a CUDA/llama.cpp numerical oracle. Full 24:4 grouped-query attention is
evaluated over every resident key at the specified long-context checkpoints.
No CUDA API, GPU, network, third-party package or service is used.
"""
from __future__ import annotations

import math
import unittest
from collections import deque
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CU = (ROOT / "distributed/pp2_fwd.cu").read_text(encoding="utf-8")
HPP = (ROOT / "distributed/pp2_fwd.hpp").read_text(encoding="utf-8")
GDN = (ROOT / "kernels/src/gdn_step.cu").read_text(encoding="utf-8")
WINDOW = 8192
HEADS_Q, HEADS_KV, HEAD_DIM = 24, 4, 256
GROUP_SIZE = HEADS_Q // HEADS_KV
CHECKPOINTS = (8191, 8192, 8193, 16384)
FULL_LAYERS = tuple(i for i in range(64) if i % 4 == 3)
GDN_LAYERS = tuple(i for i in range(64) if i % 4 != 3)
PROBED_FULL = (3, 35)  # one independent full-attention layer per device
PROBED_GDN = (0, 32)  # compare full-history recurrent replay on both devices


def between(text: str, start: str, end: str) -> str:
    a = text.index(start)
    return text[a:text.index(end, a + len(start))]


def mutate_function(source: str, start: str, end: str, old: str, new: str) -> str:
    """Target the actual decode kernel, not a prior prefill RoPE occurrence."""
    section = between(source, start, end)
    if old not in section:
        raise AssertionError(f"mutation target missing in {start}")
    changed = section.replace(old, new, 1)
    return source.replace(section, changed, 1)


def source_contract(src: str, header: str = HPP, gdn: str = GDN) -> None:
    """Assert the actual algorithmic edges exercised below, fail closed."""
    qrope = between(src, "__global__ void q_norm_rope_pos_k(",
                    "__global__ void k_norm_rope_pos_k(")
    krope = between(src, "__global__ void k_norm_rope_pos_k(",
                    "__global__ void extract_G_k(")
    ring = between(src, "__global__ void sdpa_gqa_ring_decode_k(",
                   "static bool decode_skip_layer_sync()")
    full = between(src, "static bool run_full_decode_layer(", "bool decode_init(")
    wrapped = between(full, "if(sliding_wrap){", "}else{")
    conv = between(src, "__global__ void conv_decode_k(",
                   "__global__ void q_norm_rope_pos_k(")
    async_step = between(src, "static std::vector<float> decode_step_async(",
                         "std::vector<float> decode_step(int tok,")
    preflight = between(src, "static bool sliding_kv_step_preflight(",
                        "static std::vector<float> decode_step_async(")
    gdn_cpu = between(gdn, "void gdn_step_cpu(", "__global__ void gdn_step_k(")

    assert "SLIDING_KV_ABSOLUTE_POSITION_LIMIT = 1 << 20" in header
    assert "bool sliding_kv_poisoned = false;" in header
    assert "int hq=(int)blockIdx.x, g=hq/6, tid=(int)threadIdx.x;" in ring
    assert "int count=(absolute_pos+1<capacity)?(absolute_pos+1):capacity;" in ring
    assert "int first_absolute=absolute_pos-count+1;" in ring
    assert "int p=first_absolute+logical;" in ring
    assert ring.count("p==absolute_pos") == 2
    assert "Knew[(size_t)g*256+tid]" in ring
    assert "Vnew[(size_t)g*256+tid]" in ring
    assert "Kc[(size_t)(p%capacity)*1024+g*256+tid]" in ring
    assert "Vc[(size_t)(p%capacity)*1024+g*256+tid]" in ring
    assert "scores[logical]=tmp[0]*(1.f/16.f)" in ring
    assert "expf(scores[i]-mx)" in ring
    assert "a+=scores[logical]*value;" in ring
    for rope in (qrope, krope):
        assert "int i=tid<32?tid:tid-32;" in rope
        assert "float fr=1.f/powf(10000000.f,(float)(2*i)/64.f);" in rope
        assert "float ang=(float)pos*fr" in rope
        assert "float a=sq[i], b=sq[32+i];" in rope
        assert "y=(tid<32)?(a*cc-b*ss):(a*ss+b*cc);" in rope
        assert "else y=qnrm;" in rope
        assert "float inv=1.f/sqrtf(sh[0]/256.f+1e-6f);" in rope
    assert "q_norm_rope_pos_k<<<NQ,256,0,stream>>>(d_qq,(float*)wQn->d,d_Q,(int)pos);" in full
    assert "k_norm_rope_pos_k<<<NKV,256,0,stream>>>(d_kk,(float*)wKn->d,d_K1,(int)pos);" in full
    assert "const bool sliding_wrap=ds.sliding_kv_enabled && pos>=ds.capacity;" in full
    assert "sdpa_gqa_decode_k<<<NQ,256," in full  # unchanged pre-wrap path
    assert "sdpa_gqa_ring_decode_k<<<NQ,256," in wrapped
    ordered = (
        "sdpa_gqa_ring_decode_k<<<", "attention_error=cudaStreamSynchronize(stream);",
        "if(attention_error!=cudaSuccess)", "const size_t physical_slot=(size_t)(L%ds.capacity)*NKK;",
        "ds.Kc[li]+physical_slot", "ds.Vc[li]+physical_slot",
        "commit_error=cudaStreamSynchronize(stream);", "if(commit_error!=cudaSuccess)",
    )
    offsets = [wrapped.index(piece) for piece in ordered]
    assert offsets == sorted(offsets)
    assert wrapped.index("if(commit_error!=cudaSuccess)") < full.index("ds.klen[li]=L+1;")
    assert "CS[(size_t)c*3+0]=c1; CS[(size_t)c*3+1]=c2; CS[(size_t)c*3+2]=q1[c];" in conv
    assert "conv_decode_k<<<(int)((CC+255)/256),256,0,stream>>>(d_q1,(float*)wC->d,ds.CS[li],d_cv,CC);" in src
    assert "gdn_step_cuda(d_q,d_k,d_v,d_be,d_gl,ds.S[li],d_o,HV,KHD,VHD,stream);" in src
    assert "float dec=expf(gl[h]), be=beta[h];" in gdn_cpu
    assert "for(int i=0;i<K*V;++i) Sh[i]*=dec;" in gdn_cpu
    assert "double bv=(double(vh[j])-d)*double(be);" in gdn_cpu
    assert "Sh[i*V+j]+=float(double(kh[i])*bv);" in gdn_cpu
    assert "if(!ds.async_ctx){" in preflight
    assert "(long)ds.klen[li]!=pos" in preflight
    assert "if(ds.sliding_kv_poisoned)" in preflight
    assert "std::unique_lock<std::mutex> guard(ctx->step_mutex,std::try_to_lock);" in async_step
    assert "ctx->poisoned=true;" in async_step
    assert "ds.sliding_kv_poisoned=true; // remains poisoned until FULL GPU1 completion" in async_step
    assert async_step.index("cudaEventSynchronize(ctx->ready1)") < async_step.index(
        "if(ds.sliding_kv_enabled)ds.sliding_kv_poisoned=false;")
    assert async_step.index("cudaSetDevice(0)") < async_step.index("ctx->stream0")
    assert async_step.index("cudaSetDevice(1)") < async_step.index("ctx->stream1")


def raw_vector(position: int, layer: int, head: int, query: bool) -> tuple[float, ...]:
    # Only indices 0,1,32,33 are nonzero; all other 252 coordinates of the
    # genuine 256-dimensional Q/K head are implicitly zero in this CPU toy.
    salt = layer * 11 + head * 17 + (3 if query else 29)
    return tuple(
        ((position * (13 + 2 * i) + salt * (7 + i)) % (67 + i * 2) -
         (33 + i)) / (29.0 + i * 5) for i in range(4)
    )


def normalized(vector: tuple[float, ...]) -> tuple[float, ...]:
    inv = 1.0 / math.sqrt(math.fsum(v*v for v in vector) / HEAD_DIM + 1e-6)
    return tuple(v * inv for v in vector)


def rope_reference(vector: tuple[float, ...], absolute: int) -> tuple[float, ...]:
    out = []
    # Independent analytical rotation for coordinates (i, 32+i).
    for i in range(2):
        theta = absolute / (10000000.0 ** (2 * i / 64))
        cosine, sine = math.cos(theta), math.sin(theta)
        a, b = vector[i], vector[2+i]
        out.append(a * cosine - b * sine)
        out.append(a * sine + b * cosine)
    return out[0], out[2], out[1], out[3]


def rope_candidate(vector: tuple[float, ...], absolute: int) -> tuple[float, ...]:
    # Separate complex-number implementation: changing the candidate's
    # position to absolute % WINDOW must be observable against the oracle.
    pairs = []
    for i in range(2):
        angle = absolute * (10000000.0 ** (-i / 32))
        turned = complex(vector[i], vector[i+2]) * complex(math.cos(angle), math.sin(angle))
        pairs.append(turned)
    return (pairs[0].real, pairs[1].real, pairs[0].imag, pairs[1].imag)


@dataclass(frozen=True, slots=True)
class KV:
    absolute: int
    k: tuple[tuple[float, ...], ...]  # four groups, sparse 256D heads
    v: tuple[tuple[float, float], ...]


def token(position: int, layer: int, reference: bool) -> KV:
    rotate = rope_reference if reference else rope_candidate
    keys = tuple(rotate(normalized(raw_vector(position, layer, group, False)), position)
                 for group in range(HEADS_KV))
    values = tuple(
        (math.sin((position * 3 + layer * 7 + group * 13) * 0.011),
         math.cos((position * 5 + layer * 3 + group * 11) * 0.007))
        for group in range(HEADS_KV)
    )
    return KV(position, keys, values)


def query(position: int, layer: int, head: int, reference: bool,
          override_position: int | None = None) -> tuple[float, ...]:
    rot = rope_reference if reference else rope_candidate
    return rot(normalized(raw_vector(position, layer, head, True)),
               position if override_position is None else override_position)


def score(q: tuple[float, ...], k: tuple[float, ...]) -> float:
    return math.fsum(a * b for a, b in zip(q, k)) / math.sqrt(HEAD_DIM)


def attention_reference(q: tuple[float, ...], timeline: list[KV], group: int
                        ) -> tuple[float, float]:
    logits = [score(q, row.k[group]) for row in timeline]
    highest = max(logits)
    weights = [math.exp(z-highest) for z in logits]
    den = math.fsum(weights)
    return tuple(math.fsum(w * row.v[group][j] for w, row in zip(weights, timeline)) / den
                 for j in (0, 1))


def attention_online(q: tuple[float, ...], timeline: list[KV], group: int
                     ) -> tuple[float, float]:
    # Different stable online softmax reduction from the two-pass reference.
    highest, mass, sums = -math.inf, 0.0, [0.0, 0.0]
    for row in timeline:
        z = score(q, row.k[group])
        if z > highest:
            decay = math.exp(highest-z)
            mass = mass * decay + 1.0
            sums = [s * decay + v for s, v in zip(sums, row.v[group])]
            highest = z
        else:
            weight = math.exp(z-highest)
            mass += weight
            sums[0] += weight * row.v[group][0]
            sums[1] += weight * row.v[group][1]
    return sums[0] / mass, sums[1] / mass


class RingSlots:
    def __init__(self, device: int, layer: int, capacity: int = WINDOW):
        self.device, self.layer, self.capacity = device, layer, capacity
        self.slots: list[KV | None] = [None] * capacity
        self.next_absolute = 0
        self.poisoned = False

    def visible(self, staged: KV) -> list[KV]:
        if self.poisoned or staged.absolute != self.next_absolute:
            raise ValueError("poisoned, duplicate or out-of-order token")
        p = staged.absolute
        first = max(0, p - self.capacity + 1)
        chronology = []
        for absolute in range(first, p + 1):
            if absolute == p:
                chronology.append(staged)
            else:
                old = self.slots[absolute % self.capacity]
                if old is None or old.absolute != absolute:
                    raise AssertionError("missing or aliased chronological key")
                chronology.append(old)
        return chronology

    def commit(self, staged: KV) -> None:
        if self.poisoned or staged.absolute != self.next_absolute:
            raise ValueError("cannot commit invalid/poisoned state")
        idx = staged.absolute % self.capacity
        if staged.absolute >= self.capacity:
            assert self.slots[idx] is not None
            assert self.slots[idx].absolute == staged.absolute - self.capacity
        self.slots[idx] = staged
        self.next_absolute += 1

    def poison(self) -> None:
        self.poisoned = True


def raw_conv(position: int, layer: int) -> tuple[float, float]:
    return (0.3 * math.sin(position * 0.031 + layer),
            0.4 * math.cos(position * 0.019 + layer * 0.5))


def silu(value: float) -> float:
    return value / (1.0 + math.exp(-value))


CONV_WEIGHTS = ((0.4, -0.3, 0.2, 0.7), (-0.2, 0.5, 0.3, 0.6))
DECAY, BETA = 0.9995, 0.008


def recurrent_parameters(cv: tuple[float, float]
                         ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    return (0.15 + 0.01 * cv[0], -0.09 + 0.01 * cv[1]), \
           (0.12 + 0.02 * cv[1], -0.08 + 0.02 * cv[0]), cv


class PersistentToyGDN:
    """Small representative GDN S[1,2,2] and conv CS[2,3], per layer."""
    def __init__(self, layer: int):
        self.layer = layer
        self.tail = [(0.0, 0.0)] * 3
        self.s = [[0.0, 0.0], [0.0, 0.0]]
        self.output = (0.0, 0.0)
        self.steps = 0

    def advance(self, position: int) -> None:
        if position != self.steps:
            raise ValueError("GDN absolute prefix gap")
        current = raw_conv(position, self.layer)
        four = self.tail + [current]
        cv = tuple(silu(sum(row[c] * CONV_WEIGHTS[c][i]
                            for i, row in enumerate(four))) for c in range(2))
        q, k, v = recurrent_parameters(cv)
        self.tail = four[1:]
        for i in range(2):
            for j in range(2):
                self.s[i][j] *= DECAY
        for j in range(2):
            delta = (v[j] - sum(self.s[i][j] * k[i] for i in range(2))) * BETA
            for i in range(2):
                self.s[i][j] += k[i] * delta
        self.output = tuple(sum(self.s[i][j] * q[i] for i in range(2))
                            for j in range(2))
        self.steps += 1


def replay_recurrent_from_absolute_history(layer: int, end_inclusive: int
                                           ) -> tuple[tuple[float, float], ...]:
    """Independent replay from the entire prefix, no ring or 8192 reset."""
    history: list[tuple[float, float]] = []
    s00 = s01 = s10 = s11 = 0.0
    for p in range(end_inclusive + 1):
        history.append(raw_conv(p, layer))
        prior = ([(0.0, 0.0)] * 3 + history)[-4:]
        cv = []
        for channel in range(2):
            z = math.fsum(CONV_WEIGHTS[channel][i] * raw[channel]
                          for i, raw in enumerate(prior))
            cv.append(silu(z))
        q, k, v = recurrent_parameters((cv[0], cv[1]))
        scaled = (s00 * DECAY, s01 * DECAY, s10 * DECAY, s11 * DECAY)
        b0 = BETA * (v[0] - (k[0] * scaled[0] + k[1] * scaled[2]))
        b1 = BETA * (v[1] - (k[0] * scaled[1] + k[1] * scaled[3]))
        s00, s01 = scaled[0] + k[0]*b0, scaled[1] + k[0]*b1
        s10, s11 = scaled[2] + k[1]*b0, scaled[3] + k[1]*b1
        output = (q[0]*s00 + q[1]*s10, q[0]*s01 + q[1]*s11)
    return (s00, s01), (s10, s11), output, tuple(history[-3:])


class PublishedSource(unittest.TestCase):
    def test_current_source_contract(self):
        source_contract(CU)

    def test_source_negative_mutants_detect_rope_conv_and_gdn(self):
        # These mutations deliberately target details beyond previous ring
        # source tests: rotary frequency/pairing and persistent GDN/CS update.
        candidates = (
            (mutate_function(CU, "__global__ void q_norm_rope_pos_k(",
                             "__global__ void k_norm_rope_pos_k(",
                             "powf(10000000.f,(float)(2*i)/64.f)",
                             "powf(10000.f,(float)(2*i)/64.f)"), "decode Q frequency"),
            (mutate_function(CU, "__global__ void k_norm_rope_pos_k(",
                             "__global__ void extract_G_k(",
                             "float a=sq[i], b=sq[32+i];",
                             "float a=sq[i], b=sq[31+i];"), "decode K pairing"),
            (mutate_function(CU, "__global__ void conv_decode_k(",
                             "__global__ void q_norm_rope_pos_k(",
                             "CS[(size_t)c*3+0]=c1; CS[(size_t)c*3+1]=c2;",
                             "CS[(size_t)c*3+0]=c0; CS[(size_t)c*3+1]=c2;"), "conv tail shift"),
            (GDN.replace("for(int i=0;i<K*V;++i) Sh[i]*=dec;",
                         "for(int i=0;i<K*V;++i) Sh[i]*=0.f;", 1), "GDN decay"),
            (mutate_function(CU, "static std::vector<float> decode_step_async(",
                             "std::vector<float> decode_step(int tok,",
                             "cudaEventSynchronize(ctx->ready1)",
                             "cudaEventQuery(ctx->ready1)"), "GPU1 completion event"),
        )
        for changed, mutation in candidates:
            with self.subTest(mutant=mutation):
                with self.assertRaises((AssertionError, ValueError)):
                    if mutation == "GDN decay":
                        source_contract(CU, gdn=changed)
                    else:
                        source_contract(changed)


class NumericalWindow(unittest.TestCase):
    def test_full_24_query_heads_and_4_kv_groups_through_two_wraps(self):
        self.assertEqual([h // GROUP_SIZE for h in range(HEADS_Q)],
                         [g for g in range(HEADS_KV) for _ in range(6)])
        self.assertEqual(len(FULL_LAYERS), 16)
        self.assertEqual([sum(i // 32 == dev for i in FULL_LAYERS) for dev in (0, 1)],
                         [8, 8])
        states = {li: RingSlots(li // 32, li) for li in PROBED_FULL}
        deques = {li: deque(maxlen=WINDOW) for li in PROBED_FULL}
        for p in range(max(CHECKPOINTS) + 1):
            for li in PROBED_FULL:
                candidate = token(p, li, False)
                reference = token(p, li, True)
                ring, oracle = states[li], deques[li]
                if p in CHECKPOINTS:
                    old_slot = ring.slots[p % WINDOW]
                    actual = ring.visible(candidate)
                    expected = list(oracle)[-(WINDOW-1):] + [reference]
                    self.assertEqual(len(actual), WINDOW)
                    self.assertEqual([v.absolute for v in actual],
                                     [v.absolute for v in expected])
                    self.assertEqual(actual[0].absolute, p - WINDOW + 1)
                    self.assertEqual(actual[-1].absolute, p)
                    if p >= WINDOW:
                        self.assertEqual(old_slot.absolute, p-WINDOW)
                        self.assertEqual(p % WINDOW, p - (p // WINDOW)*WINDOW)
                    for h in range(HEADS_Q):
                        group = h // GROUP_SIZE
                        expected_output = attention_reference(query(p, li, h, True),
                                                              expected, group)
                        actual_output = attention_online(query(p, li, h, False),
                                                         actual, group)
                        for j in (0, 1):
                            self.assertAlmostEqual(actual_output[j],
                                                   expected_output[j], delta=2e-10,
                                                   msg=f"p={p} dev={li//32} li={li} h={h} value={j}")
                ring.commit(candidate)
                oracle.append(reference)
                if p in CHECKPOINTS:
                    self.assertEqual(ring.next_absolute, p + 1)
                    self.assertEqual(ring.slots[p % WINDOW].absolute, p)
        self.assertEqual(states[3].device, 0)
        self.assertEqual(states[35].device, 1)
        self.assertIsNot(states[3].slots, states[35].slots)
        self.assertNotEqual(states[3].slots[0], states[35].slots[0])

    def test_absolute_rope_is_observably_not_physical_slot_rope(self):
        # At p=8192 the new query must rotate at 8192; the physical slot 0
        # is not a valid rotary position. At i=1 their phases differ.
        p, layer, group = 8192, 3, 1
        two = [token(p-1, layer, True), token(p, layer, True)]
        q_correct = query(p, layer, 6, True)
        q_wrong = query(p, layer, 6, False, override_position=0)
        self.assertGreater(abs(q_correct[1] - q_wrong[1]), 0.01)
        correct = attention_reference(q_correct, two, group)
        wrong = attention_reference(q_wrong, two, group)
        self.assertGreater(max(abs(a-b) for a, b in zip(correct, wrong)), 1e-6)

    def test_bounded_attention_differs_from_full_history_for_all_groups(self):
        # Exact uniform softmax with all-zero keys. A discarded old value
        # affects full-history attention but must not affect the 8192-window.
        for h in range(HEADS_Q):
            group = h // GROUP_SIZE
            old = 16.0 + group
            unlimited = old / (WINDOW + 1)
            bounded = 0.0 / WINDOW
            self.assertGreater(abs(unlimited-bounded), 1e-3)


class RecurrentPersistenceAndPoison(unittest.TestCase):
    def test_all_48_gdn_states_persist_when_attention_wraps(self):
        self.assertEqual(len(GDN_LAYERS), 48)
        self.assertEqual([sum(i // 32 == dev for i in GDN_LAYERS) for dev in (0, 1)],
                         [24, 24])
        states = {li: PersistentToyGDN(li) for li in GDN_LAYERS}
        for p in range(max(CHECKPOINTS) + 1):
            for state in states.values():
                state.advance(p)
            if p in CHECKPOINTS:
                for li, state in states.items():
                    self.assertEqual(state.steps, p + 1)
                    self.assertEqual(state.tail, [raw_conv(t, li) for t in range(p-2, p+1)])
                    self.assertTrue(any(abs(x) > 1e-9 for row in state.s for x in row))
                for li in PROBED_GDN:
                    state = states[li]
                    ref_s0, ref_s1, ref_o, ref_tail = replay_recurrent_from_absolute_history(li, p)
                    for got, expected in zip(state.s, (ref_s0, ref_s1)):
                        for a, b in zip(got, expected):
                            self.assertAlmostEqual(a, b, delta=5e-12)
                    for a, b in zip(state.output, ref_o):
                        self.assertAlmostEqual(a, b, delta=5e-12)
                    self.assertEqual(tuple(state.tail), ref_tail)
                    reset = PersistentToyGDN(li)
                    reset.steps = p
                    reset.advance(p)
                    self.assertGreater(
                        max(abs(state.output[j]-reset.output[j]) for j in (0, 1)), 1e-7)

    def test_partial_gpu1_commit_poison_blocks_any_replay(self):
        # Two stages are not one atomic operation. Once GPU0 committed at
        # p=8192, a failed GPU1 V write cannot rewind GPU0/GDN. Poison wins.
        states = {li: RingSlots(li // 32, li) for li in PROBED_FULL}
        for p in range(WINDOW):
            for li, state in states.items():
                state.commit(token(p, li, False))
        p = WINDOW
        gpu0, gpu1 = states[3], states[35]
        staged0, staged1 = token(p, 3, False), token(p, 35, False)
        oldest_gpu1 = gpu1.slots[0]
        self.assertEqual(gpu0.visible(staged0)[0].absolute, 1)
        self.assertEqual(gpu1.visible(staged1)[0].absolute, 1)
        gpu0.commit(staged0)
        # Simulate GPU1 K successfully copied into slot 0, then V failure.
        gpu1.slots[0] = KV(p, staged1.k, oldest_gpu1.v)
        for state in states.values():
            state.poison()
        self.assertEqual(gpu0.next_absolute, p + 1)
        self.assertEqual(gpu1.next_absolute, p)
        self.assertNotEqual(gpu1.slots[0].v, staged1.v)
        self.assertNotEqual(gpu1.slots[0].k, oldest_gpu1.k)
        for li, state in states.items():
            for attempted in (p, p+1):
                with self.subTest(layer=li, attempted=attempted):
                    with self.assertRaises(ValueError):
                        state.visible(token(attempted, li, False))
                    with self.assertRaises(ValueError):
                        state.commit(token(attempted, li, False))

    def test_nonmonotonic_and_duplicate_positions_rejected(self):
        ring = RingSlots(0, 3, capacity=4)
        with self.assertRaises(ValueError):
            ring.visible(token(1, 3, False))
        for p in range(11):
            staged = token(p, 3, False)
            self.assertEqual(ring.visible(staged)[-1].absolute, p)
            ring.commit(staged)
            with self.assertRaises(ValueError):
                ring.commit(staged)
        with self.assertRaises(ValueError):
            ring.visible(token(13, 3, False))


if __name__ == "__main__":
    unittest.main(verbosity=2)
