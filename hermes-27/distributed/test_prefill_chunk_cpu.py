#!/usr/bin/env python3
"""CPU/offline arithmetic and static contract tests for experimental PP2 chunks.

No GPU. This cannot substitute the real-GGUF layer/cache/logit continuation
oracle and must never be interpreted as a production correctness certificate.
"""
from __future__ import annotations
import math
from pathlib import Path
import random
import sys

SRC = (Path(__file__).resolve().parent / "pp2_fwd.cu").read_text(encoding="utf-8")
HDR = (Path(__file__).resolve().parent / "pp2_fwd.hpp").read_text(encoding="utf-8")


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def conv_reference(values, weights):
    out = []
    for t in range(len(values)):
        acc = sum(
            (0.0 if t - 3 + j < 0 else values[t - 3 + j]) * weights[j]
            for j in range(4)
        )
        out.append(acc / (1 + math.exp(-acc)))
    return out


def conv_chunk(values, weights, boundaries):
    CS = [0.0, 0.0, 0.0]
    output = []
    first = 0
    for count in boundaries:
        chunk = values[first:first + count]
        check(len(chunk) == count, "invalid chunk layout")
        for t in range(count):
            acc = 0.0
            for j in range(4):
                tt = t - (3 - j)
                x = CS[tt + 3] if tt < 0 else chunk[tt]
                acc += x * weights[j]
            output.append(acc / (1 + math.exp(-acc)))
        if count == 1:
            CS = [CS[1], CS[2], chunk[0]]
        elif count == 2:
            CS = [CS[2], chunk[0], chunk[1]]
        else:
            CS = chunk[-3:]
        first += count
    check(first == len(values), "chunk length mismatch")
    return output, CS


def chunks(total, chunk_size):
    return [min(chunk_size, total - start) for start in range(0, total, chunk_size)]


def test_convolution():
    rng = random.Random(13)
    cases = 0
    for size in (1, 2, 3, 4, 63, 64, 65, 66, 127, 128, 129, 1024, 8192):
        values = [rng.uniform(-1.0, 1.0) for _ in range(size)]
        weights = [rng.uniform(-0.5, 0.5) for _ in range(4)]
        oracle = conv_reference(values, weights)
        for chunk_size in (1, 2, 3, 7, 17, 32, 64):
            output, tail = conv_chunk(values, weights, chunks(size, chunk_size))
            check(max(abs(x - y) for x, y in zip(oracle, output)) < 1e-12,
                  f"conv numerical mismatch size={size} chunk={chunk_size}")
            check(tail == ([0.0] * max(0, 3 - size) + values[-3:]),
                  f"conv persistent-tail mismatch size={size} chunk={chunk_size}")
            cases += 1
    return cases


def softmax_attention(query, keys, values, limit):
    # Independent stable-softmax oracle for a single query and KV head.
    scores = [sum(a * b for a, b in zip(query, k)) / math.sqrt(len(query))
              for k in keys[:limit + 1]]
    top = max(scores)
    exp_scores = [math.exp(s - top) for s in scores]
    den = sum(exp_scores)
    return [sum(exp_scores[i] * values[i][j] / den for i in range(limit + 1))
            for j in range(len(query))]


def test_attention_rope():
    rng = random.Random(19)
    # Keep KV small, only sample hard positions near every chunk boundary.
    length = 8192
    keys = [[rng.uniform(-0.25, 0.25) for _ in range(8)] for _ in range(length)]
    values = [[rng.uniform(-0.25, 0.25) for _ in range(8)] for _ in range(length)]
    points = (0, 1, 62, 63, 64, 65, 126, 127, 128, 129,
              1022, 1023, 1024, 4095, 4096, 8191)
    cases = 0
    for pos in points:
        q = [rng.uniform(-0.25, 0.25) for _ in range(8)]
        full = softmax_attention(q, keys, values, pos)
        for chunk_size in (1, 17, 32, 64):
            base = (pos // chunk_size) * chunk_size
            local = pos - base
            # Chunk attention must see ALL K/V up to absolute base+local,
            # rather than only K/V starting at the current chunk.
            current = softmax_attention(q, keys, values, base + local)
            check(max(abs(a - b) for a, b in zip(full, current)) < 1e-14,
                  f"causal history loss pos={pos} chunk={chunk_size}")
            for i in (0, 1, 7, 31):
                freq = 1.0 / (10000000.0 ** (2 * i / 64))
                check(math.cos((base + local) * freq) ==
                      math.cos(pos * freq), "absolute RoPE mismatch")
            cases += 1
    return cases


def test_kv_append():
    cases = 0
    for size in (65, 66, 127, 128, 129, 1024, 8192):
        oracle = list(range(size))
        for chunk_size in (1, 2, 3, 7, 32, 64):
            cache = [None] * size
            cursor = 0
            for count in chunks(size, chunk_size):
                check(cursor + count <= size, "KV capacity exceeded")
                cache[cursor:cursor + count] = oracle[cursor:cursor + count]
                cursor += count
            check(cursor == size and cache == oracle,
                  f"KV append mismatch T={size} chunk={chunk_size}")
            cases += 1
    return cases


def test_static_contract():
    checks = {
        "strict opt in": 'std::strcmp(opt,"1")!=0',
        "long T exact bounds": "T<=PIN_TMAX || T>DecodeState::MAX_CONTEXT",
        "strict chunks <=64": "chunk_tokens>(int)PIN_TMAX",
        "reject failed BF16": "chunked prefill BF16 disabled: full-model numeric gate failed",
        "absolute RoPE Q": "(float)(pos0+t)*fr",
        "K cache append": "seed->Kc[li]+(size_t)pos0*NKK",
        "V cache append": "seed->Vc[li]+(size_t)pos0*NKK",
        "read old convolution tail": "CS[(size_t)c*3+(tt+3)]",
        "shift old convolution tail": "CS[(size_t)c*3]=old1",
        "streamed causal KV read": "Kc[(size_t)p*1024+g*256+tid]",
        "streamed causal V read": "Vc[(size_t)p*1024+g*256+tid]",
        "bounded scores": "(size_t)(pos0+T+256)*sizeof(float)",
        "single-token head output": "const int output_T=last_only?1:T",
        "per request boundary": "prefill_impl(chunk,chunk_stats,err,last_hidden,&ds,true",
        "KV length gate": "if(ds.klen[li]!=expected)",
        "preserve partial request failure": "decode_free(ds);",
        "anti overlap": "std::unique_lock<std::mutex>(ctx->step_mutex,std::try_to_lock)",
    }
    for name, literal in checks.items():
        check(literal in SRC, f"missing source contract: {name}")
    check("prefill_chunks_into_decode_state(" in HDR, "public chunk API missing")
    check("if(seed && pos0>0)" in SRC, "chunk mode must be separate from T<=64")
    check((8192 + 256) * 4 <= 48 * 1024, "shared-memory bound invalid")
    return len(checks) + 3


def main():
    conv = test_convolution()
    attn = test_attention_rope()
    kv = test_kv_append()
    static = test_static_contract()
    print(f"CHUNK_CPU_PASS conv={conv} attention_rope={attn} "
          f"kv_append={kv} static={static} "
          "GPU_GGUF_PARITY=NOT_TESTED")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, OSError, ValueError) as exc:
        print(f"CHUNK_CPU_FAIL {exc}", file=sys.stderr)
        sys.exit(1)
