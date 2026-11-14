#!/usr/bin/env python3
"""CPU-only negative witnesses for seeded/chunked prefill invariants.

These are independent small arithmetic models and source-contract guards.
They do not execute CUDA, compare the real GGUF, or certify model numerics.
"""

from __future__ import annotations

import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
PP2 = (ROOT / "distributed" / "pp2_fwd.cu").read_text(encoding="utf-8")
KERNEL = (ROOT / "kernels" / "src" / "nvfp4_prefill_exact.cu").read_text(encoding="utf-8")


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise AssertionError(reason)


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def coefficients(conv: float, absolute_pos: int) -> tuple[float, float, float, float, float]:
    q = 0.75 * conv + 0.11
    k = 0.33 * conv + 0.05 * math.cos(absolute_pos * 0.017)
    v = math.sin(conv * 0.7) + 0.03
    alpha = 0.71 + 0.11 * math.cos(absolute_pos * 0.009)
    beta = 0.15 + 0.04 * math.sin(absolute_pos * 0.021)
    return q, k, v, alpha, beta


def causal_attention(q: float, keys: list[float], values: list[float]) -> float:
    scores = [q * k for k in keys]
    shift = max(scores)
    weights = [math.exp(score - shift) for score in scores]
    return math.fsum(w * v for w, v in zip(weights, values)) / math.fsum(weights)


def ffn_for_position(residual: float) -> float:
    gate = residual * 1.3 + 0.1
    up = residual * 0.7 - 0.03
    return residual + 0.5 * up * gate * sigmoid(gate)


def one_shot_reference(inputs: list[float]) -> tuple[list[float], float, list[float]]:
    """Absolute-index convolution and closed-form recurrent sum, no chunk state."""
    w = (0.19, -0.07, 0.24, 0.38)
    conv = [math.fsum(w[j] * inputs[t - 3 + j]
                      for j in range(4) if t - 3 + j >= 0)
            for t in range(len(inputs))]
    params = [coefficients(value, t) for t, value in enumerate(conv)]
    keys = [p[1] for p in params]
    values = [p[2] for p in params]
    outputs = []
    last_state = 0.0
    for t, (q, _k, _v, _alpha, _beta) in enumerate(params):
        # The direct finite sum is independent of the streaming recurrence.
        last_state = math.fsum(
            params[i][4] * params[i][1] * params[i][2] *
            math.prod(params[j][3] for j in range(i + 1, t + 1))
            for i in range(t + 1)
        )
        attention = causal_attention(q, keys[:t + 1], values[:t + 1])
        residual = inputs[t] + 0.4 * q * last_state + 0.2 * attention
        outputs.append(ffn_for_position(residual))
    return outputs, last_state, inputs[-3:]


def chunked_stateful(inputs: list[float], chunk_size: int,
                     mutation: str = "") -> tuple[list[float], float, list[float], int]:
    """Carry conv-tail, GDN state and KV append across separate calls."""
    w = (0.19, -0.07, 0.24, 0.38)
    tail = [0.0, 0.0, 0.0]
    s = 0.0
    keys: list[float] = []
    values: list[float] = []
    outputs: list[float] = []
    for start in range(0, len(inputs), chunk_size):
        part = inputs[start:start + chunk_size]
        if start and mutation == "reset_convolution":
            tail = [0.0, 0.0, 0.0]
        if start and mutation == "reset_gdn":
            s = 0.0
        if start and mutation == "reset_kv":
            keys.clear()
            values.clear()
        preceding = tail + part
        for local, x in enumerate(part):
            t = start + local
            conv = math.fsum(w[j] * preceding[local + j] for j in range(4))
            pos = local if mutation == "reset_position" else t
            q, k, v, alpha, beta = coefficients(conv, pos)
            s = alpha * s + beta * k * v
            keys.append(k)
            values.append(v)
            attention = causal_attention(q, keys, values)
            residual = x + 0.4 * q * s + 0.2 * attention
            if mutation == "last_ffn_only" and local + 1 < len(part):
                outputs.append(residual)
            else:
                outputs.append(ffn_for_position(residual))
        tail = preceding[-3:]
    return outputs, s, tail, len(keys)


def test_stateful_oracle_and_fault_detection() -> int:
    rng = random.Random(20261114)
    comparisons = 0
    for size in (1, 2, 3, 63, 64, 65, 128, 129):
        x = [rng.uniform(-1.0, 1.0) for _ in range(size)]
        ref, s_ref, last = one_shot_reference(x)
        for tile in (1, 3, 7, 64):
            actual, s, tail, klen = chunked_stateful(x, tile)
            require(len(actual) == size and klen == size, f"output/KV count T={size} tile={tile}")
            require(all(close(a, b) for a, b in zip(actual, ref)),
                    f"stateful numeric mismatch T={size} tile={tile}")
            require(close(s, s_ref), f"GDN recurrence mismatch T={size} tile={tile}")
            require(tail == ([0.0] * max(0, 3 - size) + last),
                    f"conv tail mismatch T={size} tile={tile}")
            comparisons += 1

    x = [random.Random(407).uniform(-1.0, 1.0)] + [math.sin(i * 0.23) for i in range(1, 129)]
    ref, _, _ = one_shot_reference(x)
    for fault in ("reset_convolution", "reset_gdn", "reset_kv", "reset_position", "last_ffn_only"):
        bad, _, _, _ = chunked_stateful(x, 64, fault)
        require(any(not close(a, b) for a, b in zip(bad, ref)),
                f"negative witness did not detect {fault}")
    return comparisons + 5


def test_gemm_tiling_and_negative_stride() -> int:
    """Independent GEMM expression vs tiled row-major Y[T,N]."""
    rng = random.Random(982)
    n, k = 129, 64  # K matches the real GGUF block-divisibility constraint
    weights = [[rng.randint(-14, 14) / 8 for _ in range(k)] for _ in range(n)]
    checks = 0
    for t_count in (64, 65, 129):
        activations = [[rng.randint(-20, 20) / 16 for _ in range(k)] for _ in range(t_count)]
        reference = [math.fsum(x * w for x, w in zip(activations[t], weights[row]))
                     for t in range(t_count) for row in range(n)]
        for rows_per_tile in (1, 7, 128):
            tiled = [float("nan")] * (t_count * n)
            wrong_stride = [float("nan")] * (t_count * n)
            for base in range(0, n, rows_per_tile):
                rows = min(rows_per_tile, n - base)
                for t in range(t_count):
                    for local in range(rows):
                        dot = math.fsum(x * w for x, w in zip(activations[t], weights[base + local]))
                        tiled[t * n + base + local] = dot
                        wrong_stride[t * rows + base + local] = dot
            require(tiled == reference, f"column-major ldc=N tile offset error T={t_count} rows={rows_per_tile}")
            require(any(not close(a, b) for a, b in zip(wrong_stride, reference)),
                    f"negative ldc=tileRows witness failed T={t_count}")
            checks += 2
    return checks


def test_source_contract() -> int:
    statements = {
        "first GDN chunk seeds conv tail": "if(seed && pos0==0)",
        "later GDN chunk reads history": "conv_silu_chunk_k<<<",
        "later GDN chunk appends history": "append_conv_tail_k<<<",
        "GDN state lives in DecodeState": "d_S=seed->S[li]",
        "per-position GDN update": "for(int t=0;t<T;++t) gdn_step_cuda(",
        "full attention K/V append offsets": "seed->Kc[li]+(size_t)pos0*NKK",
        "full attention V append offsets": "seed->Vc[li]+(size_t)pos0*NKK",
        "KV length commit after completed layer": "if(seed) seed->klen[li]=pos0+T",
        "FFN gate all positions": "gemv_batch(d_h2b,(uint8_t*)wFg->d,d_gg,HID,FFN,T)",
        "FFN up all positions": "gemv_batch(d_h2b,(uint8_t*)wFu->d,d_uu,HID,FFN,T)",
        "FFN down all positions": "gemv_batch(d_hidb,(uint8_t*)wFd->d,d_down,FFN,HID,T)",
        "last-head-only after all layers": "const int output_T=last_only?1:T",
        "last-head token pointer": "d_x1+(size_t)(T-1)*HID",
        "T65+ chunk offsets": "(int)offset,true);",
        "large chunk BF16 fail closed": "chunked prefill BF16 disabled: full-model numeric gate failed",
    }
    for name, source in statements.items():
        require(source in PP2, f"source contract missing: {name}")
    snippet = KERNEL[KERNEL.index("ScratchResult ReusableBF16WeightScratch::enqueue("):
                     KERNEL.index("ScratchResult ReusableBF16WeightScratch::shutdown_locked(")]
    event_wait = snippet.index("cudaStreamWaitEvent(stream,ready_event_,0)")
    unpack = snippet.index("dequantize_gguf_to_bf16_vector(")
    gemm = snippet.index("st=cublasGemmEx(")
    event_record = snippet.rindex("ce=cudaEventRecord(ready_event_,stream)")
    require(event_wait < unpack < gemm < event_record,
            "shared scratch must wait, decode, GEMM, and record in stream order")
    require("&beta,y_fp32+base,CUDA_R_32F,N," in snippet,
            "tiled FP32 output must use full N leading dimension with base offset")
    require("cublasSetStream(handle_,stream)" in snippet,
            "per-slot cuBLAS stream binding missing")
    return len(statements) + 3


def main() -> None:
    state = test_stateful_oracle_and_fault_detection()
    gemm = test_gemm_tiling_and_negative_stride()
    source = test_source_contract()
    print(f"PREFILL_STATE_INVARIANTS_CPU_PASS state={state} tiled={gemm} "
          f"source={source} GPU_GGUF_PARITY=NOT_TESTED")


if __name__ == "__main__":
    main()
