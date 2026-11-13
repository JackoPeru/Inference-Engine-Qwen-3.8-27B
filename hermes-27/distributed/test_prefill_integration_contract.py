#!/usr/bin/env python3
"""CPU-only regression tripwire for PP2 seeded batched prefill.

This verifies the convolution-tail mapping independently and checks source
contracts. It cannot prove CUDA/GDN/KV numerical parity; the real GGUF test
must run on exclusive GPU before enabling production use.
"""

import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
PP2 = (ROOT / "distributed" / "pp2_fwd.cu").read_text(encoding="utf-8")
HEADER = (ROOT / "distributed" / "pp2_fwd.hpp").read_text(encoding="utf-8")
EXACT = (
    ROOT / "kernels" / "include" / "hermes" / "nvfp4_prefill_exact.hpp"
).read_text(encoding="utf-8")


def guard(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def direct_tail(qkv, length, channel):
    return [
        0.0 if length - 3 + j < 0 else qkv[length - 3 + j][channel]
        for j in range(3)
    ]


def sequential_tail(qkv, channel):
    cs = [0.0, 0.0, 0.0]
    for row in qkv:
        cs = [cs[1], cs[2], row[channel]]
    return cs


def check_tail() -> int:
    cases = (1, 2, 3, 4, 11, 64)
    for length in cases:
        data = [[float((t * 13 + k * 3) % 29) for k in range(17)]
                for t in range(length)]
        for channel in range(17):
            expected = sequential_tail(data, channel)
            actual = direct_tail(data, length, channel)
            guard(expected == actual,
                  f"conv tail mismatch T={length}, channel={channel}")
    return len(cases) * 17


def function(start, end):
    first = PP2.index(start)
    last = PP2.index(end, first + len(start))
    return PP2[first:last]


def check_source_contract():
    guard("prefill_into_decode_state(" in HEADER, "seeded public API absent")
    # With chunked prefill, both K and V must append at the absolute
    # per-layer position. When pos0=0 these checks cover the original
    # single-pass seeded API; for pos0>0 they prevent a silent overwrite.
    guard("cudaMemcpy(seed->Kc[li]+(size_t)pos0*NKK,d_K," in PP2,
          "K seed must append post-RoPE keys at pos0")
    guard("cudaMemcpy(seed->Vc[li]+(size_t)pos0*NKK,d_V," in PP2,
          "V seed must append values at pos0")
    guard("seed->klen[li]!=pos0" in PP2,
          "KV append must require previous length pos0")
    guard("seed->klen[li]=pos0+T;" in PP2,
          "host KV length must advance by chunk length")
    guard("d_S=seed->S[li];" in PP2, "GDN must use persistent request state")
    guard("seed_conv_tail_k<<<" in PP2, "conv queue tail kernel is absent")
    guard("seed->CS[li],d_qkv,T,CC" in PP2, "conv tail source is wrong")
    guard("if(!seed) cudaMemset(d_S,0" in PP2,
          "seeded GDN recurrent state is being overwritten")
    guard("if(private_staging)" in PP2, "request-local pinned staging absent")
    guard("temporary_pinned.ptr" in PP2, "private pinned ownership absent")
    guard("if(!private_staging) transfer_lock=std::unique_lock<std::mutex>(g_xfer_mu)" in PP2,
          "legacy shared pinned staging must acquire transfer mutex")
    boundary = function("// xfer GPU0->host->GPU1 pinned",
                        "// GPU1\n  for(int li=SPLIT_LAST0+1;")
    guard(boundary.index("if(!private_staging) transfer_lock=") <
          boundary.index("cudaMemcpy(stage,d_x0") <
          boundary.index("cudaMemcpy(d_x1,stage") <
          boundary.index("if(transfer_lock.owns_lock()) transfer_lock.unlock()"),
          "legacy pinned transfer lock must cover both boundary copies")
    guard("state_lock=std::unique_lock<std::mutex>" in PP2,
          "async per-request state mutex isn't protected")
    guard("if(ds.klen[li]!=0)" in PP2, "fresh-state precondition missing")
    guard("T<1 || T>64 || T>(size_t)ds.capacity" in PP2,
          "T64/capacity bounds missing")
    guard("free_bytes<512ULL*1024*1024" in PP2,
          "512 MiB fail-closed VRAM preflight absent")
    guard("const bool bf16=(option && std::strcmp(option,\"1\")==0)" in PP2,
          "BF16 experimental opt-in must require exact 1")
    guard("hermes::prefill_exact::ReusableBF16WeightScratch slot[2]" in PP2,
          "worker2 tiled BF16 scratch not integrated")
    guard("slot[dev].enqueue(w,x,y,T,N,K,0)" in PP2,
          "worker2 tiled BF16 GEMM enqueue not integrated")
    guard("slot[dev].shutdown()" in PP2,
          "BF16 scratch not safely drained before release")
    guard("if(max_abs>1.e-3 || argmax_mismatches!=0)" in PP2,
          "BF16 full-logit max/error and argmax gate absent")
    guard("if(bf16 && winner_ref!=winner_bf16)" in PP2,
          "BF16 per-token argmax accumulation absent")
    guard("decode_free(ds);" in function(
        "std::vector<float> prefill_into_decode_state(",
        "static std::vector<float> decode_step_async("),
        "failed seeded state must be invalidated")
    guard("if(ds.ever_advanced || (ctx && ctx->poisoned))" in PP2,
          "partially mutated/poisoned state must be rejected")
    guard("bf16_weight_scratch_bytes" in EXACT and "gemm_bf16_fp32" in EXACT,
          "worker-2 prefill exact API is missing")


def main():
    cases = check_tail()
    check_source_contract()
    print(f"PREFILL_INTEGRATION_CONTRACT PASS: {cases} CPU tail cases, "
          "GDN/KV/opt-in/memory/quality source gates")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, ValueError, OSError) as error:
        print(f"PREFILL_INTEGRATION_CONTRACT FAIL: {error}", file=sys.stderr)
        sys.exit(1)
