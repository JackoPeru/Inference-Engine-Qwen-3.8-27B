#!/usr/bin/env python3
"""CPU-only static guard against accidentally launching PP2 async work on stream 0.

This is a source-level tripwire, not a substitute for GPU event-race testing.
Run: python3 distributed/test_async_stream_contract.py
"""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
PP2 = (ROOT / "distributed" / "pp2_fwd.cu").read_text(encoding="utf-8")
GDN = (ROOT / "kernels" / "src" / "gdn_step.cu").read_text(encoding="utf-8")


def section(start: str, end: str) -> str:
    i = PP2.find(start)
    assert i >= 0, f"missing start anchor: {start}"
    j = PP2.find(end, i + len(start))
    assert j > i, f"missing end anchor: {end}"
    return PP2[i:j]


def verify_launches(name: str, source: str, stream_pattern: str) -> tuple[int, int]:
    matches = list(re.finditer(r"<<<([\s\S]*?)>>>", source))
    assert matches, f"{name}: no CUDA kernel launches found"
    for match in matches:
        config = match.group(1)
        assert re.search(stream_pattern, config), (
            f"{name}: launch config omits explicit request stream: {config}"
        )
    return len(matches), 0


def verify_layer(name: str, source: str) -> tuple[int, int, int]:
    kernels, _ = verify_launches(name, source, r"\bstream\b")
    gemvs = re.findall(r"gemv_batch\([^;]*;", source)
    assert gemvs, f"{name}: no GEMV calls found"
    for call in gemvs:
        assert re.search(r",\s*stream\s*\)\s*;", call), (
            f"{name}: GEMV doesn't propagate request stream: {call}"
        )
    gdns = re.findall(r"gdn_step_cuda\([^;]*;", source)
    for call in gdns:
        assert re.search(r",\s*stream\s*\)\s*;", call), (
            f"{name}: GDN step doesn't propagate request stream: {call}"
        )
    # There is one intentional sync kept for the original default-stream
    # fallback, never reachable for a nonzero per-request stream.
    syncs = source.count("cudaDeviceSynchronize()")
    assert syncs == 1 and "if(stream==0 && !decode_skip_layer_sync())" in source, (
        f"{name}: unexpected unconditional device-wide synchronization"
    )
    return kernels, len(gemvs), len(gdns)


def main() -> int:
    gdn = section("static bool run_gdn_decode_layer(",
                  "static bool run_full_decode_layer(")
    full = section("static bool run_full_decode_layer(", "bool decode_init(")
    async_decode = section("static std::vector<float> decode_step_async(",
                           "std::vector<float> decode_step(")
    gemv_wrapper = section("static void gemv_batch(",
                           "static bool use_head_bf16_warp(")

    counts = [verify_layer("GDN", gdn), verify_layer("FULL", full)]
    launch_count, _ = verify_launches(
        "async stage/head", async_decode, r"\b(?:ctx->stream0|ctx->stream1)\b"
    )
    assert "cudaDeviceSynchronize" not in async_decode, (
        "async decode contains a device-wide synchronization"
    )
    assert async_decode.count("cudaEventSynchronize(") == 2, (
        "must await GPU0 and GPU1 request completion events"
    )
    assert "cudaMemcpyAsync(ctx->host_hidden" in async_decode
    assert "cudaMemcpyAsync(ds.x1buf,ctx->host_hidden" in async_decode
    assert "cudaMemcpyAsync(ctx->host_logits" in async_decode
    assert "ctx->step_mutex" in async_decode
    assert "if(ds.async_ctx) return decode_step_async(tok,ds,pos,err);" in PP2
    assert "Nvfp4WarpExperiment::PackedU32x4,stream" in gemv_wrapper
    assert "Nvfp4GemvVariant::Warp4,stream" in gemv_wrapper
    assert "gdn_step_k<<<H,128,shm,stream>>>" in GDN

    kernels = sum(c[0] for c in counts)
    gemvs = sum(c[1] for c in counts)
    gdns = sum(c[2] for c in counts)
    assert kernels == 29, f"expected 29 decode-layer launches, found {kernels}; review updates"
    assert gemvs == 14, f"expected 14 decode-layer GEMVs, found {gemvs}; review updates"
    assert gdns == 1, f"expected 1 decode-layer GDN step call, found {gdns}; review updates"
    assert launch_count == 3, (
        f"expected 3 async-stage/head CUDA launches, found {launch_count}; review updates"
    )
    print(f"ASYNC_STREAM_CONTRACT PASS: {kernels} layer kernels, {gemvs} GEMVs, "
          f"{gdns} GDN call and {launch_count} stage/head kernels all stream-bound")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, OSError) as exc:
        print(f"ASYNC_STREAM_CONTRACT FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
