#!/usr/bin/env python3
"""STRICTLY SCRATCH-ONLY experimental PP2 Q8_1/DP4A full decode probe.

This script NEVER edits production/repository source: input and output roots
MUST differ, and --source root is checked against a clean copy from git archive.
It modifies an isolated scratch PP2 to use genuine FP32 activations (before
BF16 cast) for each of the 14 T=1 decode GEMVs. Experimental only.

NO quality guarantee: independent full-model/long context and concurrency
remain gated. No GPU initialized or run by this script.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


INCLUDE = '#include "hermes/nvfp4_gemv_warp_variants.hpp"'
NEW_INCLUDE = '#include "hermes/nvfp4_gemv_q8_1_experimental.hpp"'

# Exactly 14 T=1 decode callsites: preserve original bf16 pointer and the
# original kernels, but expose the FP32 tensor from which BF16 was produced.
DECODE_PAIRS = [
    ("d_hb", "wQkv", "d_q1", "HID", "CC", "d_h"),
    ("d_hb", "wGate", "d_zl", "HID", "VDD", "d_h"),
    ("d_hb", "wBeta", "d_b", "HID", "HV", "d_h"),
    ("d_gb", "wOut", "d_core", "VDD", "HID", "d_g"),
    ("d_h2b", "wFg", "d_gg", "HID", "FFN", "d_h"),
    ("d_h2b", "wFu", "d_uu", "HID", "FFN", "d_h"),
    ("d_hidb", "wFd", "d_down", "FFN", "HID", "d_hid"),
    ("d_hb", "wQ", "d_qq", "HID", "NQQ", "d_h"),
    ("d_hb", "wK", "d_kk", "HID", "NKK", "d_h"),
    ("d_hb", "wV", "d_vv", "HID", "NKK", "d_h"),
    ("d_gb", "wO", "d_core", "NQHD", "HID", "d_G"),
    # second FFN site in full-attention decode is a distinct identical text
    ("d_h2b", "wFg", "d_gg", "HID", "FFN", "d_h"),
    ("d_h2b", "wFu", "d_uu", "HID", "FFN", "d_h"),
    ("d_hidb", "wFd", "d_down", "FFN", "HID", "d_hid"),
]

# Need 14 distinct callsites, but three FFN lines occur twice each.
# This grouping is intentional: match total count, never replace partially.
DECODE_RULES = {}
for bi, wn, y, k, n, f in DECODE_PAIRS:
    old = f"gemv_batch({bi},(uint8_t*){wn}->d,{y},{k},{n},T,stream);"
    new = f"gemv_batch({bi},(uint8_t*){wn}->d,{y},{k},{n},T,stream,{f});"
    if old in DECODE_RULES:
        assert DECODE_RULES[old][0] == new
        prev, count = DECODE_RULES[old]
        DECODE_RULES[old] = (prev, count + 1)
    else:
        DECODE_RULES[old] = (new, 1)

GEMV_SIGNATURE = (
    "static void gemv_batch(const uint16_t* d_in, const uint8_t* d_W, "
    "float* d_out, int H, int N, int T,\n"
    "                       cudaStream_t stream=0){"
)
NEW_SIGNATURE = GEMV_SIGNATURE.replace("cudaStream_t stream=0){",
                                       "cudaStream_t stream=0,const float* d_fp32=nullptr){")

PROBE = r'''
  // EXPERIMENTAL scratch-only, fail-closed, no production route.
  static const bool q8_dp4a = [] {
    const char* v=std::getenv("HERMES_PP2_Q8_DP4A_EXPERIMENTAL");
    return v && std::strcmp(v,"1")==0;
  }();
  if(q8_dp4a){
    if(!d_fp32 || T!=1 || H<=0 || H%256 || N<=0 || g_prefill_bf16_scope)
      throw std::runtime_error(
        "Q8 DP4A PP2 requires T1 decode, genuine FP32 input and K divisible by 256");
    const size_t bytes=hermes::q8_1_experimental::q8_1_scratch_bytes(T,H);
    if(bytes==0)throw std::runtime_error("Q8 DP4A PP2 invalid scratch request");
    hermes::q8_1_experimental::Q8_1Block* q8=nullptr;
    cudaError_t rc=cudaMallocAsync((void**)&q8,bytes,stream);
    if(rc!=cudaSuccess)
      throw std::runtime_error(std::string("Q8 scratch alloc: ")+cudaGetErrorString(rc));
    rc=hermes::q8_1_experimental::project_f32_q8_1_nvfp4_dp4a_batched(
      d_fp32,d_W,q8,d_out,T,N,H,stream);
    const cudaError_t free_rc=cudaFreeAsync(q8,stream);
    if(rc!=cudaSuccess || free_rc!=cudaSuccess)
      throw std::runtime_error(std::string("Q8 DP4A launch/free: ")+
        cudaGetErrorString(rc!=cudaSuccess?rc:free_rc));
    return;
  }
'''

CMAKE_TARGET = "target_link_libraries(hermes_kernels PUBLIC CUDA::cublas)"
NEW_CMAKE_TARGET = (
    "# EXPERIMENT ONLY: independently tested Q8_1 activation GEMV. This only\n"
    "# applies in an uninstalled scratch source and requires env opt-in.\n"
    "target_sources(hermes_kernels PRIVATE "
    "kernels/src/nvfp4_gemv_q8_1_experimental.cu)\n"
    + CMAKE_TARGET
)


def replace_once(source: str, old: str, new: str, count: int = 1) -> str:
    present = source.count(old)
    if present != count:
        raise ValueError(f"source contract differs: expected {count} matches, "
                         f"got {present}, anchor={old[:125]!r}")
    return source.replace(old, new)


def transform(pp2: str, cmake: str) -> tuple[str, str]:
    if NEW_INCLUDE in pp2 or "HERMES_PP2_Q8_DP4A_EXPERIMENTAL" in pp2:
        raise ValueError("input PP2 already modified by experimental adapter")
    pp2 = replace_once(pp2, INCLUDE, INCLUDE + "\n" + NEW_INCLUDE)
    pp2 = replace_once(pp2, GEMV_SIGNATURE, NEW_SIGNATURE)
    for old, (new, n) in DECODE_RULES.items():
        pp2 = replace_once(pp2, old, new, n)
    anchor = "  if(g_prefill_bf16_scope){"
    pp2 = replace_once(pp2, anchor, PROBE + anchor)
    cmake = replace_once(cmake, CMAKE_TARGET, NEW_CMAKE_TARGET)
    return pp2, cmake


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    source = args.source.resolve(strict=True)
    scratch = args.scratch.resolve(strict=True)
    if source == scratch or source in scratch.parents or scratch in source.parents:
        ap.error("--scratch and --source must be independent existing directories")
    # Never accept a repository or a production directory as a writable
    # destination. This tool is for explicitly disposable /tmp probes only.
    if scratch.parent != Path("/tmp") or not scratch.name.startswith(
        "h27-q8-e2e-candidate-"
    ):
        ap.error("--scratch must be a disposable /tmp/h27-q8-e2e-candidate-* directory")
    for root in (source, scratch):
        for rel in ("distributed/pp2_fwd.cu", "CMakeLists.txt"):
            if not (root / rel).is_file():
                ap.error(f"missing {root / rel}")
    for rel in ("distributed/pp2_fwd.cu", "CMakeLists.txt"):
        if (source / rel).read_bytes() != (scratch / rel).read_bytes():
            ap.error(f"scratch {rel} differs from input; refusing to overwrite preexisting edits")
    pp2, cmake = transform((source/"distributed/pp2_fwd.cu").read_text(),
                            (source/"CMakeLists.txt").read_text())
    if args.check_only:
        print("PASS isolated PP2 Q8 source contracts; no changes written")
        return 0
    for file, content in ((scratch/"distributed/pp2_fwd.cu",pp2),
                          (scratch/"CMakeLists.txt",cmake)):
        file.write_text(content, encoding="utf-8")
    print("EXPERIMENT_ONLY_PATCH_APPLIED scratch="+str(scratch))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print("FAIL_CLOSED Q8 scratch adapter: "+str(exc), file=sys.stderr)
        raise SystemExit(2)
