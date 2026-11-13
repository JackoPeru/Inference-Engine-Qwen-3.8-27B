#!/usr/bin/env python3
"""Fail-closed comparison of HERMES-27 raw FP32 full-vocabulary logit traces.

Generate with hermes_generate --dump-logits-dir EXISTING_DIRECTORY. Each
prompt_P_step_S.f32 contains exactly 248320 float32 values little-endian.
The two runs MUST use the same model, prompts, generation length and seed.
This checks scalar-vs-optimized numerical parity; it is not an independent
llama.cpp oracle or proof of exact model semantics.
"""

from __future__ import annotations

import argparse
from array import array
import heapq
import math
from pathlib import Path
import re
import sys


VOCAB = 248320
NAME = re.compile(r"^prompt_\d+_step_\d+\.f32$")


def fail(message: str) -> None:
    raise ValueError(message)


def list_traces(path: Path) -> dict[str, Path]:
    if not path.is_dir():
        fail(f"not a directory: {path}")
    files = {p.name: p for p in path.iterdir() if NAME.fullmatch(p.name) and p.is_file()}
    if not files:
        fail(f"no prompt_*_step_*.f32 files in {path}")
    return files


def read_logits(path: Path, vocab: int) -> array:
    if sys.byteorder != "little":
        fail("raw .f32 traces require a little-endian host")
    if array("f").itemsize != 4:
        fail("raw .f32 traces require IEEE-754 32-bit float storage")
    if path.stat().st_size != vocab * 4:
        fail(f"{path.name}: expected {vocab * 4} bytes, got {path.stat().st_size}")
    data = array("f")
    with path.open("rb") as stream:
        data.fromfile(stream, vocab)
    if len(data) != vocab or any(not math.isfinite(v) for v in data):
        fail(f"{path.name}: truncated or non-finite logits")
    return data


def compare_pair(baseline: array, candidate: array) -> dict[str, float | int]:
    if len(baseline) != len(candidate):
        fail("trace arrays have different lengths")
    max_abs = 0.0
    total_sq = 0.0
    total_abs = 0.0
    b_top = c_top = 0
    b_max = c_max = float("-inf")
    for i, (a, b) in enumerate(zip(baseline, candidate)):
        d = abs(float(a) - float(b))
        max_abs = max(max_abs, d)
        total_sq += d * d
        total_abs += d
        if a > b_max:
            b_max, b_top = a, i
        if b > c_max:
            c_max, c_top = b, i
    return {
        "max_abs": max_abs, "rmse": math.sqrt(total_sq / len(baseline)),
        "mean_abs": total_abs / len(baseline),
        "argmax_baseline": b_top, "argmax_candidate": c_top,
    }


def softmax_metrics(baseline: array, candidate: array) -> dict[str, float | int]:
    """Diagnostic only: softmax KL(P_baseline || P_candidate) and top-k stability.

    Log-sum-exp is computed with FP64 host accumulation. These measurements
    cannot replace independent model/long-context quality verification.
    """
    if len(baseline) != len(candidate) or not baseline:
        fail("softmax input arrays must be nonempty and equally sized")
    if any(not math.isfinite(v) for v in baseline) or any(not math.isfinite(v) for v in candidate):
        fail("softmax input has non-finite values")
    a_max, b_max = max(baseline), max(candidate)
    logza = a_max + math.log(math.fsum(math.exp(v - a_max) for v in baseline))
    logzb = b_max + math.log(math.fsum(math.exp(v - b_max) for v in candidate))
    kl = math.fsum(math.exp(a - logza) * ((a - logza) - (b - logzb))
                   for a, b in zip(baseline, candidate))
    k = min(10, len(baseline))
    top_a = heapq.nlargest(k, range(len(baseline)), key=baseline.__getitem__)
    top_b = heapq.nlargest(k, range(len(candidate)), key=candidate.__getitem__)
    return {
        "kl": max(0.0, kl),  # tiny negative rounding is not a negative KL
        "top10_overlap": len(set(top_a).intersection(top_b)),
        "topk": k,
        "margin": baseline[top_a[0]] - baseline[top_a[1]] if k > 1 else 0.0,
    }


def self_test() -> None:
    a = array("f", [1.0, 4.0, -3.0])
    b = array("f", [1.0001, 4.0, -2.9])
    result = compare_pair(a, b)
    assert result["argmax_baseline"] == 1 == result["argmax_candidate"]
    assert 0.09 < result["max_abs"] < 0.11
    assert 0.05 < result["rmse"] < 0.06
    p = softmax_metrics(a, a)
    assert p["kl"] < 1e-12 and p["top10_overlap"] == 3
    p2 = softmax_metrics(a, b)
    assert p2["kl"] > 0 and p2["top10_overlap"] == 3
    print("SELF-TEST PASS: compare all floats, argmax, absolute and RMS errors")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", nargs="?", type=Path)
    parser.add_argument("candidate", nargs="?", type=Path)
    parser.add_argument("--vocab", type=int, default=VOCAB)
    parser.add_argument("--max-abs", type=float, default=0.002)
    parser.add_argument("--max-rmse", type=float, default=0.0005)
    parser.add_argument("--report-softmax", action="store_true",
                        help="also report KL divergence, top-10 overlap and top-1 margin")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.baseline is None or args.candidate is None:
        parser.error("baseline and candidate directories required")
    if (args.vocab <= 0 or not math.isfinite(args.max_abs)
            or not math.isfinite(args.max_rmse)
            or args.max_abs < 0 or args.max_rmse < 0):
        parser.error("vocab > 0; tolerances must be finite and nonnegative")
    before = list_traces(args.baseline)
    after = list_traces(args.candidate)
    if set(before) != set(after):
        missing = sorted(set(before) - set(after))
        extra = sorted(set(after) - set(before))
        fail(f"mismatched trace sets; missing={missing[:5]} extra={extra[:5]}")
    worst_abs = worst_rmse = 0.0
    failures: list[str] = []
    for name in sorted(before):
        baseline = read_logits(before[name], args.vocab)
        candidate = read_logits(after[name], args.vocab)
        measured = compare_pair(baseline, candidate)
        max_abs = measured["max_abs"]
        rmse = measured["rmse"]
        worst_abs = max(worst_abs, max_abs)
        worst_rmse = max(worst_rmse, rmse)
        passed = (max_abs <= args.max_abs and rmse <= args.max_rmse
                  and measured["argmax_baseline"] == measured["argmax_candidate"])
        details = ""
        if args.report_softmax:
            probabilities = softmax_metrics(baseline, candidate)
            details = (f" kl={probabilities['kl']:.8g}"
                       f" top10_overlap={probabilities['top10_overlap']}/{probabilities['topk']}"
                       f" top1_margin={probabilities['margin']:.8g}")
        print(f"{'PASS' if passed else 'FAIL'} {name}: "
              f"max_abs={max_abs:.8g} rmse={rmse:.8g} "
              f"top={measured['argmax_baseline']}/{measured['argmax_candidate']}{details}")
        if not passed:
            failures.append(name)
    if failures:
        fail(f"{len(failures)}/{len(before)} traces exceed tolerances or change argmax; "
             f"worst max_abs={worst_abs:.8g}; worst RMSE={worst_rmse:.8g}; "
             f"failures={failures[:8]}")
    print(f"PASS {len(before)} full-vocabulary FP32 traces; "
          f"worst max_abs={worst_abs:.8g}; worst RMSE={worst_rmse:.8g}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, EOFError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        raise SystemExit(1)
