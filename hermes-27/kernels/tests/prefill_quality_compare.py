#!/usr/bin/env python3
"""Offline, fail-closed comparison of full-model prefill logits (CPU ONLY).

This script does not generate logits or invoke a GPU/server. The manifests must
be produced by independently executed PP2 GEMV and experimental W4A4 model
paths. Neither the existing microbenchmark nor hermes_generate exports them.
See PREFILL_QUALITY_GATE.md before interpreting any result.

Exit: 0 = strict quality-only gate passed (not production approval);
      1 = observed quality regression; 2 = invalid/incomplete evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import re
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "hermes.prefill-quality.v1"
VOCAB = 248320
MIN_CASES = 32
MIN_ROWS = 512
REQUIRED_GROUPS = {"it", "en", "chat", "code", "math"}
REQUIRED_LENGTHS = (1, 4, 16, 32, 64)
MAX_ABS = 0.10
ABS_RMSE = 0.02
REL_RMSE = 0.01
MEAN_KL_NATS = 1.0e-3
MAX_KL_NATS = 5.0e-3
MIN_TOP5_SET_AGREEMENT = 0.99


class Inconclusive(Exception):
    pass


@dataclass
class Scores:
    count: int = 0
    rows: int = 0
    error_sq: float = 0.0
    reference_sq: float = 0.0
    max_abs: float = 0.0
    sum_kl: float = 0.0
    max_kl: float = 0.0
    top1_matches: int = 0
    top5_set_matches: int = 0
    continuation_matches: int = 0
    prompts: int = 0
    first_top1_flip: str | None = None
    first_continuation_flip: str | None = None

    def summary(self) -> dict[str, Any]:
        rows = max(self.rows, 1)
        rms = math.sqrt(self.error_sq / max(self.count, 1))
        norm = math.sqrt(self.reference_sq / max(self.count, 1))
        return {
            "cases": self.prompts,
            "positions": self.rows,
            "full_vocab_values": self.count,
            "max_abs": self.max_abs,
            "abs_rmse": rms,
            "relative_rmse": rms / max(norm, 1.0e-12),
            "mean_kl_nats": self.sum_kl / rows,
            "max_kl_nats": self.max_kl,
            "top1_match_rate": self.top1_matches / rows,
            "top5_set_agreement": self.top5_set_matches / rows,
            "continuation_exact_match_rate": self.continuation_matches / max(self.prompts, 1),
            "first_top1_flip": self.first_top1_flip,
            "first_continuation_flip": self.first_continuation_flip,
        }


def require(condition: bool, msg: str) -> None:
    if not condition:
        raise Inconclusive(msg)


def _parse(path: Path, *, allow_small_vocab: bool) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Inconclusive(f"manifest cannot be read: {path}: {exc}") from exc
    require(isinstance(data, dict), f"{path}: expected JSON object")
    for key in ("schema", "gguf_sha256", "vocab_size", "dtype",
                "backend", "variant", "kernel_mode", "seed_mode",
                "runner_sha256", "cases"):
        require(key in data, f"{path}: missing {key}")
    require(data["schema"] == SCHEMA, f"{path}: schema mismatch")
    require(data["dtype"] == "float32_le", f"{path}: only float32_le supported")
    require(type(data["vocab_size"]) is int and data["vocab_size"] > 1,
            f"{path}: invalid vocabulary")
    require(allow_small_vocab or data["vocab_size"] == VOCAB,
            f"{path}: expected full Qwen vocabulary ({VOCAB})")
    require(isinstance(data["gguf_sha256"], str) and
            re.fullmatch(r"[0-9a-f]{64}", data["gguf_sha256"]) is not None,
            f"{path}: missing or invalid SHA-256 for GGUF")
    require(isinstance(data["backend"], str) and bool(data["backend"]),
            f"{path}: missing backend")
    require(isinstance(data["variant"], str) and bool(data["variant"]),
            f"{path}: missing variant")
    require(data["seed_mode"] == "full_prefill",
            f"{path}: incremental token-by-token seed does NOT test prefill quality")
    require(data["kernel_mode"] in ("gguf-gemv-bf16", "cutlass-w4a4-gpuquant"),
            f"{path}: unknown or unverified kernel mode")
    require(isinstance(data["runner_sha256"], str) and
            re.fullmatch(r"[0-9a-f]{64}", data["runner_sha256"]) is not None,
            f"{path}: missing runner build fingerprint")
    require(isinstance(data["cases"], list) and bool(data["cases"]),
            f"{path}: cases must be nonempty")
    case_ids: set[str] = set()
    prompts: set[tuple[int, ...]] = set()
    for case in data["cases"]:
        require(isinstance(case, dict), f"{path}: malformed case")
        for k in ("id", "group", "prompt_ids", "positions", "logits_file",
                  "continuation_ids", "continuation_limit", "stop_reason"):
            require(k in case, f"{path}: case missing {k}")
        cid = case["id"]
        require(isinstance(cid, str) and bool(cid) and cid not in case_ids,
                f"{path}: empty or duplicated case ID {cid!r}")
        case_ids.add(cid)
        require(isinstance(case["group"], str) and bool(case["group"]),
                f"{cid}: missing group")
        ids = case["prompt_ids"]
        require(isinstance(ids, list) and bool(ids) and
                all(type(v) is int and 0 <= v < data["vocab_size"] for v in ids),
                f"{cid}: prompt_ids invalid")
        token_tuple = tuple(ids)
        require(token_tuple not in prompts, f"{cid}: duplicated prompt_ids")
        prompts.add(token_tuple)
        pos = case["positions"]
        require(isinstance(pos, list) and pos == list(range(len(ids))),
                f"{cid}: require all position logits 0..T-1, no sparse topK")
        require(isinstance(case["logits_file"], str) and
                bool(case["logits_file"]), f"{cid}: missing binary logits path")
        require(type(case["continuation_limit"]) is int and
                case["continuation_limit"] >= 16,
                f"{cid}: greedy continuation must request at least 16 tokens")
        generated = case["continuation_ids"]
        require(isinstance(generated, list) and 1 <= len(generated) <=
                case["continuation_limit"] and
                all(type(v) is int and 0 <= v < data["vocab_size"] for v in generated),
                f"{cid}: invalid continuation IDs")
        require(case["stop_reason"] in ("eos", "length"),
                f"{cid}: stop_reason must be eos or length")
        require(case["stop_reason"] != "length" or
                len(generated) == case["continuation_limit"],
                f"{cid}: truncated continuation without EOS")
        fp = Path(case["logits_file"])
        require(not fp.is_absolute(), f"{cid}: logits_file must be relative to manifest")
        base = path.resolve().parent
        resolved = (base / fp).resolve()
        require(resolved.is_relative_to(base), f"{cid}: logits path escapes manifest directory")
        require(resolved.is_file(), f"{cid}: missing logits {resolved}")
        expected = len(pos) * data["vocab_size"] * 4
        require(resolved.stat().st_size == expected,
                f"{cid}: invalid full-logit size; expected {expected} B, "
                f"found {resolved.stat().st_size} B at {resolved}")
        case["_resolved_logits"] = resolved
    return data


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _logit_pair(baseline: Path, candidate: Path, pos: int,
                vocab: int) -> tuple[tuple[float, ...], tuple[float, ...]]:
    row_bytes = vocab * 4
    unpack = struct.Struct(f"<{vocab}f")
    with baseline.open("rb") as bf, candidate.open("rb") as cf:
        bf.seek(pos * row_bytes)
        cf.seek(pos * row_bytes)
        b = bf.read(row_bytes)
        c = cf.read(row_bytes)
        require(len(b) == row_bytes and len(c) == row_bytes,
                "truncated logit row during comparison")
    return unpack.unpack(b), unpack.unpack(c)


def _top5(values: tuple[float, ...]) -> list[int]:
    # Tie-break like PP2 argmax: lowest token ID wins.
    return heapq.nlargest(5, range(len(values)),
                          key=lambda i: (values[i], -i))


def _kl_nats(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    # KL(softmax(a) || softmax(b)) with stable log-sum-exp.
    ma, mb = max(a), max(b)
    za = sum(math.exp(v-ma) for v in a)
    zb = sum(math.exp(v-mb) for v in b)
    za_log = ma+math.log(za)
    zb_log = mb+math.log(zb)
    weighted = sum(math.exp(v-ma) / za * (v-u)
                   for v, u in zip(a, b))
    return max(0.0, weighted + zb_log - za_log)


def compare(base_path: Path, cand_path: Path, gguf_path: Path,
            *, allow_small_vocab: bool = False,
            strict_minimums: bool = True) -> tuple[dict[str, Any], list[str]]:
    base = _parse(base_path, allow_small_vocab=allow_small_vocab)
    cand = _parse(cand_path, allow_small_vocab=allow_small_vocab)
    require(base["gguf_sha256"] == cand["gguf_sha256"],
            "GGUF SHA-256 differs between runs")
    require(base["vocab_size"] == cand["vocab_size"], "vocab dimensions differ")
    require(base["variant"] != cand["variant"],
            "same variant in both manifests: no W4A4 experiment proved")
    require(base["kernel_mode"] == "gguf-gemv-bf16" and
            cand["kernel_mode"] == "cutlass-w4a4-gpuquant",
            "invalid comparison: require baseline GEMV versus candidate CUTLASS W4A4")
    require(len(base["cases"]) == len(cand["cases"]), "case counts differ")
    require(gguf_path.is_file(), "GGUF file missing: cannot prove model identity")
    require(_file_hash(gguf_path) == base["gguf_sha256"],
            "GGUF file SHA-256 does not match manifest")

    scores = Scores()
    groups: set[str] = set()
    lengths: set[int] = set()
    for ca, cb in zip(base["cases"], cand["cases"]):
        for field in ("id", "group", "prompt_ids", "positions", "continuation_limit"):
            require(ca[field] == cb[field],
                    f"{ca['id']}: {field} mismatch (misaligned experiments)")
        cid = ca["id"]
        groups.add(ca["group"])
        lengths.add(len(ca["prompt_ids"]))
        scores.prompts += 1
        cont_ok = (ca["continuation_ids"] == cb["continuation_ids"] and
                   ca["stop_reason"] == cb["stop_reason"])
        if cont_ok:
            scores.continuation_matches += 1
        elif scores.first_continuation_flip is None:
            scores.first_continuation_flip = (
                f"{cid}: baseline={ca['continuation_ids']} "
                f"candidate={cb['continuation_ids']}")
        for rowpos in ca["positions"]:
            b, c = _logit_pair(ca["_resolved_logits"],
                               cb["_resolved_logits"], rowpos, base["vocab_size"])
            if any(not math.isfinite(v) for v in b) or any(not math.isfinite(v) for v in c):
                raise Inconclusive(f"{cid}@{rowpos}: non-finite logit values")
            scores.rows += 1
            scores.count += len(b)
            for v, u in zip(b, c):
                d = v-u
                scores.error_sq += d*d
                scores.reference_sq += v*v
                scores.max_abs = max(scores.max_abs, abs(d))
            idx_a, idx_b = _top5(b), _top5(c)
            if idx_a[0] == idx_b[0]:
                scores.top1_matches += 1
            elif scores.first_top1_flip is None:
                runner = idx_a[1]
                margin = b[idx_a[0]] - b[runner]
                scores.first_top1_flip = (
                    f"{cid}@{rowpos}: baseline={idx_a[0]} candidate={idx_b[0]} "
                    f"baseline_top1_margin={margin:.6g}")
            if set(idx_a) == set(idx_b):
                scores.top5_set_matches += 1
            kl = _kl_nats(b, c)
            scores.sum_kl += kl
            scores.max_kl = max(scores.max_kl, kl)

    stats = scores.summary()
    stats["model_sha256"] = base["gguf_sha256"]
    stats["baseline_variant"] = base["variant"]
    stats["candidate_variant"] = cand["variant"]
    stats["baseline_runner_sha256"] = base["runner_sha256"]
    stats["candidate_runner_sha256"] = cand["runner_sha256"]
    stats["groups"] = sorted(groups)
    stats["prompt_lengths"] = sorted(lengths)
    minimums: list[str] = []
    if scores.prompts < MIN_CASES:
        minimums.append(f"cases {scores.prompts} < {MIN_CASES}")
    if scores.rows < MIN_ROWS:
        minimums.append(f"logit positions {scores.rows} < {MIN_ROWS}")
    if not REQUIRED_GROUPS.issubset(groups):
        minimums.append("missing prompt groups: " +
                        ",".join(sorted(REQUIRED_GROUPS-groups)))
    if not set(REQUIRED_LENGTHS).issubset(lengths):
        minimums.append("missing prompt lengths: " +
                        ",".join(map(str, sorted(set(REQUIRED_LENGTHS)-lengths))))
    if strict_minimums and minimums:
        raise Inconclusive("dataset too small/incomplete: " + "; ".join(minimums))
    stats["dataset_minimums"] = minimums

    failures: list[str] = []
    if stats["max_abs"] > MAX_ABS:
        failures.append(f"max_abs {stats['max_abs']:.6g} > {MAX_ABS}")
    if stats["abs_rmse"] > ABS_RMSE:
        failures.append(f"abs_rmse {stats['abs_rmse']:.6g} > {ABS_RMSE}")
    if stats["relative_rmse"] > REL_RMSE:
        failures.append(f"relative_rmse {stats['relative_rmse']:.6g} > {REL_RMSE}")
    if stats["mean_kl_nats"] > MEAN_KL_NATS:
        failures.append(f"mean_KL {stats['mean_kl_nats']:.6g} > {MEAN_KL_NATS}")
    if stats["max_kl_nats"] > MAX_KL_NATS:
        failures.append(f"max_KL {stats['max_kl_nats']:.6g} > {MAX_KL_NATS}")
    if stats["top1_match_rate"] != 1.0:
        failures.append(f"argmax flip: {stats['first_top1_flip']}")
    if stats["top5_set_agreement"] < MIN_TOP5_SET_AGREEMENT:
        failures.append(f"top5_set_agreement {stats['top5_set_agreement']:.6g} "
                        f"< {MIN_TOP5_SET_AGREEMENT}")
    if stats["continuation_exact_match_rate"] != 1.0:
        failures.append(f"greedy mismatch: {stats['first_continuation_flip']}")
    return stats, failures


def _self_test() -> int:
    # Pure CPU unit tests: a passing match, an argmax flip, and a corrupt dump.
    # These are synthetic fixtures solely for parser/comparator mechanics.
    with tempfile.TemporaryDirectory(prefix="prefill_quality_") as td:
        d = Path(td)
        gguf = d / "dummy.gguf"
        gguf.write_bytes(b"synthetic, NOT a real GGUF")
        sha = _file_hash(gguf)
        a, b = d/"a", d/"b"
        a.mkdir(); b.mkdir()
        cases = []
        for i in range(2):
            cases.append({
                "id": f"tiny-{i}", "group": "it" if i else "en",
                "prompt_ids": [i+1, i+2], "positions": [0,1],
                "logits_file": f"tiny-{i}.f32le",
                "continuation_limit": 16, "continuation_ids": [3]*16,
                "stop_reason": "length",
            })
        meta = {
            "schema": SCHEMA, "gguf_sha256": sha, "vocab_size": 8,
            "dtype": "float32_le", "backend": "synthetic",
            "seed_mode": "full_prefill", "runner_sha256": "c"*64,
            "cases": cases,
        }
        for directory, variant, kernel in (
                (a, "gemv", "gguf-gemv-bf16"),
                (b, "w4a4", "cutlass-w4a4-gpuquant")):
            for c in cases:
                p = directory / c["logits_file"]
                p.write_bytes(struct.pack("<16f", *([0., 1., 9., 2., -1., 0., 1., 2.]*2)))
            (directory/"manifest.json").write_text(
                json.dumps({**meta, "variant": variant, "kernel_mode": kernel}),
                encoding="utf-8")
        m1, m2 = a/"manifest.json", b/"manifest.json"
        s, failures = compare(m1,m2,gguf,allow_small_vocab=True,strict_minimums=False)
        assert not failures and s["top1_match_rate"] == 1.
        p = b/"tiny-0.f32le"
        data = [0.,1.,9.,2.,-1.,0.,1.,2.]*2
        data[2]=0.;data[1]=10.
        p.write_bytes(struct.pack("<16f",*data))
        s,failures = compare(m1,m2,gguf,allow_small_vocab=True,strict_minimums=False)
        assert failures and s["top1_match_rate"] < 1.
        p.write_bytes(b"broken")
        try:
            compare(m1,m2,gguf,allow_small_vocab=True,strict_minimums=False)
        except Inconclusive:
            pass
        else:
            raise AssertionError("truncated logits must fail closed")
    print("SELF_TEST_PASS: synthetic match, argmax regression, corrupt dump rejected")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--candidate", type=Path)
    p.add_argument("--gguf", type=Path)
    p.add_argument("--json", type=Path, help="Optional CPU-only results JSON")
    p.add_argument("--diagnostic", action="store_true",
                   help="Allow small coverage; exit 2 even when values match")
    p.add_argument("--self-test", action="store_true",
                   help="Synthetic internal CPU tests only; never a model pass")
    args = p.parse_args()
    if args.self_test:
        return _self_test()
    if not args.baseline or not args.candidate or not args.gguf:
        p.error("--baseline, --candidate and --gguf are mandatory")
    try:
        scores, failures = compare(args.baseline,args.candidate,args.gguf,
                                  strict_minimums=not args.diagnostic)
    except (Inconclusive, OSError, ValueError, OverflowError) as exc:
        print(f"INCONCLUSIVE / NO PRODUCTION APPROVAL: {exc}", file=sys.stderr)
        return 2
    scores["failures"] = failures
    scores["gate"] = ("DIAGNOSTIC_ONLY" if args.diagnostic else
                      "QUALITY_FAIL" if failures else "QUALITY_PASS_ONLY")
    print(json.dumps(scores, indent=2, sort_keys=True))
    if args.json:
        args.json.write_text(json.dumps(scores, indent=2, sort_keys=True)+"\n",
                             encoding="utf-8")
    if args.diagnostic:
        return 2
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
