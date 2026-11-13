#!/usr/bin/env python3
"""Fail-closed HERMES-27 greedy A/B benchmark across optional CUDA paths.

Runs one HERMES CLI process per configuration/round and compares its actual
generated token sequence to the scalar baseline. Measurements exclude model
upload (TIMING records from the CLI), but include host overhead. This is NOT
an independent llama.cpp reference or proof of full-logit parity.

Requires exclusive GPU access on the model host; --self-test is CPU-only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROMPT_IDS = "6205,161421,303,220,17,1374,9911,1147,27053,6059,13"
TIMING = re.compile(
    r"^TIMING prompt=(\d+) seed_tokens=(\d+) seed_ms=([0-9.]+) "
    r"decode_steps=(\d+) generation_ms=([0-9.]+) decode_steps_per_s=([0-9.]+)$",
    re.MULTILINE,
)
SWITCHES = ("HERMES_NVFP4_GEMV", "HERMES_HEAD_BF16", "HERMES_DECODE_SKIP_LAYER_SYNC")
CONFIGS = {
    "scalar": {},
    "gemv_warp4": {"HERMES_NVFP4_GEMV": "warp4"},
    "gemv_head_warp4": {"HERMES_NVFP4_GEMV": "warp4", "HERMES_HEAD_BF16": "warp4"},
    "gemv_head_async": {
        "HERMES_NVFP4_GEMV": "warp4", "HERMES_HEAD_BF16": "warp4",
        "HERMES_DECODE_SKIP_LAYER_SYNC": "1",
    },
}


class BenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class Result:
    name: str
    round: int
    tokens: tuple[int, ...]
    stop_reason: str
    seed_ms: float
    generation_ms: float
    decode_steps: int
    step_rate: float


def parse_result(name: str, round_index: int, output: str, logs: str) -> Result:
    rows = [line for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        raise BenchmarkError(f"{name}: expected exactly one JSONL response, got {len(rows)}")
    try:
        result = json.loads(rows[0])
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{name}: invalid JSONL: {exc}") from exc
    if not isinstance(result, dict) or result.get("prompt_index") != 0:
        raise BenchmarkError(f"{name}: invalid prompt index/record")
    token_list = result.get("generated_token_ids")
    if not isinstance(token_list, list) or not token_list or not all(type(t) is int for t in token_list):
        raise BenchmarkError(f"{name}: invalid generated token IDs")
    if result.get("stop_reason") not in ("length", "eos"):
        raise BenchmarkError(f"{name}: unexpected stop reason")
    matches = TIMING.findall(logs)
    if len(matches) != 1:
        raise BenchmarkError(f"{name}: expected exactly one TIMING line, got {len(matches)}")
    index, seed_count, seed_ms, steps, gen_ms, rate = matches[0]
    if int(index) != 0 or int(steps) != len(token_list) - 1:
        raise BenchmarkError(f"{name}: decode step count mismatch")
    values = (float(seed_ms), float(gen_ms), float(rate))
    if not all(0 <= v < 1e9 for v in values) or float(seed_ms) <= 0:
        raise BenchmarkError(f"{name}: implausible timing values")
    if int(seed_count) < 1:
        raise BenchmarkError(f"{name}: invalid seed length")
    return Result(name, round_index, tuple(token_list), result["stop_reason"],
                  values[0], values[1], int(steps), values[2])


def run(engine: Path, gguf: Path, prompt_ids: str, max_tokens: int,
        name: str, round_index: int, timeout: float) -> Result:
    env = os.environ.copy()
    for key in SWITCHES:
        env.pop(key, None)
    env.update(CONFIGS[name])
    args = [str(engine), "--model", str(gguf), "--prompt-ids", prompt_ids,
            "--max-new-tokens", str(max_tokens)]
    try:
        process = subprocess.run(args, env=env, capture_output=True, text=True,
                                 check=False, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BenchmarkError(f"{name}: process failed: {exc}") from exc
    if process.returncode:
        raise BenchmarkError(f"{name}: exit {process.returncode}; stderr:\n{process.stderr[-1800:]}")
    return parse_result(name, round_index, process.stdout, process.stderr)


def self_test() -> None:
    output = json.dumps({"prompt_index": 0, "generated_token_ids": [7, 2], "stop_reason": "length"})
    timing = "TIMING prompt=0 seed_tokens=11 seed_ms=120.000 decode_steps=1 generation_ms=30.000 decode_steps_per_s=33.33333"
    result = parse_result("scalar", 0, output, timing)
    assert result.tokens == (7, 2) and result.step_rate == 33.33333
    for broken in (
        output.replace("[7, 2]", "[]"),
        output.replace('"length"', '"error"'),
    ):
        try:
            parse_result("scalar", 0, broken, timing)
        except BenchmarkError:
            pass
        else:
            raise AssertionError("invalid JSON unexpectedly accepted")
    try:
        parse_result("scalar", 0, output, timing.replace("decode_steps=1", "decode_steps=3"))
    except BenchmarkError:
        pass
    else:
        raise AssertionError("inconsistent decode timing unexpectedly accepted")
    print("SELF-TEST PASS (CPU only): parse, rejection, counters")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path)
    p.add_argument("--gguf", type=Path)
    p.add_argument("--prompt-ids", default=PROMPT_IDS)
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--json-out", type=Path)
    p.add_argument("--self-test", action="store_true")
    options = p.parse_args()
    if options.self_test:
        self_test()
        return 0
    if options.engine is None or options.gguf is None:
        p.error("--engine and --gguf are required, unless --self-test")
    if not 1 <= options.rounds <= 20 or not 2 <= options.max_new_tokens <= 8192:
        p.error("--rounds must be 1..20 and --max-new-tokens must be 2..8192")
    if options.timeout <= 0:
        p.error("--timeout must be positive")
    engine = options.engine.expanduser().resolve(strict=True)
    gguf = options.gguf.expanduser().resolve(strict=True)
    prompt = options.prompt_ids
    if not re.fullmatch(r"\d+(?:,\d+)*", prompt):
        p.error("--prompt-ids must be a non-empty comma-separated list of decimal IDs")
    if len(prompt.split(",")) + options.max_new_tokens - 1 > 8192:
        p.error("prompt + generated token count exceeds current MAX_CONTEXT=8192")

    results: list[Result] = []
    reference: tuple[tuple[int, ...], str] | None = None
    names = list(CONFIGS)
    for r in range(options.rounds):
        # Rotate order to reduce systematic thermal/cache/config-order bias.
        order = names[r % len(names):] + names[:r % len(names)]
        for name in order:
            current = run(engine, gguf, prompt, options.max_new_tokens, name, r, options.timeout)
            observed = (current.tokens, current.stop_reason)
            if reference is None:
                reference = observed
            if reference != observed:
                raise BenchmarkError(f"TOKEN MISMATCH in {name} round {r}: "
                                     f"expected {reference} got {observed}")
            results.append(current)
            print(f"ROUND {r+1}/{options.rounds} {name}: seed={current.seed_ms:.3f}ms "
                  f"decode={current.generation_ms:.3f}ms "
                  f"steps/s={current.step_rate:.4f} ids={list(current.tokens)}", flush=True)
    medians: dict[str, float] = {}
    print("\nA/B median decode rate (tokens generated after first step per second):")
    for name in names:
        selected = [x for x in results if x.name == name]
        medians[name] = statistics.median(x.step_rate for x in selected)
        print(f"  {name:18} {medians[name]:9.4f} steps/s "
              f"speedup vs scalar={medians[name]/medians['scalar']:.3f}x")
    if options.json_out:
        options.json_out.write_text(json.dumps({
            "checkpoint": str(gguf), "engine": str(engine),
            "prompt_ids": list(map(int, prompt.split(","))),
            "max_new_tokens": options.max_new_tokens,
            "rounds": options.rounds, "median_decode_rate": medians,
            "results": [x.__dict__ for x in results],
        }, indent=2) + "\n", encoding="utf-8")
    print("PASS identical generated token IDs and EOS/length across all variants and rounds")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (BenchmarkError, OSError, ValueError) as exc:
        print(f"FAIL / INCONCLUSIVE: {exc}", file=sys.stderr)
        sys.exit(2)
