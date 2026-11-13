#!/usr/bin/env python3
"""Independent greedy regression gate: HERMES-27 PP2 vs llama.cpp on one GGUF.

Run this script on the same host as the llama-server model file. The server
must already be running with the *exact* GGUF passed to HERMES. The script
does not start services and does not perform hidden fallback to stored golden
tokens. It runs real inference; schedule it when both GPUs are available.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Case:
    name: str
    ids: tuple[int, ...]


# Inputs are explicit token IDs so BOS, tokenizer versions, and chat-template
# rendering cannot differ between the two inference programs.
CASES = (
    Case("it-11", (6205, 161421, 303, 220, 17, 1374, 9911, 1147, 27053, 6059, 13)),
    Case("en-13", (8917, 5437, 537, 279, 10814, 314, 279, 8323, 21440, 303, 2250, 22157, 13)),
    Case("chat-20", (248045, 846, 198, 49, 158648, 13138, 384, 1147, 183687, 272, 21817, 13,
                      248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271)),
    Case("short-4", (6205, 161421, 303, 220)),
)


class GateError(RuntimeError):
    pass


def http_json(url: str, body: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.load(resp)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
            OSError, ValueError) as exc:
        raise GateError(f"llama-server request failed at {url}: {exc}") from exc
    if not isinstance(result, dict):
        raise GateError(f"llama-server response at {url} is not a JSON object")
    if result.get("error"):
        raise GateError(f"llama-server returned error: {result['error']}")
    return result


def assert_same_model(model: Path, llama_url: str, timeout: float) -> str:
    props = http_json(f"{llama_url}/props", None, timeout)
    reported = props.get("model_path")
    if not isinstance(reported, str) or not reported.strip():
        raise GateError("llama-server /props omitted model_path; cannot prove model identity")
    reference = Path(reported)
    if not reference.is_absolute():
        raise GateError(
            f"llama-server uses relative model_path={reported!r}. "
            "Start it with an absolute -m path to prove the two runtimes use the same GGUF."
        )
    if not reference.is_file():
        raise GateError(
            f"llama-server model_path={reported!r} is not visible locally. "
            "Run the gate on the llama-server host."
        )
    try:
        same = os.path.samefile(model, reference)
    except OSError as exc:
        raise GateError(f"cannot compare GGUF file identity: {exc}") from exc
    if not same:
        raise GateError(f"DIFFERENT GGUF: HERMES={model} llama.cpp={reference}")
    return str(reference.resolve())


def llama_tokens(llama_url: str, case: Case, n_predict: int, timeout: float) -> list[int]:
    # Numeric prompt does not implicitly prepend BOS in llama.cpp /completion.
    # No sampling or repetition penalty is permitted in a greedy accuracy gate.
    body = {
        "prompt": list(case.ids),
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "repeat_penalty": 1.0,
        "repeat_last_n": 0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "dry_multiplier": 0.0,
        "mirostat": 0,
        "cache_prompt": False,
        "return_tokens": True,
        "stream": False,
        "seed": 0,
        "n_probs": 0,
    }
    reply = http_json(f"{llama_url}/completion", body, timeout)
    tokens = reply.get("tokens")
    if not isinstance(tokens, list) or not tokens or not all(type(t) is int for t in tokens):
        raise GateError(f"llama-server omitted integer tokens for {case.name}: {repr(tokens)[:240]}")
    if len(tokens) > n_predict:
        raise GateError(f"llama-server returned too many tokens for {case.name}: {len(tokens)}")
    # Some servers can send fewer tokens due to EOS; HERMES should stop there too.
    return tokens


def hermes_entries(engine: Path, gguf: Path, cases: tuple[Case, ...],
                   n_predict: int, timeout: float) -> list[dict[str, Any]]:
    argv = [str(engine), "--model", str(gguf), "--max-new-tokens", str(n_predict),
            "--verify-prefill"]
    for case in cases:
        argv += ["--prompt-ids", ",".join(map(str, case.ids))]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                              check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GateError(f"HERMES invocation failed: {exc}") from exc
    if proc.returncode:
        raise GateError(f"HERMES exited {proc.returncode}. stderr tail:\n{proc.stderr[-3000:]}")
    try:
        entries = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    except (ValueError, TypeError) as exc:
        raise GateError(f"HERMES emitted invalid JSONL: {proc.stdout[-1000:]}") from exc
    if len(entries) != len(cases):
        raise GateError(f"HERMES emitted {len(entries)} JSONL records for {len(cases)} prompts")
    for i, (entry, case) in enumerate(zip(entries, cases)):
        if entry.get("prompt_index") != i or entry.get("prompt_ids") != list(case.ids):
            raise GateError(f"HERMES prompt mismatch for {case.name}: {entry}")
        tokens = entry.get("generated_token_ids")
        if not isinstance(tokens, list) or not tokens or not all(type(t) is int for t in tokens):
            raise GateError(f"HERMES invalid token IDs for {case.name}: {entry}")
        if not 1 <= len(tokens) <= n_predict:
            raise GateError(f"HERMES invalid generated length for {case.name}: {tokens}")
        if entry.get("stop_reason") not in ("eos", "length"):
            raise GateError(f"HERMES invalid stop_reason for {case.name}: {entry}")
    return entries


def compare(ref_tokens: list[list[int]], candidates: list[dict[str, Any]],
            cases: tuple[Case, ...]) -> list[str]:
    failures: list[str] = []
    for case, reference, entry in zip(cases, ref_tokens, candidates):
        actual = entry["generated_token_ids"]
        match = actual == reference
        first = next((i for i, (a, b) in enumerate(zip(actual, reference)) if a != b),
                     min(len(actual), len(reference)))
        print(f"{'PASS' if match else 'FAIL'} {case.name} "
              f"input_len={len(case.ids)} step={first if not match else '-'} "
              f"llama={reference} hermes={actual}", flush=True)
        if not match:
            failures.append(case.name)
    return failures


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", required=True, type=Path, help="Exact local GGUF path")
    p.add_argument("--engine", required=True, type=Path, help="Built hermes_generate executable")
    p.add_argument("--llama-url", default="http://127.0.0.1:8080",
                   help="Already-running llama-server, using the same GGUF (default localhost:8080)")
    p.add_argument("--tokens", type=int, default=4, help="Greedy tokens per prompt, default 4")
    p.add_argument("--timeout", type=float, default=300, help="Seconds per inference request")
    args = p.parse_args()
    if args.tokens < 1 or args.tokens > 32 or args.timeout < 1:
        p.error("--tokens must be 1..32 and --timeout >= 1")
    gguf = args.gguf.resolve(strict=True)
    engine = args.engine.resolve(strict=True)
    if not gguf.is_file() or not engine.is_file():
        p.error("--gguf and --engine must both name existing files")
    llama_url = args.llama_url.rstrip("/")
    if not llama_url.startswith(("http://", "https://")):
        p.error("--llama-url must start with http:// or https://")

    reference_model = assert_same_model(gguf, llama_url, args.timeout)
    print(f"MODEL MATCH llama.cpp + HERMES: {reference_model}", flush=True)
    print("REFERENCE llama.cpp /completion, numeric prompt IDs, temperature=0, no penalties",
          flush=True)

    # Query independent reference first. No locally stored E1/E2 or old-golden
    # fixture is allowed to substitute the independent llama.cpp output.
    refs = [llama_tokens(llama_url, case, args.tokens, args.timeout) for case in CASES]
    entries = hermes_entries(engine, gguf, CASES, args.tokens,
                             args.timeout * len(CASES))
    failed = compare(refs, entries, CASES)
    if failed:
        print(f"FAIL external-reference regression: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"PASS independent llama.cpp token-reference gate: {len(CASES)} prompts "
          f"x up to {args.tokens} greedy tokens")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GateError as exc:
        print(f"INCONCLUSIVE / BLOCKED: {exc}", file=sys.stderr)
        sys.exit(2)
