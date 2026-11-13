#!/usr/bin/env python3
"""CPU-only exact-chat-token context fixtures via the production tokenizer bridge."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def build(bridge: Path, python: Path, tokenizer: Path, gguf: Path,
          gguf_dump: Path, sizes: list[int], thinking: bool) -> dict:
    proc = subprocess.Popen([str(python), str(bridge), "--tokenizer", str(tokenizer),
                             "--model", str(gguf), "--gguf-dump", str(gguf_dump)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8")
    try:
        ready = json.loads(proc.stdout.readline())
        if not ready.get("ready"):
            raise ValueError(f"tokenizer bridge startup: {ready}")
        def count(content: str) -> tuple[int, list[int]]:
            req = {"op": "encode", "messages": [{"role": "user", "content": content}],
                   "thinking": thinking}
            proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            obj = json.loads(proc.stdout.readline())
            if "error" in obj:
                raise ValueError(f"tokenizer encode: {obj['error']}")
            return len(obj["ids"]), obj["ids"]

        records = []
        for target in sizes:
            result = None
            # The template contributes a fixed prefix. Each string is searched
            # independently to find EXACT token counts (no approximate 8K label).
            for unit in (" z", " apple", " 4", "\n", " 漢", "🧪"):
                low, high = 0, target * 5 + 64
                while low <= high:
                    mid = (low + high) // 2
                    content = "Context boundary probe. " + unit * mid
                    length, ids = count(content)
                    if length == target:
                        result = {"tokens": length, "messages": [{"role": "user", "content": content}],
                                  "thinking": thinking,
                                  "token_ids_sha256": hashlib.sha256(
                                      ",".join(map(str, ids)).encode()).hexdigest()}
                        break
                    if length < target:
                        low = mid + 1
                    else:
                        high = mid - 1
                if result is not None:
                    break
            if result is None:
                raise ValueError(f"BLOCKED cannot construct exact {target} tokens for this tokenizer")
            records.append(result)
            print(f"PASS exact context fixture tokens={target} sha256={result['token_ids_sha256'][:16]}")
        return {"schema": "h27.context-fixtures.v1", "checkpoint": str(gguf.resolve()),
                "tokenizer": str(tokenizer.resolve()),
                "size_targets": sizes, "cases": records}
    finally:
        if proc.stdin:
            proc.stdin.close()
        if proc.stdout:
            proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        if proc.stderr:
            proc.stderr.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", required=True, type=Path)
    p.add_argument("--tokenizer-json", required=True, type=Path)
    p.add_argument("--gguf-dump", required=True, type=Path)
    p.add_argument("--python", default=sys.executable, type=Path,
                   help="Python with jinja2 and tokenizers installed")
    p.add_argument("--bridge", default=Path(__file__).resolve().parents[1] /
                   "serving/tokenizer_bridge.py", type=Path)
    p.add_argument("--targets", default="129,1024,8192")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--thinking", action="store_true", help="match reasoning.enabled=true")
    args = p.parse_args()
    try:
        sizes = [int(x) for x in args.targets.split(",")]
        if not sizes or any(n < 1 or n > 8192 for n in sizes) or len(set(sizes)) != len(sizes):
            p.error("--targets must be unique integers 1..8192")
        for item in (args.gguf, args.tokenizer_json, args.gguf_dump, args.python, args.bridge):
            if not item.is_file():
                raise FileNotFoundError(item)
        output = build(args.bridge, args.python, args.tokenizer_json, args.gguf,
                       args.gguf_dump, sizes, args.thinking)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(f"PASS generated {len(sizes)} exact offline context fixtures: {args.out}")
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"BLOCKED context fixture generation: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
