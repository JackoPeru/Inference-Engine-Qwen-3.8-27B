#!/usr/bin/env python3
"""Text wrapper for hermes_generate; caller supplies tokenizer JSON explicitly."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def render_chat(model: str, executable: str, prompt: str, thinking: bool) -> str:
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    generator = Path(executable)
    inspector = generator.with_name("gguf_dump" + generator.suffix).resolve()
    template = subprocess.check_output([inspector, model, "--chat-template"], encoding="utf-8")
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

    def raise_exception(message):
        raise ValueError(message)

    env.globals["raise_exception"] = raise_exception
    return env.from_string(template).render(
        messages=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, help="path to hermes_generate")
    parser.add_argument("--model", required=True, help="path to NVFP4 GGUF")
    parser.add_argument("--tokenizer-json", required=True, help="compatible tokenizer.json")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--eos-token", type=int)
    parser.add_argument("--verify-prefill", action="store_true")
    parser.add_argument("--chat", action="store_true", help="apply the GGUF chat template via gguf_dump")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("prompt", help="plain text, or one user message with --chat")
    args = parser.parse_args()

    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("ERROR: install the existing `tokenizers` package to use text mode", file=sys.stderr)
        return 2

    try:
        tokenizer = Tokenizer.from_file(args.tokenizer_json)
        prompt = render_chat(args.model, args.executable, args.prompt, args.thinking) if args.chat else args.prompt
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not prompt_ids:
        print("ERROR: tokenizer produced empty prompt", file=sys.stderr)
        return 2

    command = [
        args.executable,
        "--model", args.model,
        "--prompt-ids", ",".join(map(str, prompt_ids)),
        "--max-new-tokens", str(args.max_new_tokens),
    ]
    if args.eos_token is not None:
        command += ["--eos-token", str(args.eos_token)]
    if args.verify_prefill:
        command.append("--verify-prefill")

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode:
        if result.stdout:
            sys.stderr.write(result.stdout)
        return result.returncode

    record = json.loads(result.stdout)
    sys.stdout.write(tokenizer.decode(record["generated_token_ids"], skip_special_tokens=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
