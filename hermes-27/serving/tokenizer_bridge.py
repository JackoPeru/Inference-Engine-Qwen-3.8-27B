#!/usr/bin/env python3
"""Persistent tokenizer/template control plane for the native Hermes server."""
import argparse
import json
import subprocess
import sys


def send(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gguf-dump", required=True)
    args = parser.parse_args()

    try:
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(args.tokenizer)
        template_text = subprocess.check_output(
            [args.gguf_dump, args.model, "--chat-template"], text=True
        )
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

        def raise_exception(message):
            raise ValueError(message)

        env.globals["raise_exception"] = raise_exception
        template = env.from_string(template_text)
    except Exception as exc:
        send({"error": str(exc)})
        return 1

    send({"ready": True})
    for line in sys.stdin:
        try:
            req = json.loads(line)
            if req.get("op") == "encode":
                prompt = template.render(
                    messages=req["messages"],
                    add_generation_prompt=True,
                    enable_thinking=req.get("thinking", True),
                )
                result = {"ids": tokenizer.encode(prompt, add_special_tokens=False).ids}
            elif req.get("op") == "decode":
                result = {"text": tokenizer.decode(req["ids"], skip_special_tokens=True)}
            else:
                raise ValueError("unknown tokenizer operation")
            send(result)
        except Exception as exc:
            send({"error": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
