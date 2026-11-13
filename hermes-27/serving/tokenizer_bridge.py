#!/usr/bin/env python3
"""Persistent tokenizer/template control plane for the native Hermes server."""
import argparse
import codecs
import json
import subprocess
import sys


def bytelevel_alphabet():
    """Inverse of the GPT-2/Rust tokenizers ByteLevel byte-to-Unicode map."""
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapped = visible.copy()
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            mapped.append(256 + extra)
            extra += 1
    return {chr(codepoint): byte for byte, codepoint in zip(visible, mapped)}


def prepare_incremental(tokenizer, tokenizer_path):
    """Enable only for a verified, strictly byte-level decoder.

    Other decoder pipelines can rewrite earlier decoded text. They retain the
    reference tokenizer.decode() path, performed once after generation.
    """
    with open(tokenizer_path, encoding="utf-8") as stream:
        config = json.load(stream)
    if (config.get("decoder") or {}).get("type") != "ByteLevel":
        return None
    alphabet = bytelevel_alphabet()
    special_ids = frozenset(
        item["id"] for item in config.get("added_tokens", []) if item.get("special")
    )
    # Check every token, once on startup. A mixed/raw Unicode vocab must use
    # the reference decoder; guessing how the decoder handles it is unsafe.
    vocabulary = tokenizer.get_vocab(with_added_tokens=True)
    if any(
        token_id not in special_ids and any(char not in alphabet for char in piece)
        for piece, token_id in vocabulary.items()
    ):
        return None
    return alphabet, special_ids


class IncrementalByteLevel:
    """Decode a token stream with stable UTF-8 output and bounded pending bytes."""

    def __init__(self, tokenizer, alphabet, special_ids):
        self.tokenizer = tokenizer
        self.alphabet = alphabet
        self.special_ids = special_ids
        self.utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def push(self, token_id):
        token_id = int(token_id)
        if token_id in self.special_ids:
            return ""
        piece = self.tokenizer.id_to_token(token_id)
        if piece is None:
            return ""  # Reference tokenizers.decode skips IDs absent from vocab.
        return self.utf8.decode(bytes(self.alphabet[char] for char in piece), final=False)

    def finish(self):
        return self.utf8.decode(b"", final=True)


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
        streaming_config = prepare_incremental(tokenizer, args.tokenizer)
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

    streams = {}
    send({"ready": True, "incremental": streaming_config is not None})
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
            elif req.get("op") == "stream_start":
                if streaming_config is None:
                    raise ValueError("incremental decoding unavailable for this tokenizer")
                stream_id = req["id"]
                if stream_id in streams:
                    raise ValueError("stream already exists")
                streams[stream_id] = IncrementalByteLevel(
                    tokenizer, streaming_config[0], streaming_config[1]
                )
                result = {"started": True}
            elif req.get("op") == "stream_push":
                result = {"delta": streams[req["id"]].push(req["token"])}
            elif req.get("op") == "stream_finish":
                result = {"tail": streams.pop(req["id"]).finish()}
            elif req.get("op") == "stream_abort":
                streams.pop(req["id"], None)
                result = {"aborted": True}
            else:
                raise ValueError("unknown tokenizer operation")
            send(result)
        except Exception as exc:
            send({"error": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
