#!/usr/bin/env python3
"""SSE parity test against an isolated CPU-only fake PP2 server.

The fake backend chooses FAKE_IDS[position] as the greedy next token. It must
export pp2::upload/decode_init/decode_step/etc. The tested server and tokenizer
bridge remain the real production source; GPU upload is not performed.

Usage:
  python test_stream_mock_http.py --server /tmp/.../build/hermes_server \
      --model /tmp/.../fake.gguf --gguf-dump /tmp/.../gguf_dump_fake.py \
      --tokenizer /opt/.../tokenizer.json --python /opt/.../bin/python
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.request


def request(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=45) as response:
        return response.status, response.read()


def one_run(args, token_ids, expected_text, label):
    env = os.environ.copy()
    env["FAKE_IDS"] = ",".join(str(token) for token in token_ids)
    command = [
        args.server, "--model", args.model, "--gguf-dump", args.gguf_dump,
        "--tokenizer", args.tokenizer, "--python", args.python,
        "--port", "0", "--context", "256",
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        env=env, text=True, bufsize=1,
    )
    try:
        while True:
            line = process.stderr.readline()
            if line.startswith("LISTEN "):
                url = line.split()[1]
                break
            if process.poll() is not None:
                raise AssertionError(f"{label}: mock server startup failed: {line}")

        def payload(stream):
            return {
                "model": "hermes-27", "messages": [{"role": "user", "content": "x"}],
                "temperature": 0, "max_tokens": len(token_ids), "stream": stream,
            }

        status, output = request(url, payload(False))
        assert status == 200, (label, status, output[:500])
        completion = json.loads(output)
        actual = completion["choices"][0]["message"]["content"]
        assert actual == expected_text, (label, repr(actual), repr(expected_text))
        assert completion["usage"]["completion_tokens"] == len(token_ids)

        status, output = request(url, payload(True))
        assert status == 200, (label, status, output[:500])
        pieces = []
        reasons = []
        done = 0
        for data in output.decode("utf-8").split("\n\n"):
            if not data:
                continue
            assert data.startswith("data: "), (label, data)
            chunk = data[6:]
            if chunk == "[DONE]":
                done += 1
                continue
            event = json.loads(chunk)
            assert event["object"] == "chat.completion.chunk", (label, event)
            choice = event["choices"][0]
            if choice["delta"].get("content") is not None:
                pieces.append(choice["delta"]["content"])
            if choice["finish_reason"] is not None:
                reasons.append(choice["finish_reason"])
        assert done == 1, (label, done)
        assert reasons == ["stop" if token_ids[-1] == 248046 else "length"], (label, reasons)
        assert "".join(pieces) == expected_text, (
            label, repr("".join(pieces)), repr(expected_text)
        )
        print(f"PASS {label}: {len(token_ids)} tokens, {len(pieces)} SSE content chunks")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "model", "gguf-dump", "tokenizer", "python"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    def encode(text):
        return tok.encode(text, add_special_tokens=False).ids

    test_cases = {
        "plain-ascii": encode("Hello world, this is an SSE test."),
        "accents": encode("Cafe\u0301, na\u00efve, fa\u00e7ade"),
        "unicode-cjk": encode("\u4f60\u597d\uff0c\u4e16\u754c"),
        "unicode-emoji": encode("\U0001f680\U0001f600\u2728"),
        "special-skipped": encode("Hello") + [248045] + encode(" world"),
        "eos-skipped": encode("end") + [248046],
    }
    from importlib.util import module_from_spec, spec_from_file_location
    module_path = Path(__file__).resolve().parents[1] / "tokenizer_bridge.py"
    spec = spec_from_file_location("hermes_tokenizer_bridge", module_path)
    bridge = module_from_spec(spec)
    spec.loader.exec_module(bridge)
    inverse = {byte: char for char, byte in bridge.bytelevel_alphabet().items()}
    for name, values in (
        ("split-emoji-bytes", [240, 159, 152, 128]),
        ("incomplete-utf8", [240, 159, 152]),
        ("invalid-utf8", [255, 128, 64, 254]),
    ):
        ids = [tok.token_to_id(inverse[byte]) for byte in values]
        assert all(token is not None for token in ids), (name, ids)
        test_cases[name] = ids

    for label, ids in test_cases.items():
        one_run(args, ids, tok.decode(ids, skip_special_tokens=True), label)


if __name__ == "__main__":
    main()
