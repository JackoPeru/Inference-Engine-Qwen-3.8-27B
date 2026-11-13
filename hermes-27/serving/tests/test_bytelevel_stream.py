#!/usr/bin/env python3
"""CPU-only parity regressions for the real Qwen ByteLevel tokenizer."""
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

BRIDGE = Path(__file__).resolve().parents[1] / "tokenizer_bridge.py"
SPEC = importlib.util.spec_from_file_location("hermes_tokenizer_bridge", BRIDGE)
bridge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bridge)


class ByteLevelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer_path = os.environ.get("HERMES_TEST_TOKENIZER")
        if not cls.tokenizer_path:
            raise unittest.SkipTest("Set HERMES_TEST_TOKENIZER to the real tokenizer.json")
        from tokenizers import Tokenizer

        cls.tokenizer = Tokenizer.from_file(cls.tokenizer_path)
        cls.decoder_config = bridge.prepare_incremental(cls.tokenizer, cls.tokenizer_path)
        if cls.decoder_config is None:
            raise unittest.SkipTest("Tokenizer is not a verified ByteLevel tokenizer")
        with open(cls.tokenizer_path, encoding="utf-8") as stream:
            cls.special_ids = sorted(
                x["id"] for x in json.load(stream)["added_tokens"] if x["special"]
            )

    def check_ids(self, ids):
        decoder = bridge.IncrementalByteLevel(self.tokenizer, *self.decoder_config)
        pieces = [decoder.push(token) for token in ids]
        pieces.append(decoder.finish())
        result = "".join(pieces)
        self.assertEqual(result, self.tokenizer.decode(ids, skip_special_tokens=True))
        # Every emitted chunk is a valid UTF-8 string, including split emoji.
        for chunk in pieces:
            chunk.encode("utf-8")

    def test_unicode_and_chat_samples(self):
        samples = [
            "Hello world!\nHello\tworld.",
            "Caf\u00e9, na\u00efve, cr\u00e8me br\u00fbl\u00e9e.",
            "\u4f60\u597d\uff0c\u4e16\u754c\u3002",
            "\u03ba\u03b1\u03bb\u03b7\u03bc\u03ad\u03c1\u03b1",
            "\U0001f600\U0001f680\U0001f389\u2728",
            "e\u0301 + combining accents\u0301\u0301",
            "<|im_start|>assistant\nThinking: no.\n<|im_end|>",
        ]
        for sample in samples:
            ids = self.tokenizer.encode(sample, add_special_tokens=False).ids
            with self.subTest(sample=repr(sample)):
                self.check_ids(ids)

    def test_special_tokens_and_missing_ids(self):
        ids = self.tokenizer.encode("Before and after", add_special_tokens=False).ids
        self.check_ids(ids[:1] + self.special_ids + ids[1:])
        # The GGUF vocabulary has unused slots; Rust tokenizers skips unknown IDs.
        self.check_ids([max(self.tokenizer.get_vocab().values()) + 100])

    def test_random_and_invalid_utf8_sequences(self):
        all_ids = tuple(self.tokenizer.get_vocab().values())
        rng = random.Random(18647)
        for n in range(1200):
            seq = rng.choices(all_ids, k=rng.randrange(1, 65))
            if n % 5 == 0:
                seq.insert(rng.randrange(len(seq) + 1), rng.choice(self.special_ids))
            self.check_ids(seq)

        # Invalid UTF-8 byte fragments must not leak a premature replacement.
        alphabet = self.decoder_config[0]
        raw_byte_ids = [
            self.tokenizer.token_to_id(char)
            for char in alphabet
            if self.tokenizer.token_to_id(char) is not None
        ]
        for count in (1, 2, 3, 4, 8, 16, 64, 128):
            self.check_ids(rng.choices(raw_byte_ids, k=count))

    def test_other_decoder_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "tokenizer.json"
            config.write_text('{"decoder":{"type":"Sequence","decoders":[]}}', encoding="utf-8")
            self.assertIsNone(bridge.prepare_incremental(self.tokenizer, str(config)))

    def test_chat_template_and_interleaved_stream_protocol(self):
        # Exercise the actual persistent bridge process, including Jinja
        # rendering and separate decoder state for interleaved requests.
        template = (
            "{%- for m in messages -%}<|im_start|>{{m.role}}:{{m.content}}"
            "<|im_end|>{%- endfor -%}"
            "{% if add_generation_prompt %}<|im_start|>assistant{% endif %}"
            "{% if enable_thinking %}<think>{% endif %}"
        )
        with tempfile.TemporaryDirectory() as tmp:
            dumper = Path(tmp) / "fake_gguf_dump.py"
            dumper.write_text(
                "#!/usr/bin/env python3\nprint(" + repr(template) + ")\n",
                encoding="utf-8",
            )
            dumper.chmod(0o755)
            with subprocess.Popen(
                [
                    sys.executable, str(BRIDGE), "--tokenizer", self.tokenizer_path,
                    "--model", str(dumper), "--gguf-dump", str(dumper),
                ],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8",
            ) as proc:
                def call(request):
                    proc.stdin.write(json.dumps(request, ensure_ascii=True) + "\n")
                    proc.stdin.flush()
                    response = proc.stdout.readline()
                    self.assertTrue(response, "tokenizer bridge exited unexpectedly")
                    result = json.loads(response)
                    self.assertNotIn("error", result, result)
                    return result

                ready = json.loads(proc.stdout.readline())
                self.assertTrue(ready["ready"])
                self.assertTrue(ready["incremental"])
                messages = [
                    {"role": "system", "content": "Rules"},
                    {"role": "user", "content": "Caf\u00e9 \U0001f680"},
                ]
                for thinking in (False, True):
                    rendered = (
                        "<|im_start|>system:Rules<|im_end|>"
                        "<|im_start|>user:Caf\u00e9 \U0001f680<|im_end|>"
                        "<|im_start|>assistant"
                        + ("<think>" if thinking else "")
                    )
                    expected = self.tokenizer.encode(
                        rendered, add_special_tokens=False
                    ).ids
                    actual = call(
                        {"op": "encode", "messages": messages, "thinking": thinking}
                    )["ids"]
                    self.assertEqual(actual, expected)

                ids_a = self.tokenizer.encode(
                    "ABC\U0001f600", add_special_tokens=False
                ).ids
                ids_b = self.tokenizer.encode(
                    "\u4f60\u597d\u00e9", add_special_tokens=False
                ).ids
                call({"op": "stream_start", "id": "a"})
                call({"op": "stream_start", "id": "b"})
                chunks = {"a": [], "b": []}
                for i in range(max(len(ids_a), len(ids_b))):
                    for stream_id, ids in (("a", ids_a), ("b", ids_b)):
                        if i < len(ids):
                            chunks[stream_id].append(
                                call(
                                    {"op": "stream_push", "id": stream_id, "token": ids[i]}
                                )["delta"]
                            )
                for stream_id, ids in (("a", ids_a), ("b", ids_b)):
                    chunks[stream_id].append(
                        call({"op": "stream_finish", "id": stream_id})["tail"]
                    )
                    self.assertEqual(
                        "".join(chunks[stream_id]),
                        self.tokenizer.decode(ids, skip_special_tokens=True),
                    )
                proc.stdin.close()
                self.assertEqual(proc.wait(timeout=5), 0)


if __name__ == "__main__":
    unittest.main()
