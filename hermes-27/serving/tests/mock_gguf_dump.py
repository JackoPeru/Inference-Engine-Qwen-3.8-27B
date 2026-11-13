#!/usr/bin/env python3
"""CPU-only chat-template source for mock HTTP serving tests.

This intentionally does not parse GGUF or access GPUs. It allows the real
tokenizer_bridge.py to render a deterministic one-token seed prompt.
"""
import sys

if len(sys.argv) != 3 or sys.argv[2] != "--chat-template":
    print("usage: mock_gguf_dump.py FAKE_MODEL --chat-template", file=sys.stderr)
    raise SystemExit(2)

print("{{ messages[-1].content }}")
