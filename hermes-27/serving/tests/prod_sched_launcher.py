#!/usr/bin/env python3
"""Test-only launcher enabling the opt-in continuous backend.

Set HERMES_TEST_SERVER_BIN to a CPU-only scratch server path. Never point
it at a live service or an executable backed by CUDA during CPU-only tests.
"""
import os
import sys


def main():
    binary = os.environ.get("HERMES_TEST_SERVER_BIN")
    if not binary or not os.path.isabs(binary):
        raise SystemExit("HERMES_TEST_SERVER_BIN must be an absolute CPU-mock binary path")
    os.execv(binary, [binary, "--continuous-scheduler", *sys.argv[1:]])


if __name__ == "__main__":
    main()
