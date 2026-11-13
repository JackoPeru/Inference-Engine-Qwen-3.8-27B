#!/usr/bin/env python3
"""Check Hermes-27 greedy PP2 on the target checkpoint and two idle GPUs."""

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    command = [str(args.executable.resolve()), "--model", str(args.model.resolve())]
    italian = "6205,161421,303,220,17,1374,9911,1147,27053,6059,13"
    english = "8917,5437,537,279,10814,314,279,8323,21440,303,2250,22157,13"
    # GGUF chat template for “Rispondi solo con la parola ciao.”, thinking disabled.
    chat = "248045,846,198,49,158648,13138,384,1147,183687,272,21817,13,248046,198,248045,74455,198,248068,271,248069,271"

    result = subprocess.run(
        command + ["--verify-prefill", "--max-new-tokens", "4",
                   "--prompt-ids", italian, "--prompt-ids", english,
                   "--prompt-ids", italian, "--prompt-ids", chat],
        capture_output=True, text=True, check=True, timeout=120,
    )
    entries = [json.loads(line) for line in result.stdout.splitlines()]
    expected = [[271, 248068, 198, 760], [271, 760, 8323, 21440],
                [271, 248068, 198, 760], [66, 21817, 248046]]
    if [entry["generated_token_ids"] for entry in entries] != expected:
        raise SystemExit(f"Unexpected output tokens: {entries}")
    if any(entry["prompt_index"] != i or entry["stop_reason"] != ("eos" if i == 3 else "length")
           for i, entry in enumerate(entries)):
        raise SystemExit(f"Unexpected prompt indices or finish reasons: {entries}")

    result = subprocess.run(
        command + ["--max-new-tokens", "4", "--prompt-ids", italian, "--eos-token", "271"],
        capture_output=True, text=True, check=True, timeout=45,
    )
    entry = json.loads(result.stdout)
    if entry["generated_token_ids"] != [271] or entry["stop_reason"] != "eos":
        raise SystemExit(f"Unexpected EOS behavior: {entry}")
    print("PP2 greedy, full-prefill parity, prompt isolation and EOS PASS")


if __name__ == "__main__":
    main()
