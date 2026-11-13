# GGUF regression gates before performance tuning

These two gates have different purposes. Both use the real GGUF and produce a
nonzero exit code on failure. Neither contains hard-coded "expected output"
from the earlier E1/E2 runs.

## 1. Cheap mathematical and source preflight

Run from the repository on the host where the model is available:

```bash
python3 tests/gate_math_gguf.py \
  --gguf /home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf
```

It reads the GGUF header and three small F32 vectors, without loading weights
to CPU/GPU. It checks that `ssm_a` is a pretransformed negative decay, rejects
`-exp(ssm_a)` in CPU E2, rejects GDN Q/K head mapping `h/3`, and checks that
active E1 scripts do not use `repeat_interleave(3,dim=1)`. For this layout,
`repeat(1,3,1)` is correct because it maps output head `h` to input `h%16`.

For a quick source-only audit without a GGUF:

```bash
python3 tests/gate_math_gguf.py --source-only
```

This preflight is intentionally strict and source-specific. If refactoring
changes the syntax of a correct implementation, update the check only after
reviewing the math. A pass does not establish model accuracy.

## 2. Independent llama.cpp golden comparison

The llama.cpp reference **must be an actual `llama-server` instance** using
the same GGUF file as HERMES. Start it separately with the absolute path
to that GGUF. Run the comparison on the same filesystem host as the server:

```bash
python3 tests/compare_llama.py \
  --gguf /home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf \
  --engine /home/matteo/hermes-27/build/hermes_generate \
  --llama-url http://127.0.0.1:8080 --tokens 4
```

Only run this inference test when both GPUs and sufficient host RAM are
available, with no other benchmark competing for them. It does not start,
stop or modify any server.

The script verifies that `GET /props` reports a `model_path` referring to
the exact same local file (`os.path.samefile`). It requests `POST /completion`
with integer token IDs, temperature zero, no repetition penalty, and
`return_tokens=true`. It then runs `hermes_generate --verify-prefill` on
identical token-ID prompts and compares every greedy token, stopping at the
first reported mismatch within each prompt. Cases cover Italian, English,
chat-template IDs and a shorter prefix. There is no tokenizer or template
conversion in either request.

Exit codes: `0` = all checks passed, `1` = a demonstrated mismatch,
`2` = incomplete/unverifiable setup or execution failure. A missing or
different llama.cpp model path is never silently treated as a pass.

## Legacy references

- The initial E1/E2 top-5 starting with `898` came from obsolete
  `(1+w)` RMSNorm. It is invalid.
- The subsequent E1/E2 fixture starting with `220` was previously called
  equivalent, but the old CPU E2 source used `-exp(ssm_a)` and `h/3` while
  E1 used direct `ssm_a` and the correct `repeat(1,3,1)`. Any pre-fix E2
  artifact requires revalidation. A shared top-1 does not prove full-logit
  agreement with llama.cpp.
- A new fixture becomes an oracle only after passing these gates against
  llama.cpp on the exact checkpoint. Record the GGUF checksum, llama.cpp
  version, prompt IDs, per-step token IDs and command line in the run report.

Remaining gates before speculative/performance tuning: full-logit or top-K
comparison where llama.cpp instrumentation supports it; multi-prompt
prefill/decode parity through GDN and RoPE transitions; MTP K=1..5
acceptance including target hidden-state evolution; rollback preserving
GDN/conv/KV and draft KV; sampling verification using the residual
`max(0, p_target - p_draft)` after rejection.
