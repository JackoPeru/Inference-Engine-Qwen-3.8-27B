# HERMES-27: strict full-vocabulary prefill-result cardinality

**2026-11-14. Correctness/security fix for an opt-in diagnostic path.**
No real GGUF inference, CUDA kernel execution, service modification,
throughput measurement or release acceptance was performed.

## Source-level defect and actual RED

`tools/hermes_generate.cpp` allows `--verify-prefill` to calculate an
independent full-prefill reference before building a decode state. Previously
it only checked `full.empty()` and then did:

```cpp
const std::size_t offset = (prompt.size()-1)*kVocab;
reference.assign(full.begin()+offset, full.begin()+offset+kVocab);
```

For a nonempty but truncated `std::vector<float>` returned by `pp2::prefill`,
the last iterator can exceed `end()`, causing undefined behavior instead
of a clean rejection. For multi-token prompts, returning the first row only
is enough to trigger the defect.

A **real CPU-only RED** executable was compiled from the previously published
`hermes_generate.cpp` SHA-256
`166567adac3dddc3e1bc368322a43d6b0364d81388ab47480af4ec39f3e26139`
with a separately linked synthetic `pp2::prefill` that returned
`T * 248320 - 1` floats. `pp2::upload` was a stub and no GPU/real
model was loaded. Command requested prompt `[6205,161421]` with
`--verify-prefill`. Observed exit 1 with
`AddressSanitizer: heap-buffer-overflow` in `memmove`, not a handled
backend failure.

## Correction

Before constructing the reference slice, require the returned vector to
contain **exactly** `prompt.size() * 248320` floats. A shorter **or
larger** nonempty return is rejected with a bounded error, state cleanup
and exit 1. The prior `empty()` diagnostic remains. The regular decode
path and all numerical tolerances are unchanged. This adds one host-side
shape check only when `--verify-prefill` is explicitly requested.

## GREEN verification

- New `tools/tests/mock_generate_prefill_shape_cpu.cpp` implements only
  the PP2 functions needed by a fake executable, returning synthetic empty,
  short, extra and exact logits arrays. Its `decode_init` always refuses
  a real state, so no CUDA context can be opened.
- New `tools/tests/test_generate_prefill_shape_cpu.py` runs the fake CLI
  with each case. Genuine CPU GCC C++20 compilation with
  `-Wall -Wextra -Werror -fsanitize=address,undefined` succeeded.
  **4/4 dynamic tests PASS**: short and extra reject before decoder
  allocation, empty keeps its existing error, and exact length reaches the
  deliberately refusing mock `decode_init`.
- No check represents real GGUF correctness; full-vocabulary independent
  numerical **FAIL 12/12** remains unresolved. The model path in every
  fake invocation is `/no-real-model-loaded.gguf`, which is never opened.

**Gate:** source-level out-of-bounds hazard repaired under CPU verification;
overall HERMES-27 release still **NO-GO**.
