# HERMES-27: independent layer0 comparator versus current three-marker tracer

**2026-11-14 — CPU/OFFLINE proof only.** This repairs a concrete compatibility
failure between the published independent comparator and the already-published
HERMES scratch tracer, without changing production CUDA, the comparison math,
the numerical thresholds, the genuine model, or any live service.

## Observed defect and RED witness

- The comparator's HERMES provenance check accepted exactly **two**
  occurrences of `HERMES_SCRATCH_LAYER0_TRACE_DIR` in the patched PP2 source.
- The frozen published scratch patcher (SHA-256
  `80c0be0c1e29faf5db837da306af59ee729d4207c9bc21022163cfdef049c886`)
  emits exactly **three** occurrences. The third one is the early
  async/sliding dispatch refusal, added after the circular-KV integration.
- With the *actual* published `distributed/pp2_fwd.cu` (SHA-256
  `0b8b8bcbcb6d60ee1c49fb240d7e925772fafc96def5c03151fe02b0d50b93ed`)
  and its genuine three-marker scratch transformation, the old comparator
  rejected a rehashed, fully shaped **synthetic CPU manifest** with
  `INVALID_PROVENANCE: scratch PP2 source did not preserve unmodified
  production source`. This **RED** observation used no hidden activations,
  CUDA kernels, or model runtime. It proves the admission bug, not numerical
  model accuracy.

## Minimal fail-closed repair

- `tests/reliability_layer0_compare_authentic.py` now validates the actual
  three-marker source and **reconstructs the complete expected patch** using
  the SHA-pinned source-only patcher. A changed original source, marker-only
  counterfeit, altered hook body, or stale two-marker scratch file is refused,
  even if the attacker rehashes the manifest and keeps all three markers.
- Both source files must be distinct regular `pp2_fwd.cu` files; the
  production original may not contain the scratch environment tag.
- Capture-directory inventory refuses symlinks, unknown subdirectories and
  other nonregular entries; only the 22 expected tensor files, two exact
  token/position sidecars, and optional manifest in that same directory may
  be present.
- The comparator still computes numerical errors **independently** using
  physical HERMES/llama tensor bytes, not by importing a runtime's own
  verifier or generated values. The pinned scratch patcher is used only to
  authenticate source correspondence.
- The test fixture now uses real frozen `pp2_fwd.cu` and its exact scratch
  transformation, while the tiny GGUF, numerical tensors and binaries in
  fixture trees remain unequivocally **synthetic**. They cannot satisfy the
  production CLI's fixed real GGUF identity.

## Executed verification

Isolated Ubuntu CPU scratch:
`/tmp/h27-three-marker-comparator-cpu-20261114`.
Copied both revised test/comparator files from the shared Windows source;
copied the unmodified published PP2 and patcher into the same scratch tree.
Independent SHA-256 of the four files was printed before execution.

```bash
cd /tmp/h27-three-marker-comparator-cpu-20261114
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/tests" \
  python3 -B -m unittest discover \
  -s tests -p test_reliability_layer0_compare_authentic.py -v
```

**GREEN: 27/27 PASS, exit 0, 1.738 seconds.** This includes the prior 23
positive/negative cases and four new cases: authentic current patch admitted,
marker-preserving tampered scratch rejected even with rehashed manifest,
changed original rejected with rehashed manifest, and unknown subdirectory
rejected. The previously tested Q/K unsupported-stage and token-axis checks
remain enforced. No numerical threshold was changed.

**Remaining gate:** authentic HERMES v2 sidecars and contiguous llama.cpp
downstream callbacks do not yet exist. The physical model identity and
prior authentic llama position-0 capture remain untouched. No GPU context
was acquired. The full 248320-logit independent gate remains FAIL 12/12,
and this repair alone is insufficient for GO.
