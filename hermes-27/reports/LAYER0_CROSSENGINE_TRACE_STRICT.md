# HERMES-27 layer0: strict cross-engine trace provenance and mapping (worker10)

**2026-11-14. CPU/OFFLINE completed, 14/14 isolated Ubuntu regression tests PASS.**
No GPU, model inference, CUDA, service, installed-library or Git operations
were performed. No authentic two-token HERMES stage tensor capture occurred.
All edits are exclusively to the authorized verifier, two test files and this
report; the published scratch patcher, llama downstream guard and independent
comparator remain unchanged by this worker.

## Physically observed evidence and limits

- On the existing Ubuntu server (`matteo@hermes`, read-only SSH), the **entire**
  physical GGUF at
  `/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf`
  was hashed with `sha256sum`, with **exit 0** and digest
  `db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1`.
  Physical length is **19,694,360,544 bytes**. This reaffirms checkpoint
  identity only; it does not run or numerically verify the model.
- The genuine independent **llama pos0** manifest at
  `/tmp/h27-layer0-llama-pos0-prime3/manifest.json` was rehashed on Ubuntu:
  `49b0052027532e1e8572d9d865517b2beef92c0e8176a02adacd35efe46d540b`.
  Its published verified contract is input **6205** at absolute position 0,
  greedy next token **161421**, and physical callbacks `attn_norm-0`,
  `linear_attn_qkv_mixed-0`, `conv_output_silu-0`; it contains no actual
  `beta_sigmoid-0`, `gate-0`, `attn_output-0` or normalized Q/K callback.
  This is a real *single-token* trace; its bytes cannot be sliced into a
  synthetic position-1 oracle.
- Read-only Ubuntu `find /tmp -maxdepth 5 -type f -name p0_context.txt`
  returned no matching files. Scoped inspections of
  `/tmp/h27-trace-v2-SwhR7R` and
  `/tmp/h27-trace-v2-candidate-20261114` likewise found no actual
  HERMES `p0_context.txt`, `p1_context.txt` or HERMES v2 manifest.
  **No physically authenticated HERMES v2 runtime sidecars exist in the
  inspected temporary locations.** Do not infer that any hypothetical
  two-token activations were captured. The source/test fixtures in this
  report contain only artificial CPU data.
- The current frozen `experiments/pp2_layer0_trace_scratch_patch.py`
  changed during this worker's first inspection, from an older hash
  `d85f81899b0c7f1e34bc98e689d988a96aae635ecbb035e8c8aebb915f77060c`
  to physically read SHA256
  `80c0be0c1e29faf5db837da306af59ee729d4207c9bc21022163cfdef049c886`.
  The newer file validates **five** exact source anchors including
  `ASYNC_POINT`, and emits **three** occurrences of the tracing environment
  variable, due to the new async/sliding preflight. The prior two-marker
  scratch binaries do not satisfy this stricter source correspondence;
  they need a separately authenticated new compile if a future GPU run
  is authorized. The worker did not edit this upstream patcher.

## Changes to `experiments/verify_hermes_layer0_trace.py`

1. **Physical GGUF identity is fixed for genuine captures.** For its normal
   `verify` and CLI paths, caller overrides of expected GGUF length or SHA
   are rejected before reading a trace. The only way to inspect a short
   *mock* in unit tests is explicit Python `test_fixture=True`; it always
   emits **different** schema
   `h27.hermes-scratch-layer0-synthetic-fixture.v1`, backend
   `hermes-27-synthetic-fixture`, and status
   `SYNTHETIC_CPU_FIXTURE_NOT_PHYSICAL_RUNTIME`. The normal CLI exposes
   no fixture option. The unchanged independent comparator requires
   `h27.hermes-scratch-layer0-capture.v2` and the real SHA/bytes, so it
   refuses a synthetic manifest before comparing numerical values.
2. **Exact CPU-verifiable patch provenance.** Both C++ input files must be
   named `pp2_fwd.cu`, distinct and regular, with a hook-free original.
   The checker pins the physical published patcher SHA and reconstructs
   the *complete* patched source bytes from the original using the
   patcher's own pure source transformation. Any extra C++ changes,
   including changes that preserve all three environment-variable marker
   strings, are rejected. The source and patch are hashed again after
   reading. The independent scratch executable is hashed and recorded.
   **A source SHA and executable SHA alone cannot prove the executable
   was produced by that source**: a separately trusted compile provenance
   and actual runtime attestation are still required.
3. **Exactly 24 raw scratch outputs.** The capture directory must have
   exactly `p0_context.txt`, `p1_context.txt` and the 22 specific
   `p{position}_{stage}.f32` files, all regular non-symlink files. Unknown
   extensions, stray `.partial` files, hidden files and subdirectories
   are refused. Each sidecar must match the exact bytes:

   ```text
   p0_context.txt: token_id=6205\nabsolute_position=0\nsequence=0\n
   p1_context.txt: token_id=161421\nabsolute_position=1\nsequence=1\n
   ```

   `\n` above denotes a real LF byte, including the final LF. Swaps,
   omissions, changed token/position/sequence, extra bytes and symlinks
   fail closed. The scratch patch uses `L0TraceScope(tok,pos,&ds)` and
   rejects a different `DecodeState` between the two token snapshots.
   This establishes **the instrumented HERMES runtime's self-report**,
   not independent llama identity or a hardware execution attestation.
4. **Physical tensor layout is not invented.** Each HERMES stage is stored
   as one bounded, finite, little-endian FP32 flat vector with source
   width 48, 5120, 6144 or 10240. A native Python `array('f')` byte-order
   assumption was replaced with explicit `struct.iter_unpack('<f', ...)
   ` plus a little-endian host check; raw tensor SHA and exact bytes remain
   reported. Added diagnostic fields explicitly record `runtime_identity_scope`,
   `independent_llama_tensor_identity_verified=false`,
   `cross_engine_comparison_status=BLOCKED_NO_INDEPENDENT_LLAMA_CALLBACK`,
   and a Q/K head-repetition and scaling ambiguity. These fields do not
   alter the published v2 schema required by the frozen comparator.

## Stage and axis mapping: what may be compared

| Stage | HERMES v2 per-position F32 width | independent llama expected shape for **two-token batch** | GGML time axis | Evidence state |
| --- | ---: | --- | ---: | --- |
| `attn_norm` | 5120 | `[5120,2,1,1]` | 1 | genuine llama pos0 only |
| `linear_attn_qkv_mixed` | 10240 | `[10240,2,1,1]` | 1 | genuine llama pos0 only |
| `conv_output_silu` | 10240 | `[10240,2,1,1]` | 1 | genuine llama pos0 only |
| `beta_sigmoid` | 48 | `[1,48,2,1]` | 2 | callback source only; actual bytes absent |
| `gate` | 48 | `[48,2,1,1]` | 1 | callback source only; actual bytes absent |
| `attn_output` | 6144 | `[128,48,2,1]` | 2 | callback source only; actual bytes absent |
| HERMES `q_norm_scaled` / llama `q_conv_predelta` | 6144 | `[128,16,2,1]` or `[128,48,2,1]` | 2 | **BLOCKED: scaling/head repetition unproven** |
| HERMES `k_norm` / llama `k_conv_predelta` | 6144 | `[128,16,2,1]` or `[128,48,2,1]` | 2 | **BLOCKED: head-axis/normalization unproven** |

HERMES flat width 6144 supplies **no physical GGML `ne[]` axis evidence**.
Even a 6144-F32 array exactly matching in length cannot be treated as
equivalent to `[128,48,2,1]` or to a 16-head Q/K callback without proving
head repetition, normalization point, Q scaling and exact position slice.
For a real two-token llama callback, position `pos` begins at flattened
offset `pos * product(ne[:token_axis])` (GGML `ne0` fastest). The frozen
comparator already blocks Q/K requests as `BLOCKED_UNSUPPORTED_STAGE`.
This worker preserved that gate and added independent tests to prevent
accidental admission of equal-width, fake-Q/K fixtures.

## Physical CPU test receipt

Isolated directory on Ubuntu:
`/tmp/h27-layer0-crossengine-w10-20261114-1145/`. This contains copies of
the three edited Python files and the two untouched published Python
dependencies; the source repository, services and GPU remain untouched.

```bash
python3 -m unittest discover -v \
  -s /tmp/h27-layer0-crossengine-w10-20261114-1145/experiments/tests \
  -p 'test_*.py'
```

**Observed exit 0: `Ran 14 tests in 0.055s`, `OK` (14/14 PASS).** Negative
cases include fake GGUF override, missing/changed/swap sidecars, wrong
sequence and token IDs, stage file inventory and NaN, three-marker patch
tamper, Q/K flattened-vs-headed shape and time-axis confusion, dummy
Q/K arrays rejected by the immutable comparator, and synthetic HERMES
manifest rejected by its real v2 parser.

Exact SHA256 frozen on shared Windows source, and independently matched
to the isolated Ubuntu copies after the successful run:

```text
experiments/verify_hermes_layer0_trace.py
  d06c4762bc79bca8bb79ef840d7fbdaa94b5297ce2d3909b2a585bbb6fdfb333
experiments/tests/test_verify_hermes_layer0_trace.py
  45677d7f3f27bb26b4321b721710c0d506bcde00432451d0bb56e14b7503883e
experiments/tests/test_layer0_crossengine_mapping_contract.py
  02448958a843cbd926146ffcbf377a3257a795d183d75c9487b7f8d91eb5cd28
frozen read-only experiments/pp2_layer0_trace_scratch_patch.py
  80c0be0c1e29faf5db837da306af59ee729d4207c9bc21022163cfdef049c886
frozen read-only tests/reliability_layer0_compare_authentic.py (Ubuntu copy)
  7530e5b336ce22e848a27c1a3b2b10981f7c5f372481589d142b5e0f2cb0290e
```

Earlier runs yielded a transient 11-error batch because the patcher was
revised during the first read. After re-reading and independently hashing
the revised patcher, the next run reached 13/14; the remaining failure
was an expected negative-path error-message wording change for a symlink.
The final run above is **14/14 PASS**, without lowering test assertions.

## Admission / next step

The current test proves failure behavior and full GGUF identity; it does
**not** prove the 27B engine's logits, downstream GDN output or release
readiness. Before any real comparison, the prime must obtain exclusive
physical ingress/GPU attestation (currently absent), create a freshly
compiled scratch `hermes_generate` from the reviewed 3-marker patcher,
generate genuine HERMES v2 sidecars plus all 22 tensor files, and obtain
the independent genuine llama **contiguous** `[6205,161421]` batch
callbacks. Every real capture must be physically checked with the fixed
GGUF and separate exporter/runtime provenance. Only then may the frozen
comparator compute per-position errors for source-proven equivalent stages.
Never claim Q/K equivalence, model-wide accuracy, or a throughput gain
from these CPU-only fixtures.
