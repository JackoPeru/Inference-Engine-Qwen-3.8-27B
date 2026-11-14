# HERMES-27 — llama downstream layer-0 capture: exact offline gate

**2026-11-14, worker10. State: CPU source audit and preparation only.** No GPU,
network, SSH, installed runtime changes, or model inference was performed.
The physical Ubuntu scratch files were not accessible from this Windows worker.
Windows `python`, `py`, `python3` and Node were absent from PATH; `wsl --list
--quiet` reports no installed Linux distribution. Thus neither the new 20
unit tests nor the full physical preflight have been executed here; they are
**BLOCKED**, not PASS. No new real llama tensor was produced.

## What is actually known

- Previously physically checked **pos0** llama callback:
  `/tmp/h27-layer0-llama-pos0-prime3/manifest.json`, manifest SHA256
  `49b0052027532e1e8572d9d865517b2beef92c0e8176a02adacd35efe46d540b`.
  Only `attn_norm-0`, `linear_attn_qkv_mixed-0` and `conv_output_silu-0`
  were present. Token **6205** was at absolute position **0**, and the
  greedy token was **161421**. The old v1 manifest has an exporter SHA
  but **no exporter path**, so it cannot certify the old executable's
  physical identity without an additional real exporter file.
- The complete real GGUF was previously hashed:
  `19,694,360,544` bytes, SHA256
  `db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1`.
  Real `libllama.so.0.5.0` SHA256 was
  `fd9e3e0810c6eca2e37b57abf443bcf15963671cbd0ee8489a37461a521094d4`.
  All five installed library hashes were recorded in
  `reports/RELIABILITY_LAYER0_DOWNSTREAM_CAPTURE_PLAN.md`.
- Existing `tests/reliability_layer0_downstream_callback_trace.cpp:591-621`
  constructs **one** contiguous llama batch with `[6205,161421]`,
  positions `[0,1]`, `n_seq_id=1` and `seq_id=0` for both tokens.
  Its callback (`:236-353`) uses GGML actual stage names and bounded
  backend reads; it records exact source four-axis shapes and strides.
  Its manifest (`:357-425`) is only emitted after success (`:635-651`).
  The existing strict independent verifier is
  `tests/reliability_layer0_downstream_callback_verify.py`.
- Installed source SHA pins were previously observed in read-only Ubuntu
  inspection: `qwen35.cpp` =
  `ef973e9c789eb52ef27a77e0b1f36c6f41a378733a994d41b5924e3be012f671`,
  `delta-net-base.cpp` =
  `5e3b318953f854b3825dd7cfe4b483ca7935cd5f6680438a3ece4be9b4d5428f`,
  `llama.h` =
  `7279a2065cf7b7a081be9138dbe2be1bf8e0ce527e7df74429a7a982ae877a47`.
  Checking these exact source files does **not** prove their compiled
  correspondence to a pre-existing installed library.

## New exclusive files and checks

`experiments/llama_layer0_downstream_capture_guard.py` (CPU-only) pins the
physical model size and full hash, all five installed libllama/libggml ELF
hashes, old true v1 manifest and three physical tensor hashes, installed
source SHAs, frozen tracer source/header/verifier, and a **separately approved**
SHA of the freshly compiled exporter. It validates an exact five-library
runtime lock and invokes the exporter **only** with `--inspect-plan`,
`--inspect-runtime` and `--selftest`. It will never pass `--capture` to a
subprocess, or load the GGUF for inference. Any missing, changed or extra
identity produces `BLOCKED_OFFLINE_LLAMA_DOWNSTREAM_PREFLIGHT` and exit 2.
An optional postcapture mode executes the already established independent
physical hash/FP32 verifier and checks source-derived head/position axes.

`experiments/tests/test_llama_layer0_downstream_trace_contract_cpu.py`
contains **20 defined CPU tests** for exact stage set, both Q/K head
layout cases, flattened-head rejection, axis confusion, token replay,
bad absolute positions, two separate contexts, pos0 argmax change,
callback completeness, duplicate/missing stages, raw-byte size, schema,
runtime-lock tamper, full file hash/size, forbidden CLI mode and duplicate
JSON metadata. These tests use only artificial **metadata/bytes**, never
synthetic data presented as a genuine llama numeric oracle.

Frozen **Windows SHA256** for the three previously published source inputs:

```text
tests/reliability_layer0_downstream_callback_trace.cpp        ecaebd72739a2cc443147139b899e58717b532d9bea80da65cb8d11204e46429
tests/reliability_layer0_downstream_callback_sha256.hpp       eebe48b6c3518f84c8928b9613e13cb9f92bdb9fb7ceaae06a890c38bf2c37df
tests/reliability_layer0_downstream_callback_verify.py        2d9cc3da5fcb454672efcd63444f55e3da38ad57094f0a5f1bf3112e5087ae10
```

Guard source and unit-test SHA are to be frozen after the final review.

## Absolute positions, GGML axes and operator mapping

The smallest causal downstream capture should request these **eight** labels
in a single two-token decode. Source-derived expected `ne[0..3]` and
`token_axis` are admission rules, **not** a claim of measured runtime shape:

| llama callback | expected GGML `ne[0..3]` | token axis | interpretation |
|---|---|---:|---|
| `attn_norm-0` | `[5120,2,1,1]` | 1 | prior anchored upstream |
| `linear_attn_qkv_mixed-0` | `[10240,2,1,1]` | 1 | prior anchored upstream |
| `conv_output_silu-0` | `[10240,2,1,1]` | 1 | prior anchored upstream |
| `beta_sigmoid-0` | `[1,48,2,1]` | 2 | sigmoid beta per value head |
| `gate-0` | `[48,2,1,1]` | 1 | recurrent decay/alpha gate |
| `q_conv_predelta-0` | `[128,16,2,1]` **or** `[128,48,2,1]` | 2 | L2-normalized Q; 16 heads possibly repeated to 48 |
| `k_conv_predelta-0` | `[128,16,2,1]` **or** `[128,48,2,1]` | 2 | L2-normalized K; same condition |
| `attn_output-0` | `[128,48,2,1]` | 2 | computed recurrent GDN output |

The GGML file is flattened **ne0-fastest**. A position slice starts at
`position * product(ne[:token_axis])` for these two-token, single-sequence
shapes. A flattened `[6144,2,1,1]` is **not** acceptable evidence for
`attn_output-0`; required shape is `[128,48,2,1]`. The llama Q/K callbacks
must **not** be assumed numerically equal to HERMES `q_norm_scaled` and
`k_norm`: HERMES may apply scale at a different point and llama may repeat
heads differently. Both operators need an explicit normalization/head
mapping proof before any comparative verdict.

Only named `attn_output-0` is exposed for the fused-GDN path. The installed
`delta-net-base.cpp` graph does not name a callback on raw `gdn_out` updated
state snapshots; this capture cannot manufacture such a stage. The separate
callback-on-GDN output does **not** prove equivalent hidden recurrent state.

## Reproducible commands: Ubuntu CPU only, after source sync

Run only on the pre-existing Ubuntu host **without GPU ownership**. No
network, model inference or installed-file modifications are needed. Copy the
newly created Python files along with the frozen tracer files into the already
authorized `/home/matteo/hermes-27` before executing these commands.

```bash
set -eu
root=/home/matteo/hermes-27
llama=/opt/hermes/runtimes/llama-mainline
libdir="$llama/build-cuda/bin"
model="$root/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf"
s=/tmp/h27-layer0-downstream-cpu-prep-20261114
out=/tmp/h27-layer0-downstream-prime-capture-20261114
lock="$s/runtime.lock"
mkdir "$s" "$out"  # Fresh, otherwise stop; never overwrite real evidence.

python3 "$root/experiments/tests/test_llama_layer0_downstream_trace_contract_cpu.py" -v

g++ -std=c++20 -O2 -Wall -Wextra -Werror \
  -I"$llama/include" -I"$llama/ggml/include" -I"$root/tests" \
  "$root/tests/reliability_layer0_downstream_callback_trace.cpp" \
  -L"$libdir" -Wl,-rpath,"$libdir" \
  -lllama -lggml -lggml-cpu -lggml-base -ldl -pthread \
  -o "$s/trace"

cat > "$lock" <<'EOF'
libggml-base.so.0.25.3 63be6a671ca0bdc97d5ceb4ba4406f9a267bff4ae9ad9a9b5ba4b84ace24c876
libggml-cpu.so.0.25.3 01e92fe889efdd9da239d43dd7154a9fbb66f52193d77c45144fb849dbb0a191
libggml-cuda.so.0.25.3 27dff5b6ded0e179b6fbe4504e7c1f0f5751598cdd35759000bbe4df30b20de3
libggml.so.0.25.3 cd3766cae482625d6e8a309a6536b00c8285c80aa986b04095f85b51af99f794
libllama.so.0.5.0 fd9e3e0810c6eca2e37b57abf443bcf15963671cbd0ee8489a37461a521094d4
EOF

sha256sum "$s/trace"  # Give this receipt to prime for independent approval.
# The pin below MUST come from the reviewed compile receipt, not from a
# digest silently generated by the same preflight process.
trusted_exporter_sha=REPLACE_WITH_PRIME_APPROVED_64_HEX_SHA256

python3 "$root/experiments/llama_layer0_downstream_capture_guard.py" \
  --model "$model" --lib-dir "$libdir" --llama-source "$llama" \
  --tracer-source "$root/tests/reliability_layer0_downstream_callback_trace.cpp" \
  --tracer-helper "$root/tests/reliability_layer0_downstream_callback_sha256.hpp" \
  --verifier "$root/tests/reliability_layer0_downstream_callback_verify.py" \
  --tracer "$s/trace" --trusted-exporter-sha "$trusted_exporter_sha" \
  --prior-manifest /tmp/h27-layer0-llama-pos0-prime3/manifest.json \
  --runtime-lock "$lock" --capture-dir "$out"
```

Exit 0 `PASS_CPU_PREFLIGHT_ONLY` proves only pinned physical inputs plus
safe compiled-exporter inspection. A source hash and binary hash are
separate checks and do not, without reproducible-build proof, establish
that one exact source tree produced that executable.

## **GPU-exclusive** next command — not executed by worker10

Only after prime has verified the CPU receipt, approved the executable hash,
confirmed exclusive GPU ownership, and confirmed that `$out` is fresh/empty:

```bash
timeout 180 "$s/trace" --capture --gpu-exclusive \
  --model "$model" --llama-lib "$libdir/libllama.so.0.5.0" \
  --runtime-lock "$lock" \
  --expected-model-sha db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1 \
  --expected-libllama-sha fd9e3e0810c6eca2e37b57abf443bcf15963671cbd0ee8489a37461a521094d4 \
  --out-dir "$out" --token0 6205 --token1 161421 --n-gpu-layers 80 \
  --threads 4 --max-total-mib 16 --max-tensor-mib 2 \
  --include attn_norm,linear_attn_qkv_mixed,conv_output_silu,beta_sigmoid,gate,q_conv_predelta,k_conv_predelta,attn_output
```

The tracer writes `manifest.json` only for a completed callback, valid
logits and complete stage set; discard partial `.f32` on errors. A successful
exit alone does not establish source/hardware or numerical equivalence.
Then **CPU-only** postcapture independent physical admission:

```bash
python3 "$root/experiments/llama_layer0_downstream_capture_guard.py" \
  --model "$model" --lib-dir "$libdir" --llama-source "$llama" \
  --tracer-source "$root/tests/reliability_layer0_downstream_callback_trace.cpp" \
  --tracer-helper "$root/tests/reliability_layer0_downstream_callback_sha256.hpp" \
  --verifier "$root/tests/reliability_layer0_downstream_callback_verify.py" \
  --tracer "$s/trace" --trusted-exporter-sha "$trusted_exporter_sha" \
  --prior-manifest /tmp/h27-layer0-llama-pos0-prime3/manifest.json \
  --runtime-lock "$lock" --capture-dir "$out" \
  --capture-manifest "$out/manifest.json"
```

The postcapture guard verifies full physical GGUF, all loaded libraries,
exporter, lock and tensor FP32 byte hashes through the existing validator;
then it independently checks the eight true runtime shapes, axes, causal
sequence, callback completeness and prior pos0 greedy result. Exit 0 says
`VERIFIED_CAPTURE_STAGES_ONLY`: it is **not** a release gate or an assertion
that HERMES and llama outputs match. Actual comparison requires an
authenticated HERMES v2 trace for both positions, the verified direct
operator mapping, and a result from
`tests/reliability_layer0_compare_authentic.py`.

## Release blockers / falsification targets

1. **No authorized GPU capture yet**: eight real downstream tensors and true
   `ne[]` axes remain unknown. A missing requested callback or alternative
   fused-GDN graph must yield `BLOCKED`, never a made-up tensor.
2. **No physically authenticated HERMES scratch v2** with two runtime
   token/position sidecars and all 22 tensors: cross-engine numeric
   comparison remains blocked.
3. **Export binary proof is limited**: independent approval of new SHA
   identifies that executable, but does not prove it came from the pinned
   callback source unless a reproducible-build/attested compile is added.
4. **Q/K semantic mapping is unproven**: normalize, repeat heads and scale
   explicitly before comparing. A mismatch here is not automatically a
   PP2 full-logit bug.
5. **CPU test execution currently BLOCKED on worker Windows host** due to
   missing Python. Run the exact isolated CPU unittest command on Ubuntu
   before the GPU gate. If any pinned SHA or library list differs, re-audit
   the true installed build; do not update a pin by copying the observed
   untrusted value into an allowlist.

## Prime Ubuntu CPU integration after worker freeze — 2026-11-14

The preceding Windows-worker test execution blocker has been resolved **for
CPU preflight only**. Prime transferred the frozen guard/test/report into an
isolated Ubuntu Git-archive snapshot and reran the 20 new unit tests with
`python3 -B -m unittest discover -s experiments/tests -p
test_llama_layer0_downstream_trace_contract_cpu.py -v`: **20/20 PASS**.

The existing C++ callback tracer was compiled separately with the pinned
installed llama.cpp/GGML headers and libraries, without `--capture`, with
`g++ -std=c++20 -O2 -Wall -Wextra -Werror`. CPU `--inspect-plan`,
`--inspect-runtime` and `--selftest` all returned exit **0**. The newly
compiled exporter SHA256 is
`353e7f5f34f4a6b343b1bef4ff41068b1f79cc48271548e8cd38ace5d57f3231`.

Prime separately supplied this exact build-receipt SHA to the frozen guard
via `--trusted-exporter-sha`. A **full** offline physical preflight hashed
the entire 19,694,360,544-byte GGUF, all five installed llama/GGML runtime
libraries, source/tracer/verifier files, previous pos0 manifest and all
three previously captured pos0 tensors. It passed with the explicit status
`PASS_CPU_PREFLIGHT_ONLY`; executable scratch path was
`/tmp/h27-llama-downstream-cpu-20261114/trace` and the checked runtime
lock is alongside it. All checks and the empty future capture directory
are isolated under `/tmp`.

**No** `--capture`, CUDA model load, GPU inference, new tensor collection,
service restart, or production change occurred. The first true downstream
numeric divergence, full-vocabulary 12/12 gate, and physical two-token
HERMES/llama compare remain **BLOCKED/FAIL**, exactly as before. The build
receipt alone is not a reproducible-build attestation.
