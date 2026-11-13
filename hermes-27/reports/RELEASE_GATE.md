# HERMES-27 — Independent Reliability Release Gate

**Scope:** target Qwen3.8-27B-Uncensored-NVFP4-MTP GGUF, HERMES PP2, two
RTX 5060 Ti 16 GiB, greedy token generation. **Test date:** 2026-11-13.
**Decision as implemented today: NO-GO for production.** Some checks are
demonstrably failing and other GPU/soak checks are not yet certified. A
successful selftest or HERMES-vs-HERMES trace comparison is **not** model
validation.

## Measured today, independently

| Gate | Status | Evidence |
|---|---|---|
| GGUF schema/model semantics | **PASS (CPU)** | Real GGUF metadata: 866 tensors, 448 NVFP4, vocab 248320, EOS 248046; GGUF `ssm_a` negative, Q/K `hh%16`; `gate_math_gguf.py` |
| llama.cpp C-API metadata import | **PASS (CPU; weights not loaded)** | Actual model, `--check-gguf-only`; verified arch qwen35, vocab and EOS, no kernels |
| Corrected block0 oracle | **PASS (CPU generation only)** | Independent GGUF NumPy calculation, no old E1 fixture; CUDA P1 hardware not implied |
| Exact context fixture construction | **PASS (CPU)** | Production template/tokenizer found token sequences of length **129, 1024, 8192**; no inference at those lengths yet |
| Standalone QA harness | **PASS (mock CPU)** | Six Python unittest cases cover HTTP/SSE, Unicode, N1/N2/N4/N8, metrics, full FP32 tampering and fail-closed reports |
| Server artifact parity | **FAIL — deployment regression** | Current source-built `gguf_dump --chat-template`: valid 8945-byte Jinja. Installed `/home/matteo/hermes-27/build/gguf_dump` emits 5546-byte metadata dump; tokenizer bridge rejects it as invalid template. Deploy gate returns NO_GO |
| Same-checkpoint independent llama.cpp full-logit comparison | **FAIL — numerical** | Real independent llama.cpp C API and real HERMES CLI each generated three 248320-element FP32 traces for the same 11 token IDs. All three greedy IDs matched, **all three FP32 traces exceeded threshold**: worst max_abs **0.56496239** (limit 0.20), worst RMSE **0.05635403** (limit 0.02). See detailed trace below |
| Same-checkpoint independent llama.cpp greedy tokens | **PASS (real hardware)** | **4 prompts × 3 greedy steps = 12/12 exact tokens**, including an EOS transition. Captures bound to real GGUF SHA256 `db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1` and exporter binaries via fresh per-file SHA256 manifests. This does not supersede FP32 FAIL |
| Batched prefill vs token-by-token decode (4 prompts) | **NOT RUN** | New `verify_release_prefill.py` calls the production `--verify-prefill` path and checks full 248320 logits with max abs ≤0.002; requires new GPU-exclusive approval |
| Real N1/N2/N4/N8 server workload in this QA suite | **NOT RUN** | Requires a fresh isolated server on the current binary; do not treat older performance tests as a substitute |
| Actual server Unicode/SSE/cancel/retry/auth | **NOT RUN** | Mock passed only. Live deploy cannot be certified with stale tokenizer helper |
| Memory-leak soak/long context | **NOT RUN** | Requires exclusive GPU and 8192-token request; expensive |
| EOS forced to first greedy ID | **NOT RUN** | Two independent HERMES CLI launches planned; no evidence yet |

### Real independent full-FP32 result (11-token Italian case)

Both programs used the **same actual GGUF** with path
`/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf`
and exact numeric input IDs:

```
6205,161421,303,220,17,1374,9911,1147,27053,6059,13
```

llama.cpp independent native C API: 64 decoder layers, GPU split across
GPU0/GPU1, `n_gpu_layers=80`, token-by-token input, FP32 output logits.
HERMES: `hermes_generate --dump-logits-dir`, also token-by-token input.
**The processes ran sequentially** (never two complete copies of the
checkpoint in VRAM together); both exited successfully with no service
disruption. Both produced greedy IDs `[271, 248068, 198]`.

| step | worst abs | RMSE | top-10 overlap | KL(llama ∥ HERMES) |
|---|---:|---:|---:|---:|
| 0 | 0.56496239 | 0.05351347 | 10/10 | 5.99e-5 |
| 1 | 0.23819351 | 0.04426458 | 9/10 | 1.19e-7 |
| 2 | 0.29322016 | 0.05635403 | 10/10 | 1.81e-4 |

CPU scalar HERMES and the current CLI matched exactly. Warp4 and packed head
variants exhibited similar deviation versus independent llama.cpp:

- llama vs HERMES scalar: worst abs 0.56496, worst RMSE 0.05635
- llama vs HERMES Warp4: worst abs 0.49401, worst RMSE 0.05623
- llama vs HERMES packed FP64-head: worst abs 0.55357, worst RMSE 0.05602

These numbers do **not** establish an optimization-induced precision
regression. They also do **not** establish acceptable accuracy. The strict
full-logit gate stays **FAIL** pending mathematical investigation, a
documented correction, or an explicit and independent numerical error budget
justified by the model's real-world quality.

Raw scratch evidence from this run:
`/tmp/h27-release-worker1-X3ZZUN/llama-live/` and
`/tmp/h27-release-worker1-X3ZZUN/hermes-live/`, with corresponding
`llama-live.jsonl` and `hermes-live.jsonl`. The data is **not**
stored as a permanent blessed golden and expires under the QA scripts.

### Expanded four-prompt independent result

One additional llama.cpp C-API run and one additional HERMES run produced
another nine raw FP32 outputs (English 13 IDs, Qwen chat-template 21 IDs,
short 4-ID prefix). Their twelve complete traces are available in
`/tmp/h27-release-worker1-X3ZZUN/{llama-all,hermes-all}/` with
corresponding `*-all.jsonl`, `llama-manifest.json` and
`hermes-manifest.json`. The provenance checker verified the **same SHA256
GGUF**, correct exporter binaries, exact prompt IDs, greedy continuation and
12 complete finite logits arrays. The independent full-logit comparison
completed with exit code **1** and wrote
`independent-fp32-result.json` with `passed: false`.

| prompt class | 3 greedy token IDs (llama = HERMES) | worst max abs | worst RMSE |
|---|---|---:|---:|
| Italian, 11 tokens | 271, 248068, 198 | 0.56496 | 0.05635 |
| English, 13 tokens | 271, 760, 8323 | 0.20275 | 0.04241 |
| Chat-template, 21 tokens | 66, 21817, **248046 (EOS)** | 0.32821 | 0.04277 |
| Short prefix, 4 tokens | 20, 173429, 58902 | 0.26328 | 0.05112 |

The independent greedy-only audit
`scripts/verify_release_independent.py --llama-manifest ... --hermes-manifest ...`
returned **PASS** with 12/12 matched, model SHA256 validated and actual
exporter binaries still present. All 12 numerical comparisons independently
returned **FAIL** (RMSE >0.02). They are **separate gates**, intentionally
not merged into a misleading overall PASS.

## Release policy (fail-closed)

| Gate | Acceptance criteria | Impact |
|---|---|---|
| Identity | Source GGUF SHA256 identical across HERMES/llama exports; independent exporter binary hash and run ID recorded; trace IDs/history match | **Mandatory** |
| Model semantics | Independent llama.cpp C API 4 prompts × at least 3 generated steps; full logits all finite, argmax identical, max abs ≤0.20, RMSE ≤0.02 on every 248320-element output | **Mandatory** |
| Prefill/decode | `hermes_generate --verify-prefill` for <=64-token prompts and independent continuation at GDN+attention transitions; never use stale E1/E2 | **Mandatory** |
| Concurrency | Serial greedy baseline identical to N=1,2,4,8 simultaneous; repeat 2 times; responses, token counts and EOS/finish reason identical; no swapped states | **Mandatory** |
| API correctness | Eight UTF-8/Unicode/RTL/emoji inputs; SSE/non-SSE text byte-for-byte; exactly one SSE terminal event; cancellation, schema errors, 404, 400, 401, 429 and retry-after | **Mandatory** |
| EOS | Valid metadata EOS and forced first-token early-stop: exactly one output token and `stop_reason=eos` | **Mandatory** |
| Context | Genuine **129, 1024 and 8192** prompt tokens generated with exact same chat template/tokenizer; each yields one token and correct usage; server `--context 8192` | **Mandatory** |
| Memory | Idle/steady VRAM after each of >=2 cycles; growth ≤128 MiB/GPU both internal `/metrics` and `nvidia-smi`; hard cap 15360 MiB/GPU, no OOM | **Mandatory** |
| Deploy | Compile GGUF parser from **current source** in scratch, Jinja render/Unicode bridge succeeds, deployed `gguf_dump --chat-template` byte-identical to current source | **Mandatory** |
| Operations | 24h production-like soak, forced process crash/restart, OOM recovery, observability, rollback and external auth validation | **Mandatory for broad production**, not implied by controlled canary gate |

**Interpretation:** Any `FAIL` is **NO-GO**; any mandatory `BLOCKED` or
`NOT_RUN` is **NO-GO (evidence incomplete)**; only all mandatory PASS is
**GO FOR CONTROLLED CANARY**, *not* unrestricted production. The standalone
summary CLI enforces this. No thresholds may be relaxed silently.

## CPU-only commands now safe

Run from the current project checkout; all scripts produce structured JSON
with 0=pass, 1=fail, 2=blocked. None of these starts a server or loads weights
to CUDA.

```bash
python3 tests/reliability_selftest.py

python3 scripts/verify_release_static.py \
  --gguf /path/to/checkpoint.gguf --engine /path/to/hermes_generate \
  --out /tmp/h27-static.json

python3 scripts/verify_release_deploy.py \
  --gguf /path/to/checkpoint.gguf \
  --tokenizer-json /path/to/tokenizer.json \
  --python-bin /path/to/python-with-tokenizers-and-jinja2 \
  --deployed-gguf-dump /path/to/deployed/gguf_dump \
  --out /tmp/h27-deploy.json

python3 scripts/verify_release_contexts.py \
  --gguf /path/to/checkpoint.gguf \
  --tokenizer-json /path/to/tokenizer.json \
  --python /path/to/python-with-tokenizers-and-jinja2 \
  --gguf-dump /path/to/current/gguf_dump \
  --out /tmp/h27-contexts.json
```

The deployment check deliberately compiles the GGUF parser itself with
`g++` in a temporary directory. It fails against a stale deployed binary
even if the source is correct. Do not use the stale `gguf_dump` to generate
context fixtures or run the server.

## Offline independent llama.cpp exporter

Compile the isolated C API probe with a **separate llama.cpp installation**:

```bash
LLAMA=/opt/hermes/runtimes/llama-mainline
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
  -I"$LLAMA/include" -I"$LLAMA/ggml/include" \
  tests/reliability_llama_logits.cpp \
  "$LLAMA/build-cuda/bin/libllama.so" -o /tmp/reliability_llama_logits
LD_LIBRARY_PATH="$LLAMA/build-cuda/bin" CUDA_VISIBLE_DEVICES=-1 \
  /tmp/reliability_llama_logits --model /path/to/checkpoint.gguf --check-gguf-only
```

`--check-gguf-only` is a strict CPU metadata/import gate; it does not
materialize weights or run an inference. Actual independent full logits:

```bash
# Run ONLY after GPU-exclusive approval, with empty output directories.
# llama.cpp and HERMES must NOT load the checkpoint simultaneously.
LD_LIBRARY_PATH="$LLAMA/build-cuda/bin" \
  /tmp/reliability_llama_logits --model /path/to/checkpoint.gguf \
  --out-dir /tmp/h27-llama --prompt-ids 6205,161421,303,220,17,1374,9911,1147,27053,6059,13 \
  --max-new-tokens 3 --threads 4 --n-gpu-layers 80 --gpu-exclusive > /tmp/h27-llama.jsonl

hermes_generate --model /path/to/checkpoint.gguf \
  --prompt-ids 6205,161421,303,220,17,1374,9911,1147,27053,6059,13 \
  --max-new-tokens 3 --dump-logits-dir /tmp/h27-hermes > /tmp/h27-hermes.jsonl
```

Never run both engines concurrently on the two 16GiB GPUs; the checkpoint
is ~19.69 GB. CPU llama mode (`--n-gpu-layers 0`) is available but may be
slow and consume >20GB host memory. No services must be stopped as a side
effect of testing.

Use `scripts/verify_release_logits.py manifest` **once per backend** with
`--backend`, `--exporter-binary`, `--exporter-version`, `--run-id`,
`--gguf`, `--jsonl`, `--dump-dir`, `--out`. Then compare with:

```bash
python3 scripts/verify_release_logits.py compare \
  --hermes-manifest /tmp/h27-hermes-manifest.json \
  --llama-manifest /tmp/h27-llama-manifest.json \
  --max-abs 0.20 --max-rmse 0.02 --out /tmp/h27-independent-fp32.json
```

The independent gate hashes the entire GGUF and every trace, compares all
floats via `tools/compare_fp32_logits.py`, checks exact IDs/step sets,
verifies both exporters' binary hashes, rejects stale files and different
backends. **Greedy token equality alone cannot make this pass.**

## Live HTTP suite — GPU-exclusive only

```bash
# Start a TEST INSTANCE of the rebuilt hermes_server with --context 8192,
# --active >=8 and --api-key; use a non-production port.
python3 scripts/verify_release_online.py \
  --base-url http://127.0.0.1:8089 --model hermes-27 --api-key TESTKEY \
  --gpu-exclusive --run-long-contexts --context-fixtures /tmp/h27-contexts.json \
  --repeats 2 --memory-cycles 2 --out /tmp/h27-online.json

# Separately restart a TEST INSTANCE with --active 1 --queue 1
# so 429 and Retry-After can actually be exercised.
python3 scripts/verify_release_online.py \
  --base-url http://127.0.0.1:8089 --model hermes-27 --api-key TESTKEY \
  --gpu-exclusive --backpressure-only --out /tmp/h27-queue.json

python3 scripts/verify_release_eos.py \
  --gguf /path/to/checkpoint.gguf --engine /path/to/hermes_generate \
  --gpu-exclusive --out /tmp/h27-eos.json

python3 scripts/verify_release_prefill.py \
  --gguf /path/to/checkpoint.gguf --engine /path/to/hermes_generate \
  --gpu-exclusive --out /tmp/h27-prefill.json
```

For independent greedy reference via existing llama-server
`/props` and `/completion`, run `verify_release_independent.py`
with `--gguf --engine --llama-url --gpu-exclusive --out`. llama-server
must expose the exact same absolute GGUF path (`os.path.samefile`).
If unavailable, mark the test BLOCKED; do not replace it with older golden.

Finally aggregate all reports using
`scripts/verify_release_summary.py --gguf --static --deploy --online --queue
--llama-tokens --eos --prefill --llama-fp32 --out` (each option needs its evidence
path). This rejects missing, stale, failed or incomplete mandatory reports.

## Resource scope and outstanding risks

Real 8192-token sequential decode currently costs thousands of complete
layer evaluations, not the 64-token batched prefill path. It is **deliberately
opt-in**, should be run only in a separate GPU-exclusive period, and must not
be conflated with cheap `decode_init(capacity=8192)` allocation. The
memory check measures **quiescent growth only**: it does not certify
transient high-water marks or CPU/RSS leaks. Additional monitor/high-water
measurements and crash/restart testing are required before broad production
exposure.

**Prioritize fixing the stale deployed gguf_dump and investigating the actual
llama.cpp-vs-HERMES numerical deviation.** Neither is solved by a mock test,
an extra throughput benchmark or a relaxed threshold.
