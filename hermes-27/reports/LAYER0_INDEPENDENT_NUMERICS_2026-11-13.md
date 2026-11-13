# HERMES-27: first-layer independent llama.cpp numeric diagnosis

**Date:** 2026-11-13. **Status:** DIAGNOSTIC; independent full-model
logit release gate remains **FAIL / NO-GO**.

## Provenance and scope

The reference is the separately installed llama.cpp C API, **not HERMES**.
Its `cb_eval` callback exported three actual tensors during a single
`llama_decode` of token ID **6205 at absolute position zero**, layer zero,
from the same real GGUF:

- GGUF SHA256:
  `db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1`.
- llama.cpp shared library SHA256:
  `fd9e3e0810c6eca2e37b57abf443bcf15963671cbd0ee8489a37461a521094d4`.
- Actual capture manifest: `/tmp/h27-layer0-llama-pos0-prime3/manifest.json`.
- Exported stages: `attn_norm-0` (5120 FP32), `linear_attn_qkv_mixed-0`
  (10240 FP32), `conv_output_silu-0` (10240 FP32).
- The initially requested `post_ffn-0` did not appear in the actual
  callback and correctly caused a **BLOCKED** capture. A fresh capture
  limited to those three observable tensors exited **0** and wrote
  a provenance manifest with per-tensor SHA256. Do not treat the first
  capture's unmanifested files as reference evidence.

The new independent host diagnostic is
`tests/oracle_layer0_cpu_compare.py`; it validates the capture hashes,
shape, token, position and checkpoint SHA256 before calculation. It uses
read-only GGUF mmap through `scripts/block0_corrected_ref.py`, without
calling HERMES, llama kernels or CUDA. Host FP32 RMSNorm, NVFP4 unpacked
weights and matrix multiplication generate the first three stage
predictions. A second projection rounds the exact same normalized
activations to BF16 round-to-nearest-even, reconstructs FP32 and multiplies
by the **same** decoded weights. This approximates HERMES's activation
rounding, but is **not** a CUDA GEMV replay.

## Real result: one token, three stages

| Compared arrays (all elements) | Max absolute difference | RMSE |
| --- | ---: | ---: |
| CPU FP32 `attn_norm` vs llama `attn_norm` (5120) | **0** | **0** |
| CPU FP32 QKV vs llama QKV (10240) | **0.03521538** | **0.00570704** |
| CPU FP32 conv/SiLU vs llama conv/SiLU (10240) | 0.00703979 | 0.00020454 |
| CPU BF16-RNE QKV vs CPU FP32 QKV | 0.01772308 | 0.00202889 |
| CPU BF16-RNE QKV vs llama QKV | **0.03662157** | **0.00596865** |
| CPU BF16-RNE conv/SiLU vs llama conv/SiLU | 0.00882435 | 0.00030089 |

The BF16-RNE conversion itself changes normalized activations by
max_abs=0.00776076, RMSE=0.00157434. The initial **FP32**
normalization is bit-exact against independent llama.cpp, but the
QKV projection is already different. Applying BF16 rounding **does
not eliminate the difference** and slightly increases the QKV RMSE
on this one position.

**Conclusion supported by measurements:** the observed first difference
between the CPU GGUF mathematical oracle and llama.cpp occurs at the
first NVFP4 QKV projection, before convolution and GDN recurrence.
Neither an embedding mismatch nor an incorrect FP32 attention norm is
needed to explain that difference on the measured position.

**Not yet proven:** the first difference between HERMES's *actual
GPU-produced* layer0 QKV and llama, the specific llama CUDA matmul path
(MMQ/MMVQ/FP4/Q8_1), or how the per-projection errors propagate to
64-layer logits. A future actual HERMES layer callback or opt-in debug
capture and a matched llama tensor comparison are essential.
The independent 12/12 complete full-vocabulary logit comparisons still
FAIL the thresholds despite 12/12 identical greedy token IDs.
Do not relax quality thresholds or enable BF16/W4A4 based on this test.

## Reproduce on the real server (CPU only after existing capture)

```sh
PYTHON=/opt/hermes/runtimes/exllamav3-venv/bin/python
SRC=/home/matteo/hermes-27-gh/hermes-27
GGUF=/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf
LLAMA_TRACE=/tmp/h27-layer0-llama-pos0-prime3
"$PYTHON" -B "$SRC/tests/oracle_layer0_cpu_compare.py" --selftest
"$PYTHON" -B "$SRC/tests/oracle_layer0_cpu_compare.py" \
  --llama-dir "$LLAMA_TRACE" --model "$GGUF" \
  --out /tmp/h27-layer0-cpu-vs-llama-20261113.json
```

Creating the actual llama capture requires GPU exclusivity and the guarded
`tests/reliability_layer_llama_cb.cpp` exporter as documented in
`reports/RELIABILITY_LAYER_BISECT.md`. The scratch capture is evidence,
not a versioned giant binary golden; only the diagnostics/source are
committed. Results are specific to token 6205 at position zero and do
not cover subsequent positions, different input distributions or
multiuser state.
