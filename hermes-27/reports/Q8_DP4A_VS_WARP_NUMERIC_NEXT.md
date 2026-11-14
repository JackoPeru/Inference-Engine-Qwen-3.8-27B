# Q8_1 scalar/DP4A vs BF16 warp — authenticated CPU same-capture comparison

Date: 2026-11-14. **Result: PASS (bounded CPU source/mathematical contract), full-model and CUDA parity remain NO-GO.** This report does not supersede `Q8_PP2_FULL_MODEL_DIFFERENTIAL_2026-11-13.md` (its real full-model 3/3 logit failures still stand).

## Exact test and identities

- New owned file: `experiments/tests/test_q8_same_capture_contract_cpu.py` — SHA256 `ba56c0828181ba9d4e9717b7e35163bbe3740c3611d0dc3565060ff54e55538d`.
- Existing scratch adapter **unchanged**: `experiments/q8_dp4a_pp2_scratch_patch.py` SHA256 `27d9b8fe31f3bc0ca8c5fb120769e93e20d466bdc07878310b6d0cccfe0b598c`. No RED defect of adapter/kernel semantics was demonstrated; no speculative kernel or PP2 edit was made.
- Source checkouts: Windows `/saas/Inference Engine Qwen 3.8 27B/hermes-27` (not a Git repository); read-only Ubuntu Git `/home/matteo/hermes-27-gh/hermes-27` was **HEAD `6da5f533`** when inspected, updated from the assignment's expected `efad3f42` by concurrent work. Both inspected CUDA source files and the scratch adapter have byte-identical local/Ubuntu SHA256.
- Physical checkpoint: `/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf`, independently rehashed by the original validator: `db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1`.
- Genuine independent llama.cpp callback manifest: `/tmp/h27-layer0-llama-pos0-prime3/manifest.json`, SHA256 `49b0052027532e1e8572d9d865517b2beef92c0e8176a02adacd35efe46d540b`. The existing validator verifies `libllama.so.0.5.0` SHA256 `fd9e3e0810c6eca2e37b57abf443bcf15963671cbd0ee8489a37461a521094d4`, physical model path/hash, token **6205**, absolute position **0**, layer **0**, GPU callback CUDA0 source buffer label, little-endian FP32 tensor type/strides and all three tensor SHA256 values. The manifest's exporter SHA is a capture claim and has no independently attested physical path.
- Source numerical pins in new harness: `nvfp4_gemv_q8_1_experimental.cu` `7cd1e3010d3d038caeb19d31cf2dcbd1f6e619477917f88ddea2958e37922d7c`, `nvfp4_gemv_warp_variants.cu` `bb3ba6c14ebda27e17cec39d294ca4357ac3dae675423ff0804c321df5cbcfc9`, original llama validator `f41b46eeef2818cd8f15bff04e23b2e19629744bb28d696e5d07b835c7d062fc`, GGUF parser `b1ef0c53c0e3b426754f70f124a054e897a814ce17dbce2b5c4731677f61370b`. Source drift fails closed.

The source data is the same 5120-element captured `attn_norm-0.f32` for every path, with the same **raw GGUF NVFP4 `blk.0.attn_qkv.weight` [10240,5120]** bytes and 10240-element independently captured `linear_attn_qkv_mixed-0.f32` oracle. No synthetic data contributes to the main metrics. Three numerical modes share physical input and weights: (1) Q8_1 round-away-to-int8 on each FP32 group of 32, FP32 `amax/127`, **FP16 stored d**, signed E2M1-times-two/UE4M3 scale with zero and 0x7f reserved codes; (2) original CUDA scalar Q8 source's 8-integer subtotal and four-scale FLOAT reduction ordering; (3) DP4A source's four-GGUF-block, 32-lane, two-4-int subgroup sum and float reduction ordering. BF16 input is independently rounded to nearest even before FP64 matrix reference; all six BF16 warp variant **indexing and approximate FP32 accumulation orders** are also emulated against six selected real GGUF output rows.

## Full 10240-row real QKV matrix numerical comparisons

| CPU path against independent llama capture | RMSE | max abs | max error index |
|---|---:|---:|---:|
| Q8_1 **DP4A lane/accumulation emulation** | **1.8690822708717782e-7** | **3.814697265625e-6** | 5919 |
| Q8_1 scalar source lane/accumulation emulation | 2.879553446833076e-7 | 5.7220458984375e-6 | 5365 |
| Independent Q8_1 × decoded NVFP4 FP64 math | 1.5764864708187002e-7 | 3.814697265625e-6 | 4585 |
| BF16 RNE × decoded NVFP4 FP64 math | **0.005968646254418383** | **0.036622047424316406** | 3604 |

Direct **Q8 scalar vs Q8 DP4A** (not vs llama): RMSE `2.910307328163108e-7`, maxabs `5.7220458984375e-6` (index 5365), over all 10240 values. This is expected float accumulation-order variation in the CPU model. The Q8 benefit is local to this first QKV stage: BF16 RMSE / Q8 DP4A RMSE is approximately **31,933×**; it is not a full-model quality or performance ratio.

## BF16 warp variants on actual GGUF samples

For each variant, source-shaped CPU indexing and FP32 FMA-order approximation was run on authentic output rows `0,1,31,128,4585,10239` with the same BF16 input and same real raw GGUF weights. The comparison is against the independent BF16 × NVFP4 FP64 mathematical result for those rows (not a live GPU kernel).

| Variant (`Nvfp4WarpExperiment`) | maxabs on six real rows |
|---|---:|
| `PackedU16x2` | 5.21540641784668e-8 |
| `PackedU32x4` | 3.814697265625e-6 |
| `Unroll2` | 1.1920928955078125e-7 |
| `WarpSplit2` | 1.1920928955078125e-7 |
| `WarpSplit4` | 2.384185791015625e-7 |
| `Warp8Rows` | 1.1920928955078125e-7 |

`Warp8Rows` is the same Unroll2 arithmetic with a different CUDA rows-per-CTA launch; the software model correctly reproduces this. A **real head tensor** comparison was not available: no authenticated final head activation/target capture was provided. The mapping tests here cover the same kernel variants applied to the actual first QKV tensor shape, not head/whole-logit correctness.

## Executed tests and negative controls

Executed **without GPU or model load**, by streaming the exact Windows test script over read-only SSH standard input into the already installed Ubuntu Python `/opt/hermes/runtimes/exllamav3-venv/bin/python -B -`. This created no scratch files and changed no Ubuntu source, Git history, SSH configuration, services or GPU state.

```powershell
# Executed from the Windows shared source root; both exited 0:
Get-Content -LiteralPath experiments/tests/test_q8_same_capture_contract_cpu.py -Raw |
  ssh -o BatchMode=yes -o ConnectTimeout=8 hermes-tailscale '/opt/hermes/runtimes/exllamav3-venv/bin/python -B - --selftest'

Get-Content -LiteralPath experiments/tests/test_q8_same_capture_contract_cpu.py -Raw |
  ssh -o BatchMode=yes -o ConnectTimeout=8 hermes-tailscale 'OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /opt/hermes/runtimes/exllamav3-venv/bin/python -B - --manifest /tmp/h27-layer0-llama-pos0-prime3/manifest.json --gguf /home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf --source-root /home/matteo/hermes-27-gh/hermes-27'
```

- `SELFTEST_PASS`: deterministic synthetic positive controls of Q8 per32 zero/unit quant, FP16 stored-scale value **0.99993896484375** for ones, signed E2M1 `[-12,12]` twice-code, UE4M3 zero/0x7f/subnormal, scalar/DP4A integer grouping, **six BF16 warp variants**, deterministic repeats; malformed Q8 input and incompatible DP4A K shape rejected.
- Genuine full replay: **exit 0**, schema `h27.q8-same-capture-cpu.v1`, status `CPU_ONLY_AUTHENTIC_SOURCE_CONTRACT_NOT_CUDA_PARITY`; all 10240 QKV rows finite and Q8 mismatch guard `<1e-3` passed.
- **Negative RED**, missing authentic manifest: exit **2**, `FAIL_CLOSED_Q8_SAME_CAPTURE: manifest hash differs from first independently authenticated llama capture`.
- **Negative RED**, unaudited source root `/tmp`: exit **2**, `FAIL_CLOSED_Q8_SAME_CAPTURE: changed/unavailable audited numerical source: experiments/q8_dp4a_pp2_scratch_patch.py` (first missing source pin).
- Re-read source and manifest SHA256 on Ubuntu before real execution, all expected exact. SHA256 of the new test after final source-pin additions: `ba56c0828181ba9d4e9717b7e35163bbe3740c3611d0dc3565060ff54e55538d`.

## Decision and remaining numerical gaps

The result proves that the **CPU model of** the Q8_1 DP4A and scalar integer/address layout agrees tightly with llama's authentic first QKV stage and that the BF16 input rounding explains its first-stage mathematical offset. It does **not** prove bitwise CUDA/DP4A behavior, CTA launch correctness, CUDA stream safety, numerical stability for nonzero causal state, FFN, full attention, MTP, head variants on real head activations or user concurrency. The earlier full-model Q8 scratch execution still failed 3/3 independent logit acceptance traces. No new CUDA runtime execution, throughput benchmark or physical owner grant was used.

**Gate: retain NO-GO.** Given this bounded positive QKV result and the existing full-model failures, the next useful experiment is an independently authenticated later-stage (GDN recurrent/out and full-attention/head) comparison at identical token, position, GGUF and runtime identities, under a separately proven GPU ownership grant. There is no demonstrated Q8 source defect that justifies modifying the scratch adapter or production kernels now.
