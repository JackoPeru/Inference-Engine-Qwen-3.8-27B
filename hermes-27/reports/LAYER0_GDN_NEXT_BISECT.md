# HERMES-27 — Layer 0 GDN next downstream bisect (CPU/OFFLINE)

**Frozen 2026-11-14.** Outcome: isolated CPU numerical checks **PASS**, but the first post-convolution **independent llama.cpp comparison is BLOCKED by absent authenticated tensors**. No model-wide inference, live services, GPU, PP2, serving, CMake or other production source was touched. This is NOT an independent full-model parity result.

## Provenance and restricted scope

Authentic llama.cpp callback trace: `/tmp/h27-layer0-llama-pos0-prime3/`, token **6205**, absolute position **0**, layer **0**, exact GGUF model **SHA256 `db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1`**, installed libllama **SHA256 `fd9e3e0810c6eca2e37b57abf443bcf15963671cbd0ee8489a37461a521094d4`**. Manifest **SHA256 `49b0052027532e1e8572d9d865517b2beef92c0e8176a02adacd35efe46d540b`**; its strict validator checks physical file, individual tensor hashes/shapes, source backend and prompt identity.

**Only three authentic llama tensors:** `attn_norm-0` FP32 [5120], `linear_attn_qkv_mixed-0` FP32 [10240], `conv_output_silu-0` FP32 [10240]. Previous frozen source-independent CPU stage JSON: `/tmp/h27_w5_layer0_gdn_q8_result.json`, **SHA256 `faed57314848224e9c41514b7ce4d2dd9368a29c28de0f11a53cb5c9c7ff893f`**. New `tests/reliability_layer0_gdn_next_bisect.py` requires exact frozen SHA and capture provenance before computing.

Previously proven comparisons against **real captured llama values** remain:

| Actual captured stage | Q8_1 CPU RMSE vs llama | BF16 CPU RMSE vs llama | Status |
| --- | ---: | ---: | --- |
| `attn_norm-0` | 0 (old GGUF CPU RMS oracle) | — | PASS |
| `linear_attn_qkv_mixed-0` | **1.57648647e-7** | 0.005968646 | PASS Q8 |
| `conv_output_silu-0` | **2.07688537e-8** | 0.0003008902 | PASS Q8 / BF16 fails matching |

The **new** test fixes Q/K/V to the **captured llama convolution output**. It reads real GGUF `blk.0.ssm_beta.weight`, `blk.0.attn_gate.weight`, `blk.0.ssm_norm.weight`, `blk.0.ssm_alpha.weight`, `blk.0.ssm_a` and `blk.0.ssm_dt.bias`. The same authentic `attn_norm-0` is quantized as per-32 GGML Q8_1 (FP16 stored scale), BF16-RNE or unrounded FP32 before beta/gate NVFP4 projections; only beta/gate precision is varied. All values in this section are full-vector **CPU-to-CPU comparisons**, not a downstream llama oracle.

## First isolated candidate differences (single token, full arrays)

| Comparison, authentic llama conv held fixed | Elements | RMSE (vs CPU Q8_1) | Max abs |
| --- | ---: | ---: | ---: |
| `beta_sigmoid` BF16 vs Q8_1 | 48 | **0.0006009598276522681** | 0.001586258411 (head 32) |
| Raw gate `z` BF16 vs Q8_1 | 6144 | **0.006180220276921823** | 0.024942755699 (index 3435) |
| GDN output after beta-only BF16 | 6144 | **0.0000063413921769** | see machine JSON |
| Gated RMS after beta-only BF16 | 6144 | **0.0000303417305287** | see machine JSON |
| GDN output after gate-only BF16 | 6144 | **0, exact** | **0, exact** |
| Gated RMS after gate-only BF16 | 6144 | **0.0000763759626052** | see machine JSON |
| Gated RMS after both beta/gate BF16 | 6144 | **0.0000847918805416** | see machine JSON |
| Gated RMS after both beta/gate FP32 | 6144 | **0.0000778922433894** | see machine JSON |
| Deliberately **wrong** Q/K grouping `h//3` vs correct `h%16` at GDN output | 6144 | **0.008711138213041973** | **0.36089906841516495** |

The passing **factorial assertions** show that beta changes GDN output **before** gated RMS; changing only z must not alter the GDN recurrence output and does not (RMSE exactly 0), but z changes gated RMS. Therefore a larger raw-gate projection error cannot be assumed to cause an earlier GDN recurrence mismatch. The incorrect grouping demonstrates a large effect, **but does not justify altering actual PP2 grouping**. PP2 `distributed/pp2_fwd.cu:140-165` uses `g=hh%(KD/KHD)`. Actual independent llama fused CUDA `ggml/src/ggml-cuda/gated_delta_net.cu` uses `fastmodulo(h_idx, neqk1_magic)`; its unfused Qwen35 graph repeats 16 Q/K heads across 48 V heads (`src/models/qwen35.cpp:427-445`).

At pos0, the previous GDN S state is zero. Thus `S_new = k ⊗ (beta*v)` and `o=S_new^T q` regardless of `exp(ssm_a * softplus(alpha+dt))`. The test checks that changing the zero-state decay makes no difference. **Pos0 cannot verify recurrent decay or a nonzero-history convolution tail.** The Q8 CPU GDN output using the genuine llama conv has **SHA256 `277cf59bd3e954daffa0d998c1108171edc79f56c810fe7a5db3d2daa28712f6`**, confirmed by a separate read-only SHA comparison against the previously frozen *CPU* oracle. This is repeatability, not an authenticated llama GDN output.

A minor additional **internal CPU rounding** distinction exists: `z/(1+exp(-z))` versus `z*(1/(1+exp(-z)))` SiLU expressions have different FP32 gated-RMS SHA values (`3c05b038cc9d973461bbc31d564de9aecfbe581bd9fa061acd74ee4157487eaf` versus `b56dac6751f6464006a5834789af6afa6b5d9fcb981dfe383281fd4e32102315`). A hash mismatch alone cannot quantify its error or establish a llama discrepancy. This distinction is **not** the proven full-logit root cause.

## Real test, reproducibility and failures

New owned test: `tests/reliability_layer0_gdn_next_bisect.py`. Only new report: `reports/LAYER0_GDN_NEXT_BISECT.md`.

Executed on Hermes as isolated copied source `/tmp/h27_w5_layer0_gdn_next_bisect.py`; no HERMES or llama inference invoked. Commands previously run:

```bash
PY=/opt/hermes/runtimes/exllamav3-venv/bin/python
$PY -B /tmp/h27_w5_layer0_gdn_next_bisect.py --selftest
# SELFTEST_PASS zero state, independent beta/gate, SiLU-zero, SHA

OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
$PY -B /tmp/h27_w5_layer0_gdn_next_bisect.py \
  --manifest /tmp/h27-layer0-llama-pos0-prime3/manifest.json \
  --gguf /home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf \
  --source-root /home/matteo/hermes-27-gh/hermes-27 \
  --llama-root /opt/hermes/runtimes/llama-mainline \
  --frozen /tmp/h27_w5_layer0_gdn_q8_result.json \
  --output /tmp/h27_w5_layer0_gdn_next_result.json
# PASS_CPU_NO_LLAMA_DOWNSTREAM_ORACLE, exit 0
```

New result: `/tmp/h27_w5_layer0_gdn_next_result.json`, **SHA256 `abf8cbec4ce4302b20944c68074436a6905635aaa0dd7584a4c36f8bbfaf7688`**. This hash was independently read-only verified on 2026-11-14; no repeat inference. Test source in shared Windows workspace **SHA256 `969cf0d35473952593c4824e89d39b55db92c1280300df710a2b088dae5e5b81`**. The server /tmp copy is **16,360 bytes** (SHA256 `6b75a073fca9ee779f3b23b419ba2cea08e024cdd6fe9af72e8da9344ceba4a7`), exactly the 16,358-byte local source with **two trailing CRLF bytes** appended by the PowerShell→SSH transfer. Read-only `head -c -2 ... | sha256sum` returned the **identical local source SHA256**. Thus the executable Python source content was unchanged.

The test **PASS** asserts: authenticated source provenance; frozen JSON hash/schema, required llama tensor set and original Q8-vs-BF16 upstream gate; real GGUF tensor shapes; Q/K modulo16; zero-state algebra and finite arrays; positive beta/gate Q8-vs-BF16 divergence; gate-only GDN output exact equality and nonzero gated-RMS difference; incorrect consecutive head grouping detectable. The **deliberately wrong head grouping** has an expected significant discrepancy, not a test-suite failure. The strict test refuses to invent any llama tensor after convolution.

## Next smallest decisive experiment, not an authorized step in this task

**Earliest unverified independent downstream stages:** beta sigmoid and raw gate `z` projections parallel to QKV from the same `attn_norm-0`, plus normalized `q_conv_predelta`, `k_conv_predelta` and `attn_output`. They are **not present** in the authenticated capture. A fresh independently authenticated llama C-API callback (exact token6205 pos0, same verified GGUF/runtime library/exporter SHA and tensor shape/stride) should capture `beta_sigmoid-0` and `z-0` first, then normalized Q/K and GDN output if available. Only after those values exist can one determine the **first actual semantic or arithmetic mismatch** against CPU/PP2. Position 1 (not another pos0 replay) is needed to test temporal decay and nonzero conv history.

**Minimal correction recommendation:** do **not** change GDN head grouping, convolution tap order, beta/softplus sign, recurrent equations, or numeric release tolerances. Prioritize proving actual llama NVFP4 matvec operand quantization for beta and gate and capturing their outputs; retain Q8_1 activation quantization as the leading **testable precision-contract hypothesis**, not as a proven whole-model repair. GPU execution is prime-exclusive, not granted here. The independent full-logit release gate remains **NO-GO**.
