# HERMES-27 — Registro di validazione (verità misurate, niente mock)

## Formato e layout (verificati sul file reale 19.69GB)
- Parser GGUF strict: PASS su `Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf`
  (v3, 866 tensori, 50 KV, align 32). Bug trovati: ARR STR early-break,
  limite array, arch `qwen35`.
- NVFP4: blocco 36B = 4x UE4M3 + 32B E2M1. Ordine nibble GGUF = halves
  `[L0..L7, H0..H7]` (confermato da llama.cpp `dequantize_block_nvfp4` +
  hand-decode). **Trappola documentata**: ordine pair `(L,H)` è sbagliato;
  E1 lo usava (golden invalidato e rigenerato).
- q_proj: split **interleaved per-head** `[q_h(256) | g_h(256)]` (non halves!),
  verificato vs layer transformers con hook (Q pre-rope match, poi full-block
  err 1.7e-7). Stessa trappola in E1/E2/numpy (tutti e tre 일치 dopo fix).

## Kernel CUDA SM120a (ctest su GPU1)
- rmsnorm/rope/mlp/attn/nvfp4_gemv/gdn_step: PASS (err ≤3e-2, GDN 1e-8).
- RoPE = split-half GPT-J (non interleaved); mRoPE-text = stream identici.
- RMSNorm semplice `(1+w)`; RMSNormGated `w*silu(z)`; qk-l2norm = L2 vera.

## CUTLASS W4A4 NVFP4 (79a, v4.2.2 + CUDA 12.8, arch 120a)
- 79a esempio: 512³ 44.5 TFLOPS, 2048³ 257 TFLOPS (Passed).
- Wrapper engine: repack SFA/SFB via layout CUTLASS + idx2crd (modi M,K,L
  verificati con probe; m/n broadcast confermati). Nibble lineari low-first.
- Test: const 8192 esatto; order 1408 (no swap); buildB 0.0; pattern 0.0;
  w4a4 random PASS (vs dequant **0.0038**; vs fp32 rel 0.46 = rumore 4+4bit).
- Bug trovati: nibble A GGUF-order in quant, `rs` raddoppiato nel test,
  layout call con rank errato.

## Gated DeltaNet (equazioni esatte)
- vs fla.naive_recurrent: 1e-7; chunk vs recurrent fla: 4e-9;
  vs transformers tiny block (conv+gate+recurrence+norm+out): prefill 1.8e-7;
  token-by-token (conv+state carry) vs prefill: 3.7e-8.
- Gate: beta=sigmoid, g=-exp(A)*softplus(a+dt_bias) (confermato in fla gate.py).
- Full-attn tiny vs layer transformers: 1.7e-7 (dopo fix split).

## Modello reale (2026-11-13: E1==E2 PASS)
- E1 (torch CPU fp32): prefill golden top5 `[898,34566,22197,29069,73]`.
- E2 (C++ indipendente): `test_model_fwd` top5 identici → **MATCH E1 PASS**.
- Decode: `test_decode_quick` T=3 prefill_vs_incr 1.57e-03 PASS,
  decode_vs_teacher 1.85e-04 PASS. Teacher-forcing E1 (`e1_teacher.py` rot fix) in run.
- Bug chiusi: softplus segno, gate full-attn SiLU→sigmoid, decode Q doppio-norm,
  decode GDN doppio-GEMM (dettagli `E2_FIXES_2026-11-13.md`).
- MTP blk.64: 15 tensori (full-attn BF16 + eh_proj/enorm/hnorm/shared_head_norm);
  draft `mtp_draft` implementato in `model_fwd`, build OK, validazione dopo teacher.

## Decisioni architetturali (misurate)
- TP2 pesi: 10.35 GiB/GPU (sharding validato, nessun BAD).
- P2P assente → allreduce host 145µs/layer (~18.6ms/tok) → **default PP2**.
- Sim scheduler calibrata: conc4 pp2 103.5 vs 121 misurati.
