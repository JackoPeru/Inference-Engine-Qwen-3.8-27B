# E1/E2 — Full-model su checkpoint reale (stato 2026-11-13: MATCH diretto, full-logits, MTP ACCEPT)

## Golden vigente (convenzione diretta `rms(x)*w`, cfr `NORM_CONVENTION.md`)
- E1 torch CPU fp32, prefill 11 token (`Spiega in 2 frasi la gravita.`):
  top5 `[220,1374,27053,17,1147]` vals `[15.74,12.96,12.41,12.19,11.2]`,
  mean -1.902 std 2.027. Golden precedente (1+w) `[898,...]` INVALIDATO.
- Script: `scripts/e1_runner.py` (+`E1_SAVE_LOGITS` salva `/tmp/e1_logits.pt`).
  Decode con carry E1: crash shape noto, non bloccante (carry validato su E2 vs prefill E1).

## E2 (C++ `ModelFwd`)
- `test_model_fwd` top5 identici al golden → **MATCH E1 PASS** (log `/tmp/e2_full_direct.log`).
- Blk.0 diretto E1==E2: qkv 2.29e-05, cv 7.63e-06, beta 3.58e-07, gl 4.77e-06,
  o 7.15e-07, core 5.72e-06, y 5.72e-06; y mean 0.00267 std 0.21054 PASS.
- Bug chiusi 2026-11-13: softplus segno, gate full-attn SiLU→sigmoid, norme (1+w)→(w),
  decode Q doppio-norm→singolo, decode GDN doppio-GEMM→singolo (`E2_FIXES_2026-11-13.md`).
- Decode T=3 diretto: prefill_vs_incr 1.57e-03 PASS; decode_vs_teacher 1.85e-04,
  top 101265==101265 PASS (`/tmp/e2_decode_direct.log`).
- Esteso T=12: E2 top=59 = pred E1 teacher MATCH.
- Full-logits 11×248320: maxerr 7.75e-03 (E1 salvato fp16: floor ~1e-3),
  meanerr 4.17e-04; argmax 11/11 SAME, top50 50/50 ovunque (`cmp_logits.py`).
- Lunghezze coperte: 3, 4, 11, 12, 13, 14, 15 (secondo prompt distinto: pending fase GPU).
- MTP K=1 diretto: draft 17 == teacher 17 → **ACCEPT exit 0** (`test_mtp_accept`,
  `MTP_WIRING_AUDIT.md`). Wiring vs llama `graph_mtp` confermato riga per riga.

## W4A4 CUTLASS (chiuso)
- `test_nvfp4_w4a4` PASS: vs fp32 rel 0.46, vs dequant max err **0.0038**.
- 79a: 512³ 44.5 TFLOPS, 2048³ 257 TFLOPS (MMA vere, non stima end-to-end).

## Prossimo
P1 runtime GPU (PP2-first): single-layer CUDA pesi reali vs golden → forward → decode → benchmark. Finestra GPU: 15.3/14.5GB liberi rilevati, Tabby con modello scarico.
