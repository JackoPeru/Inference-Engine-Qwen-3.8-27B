# E1/E2 — Full-model su checkpoint reale (stato 2026-11-13: PRE FILL MATCH, DECODE PASS)

## E1 (Python/torch CPU fp32, equazioni transformers verificate)
- Prefill 11 token (`Spiega in 2 frasi la gravita.`, ids raw senza template):
  golden corrente (nibble halves + split interleaved): top5 [898,34566,22197,29069,73]
  vals [19.23,18.79,18.34,17.4,17.36], mean -2.05 std 3.30.
  (Il vecchio log `benchmarks/e1_prefill.log` con top5 [3695,...] è stantio: pre-fix nibble/split.)
- Script: `scripts/e1_runner.py`. Decode con carry: `e1_runner.py` crash shape (non bloccante, vedi sotto).

## E2 (C++ `ModelFwd`, stessa matematica, confronto top5 vs E1)
- Loader: riusa parser strict M1 + mmap; dequant NVFP4 on-the-fly per riga.
- Head BF16 shardata in 8 (evita temp 5GB).
- Bug trovati e fixati 2026-11-13 (dettagli in `E2_FIXES_2026-11-13.md`):
  softplus `exp(-v)`→`exp(v)`; gate full-attn SiLU→sigmoid; decode Q doppio RMS→singolo;
  decode GDN doppio GEMM→singolo con buffer `oe`; dump BLK0 scope fix.
- Test `test_model_fwd`: top5 `[898,34566,22197,29069,73]` vals `[19.23,18.80,18.34,17.39,17.36]` → **MATCH E1 PASS**.
- Bisect blk.0: qkv 3.8e-05, cv 1.9e-05, beta 5e-07, gl 6.2e-06, o 1.4e-06, core 7.6e-06, y 1.5e-05 PASS.
- Decode T=3 (`/tmp/test_decode_quick`): prefill_vs_incr 1.57e-03 PASS; decode_vs_teacher 1.85e-04 top 101265==101265 PASS.

## W4A4 CUTLASS (chiuso)
- `test_nvfp4_w4a4` PASS: vs fp32 rel 0.46 (rumore quant 4+4bit atteso),
  vs dequant max err **0.0038** (kernel+builder esatti).
- Bug reali trovati: nibble order A (j/j+8 -> 2j/2j+1), `rs` test raddoppiato.
- Struct probe: SFA/SFB mapping esatti, const 8192, order 1408, buildB 0.0.

## Prossimo
E2-decode (carry) vs E1, MTP draft (wiring NextN documentato), server M3.
