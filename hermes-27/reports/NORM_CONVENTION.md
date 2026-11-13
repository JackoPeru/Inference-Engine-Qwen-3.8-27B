# NORM_CONVENTION — Convenzione RMSNorm per Qwen3.8-27B-NVFP4-MTP.gguf

Stato 2026-11-13: per QUESTO file GGUF la convenzione corretta e` moltiplicazione
diretta `y = rms(x) * w`. La convenzione `(1+w)` e` INVALIDATA per questo file.
Dettaglio completo sotto; golden precedente conseguentemente invalidato.

## 1. Medie pesi norm misurate (scala DIRETTA ~1, nessuno ~0)

Medie misurate sui tensori norm del file reale
`Qwen3.8-27B-NVFP4-MTP.gguf`:

| Tensore | Media misurata |
|---|---|
| blk.0.attn_norm | +0.967 |
| blk.3.attn_q_norm | +1.23 |
| blk.64.attn_norm | +1.04 |
| blk.64.attn_q_norm | +1.79 |
| nextn.hnorm | +0.84 |
| nextn.enorm | +0.54 |
| nextn.shared_head_norm | +2.25 |
| output_norm | +1.94 |
| blk.0.ssm_norm | +0.87 |

Lettura: pesi in scala diretta ~1 (range ~0.5–2.3). Nessun tensore ha media ~0.
Con pesi zeros-init ci si aspetterebbe media ~0; qui non e` il caso.

## 2. Riferimento llama.cpp `build_norm`

`build_norm` RMS in llama.cpp fa `rms(x) * w` SENZA `+1` (verificato nel sorgente).
Quindi per pesi in scala diretta ~1 (tabella sopra) la replica corretta e`
moltiplicazione diretta `*w`, senza `plus_one`.

## 3. Perche' `(1+w)` era sbagliato qui (nonostante i test sintetici passassero)

- `transformers` `Qwen3_5RMSNorm` usa zeros-init + `(1+w)`, ma vale per checkpoint
  HF zeros-init, non per questo GGUF (medie ~1, vedi tabella).
- E1 (torch) ed E2 (C++) usavano `(1+w)` ovunque -> scaling ~2x a ogni norm.
- E1 ed E2 erano autoconsistenti (E1==E2 MATCH) perche' applicavano entrambi la
  stessa convenzione sbagliata; i test sintetici passavano per lo stesso motivo
  (quei test usavano pesi zeros-init, autoconsistenti con `(1+w)`).
- Autoconsistenza E1==E2 non prova compatibilita` con llama.cpp / file reale.
  Con pesi reali ~1, `(1+w)` raddoppia circa ogni norm ed e` incompatibile
  con llama.cpp.

## 4. File cambiati (fix a diretta `*w`)

- `scripts/e1_runner.py`, `scripts/e1_teacher.py`, `scripts/blk0_stats.py`,
  `scripts/qkv_cmp.py`: passati a `*w` diretto.
- `architecture/src/model_fwd.cpp`: `rms()` ora ignora `plus_one`
  (sempre diretto) + q/k norm prefill/decode/MTP passate a `qn[d]`/`kn[d]`
  diretti; `ssm_norm` era gia` diretto.

## 5. Stato golden

- Golden precedente E1 top5 `[898,34566,22197,29069,73]` era in convenzione
  `(1+w)` -> INVALIDATO.
- Golden E1 diretto in corso di rigenerazione sul server (non ancora disponibile).
- Evidenza fresca blk0 (diretta, E1==E2) sotto vale come truth corrente
  in attesa del nuovo golden prefill completo.

## 6. Evidenza blk0 diretta (E1==E2, convenzione `*w`)

Bisect blk.0 11 token, convenzione diretta, E1==E2:

- qkv 2.289e-05, cv 7.629e-06, beta 3.576e-07, gl 4.768e-06,
  o 7.153e-07, core 5.722e-06, y 5.722e-06
- y mean 0.00267 std 0.21054 min -0.87 max 19.35 identici E1/E2.
