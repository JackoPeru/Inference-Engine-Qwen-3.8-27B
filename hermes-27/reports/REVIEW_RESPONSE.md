# Risposta alla revisione indipendente (2026-11-13, lavoro in corso)

Stato server verificato prima di agire: golden E1 diretto rigenerato, E2 rebuild, nessun lavoro duplicato. Servizi produzione intatti (nessun restart, nessuna modifica a TabbyAPI/ExLlama/GPU-Manager/ComfyUI).

## A. Budget VRAM e GDN — CHIUSO (doc), misure GPU pending
- `reports/VRAM_BUDGET_V2.md`: riconciliazione 9.5 vs 10.35 GiB (stima ideale vs misurato con embed replicato 2.37 GiB), PP-B raccomandata 8.97/9.36 GiB, TP2 10.35 GiB/GPU.
- Per-request: GDN 150MB fissi + KV 32KB/tok; tabelle 1/2/4/8/16 req @1K/8K/32K; ammissione max 16@1K, 8@8K, 2@32K, coda 32/timeout 60s poi 429; stime vs misurato separati; repack naive +0.7 vietato (single-copy).
- Da misurare con GPU libera: cudaMemGetInfo, workspace CUTLASS reale, KV/GDN sharded, PP-B vs PP-A.

## B. Overhead kernel CUDA — PARZIALE
- `kernels/KERNEL_AUDIT.md`: malloc/free/sync rimossi dal percorso caldo (`run_ws` + `Workspace` persistente, API compatibile); conversione layout una volta al load dove possibile.
- Quantizzazione attivazioni device-side: prototipo bit-esatto disabilitato di default (`HERMES_ENABLE_DEVICE_QUANT`), repack interleaved ancora host; verifica tiny host-vs-device prima di abilitare.
- I 257 TFLOPS (2048³) restano prova di correttezza MMA, NON stima end-to-end (scritto nel report).

## C. Correttezza kernel e test — PARZIALE
- `kernels/src/gdn_step.cu`: eliminato `return` prima di `__syncthreads` (predicato `active`, tutti i thread alla barriera). `test_gdn_step` PASS incl. caso V=16<128 che hangava.
- Validazione modello: top5 non basta — saver full-logits pronti (`E1_SAVE_LOGITS`, `test_logits_dump`, registrati ma fuori ctest); confronto full-logits + multi-prompt + chunked accodati dopo MTP.
- Convenzione norme corretta a diretto `rms(x)*w` (llama `build_norm`, pesi GGUF ~1): golden (1+w) invalidato e rigenerato; dettagli `reports/NORM_CONVENTION.md`.
- Baseline EXL3 solo per perf/comportamento, mai per numerica (checkpoint diverso).

## D. MTP — semantica fixata, validazione in corso
- `architecture/tests/test_mtp_accept.cpp` (da /tmp, registrato in CMake): exit 0 = finito+sano+identita` greedy K=1; 1 = invalido numerico; 2 = valido ma mismatch. Rollback/sampling-residua e sweep K=1..5 pending (spec `speculation/MTP_WIRING_AUDIT.md`).
- Codice MTP conservato; priorita` a inferenza GPU completa prima delle ottimizzazioni MTP.
- K=1 re-run in convenzione diretta in corso (il REJECT precedente era sotto (1+w), invalidato).

## P1-P5
- P1 (runtime GPU PP2-first, TP2 mantenuto): disegno in `docs/M3_PP2_PLAN.md`; implementazione dopo chiusura validazione CPU/GPU-numerica. Prerequisito: finestra GPU libera (`CUDA_VISIBLE_DEVICES=1`), servizi da fermare comunicati prima — nessuna validazione dichiarata senza test reale.
- E2 full diretto: top5 `[220,1374,27053,17,1147]` = golden E1 → MATCH PASS (log `/tmp/e2_full_direct.log`).
- Decode T=3 diretto: prefill_vs_incr 1.57e-03 PASS; decode_vs_teacher 1.85e-04 top 101265 PASS.

## Corrispondenza Windows <-> server
- Sincronizzati: CMakeLists, model_fwd hpp/cpp, gdn_step.cu, nvfp4_gemm hpp/cu, test_model_fwd/mtp_accept/logits_dump, KERNEL_AUDIT, tutti i reports, tutti gli scripts/, MTP_WIRING_AUDIT, M3_PP2_PLAN, e1_runner.py.
- Eccezione nota: root contiene helper locali (test_*.cpp, *_dump.py, patch_*.py) non nel repo server — intenzionale, non parte del runtime.
