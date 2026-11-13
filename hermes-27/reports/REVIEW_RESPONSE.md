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

## C. Correttezza kernel e test — QUASI CHIUSO
- `kernels/src/gdn_step.cu`: eliminato `return` prima di `__syncthreads` (predicato `active`). `test_gdn_step` PASS incl. caso V=16<128 che hangava.
- Full-logits E1vsE2 11×248320 ESEGUITO: maxerr 7.75e-03 (E1 fp16), meanerr 4.17e-04; argmax 11/11 SAME, top50 50/50 (`cmp_logits.py`). Lunghezze 3,4,11,12,13,14,15 coperte; secondo prompt distinto rimandato a fase GPU (no CPU ridondanti).
- Convenzione norme diretta `rms(x)*w` (llama `build_norm`, pesi ~1): golden (1+w) invalidato e rigenerato (`NORM_CONVENTION.md`).
- Baseline EXL3 solo per perf/comportamento, mai per numerica.

## D. MTP — semantica fixata, K=1 ACCEPT
- `test_mtp_accept`: exit 0/1/2 (identita`/invalido/mismatch). K=1 diretto: draft 17 == teacher 17 → **ACCEPT exit 0**; wiring vs llama `graph_mtp` confermato (`MTP_WIRING_AUDIT.md`).
- Rollback/sampling-residua e sweep K=1..5 pending. Codice MTP conservato; ottimizzazioni dopo P1.

## P1-P5
- P1 (runtime GPU PP2-first, TP2 mantenuto): disegno in `docs/M3_PP2_PLAN.md`; implementazione dopo chiusura validazione CPU/GPU-numerica. Prerequisito: finestra GPU libera (`CUDA_VISIBLE_DEVICES=1`), servizi da fermare comunicati prima — nessuna validazione dichiarata senza test reale.
- E2 full diretto: top5 `[220,1374,27053,17,1147]` = golden E1 → MATCH PASS (log `/tmp/e2_full_direct.log`).
- Decode T=3 diretto: prefill_vs_incr 1.57e-03 PASS; decode_vs_teacher 1.85e-04 top 101265 PASS.

## Corrispondenza Windows <-> server
- Sincronizzati: CMakeLists, model_fwd hpp/cpp, gdn_step.cu, nvfp4_gemm hpp/cu, test_model_fwd/mtp_accept/logits_dump, KERNEL_AUDIT, tutti i reports, tutti gli scripts/, MTP_WIRING_AUDIT, M3_PP2_PLAN, e1_runner.py.
- Eccezione nota: root contiene helper locali (test_*.cpp, *_dump.py, patch_*.py) non nel repo server — intenzionale, non parte del runtime.

## P4 GPU decode — CHIUSO (2026-11-13)
- Causa radice del FAIL iniziale (0.387 vs 2e-3): i due kernel conv GDN
  (batched vs incremental) generavano FMA in modo diverso (1 ulp a tok3,
  primo window pieno; amplificato a 1.6e-4 layer26 e 0.1-0.5 ai logit).
  Fix: catene `__fmaf_rn` esplicite identiche in entrambi (stesso ordine j).
- Verifiche post-fix (GPU1, pesi reali): bisect11 11/11 BIT-EXACT vs prefill;
  layerbisect 0/64 divergenze; greedy 4 tok [220,17,1374,220] tutti finite,
  ~654ms/step, VRAM stabile (no leak); GATE3 top5 MATCH E2; GATE4 invariato
  (maxerr 6.55, argmax 9/11: rumore bf16-activation+W4A4, non regressione).
- Dettagli: `distributed/P4_DECODE.md` sez. 6.

## P5 batching + secondo prompt GPU (2026-11-13)
- Decode concorrente N=2/4 bit-exact vs single-stream; agg 1.53→1.56 tok/s
  (x1.00-1.02): collo host-bound (malloc/sync per layer), non GPU.
  Workspace persistenti introdotti (2.07 MiB/req, no leak); passo successivo
  CUDA Graphs + W4A4 hot path, non altro batching.
- Secondo prompt EN su GPU PP2: top5 `[2250,279,271,8323,198]` = E1 MATCH.
- Servizi produzione intatti per tutta la sessione (nessun restart).
