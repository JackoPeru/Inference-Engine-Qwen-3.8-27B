# M3 PP2 plan — default pipeline-parallel a 2 stadi

Stato: piano misurabile, niente mock. Prerequisito: finestra GPU libera (vedi sez. 5).
Baseline e sim citate senza inventare nuovi numeri: ogni nuovo run riporta output grezzo.

## 1. Decisione: default PP2

- Topologia misurata (M0): `can_device_access_peer` 0->1 e 1->0 false, topo PHB.
  P2P assente: ogni allreduce TP passa per host pinned.
- Costo TP2 misurato (`scripts/tp2_shard.py`, `host_comm.py`, GPU1):
  host-staged ~145 us/layer per messaggi piccoli, quindi 2 allreduce/layer x 64 layer
  danno ~18.6 ms/tok di soli allreduce. PP2 xfer ~0.6 ms per confine stadio.
- Sim scheduler calibrata (`scheduler/sim.py`, CHUNK 512, MAXTOK 4096):
  conc4 tp2 70.7 vs pp2 103.5 tok/s (misurato EXL3 121);
  conc8 tp2 84.5 vs pp2 104.3 (misurato EXL3 101.7).
  PP2 batte TP2 a ogni concorrenza nel sim e resta sotto il misurato a conc4,
  quindi il sim e` conservativo e non ottimistico.
- Baseline EXL3 (TabbyAPI, MTP attivo, non distruttivo):
  single ~71 tok/s, TTFT stream 0.265-0.27 s;
  long 1K prefill 400 tok/s decode 101.7 tok/s;
  long 8K prefill 658.8 tok/s decode 85.0 tok/s;
  conc2 93.5, conc4 121.3, conc8 101.7 tok/s agg, p50 2.34 s p95 2.51 s, 0 errori.
- Conclusione: PP2 e` il default M3. TP2 resta percorso alternativo solo se un run
  futuro su finestra libera dimostrasse allreduce reale sotto il budget qui fissato.

## 2. Budget VRAM per scheda (tetto 16 GB, backpressure obbligatoria)

Usabile reale: 16311 MiB (nvidia-smi) / 15849 MiB (torch). Niente pool unificato.

| Voce | PP2 stadio (32 layer) | Nota |
|------|----------------------|------|
| Pesi shard | 10.35 GiB/GPU (misurato `tp2_shard.py`, 866 tensori, BAD none; supera la vecchia stima 9.5) | PP2 split 32+32 layer, stesso ordine di grandezza |
| KV pool FP8 paged | ~2.5 GB | 16 full-attn layer x 4 KV head x 256 dim; cap + paged, mai preallocare 262K nativi |
| DeltaNet + conv state pool | ~0.5 GB | stati separati da KV, CoW solo se matematicamente identici |
| Transient prefill (chunk 512) | ~1.0 GB | chunk fisso, non 8K interi |
| MTP draft + graph | ~0.5 GB | 1 layer blk.64 |
| Riserva CUDA + margine | ~1.0 GB | include reserve 96 MB/stadio stile autosplit |
| Totale | ~15.0-15.5 GB < 16 GB | OK solo con backpressure; oltre il budget: admit/drop, mai OOM |

Regola: ogni run M3 stampa il budget effettivo e il punto di intervento del backpressure.
Prefix caching con stati ricorrenti solo se identita` matematica provata (cfr MTP_WIRING_AUDIT V5
per il principio applicato agli stati).

## 3. Coda: continuous batching + chunked prefill

- Parametri fissati dal sim: CHUNK 512, MAXTOK 4096.
- Politica: coda con admit/drop, fairness documentata, KV/state budget check prima di ammettere;
  prefill in chunk da 512 token interleavati al decode; TTFT target <= sim conc4 ~0.5 s
  su finestra libera (confronto, non SLO contrattuale finche` non misurato sul runtime).
- Cosa riusare: contratto `serving/validate.py` (PASS: validazione OpenAI + SSE) come specifica
  del layer HTTP nativo; sim `scheduler/sched_result.json` come riferimento policy, non come prova runtime.

## 4. Gap runtime (manca per M3-done)

Continuous batching reale nel runtime, chunked prefill, KV/state budget, admit/drop/backpressure,
API streaming OpenAI-compat, `/health`, `/metrics`, cancel, rate-limit, CUDA Graphs per batch shape,
prefix caching condizionale. Richiede M2-done-GPU (forward full-model su GPU vs golden) + finestra GPU.

## 5. Prerequisiti finestra GPU (bloccanti)

- GPU occupate dalla produzione: `hermes-tabby`, `kokoro-tts`, `uninote-stt`
  (~12-13.5 GB/16 GB per scheda al momento degli audit; GPU0 ~208 MB free).
- Ogni run GPU: `CUDA_VISIBLE_DEVICES=1`, non distruttivo, senza fermare i servizi;
  se la VRAM libera su GPU1 e` insufficiente, il run non parte e M3-runtime resta bloccato.
- Sequenza: prima M2-done-GPU (logit GPU == golden), poi run M3. Nessun test GPU dichiarato
  PASS senza output su hardware reale.

## 6. Milestones M3-done (criteri misurabili, niente mock)

- M3-a Coda continuous batching: soak conc2/4/8 sullo stesso carico del baseline EXL3
  (64 tok/req single, 256 tok long, 32 tok/req conc8), 0 errori, p50/p95 riportati,
  confronto agg tok/s vs 93.5 / 121.3 / 101.7. PASS se il confronto e` pubblicato con log grezzi.
- M3-b Chunked prefill CHUNK 512: long 1K e 8K con prefill/decode tok/s riportati
  vs 400/101.7 e 658.8/85.0. PASS se nessun OOM e TTFT misurato riportato.
- M3-c Budget + backpressure: run oltre budget (KV pool saturo) deve rifiutare/accodare,
  mai OOM. PASS se il punto di intervento e` nel log con contatori `/metrics`.
- M3-d API: OpenAI-compat streaming + `/health` + `/metrics` + cancel + rate-limit,
  verificati da `serving/validate.py` esteso sul runtime nativo. PASS se tutti i casi passano
  contro il server reale (non mock).
- M3-e CUDA Graphs: decode shape fissa con graph replay, identita` logit vs non-graph
  entro tolleranza fissata nel test. PASS con err riportato.
- M3-f Prefix caching: solo se hit-rate e identita` matematica provate su stati
  DeltaNet + KV; altrimenti resta disabilitato e M3 chiude senza di esso.
- M3-g Decisione PP2 finale: PP2 vs TP2 misurati sullo stesso carico conc4/conc8 su finestra
  libera. PASS se PP2 >= TP2 in agg tok/s o se la tassa allreduce resta ~18.6 ms/tok.

## 7. Comandi di run (placeholder, output non precompilati)

Eseguire solo in finestra GPU libera, con `CUDA_VISIBLE_DEVICES=1`. Output attesi: nessuno
precompilato qui; ogni run allega log grezzo.

- `cmake -S . -B build && cmake --build build -j6`
- `CUDA_VISIBLE_DEVICES=1 ctest --output-on-failure`
- `python scheduler/sim.py --chunk 512 --maxtok 4096`
- `python serving/validate.py --base-url http://127.0.0.1:8000`
- Soak M3 (placeholder, carico = baseline M0): `python scripts/bench_m3_soak.py --conc 2,4,8`
