# P5 BATCH concorrente PP2 — GATE1+GATE2 PASS (misurato, 2026-11-13)

Obiettivo: decode concorrente reale su PP2 (2x RTX 5060 Ti), misurato.
Nessun servizio toccato (no stop/restart/kill, no Tabby/GPU-Manager).
VRAM <15GB/GPU sempre, abort se superi. 1 job E1 in background (~17GB) non disturbato:
solo lavoro CPU leggero, build -j4 brevi, nessun test CPU pesante.

## 0. Disegno (scelta documentata)

Estensione `distributed/` (riuso pesi/kernel esistenti, nessuna duplicazione pesi):
- `pp2_batch.hpp/.cu` (nuovi): `BatchState { N, vector<DecodeState> reqs }` = N stati
  S/CS/KV separati per request (estensione di `DecodeState` 1-richiesta).
  `batch_init/free` (loop `decode_init/free` + check VRAM>15GB abort),
  `batch_step` sequenziale (loop `decode_step`, v1 onesta),
  `batch_step_parallel` (N host threads su `decode_step`, overlap pipeline reale:
  mentre GPU1 fa seconda meta` di req A, GPU0 fa prima meta` di req B),
  `admit_check` (16KB/tok/GPU +75MB/req/GPU, cap 16@1K/8@8K/2@32K da VRAM_BUDGET_V2),
  `vram_used` (cudaMemGetInfo per report no-leak).
- `pp2_fwd.cu` (modifica minima additiva, 3 righe): `+ #include <mutex>`,
  `+ static std::mutex g_xfer_mu`, xfer `g_pinned` reso atomico sotto lock in
  `decode_step` (stesso pinned condiviso, compute GPU0/GPU1 fuori dal lock ->
  overlap reale, single-stream invariato a meno di lock 0.06ms).
- `test_pp2_batch.cu` (nuovo): `g1` correttezza N=2, `g2` throughput N=1/2/4 seq/par,
  `all` entrambi. Schema esistente, build -j4.
- `CMakeLists.txt`: solo aggiunta target `test_pp2_batch` (+ `find_package(Threads)`),
  schema esistente.

Prompt (stessa lunghezza 11 per pos allineate, documentati):
- A baseline P2/P4: `[6205,161421,303,220,17,1374,9911,1147,27053,6059,13]`
- B = reverse(A): `[13,6059,27053,1147,9911,1374,17,220,303,161421,6205]`
- C = rotl(A,3): `[220,17,1374,9911,1147,27053,6059,13,6205,161421,303]`
- D = rotl(A,6): `[9911,1147,27053,6059,13,6205,161421,303,220,17,1374]`

Comandi esatti (PowerShell locale, remote fra doppie virgolette, mai `|` fuori):
```
scp .\distributed\pp2_batch.hpp matteo@hermes:/home/matteo/hermes-27/distributed/pp2_batch.hpp
scp .\distributed\pp2_batch.cu matteo@hermes:/home/matteo/hermes-27/distributed/pp2_batch.cu
scp .\distributed\test_pp2_batch.cu matteo@hermes:/home/matteo/hermes-27/distributed/test_pp2_batch.cu
scp .\distributed\pp2_fwd.cu matteo@hermes:/home/matteo/hermes-27/distributed/pp2_fwd.cu
scp .\CMakeLists.txt matteo@hermes:/home/matteo/hermes-27/CMakeLists.txt
ssh matteo@hermes "cmake -S /home/matteo/hermes-27 -B /home/matteo/hermes-27/build"
ssh matteo@hermes "cmake --build /home/matteo/hermes-27/build --target test_pp2_batch -j4"
ssh matteo@hermes "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv"
ssh matteo@hermes "timeout 600 /home/matteo/hermes-27/build/test_pp2_batch g1"
ssh matteo@hermes "timeout 600 /home/matteo/hermes-27/build/test_pp2_batch g2"
# util sampled (leggero, non secondo run pesante):
ssh matteo@hermes "nvidia-smi dmon -s u -c 200 > /tmp/dmon_batch.log 2>&1 & timeout 550 /home/matteo/hermes-27/build/test_pp2_batch g2; kill %1; cat /tmp/dmon_batch.log"
```

## 1. GATE1 correttezza N=2 — PASS (1 iterazione, 0 debug)

Misurato (mai stimato):
```
UPLOAD sec=15.70 (cold, poi 6.95 warm) bytes0=9416946688 (8.770 GiB) bytes1=10266418176 (9.561 GiB)
 checksum blk.0.attn_norm 4949.16973 vs 4949.16973 diff 0.0, output_norm 9953.98611 vs 9953.98611 diff 0.0
VRAM post-upload used0=9748 MiB used1=11361 MiB
REF A gen=[220,17,1374,220] wall15=9885.8ms (11 warmup+4 greedy, 659.1ms/step, atteso 654ms)
REF B gen=[17,220,303,17] wall15=9868.5ms (657.9ms/step)
TOK12 single A=220 B=17
batch N=2 used0=9964 MiB used1=11577 MiB (delta +216/+216 vs post-upload, <15360 PASS)
GATE1 k=0 pos11 A maxerr=0.000000 mean=0.0 arg 17 vs 17 SAME | B 0.000000 arg 220 vs 220 SAME PASS
GATE1 k=1 pos12 A 0.000000 arg 1374 vs 1374 SAME | B 0.000000 arg 303 vs 303 SAME PASS
GATE1 k=2 pos13 A 0.000000 arg 220 vs 220 SAME | B 0.000000 arg 17 vs 17 SAME PASS
GATE1 k=3 pos14 A 0.000000 arg 17 vs 17 SAME | B 0.000000 arg 220 vs 220 SAME PASS
BATCH N=2 4-step wall=5206.2ms (8 tok) agg=1.537 tok/s
PAR sanity pos11 A 0.000000 SAME | B 0.000000 SAME PASS (thread-safe, stessi kernel)
VRAM dopo free 9750/11363 MiB (delta +2/+2 vs post-upload, no leak)
smi fuori run 486/1289 MiB prima==dopo, servizi intatti
GATE1 PASS (soglia 1e-4 bit-exact come decode-vs-prefill)
```

Nota gen: `GEN batch A=[17,1374,220,17] vs ref [220,17,1374,220]` e` shift di definizione
(ref gen[0]=tok12 input, batch gen[0]=output pos11); i confronti logits sono allineati
(stesso input tok12, stesso pos) e bit-exact 0.0, arg SAME ovunque.

## 2. GATE2 throughput N=1/2/4 — PASS (misura onesta, no speedup v1)

Baseline single-stream P3: 1.53 tok/s (654ms/tok). Agg = N*steps/wall.
```
N=1 seq 4-step wall=2621.3ms (4 tok) agg=1.526 tok/s (x1.00) ms/tok_agg=655.3 vram 9856/11469
N=2 seq wall=5184.4ms (8 tok) agg=1.543 tok/s (x1.01) ms/tok=648.0 vram 9964/11577 prompts=A,B
N=2 par wall=5204.2ms (8 tok) agg=1.537 tok/s (x1.00) ms/tok=650.5 vram 9964/11577 prompts=A,B
N=4 seq wall=10411.2ms (16 tok) agg=1.537 tok/s (x1.00) ms/tok=650.7 vram 10178/11791 prompts=A,B,C,D
N=4 par wall=10258.8ms (16 tok) agg=1.560 tok/s (x1.02) ms/tok=641.2 vram 10178/11791 prompts=A,B,C,D
```
VRAM sempre <15360 MiB (max 10178/11791 N=4), post-free 9750/11363 (delta -428/-428 vs pre,
torna a post-upload +2/+2, no leak). smi fuori 486/1289 prima==dopo.

GPU util (dmon -s u, 200 campioni durante g2):
- pattern dominante alternato 0/99 (come P3 baseline, pipeline senza overlap di microbatch),
- con `par` campioni con entrambe GPU 98-99% simultanee (overlap reale GPU0 reqB + GPU1 reqA),
  ma wall quasi identico a seq (x1.00-1.02): l'overlap e` limitato da malloc/free per-layer
  per-step + cudaDeviceSynchronize per-layer in `run_*_decode_layer` (host-bound, non fuso).
  Misura onesta: loop v1 non basta per speedup; fusione solo se i numeri la giustificano ->
  qui non la giustificano senza refactoring (vedi §4).

Ammissione (da VRAM_BUDGET_V2, verificata in test):
```
16@1K ADMIT (need 1200MB/GPU), 17@1K REJECT, 8@8K ADMIT, 2@32K ADMIT, 4@32K REJECT
N=1 need 75MB/GPU ADMIT, N=2 150MB ADMIT, N=4 300MB ADMIT
```

## 3. File creati/modificati, tempi, blocchi

Creati (nuovi, mai toccata produzione `architecture/*`, `scripts/*`, `reports/*`, kernel):
- `distributed/pp2_batch.hpp` (BatchState, batch_init/free/step/par, admit_check, vram_used)
- `distributed/pp2_batch.cu` (loop seq + threads par, check 15GB, cap 16@1K/8@8K/2@32K)
- `distributed/test_pp2_batch.cu` (g1/g2/all, prompt A/B/C/D documentati)
- `distributed/P5_BATCH.md` (questo)

Modificati (solo aggiunte, nessuna logica single-stream cambiata):
- `distributed/pp2_fwd.cu` (+ `#include <mutex>`, `+ static mutex g_xfer_mu`,
  xfer `g_pinned` atomico sotto lock, compute fuori lock per overlap)
- `CMakeLists.txt` (+ `find_package(Threads)`, `+ test_pp2_batch` con `pp2_batch.cu`, schema esistente)

NO commit (fa orchestratore). Mai eseguiti job vietati
(`test_model_fwd`, `test_logits_dump*`, `test_mtp_accept`, `test_decode_quick`, `e1_*.py`, `blk0_stats.py`).
Mai due run GPU pesanti in contemporanea (g1 poi g2 sequenziali, dmon leggero in parallelo).
Totale task <120min: build -j4 OK (solo warning pre-esistenti), g1 ~60s, g2 ~150s, 0 iterazioni debug
(1 build PASS, GATE1 PASS prima prova, GATE2 misura diretta).

## 4. Cosa resta (P5 serving, non in questo stage)

- Ammit/queue/backpressure vera: `admit_check` qui e` solo calcolo (cap 16/8/2) + log,
  manca coda (max 32, timeout 60s, 429+Retry-After), contatori `kv_tokens_attivi/gdn_slots`,
  scheduler continuous batching + fairness (decode>prefill, mai evict attivi), arena `memory/`
  per-device. Stima §2.2 V2 resta STIMA (runtime 0.75, transient 0.93 da validare con
  `cudaMemGetInfo` a vuoto/post-upload/warmup).
- CUDA Graphs per decode step: oggi ogni step fa decine di `cudaMalloc/Free` + `cudaDeviceSynchronize`
  per layer (dominano 654ms, GEMV-bound ma host-bound). Servono: workspace persistente,
  stream dedicati + events (non default + sync), graph capture del loop 64L, pinned pool per-req
  (non singolo `g_pinned` + lock). Senza questo, overlap `par` resta x1.02 (misurato).
- Prefix caching con stati GDN: S non condivisibile tra prefissi diversi (a differenza di KV),
  CoW solo se matematicamente identico; da progettare con hash prefisso + refcount.
- MTP sweep K=1..5 acceptance + sampling con distribuzione target: ancoraggio noto
  (MTP teacher next=17 dopo top 220, P3), ma sweep non eseguito (fuori stage batch).
- Continuous batching vera (MTP sweep la blocca in parte): serve pipeline con lunghezze
  diverse (poss per-request gia` supportate in `batch_step_parallel`, ma test qui usa
  stessa lunghezza 11 per pos allineate; prompt misti + early-exit + padding/chunked prefill
  8K restano da implementare e misurare).

Cosa blocca il resto:
- MTP sweep: richiede `test_mtp_accept` (vietato ora, CPU pesante) + draft/verify su GPU
  con stati KV/GDN separati per draft; senza sweep, nessun dato acceptance per dimensionare
  speculazione in serving.
- Continuous batching vera: bloccata da (a) mancanza code/backpressure/serving SSE
  (`/health /metrics`), (b) mancanza CUDA Graphs/workspace persistente (per-step overhead
  uccide il guadagno con N variabile), (c) prefix caching GDN non banale. Il batch N fisso
  qui prova correttezza stati separati + misura onesta (x1.00), ma non e` serving dinamico.
