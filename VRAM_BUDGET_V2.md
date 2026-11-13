# HERMES-27 VRAM_BUDGET_V2 — per-GPU 2x RTX 5060 Ti 16GB (PP2 default, TP2 alt)

Data: 2026-11-13. Metodo: SOLO letture leggere (header GGUF, mai modello caricato).
Legenda: **MISURATO** = byte esatti da header GGUF / log esistenti; **STIMA** = da validare con `cudaMemGetInfo` a GPU libera.
Limite hard: `nvidia-smi` riporta 16311 MiB/GPU = 15.93 GiB (MISURATO oggi: GPU0 13478 MiB usati, GPU1 11843 MiB — validazione 16GB in corso, non disturbare).
Target: ogni GPU resta <16 GiB nominali con margine; arene separate, niente pool condiviso. Torch usable 15849 MiB = 15.48 GiB (da `docs/VRAM_BUDGET.md`).

Fonti: `scripts/tp2_shard.py` (letto), `docs/VRAM_BUDGET.md`, `reports/M2.md`, `kernels/src/nvfp4_gemm_w4a4.cu` (letto), `/tmp/list_tensors.py` + `/tmp/vram_acct.py` (solo header).

## 1. Riconciliazione 9.5 vs 10.35 GiB (MISURATO)

- File: 19694360544 B = 18.3418 GiB. Tensori: 19683364864 B = **18.3316 GiB MISURATO** (delta 10.5 MB header/metadata).
  Split tipo MISURATO: NVFP4 12.7503 GiB (69.5%), BF16 5.5273 GiB (30.1%), F32 0.0539 GiB (0.3%).
- Ideale dimezzato: 18.3316/2 = **9.1658 GiB**.
- TP2 reale (stesse regole `tp2_shard.py`, BAD none): **10.3548 GiB/GPU MISURATO** (11118424064 B, 866 tensori: col 374 / row 131 / rep 361).
- Delta ideale->reale: **+1.1890 GiB = replicati/2**. Replicati totali MISURATO: **2.3781 GiB**, di cui `token_embd.weight` 2.3682 GiB (5120x248320 BF16) domina; resto ~10 MiB (norm vettoriali `[256]`, bias, `ssm_conv1d` F32 0.16 MiB x48, `ssm_a/dt`).
  Nota: `output.weight` (2.3682 GiB) è **row-split** quindi dimezzato in TP2, non replicato — solo l'embed penalizza.
- Da dove viene “9.5”: `docs/VRAM_BUDGET.md` era upper-bound pre-dump: ~27B param, 90% NVFP4 @0.5625 B/elem (~13.5 GB) + 10% BF16 (~5.4 GB) = ~18.9 GB /2 ≈ 9.5 GB decimali (= 8.85 GiB). Errori: (a) assume split perfetto, ignora replicati (+1.19 GiB); (b) mix reale 69.5/30.1 non 90/10 — MTP blk.64 BF16 0.7911 GiB + embed/head BF16 pesano; (c) confonde GB/GiB (~7%): 9.5 GB = 8.85 GiB vs 9.1658 GiB ideale. Quindi 9.5 era STIMA, 10.35 è MISURATO header-only al byte.

Dettagli MISURATO (`/tmp/vram_acct.py`): embed 2.3682 GiB, output+norm 2.3682 GiB, MTP blk.64 0.7911 GiB (810.10 MiB), avg layer 0-63 204.87 MiB (blk.0 206.58 MiB GDN, blk.3 199.73 MiB full-attn), sum blk0-63 12.8041 GiB.

## 2. Budget per GPU

### 2.1 Pesi per partizione (MISURATO pesi, resto STIMA)

PP2 = split per layer, niente replicati embed/head (ognuno su una sola GPU):

- **PP-A 32/32** (embed+0-31 | 32-63+MTP+head): GPU0 **8.7702 GiB**, GPU1 **9.5613 GiB**, max 9.5613.
- **PP-B 33/31** (embed+0-32 | 33-63+MTP+head): GPU0 **8.9720 GiB**, GPU1 **9.3596 GiB**, max 9.3596 — **RACCOMANDATA** (minimizza max, +0.20 GiB margine vs A; KV full-attn ancora 8+8; GDN 25 vs 23 layer, sbilancio 3 MB/req trascurabile).
- PP-C 31/33 (embed+0-30 | 31-63+MTP+head): GPU0 8.5752, GPU1 9.7564, max 9.7564 — scartata (GPU1 troppo piena con MTP+head).
- **TP2**: **10.3548 GiB/GPU** entrambi.

Raccomandazione: **PP-B default** (coerente con PP2 favorito: P2P assente, xfer PP2 ~0.6ms MISURATO vs TP2 18.6ms/tok allreduce host 145us/layer — da `reports/M2.md`). PP-A accettabile se si vuole simmetria pipeline 32/32 a costo di 0.20 GiB margine.

### 2.2 Voci comuni (per GPU, GiB)

| Voce | TP2 /GPU | PP-B GPU0 / GPU1 | Fonte |
|---|---|---|---|
| Pesi | 10.3548 / 10.3548 | 8.9720 / 9.3596 | MISURATO header |
| Runtime CUDA (ctx, cublas, graph) | 0.75 | 0.75 / 0.75 | STIMA (torch 15849 vs smi 16311 = 0.45 + 96 MB reserve citata + driver variance; validare con `cudaMemGetInfo` a vuoto) |
| Transienti/attivazioni (prefill chunk 8K: 81.9 MB/tensore 5120x8K BF16 x3-5 live + attn tmp) | 0.93 | 0.93 / 0.93 | STIMA (1.0 GB da vecchio budget; chunk 8K fisso, decode T=1 trascurabile) |
| KV cache pool (cap, vedi §3) | variabile | variabile | MISURATO formula: full 32 KB/tok/req (16 layer x4 head x256 x2 x1B FP8); sharded **16 KB/tok/GPU** |
| Stati GDN+conv (FP32 48x48x128x128x4B=144MiB + conv 5.76MB≈6MB =150MB/req full) | variabile (75 MB/req/GPU) | variabile (GPU0 25 layer, GPU1 23 layer: ~78/72 MB/req) | MISURATO formula, da confermare dimesito con sharding teste (col-split `in_proj/ssm_beta`) |
| MTP extra runtime (draft verify, KV propria) | 0.20 | 0.00 / 0.20 | STIMA trascurabile pesi già inclusi; validare |
| Workspace CUTLASS persistente | 0.06 | 0.06 / 0.06 | STIMA da `nvfp4_gemm_w4a4.cu`: `run()` fa `cudaMalloc(ws)` transient per GEMM (`get_workspace_size` SM120 128³ tip. KB); 64 MB copre pool + Aq(20MB)+SFA(2.5MB) prefill 8K |
| Copie/repacking residenti | 0.00 | 0.00 / 0.00 | STIMA disegno single-copy: `build_B_from_gguf` produce Bq lineare (0.5 B/elem) + SFB swizzled (~0.0625 B/elem) = 0.5625 B/elem = stessa size file; 0 extra se si tiene solo copia device swizzled + mmap host. Naive double-copy costerebbe +0.7 GiB/GPU TP2 — VIETATO |
| Margine sicurezza | 1.00 | 1.00 / 1.00 | Policy (vecchio budget), non allocato |

Esempio totale GPU con 4 req @1K (KV 16MB + GDN 75MB = 91MB/req/GPU ≈ 0.36 GiB):
- TP2: 10.3548+0.75+0.93+0.36+0.20+0.06+1.00 = **13.65 GiB** < 15.93 OK.
- PP-B GPU1 (worst): 9.3596+0.75+0.93+0.34+0.20+0.06+1.00 = **12.64 GiB** OK. GPU0 ~12.25 GiB.
Margine residuo: TP2 ~2.3 GiB, PP-B ~3.3 GiB prima del cap (§3 dettaglia saturazione).

## 3. Tabella concorrenti (contesto tipico 1K; GDN fisso, KV ∝ T)

Per-GPU sharded @1K: 91 MB/req (16 MB KV + 75 MB GDN). Totali per GPU = pesi + fissi (runtime 0.75+transient 0.93+ws 0.06+MTP 0.20 TP2 / 0.20 solo GPU1 PP) + 1.00 margine + variabile. Fissi TP2 = 12.2948 + margine; PP-B GPU1 = 11.2996 + margine.

| req | var/GPU @1K | TP2 tot/GPU | PP-B GPU0 tot | PP-B GPU1 tot (worst) | esito <16GiB |
|---|---|---|---|---|---|
| 1 | 0.08 GiB | 12.68 | 11.63 | 12.02 | OK entrambi, margine >3.9 |
| 2 | 0.17 | 12.77 | 11.72 | 12.11 | OK |
| 4 | 0.36 | 13.65* | 12.07 | 12.64 | OK (esempio §2.2; *TP2 include MTP 0.20) |
| 8 | 0.71 | 14.01 | 12.77 | 13.34 | OK; GDN pool 0.6 GB occupato |
| 16 | 1.36 | **14.66** | 13.48 | **14.05** | OK @1K con backpressure; TP2 margine residuo 1.27 GiB (smi) |

Nota 8K/32K (KV scala con T, GDN no). Per-GPU/req: @8K = 128 MB KV + 75 MB = 203 MB; @32K = 512 MB + 75 MB = 587 MB.
- @8K: 4 req = 0.79 GiB → TP2 14.09 OK; 8 req = 1.59 GiB → TP2 14.89 OK limite; **16 req = 3.18 GiB → TP2 16.48 OOM, PP-B GPU1 15.48 borderline — VIETATO, serve cap**.
- @32K: 1 req = 0.57 GiB → TP2 13.87 OK; 4 req = 2.29 GiB → TP2 15.59 borderline torch (15.48) — ammettere solo con coda; 8 req = 4.58 GiB → OOM certo.
Regola: ridurre max req proporzionalmente a T (vedi §4). 262K nativi NON preallocare (obiettivo, non size iniziale — da vecchio budget).

## 4. Ammissione / backpressure (niente OOM, niente pool condiviso)

- Contatori per GPU (arena separata): `kv_tokens_attivi`, `gdn_slots_attivi`, `req_attive`. Stima memoria prima di admit: `need = T_ctx*16KB + 75MB (+78/72MB PP-B)` per GPU; ammetti solo se `need <= pool_residuo`.
- Cap raccomandati (STIMA da validare in `memory/` arena): **max_req_attive 16 @≤1K, 8 @≤8K, 2 @32K; max_token_attivi 16K @1K (16x1K), 32K @8K (4x8K o 8x4K), 64K @32K (2x32K)**. Prefill chunk 8K fisso; chunked prefill con defer se transient insufficiente.
- Comportamento oltre cap: **coda (priorità decode > prefill), mai crescita incontrollata**. Coda attesa max 32, timeout 60s → **rifiuto esplicito 429 + `Retry-After`** (non drop silente, non OOM). Scheduler: continuous batching + fairness; prefill può essere rimandato/segmentato, decode in corso mai evicted salvo `max_token` superato → allora rifiuta nuovo, non uccide attivi. Prefix CoW solo se matematicamente identico (stati GDN non condivisibili tra prefissi diversi).
- PP2: admission decisa da GPU più piena (GPU1); TP2: identica su entrambe (pesi simmetrici). Backpressure propaga a `serving/` SSE `/health /metrics`.

## 5. Vincolo <16 GiB e margine

- Hard: 16311 MiB smi (15.93 GiB) MISURATO; torch 15849 MiB (15.48 GiB). Budget sopra usa GiB (1024³) e include **1.00 GiB margine non allocato** + fissi conservativi (runtime 0.75, transient 0.93). Residuo @1K/16req: TP2 1.27 GiB, PP-B 1.88 GiB prima di toccare smi — assorbe variance driver/CUTLASS ws.
- Dove sta il margine: (a) PP-B lascia 0.20 GiB vs PP-A; (b) single-copy repack evita +0.7 GiB; (c) chunk 8K fisso evita picchi 32K prefill; (d) pool KV 2.5 GB + GDN 1.5 GB cap impediscono deriva.
- Niente pool condiviso: arene `memory/` per device con limiti configurabili indipendenti; TP2 host-pinned comm (no P2P) non usa VRAM extra oltre staging pinned host (già in 145us/20KB MISURATO).
- Cosa resta da MISURARE su GPU libera (vietato ora — run 16GB in corso): (1) `cudaMemGetInfo` a vuoto / post-upload pesi / post-warmup per fissare runtime reale vs 0.75 STIMA; (2) `Gemm::get_workspace_size` reale per nostre (M_T,N,K) + picco `Aq+SFA` prefill 8K; (3) KV FP8 + GDN FP32 effettivi per req con arena (conferma sharding teste 16KB/75MB); (4) MTP draft overhead decode; (5) conferma PP-B bilanciamento con pesi caricati + TTFT/decode vs PP-A (sim M3).
