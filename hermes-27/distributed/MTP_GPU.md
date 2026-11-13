# MTP GPU draft NextN blk.64 — GATE1/2/3/4 (misurato, no stime)

Data: 2026-11-13. Obiettivo: draft MTP su GPU (blk.64) + verifica + sweep accept K=1..4.
Nessun servizio toccato (no stop/restart/kill, no Tabby/GPU-Manager).
VRAM sempre <15 GB/GPU. Solo lavoro CPU leggero (upload escluso, build -j4).
Mai eseguiti job vietati (`test_model_fwd`, `test_logits_dump*`, `test_mtp_accept`,
`test_decode*`, `e1_*.py`, `blk0_stats.py`, golden). Nessun run GPU >10 min filato.

## 0. Disegno

File nuovi (mai toccata produzione `architecture/*`, `scripts/*`, `reports/*`):
- `distributed/mtp_gpu.hpp` — API `mtp_gpu::init/free/reset/draft_step`
  (`MtpState` KV draft propria `[MAXL=128 x 1024]` fp32 su GPU1), utility greedy.
- `distributed/mtp_gpu.cu` — un passo draft completo, stesse equazioni di
  `ModelFwd::mtp_draft` (cfr `speculation/MTP_WIRING_AUDIT.md`):
  `h_norm=RMS(h,hnorm)`, `e_norm=RMS(emb[tok],enorm)`,
  `x=eh_proj(concat[e_norm;h_norm])` [10240->5120],
  q split interleaved `[q|g]` per-head, q/k RMSNorm, RoPE mRoPE-text pos assoluta,
  `att*sigmoid(gate)`, out_proj, residui, FFN, `shared_head_norm` + head condivisa.
  Gate=sigmoid (NON silu). QK split interleaved (NON halves). Norme dirette `*w`.
- `distributed/test_mtp_gpu.cu` — binario `test_mtp_gpu [k1|k3|sweepA|sweepB|sweep|all]`,
  ferma al gate fallito.
- `CMakeLists.txt` — solo aggiunta (lib `hermes_mtp` + exe + test, stesso schema).
- `distributed/pp2_fwd.hpp/.cu` — solo aggiunta minimale (accessor, non logica):
  `prefill(..., h_last_out=nullptr)` (ultimo hidden pre-output_norm, +20 KB D2H,
  default-arg: nessun caller esistente cambia comportamento),
  `pp2::get_weight_ptr(...)` (raw device pointer, no copy, no ownership).

Scoperta tipi (via parser GGUF, non indovinata): lineari blk.64 tutte **BF16**
(`attn_q/k/v/output`, `ffn_gate/up/down`, `eh_proj`), norme F32, head condivisa BF16.
Il path GEMV NVFP4 (`nvfp4_gemv_cuda`) NON vale per MTP: nuovo `bf16_gemv_k`
(y=W*x, W row-major BF16, x BF16, accumulo fp64 come `head_bf16_k`).
Pesi riusati residenti PP-A (GPU1, single-copy, +0 byte): solo Kc/Vc draft
(~1 MiB) + workspace per-step (~1.7 MiB, malloc/free per step come `run_full_layer`).

KV draft: solo storia draft (append K post-norm/post-RoPE, `len++`), GQA x6,
pos RoPE assoluta. Conv N/A (full-attn puro). Per K>1 si riusa `h_last` target
fissa + tok precedente (assunzione documentata, vedi sez. 5).

Comandi esatti:
```
cmake -S ~/hermes-27 -B ~/hermes-27/build
cmake --build ~/hermes-27/build -j4 --target test_mtp_gpu
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv   # prima/dopo
timeout 600 ~/hermes-27/build/test_mtp_gpu k1
timeout 600 ~/hermes-27/build/test_mtp_gpu k3
timeout 600 ~/hermes-27/build/test_mtp_gpu sweepA
timeout 600 ~/hermes-27/build/test_mtp_gpu sweepB
timeout 600 ~/hermes-27/build/test_mtp_gpu all
```

## 1. GATE1 — draft singolo K=1 (PASS, 1 iterazione, RC=0)

Prompt base 11-tok `[6205,161421,303,220,17,1374,9911,1147,27053,6059,13]`.
```
prefill11 ms=6792.3 top=220 top5=[220,1374,27053,17,1147] vals=[16.09,13.26,12.79,12.31,11.74]  # match P2/GOLDEN
teacher prefill12 (base+[220]) ms=7401.7 next=17 top5=[17,16,18,19,20]
draft K=1 pos=11 ms=34.2 top=17 val=14.55 mean=-1.468 min=-9.67 max=14.55 finite=1
draft top5=[17,16,220,18,19] overlap_teacher=4/5
K=1 accept: draft 17 vs teacher 17 -> ACCEPT
GATE1 PASS (draft==teacher, overlap 4/5)
```
Soglie: finito=1, max 14.55 in [~15-20] sana (<1e4), argmax match, top5 overlap 4/5 ≥ 4/5.
Draft ~34 ms vs step target ~654 ms (~19x, ~1 layer vs 64, atteso).
VRAM: post-upload 9748/11361 MiB, mtp-init 9750/11363 MiB, post-gate 9750/11363 MiB
(stabile, no crescita; smi fuori run 486/1289 MiB identici prima/dopo, no leak).

## 2. GATE2 — autoregressivo K=3 (PASS, RC=0)

Stesso h_last, tok precedente, carry MtpState (KV draft propria), pos 11..13:
```
draft t+1 pos=11 ms=33.8 top=17  val=14.55 mean=-1.468 min=-9.67 max=14.55 finite=1
draft t+2 pos=12 ms=33.6 top=220 val=14.25 mean=-1.721 min=-8.89 max=14.25 finite=1
draft t+3 pos=13 ms=33.6 top=17  val=16.18 mean=-1.472 min=-9.85 max=16.18 finite=1
GATE2 PASS 3 draft token finiti: [17,220,17]
```
Soglia: tutti finiti, scala sana (<1e4). VRAM stabile 9750/11363 MiB.

## 3. GATE3 — verify+accept (protocollo corretto, RC=0)

Verify = prefill GPU full su `P + [tok0] + D` (tok0=top di prefill(P)); confronto
greedy `argmax(logits[L+j])` vs `D[j]` (D[0] e` in posizione L+1, NON L — senza
tok0 si misura 0/K spurio, bug trovato e fixato in 1 iterazione, vedi sez. 5).
Exit: 0 sempre se numeri finiti (rate riportata onesta, mai falsata);
2 solo per mismatch K=1 (stesso criterio `test_mtp_accept`), 1 su errore numerico.

## 4. GATE4 — sweep K=1..4 su 2 prompt (RC=0, tabella onesta)

P-A11: base 11-tok. P-B13: base11 + teacher top2 `[220,1374]` (13 tok, teacher-forced).

P-A11 (verify prefill 8.0..9.8 s, draft ~34 ms/step):
```
K  drafts         accept  rate  draft_ms  verify_ms  eff_tok/s
1  [17]           1/1     1.00  34.2      8020.5     0.25
2  [17,220]       1/2     0.50  67.2      8623.0     0.23
3  [17,220,17]    1/3     0.33  100.7     9237.0     0.21
4  [17,220,17,220] 1/4    0.25  134.6     9846.1     0.20
```
P-B13 (verify 9.2..11.1 s):
```
K  drafts         accept  rate  draft_ms  verify_ms  eff_tok/s
1  [17]           1/1     1.00  33.9      9238.4     0.22
2  [17,220]       1/2     0.50  67.2      9848.4     0.20
3  [17,220,17]    1/3     0.33  100.8     10459.2    0.19
4  [17,220,17,220] 1/4    0.25  134.4     11073.7    0.18
```
`eff_tok/s = (accepted+1)/wall(draft+verify)` (+1 bonus token). Baseline non-spec 1.53 tok/s.

Conclusione onesta: **NO, non conviene con i kernel attuali, per nessun K.**
K=1 ha accept 100% su entrambi i prompt (wiring confermato) ma eff 0.22-0.25 tok/s
<< 1.53 perche' il verify-prefill GEMV-bound (~620 ms/tok, come il decode) domina:
verify T=12..17 = 8-11 s contro K*654 ms del decode diretto. K≥2 accetta solo il
primo token (1/K su entrambi i prompt — sistematico, non rumore: vedi sez. 5),
quindi l'eff cala ulteriormente (0.18-0.23). Il draft e` economico (34 ms) ma
inutile se verify costa piu` del decode che sostituisce. Servirebbero, prima di
riprovarlo: verify incrementale che riusa KV (no prefill-full) e/o prefill
molto piu` veloce (GEMM batched/CUTLASS), piu` h-carry (sez. 5).

## 5. Ipotesi scartate, debug, cosa blocca il resto

- Bug verify trovato e fixato (1 iterazione, non conteggiata come debug wiring:
  era nel test, non nel draft): `ext = P + D` + check a `L-1+j` dava 0/K spurio
  perche' D[0] e` in posizione L+1; corretto a `ext = P + [tok0] + D` + check a
  `L+j`. Dopo fix K=1 = 1/1 coerente con GATE1 (validazione protocollo).
- Iterazioni debug wiring usate: **0 di 2** (K=1 ACCEPT al primo colpo: concat
  [e;h], norme dirette, RoPE assoluta, GQA x6, sigmoid gate tutti confermati;
  nessuna alternativa provata, nessuna necessaria).
- Perche' K≥2 = 1/K sistematico su 2 prompt: ipotesi prima e` semantica h-carry,
  NON bug numerico (tutti finiti, scale sane, KV append verificato dal PASS K=1
  e dal carry K=3 finito). Con h fissa, D[1]=220 da (h_last,17) viene rifiutato
  dal target dopo P+tok0+17. Variante non provata: h evolvente (hidden draft
  post-blocco come h dello step successivo, stile DeepSeek-MTP). Richiede che
  `draft_step` esponga l'hidden (1 D2H 20 KB o buffer su device) + re-sweep:
  1 run stimato, NON eseguito per restare nel budget (~120 min) e perche'
  GATE2/3/4 non lo richiedono (solo report onesto). Prossimo passo concreto.
- Cosa blocca il resto (fuori scope di questo task, non iniziato):
  sampling con distribuzione target (acceptance `min(1,p_target/p_draft)`,
  test distribuzionale non-identita`, seed fissato); rollback KV/GDN pieno
  (tronca `SeqState`+`MtpState` alla posizione accettata + rigenera, vedi
  P4_DECODE FAIL layer26 per il carry GDN — il draft MTP non ne dipende ma il
  rollback target si); DFlash2 esterno (non necessario al primo rilascio,
  draft esterno non verificato contro checkpoint abliterato).
- Tempi misurati: build `-j4 --target test_mtp_gpu` OK (unico errore: `&over`
  vs `over`, fixato); upload warm 2.88 s (cold 9.60 s in P2); conversione 1
  matrice 25.18 ms; k1 ~15 s, k3 ~10 s, sweepA ~60 s, sweepB ~65 s (tutti <10 min).
  VRAM max usata 9750/11363 MiB < 15360, smi prima/dopo 486/1289 MiB (no leak).

File creati: `distributed/mtp_gpu.hpp`, `distributed/mtp_gpu.cu`,
`distributed/test_mtp_gpu.cu`, `distributed/MTP_GPU.md` (questo).
Modificati (solo aggiunte): `distributed/pp2_fwd.hpp/.cu` (accessor h_last +
`get_weight_ptr`), `CMakeLists.txt` (8 righe, stesso schema).
NO commit (fa orchestratore). Mai modificati `architecture/*`, `scripts/*`, `reports/*`.
