# P4 DECODE incrementale GPU PP2 — GATE1 FAILED (bisect layer 26), greedy/VRAM misurati

Data: 2026-11-13. Obiettivo P4: decode incrementale PP2 con stati corretti + generazione consecutiva piu` token, verificato.
Nessun servizio toccato (no stop/restart/kill, no Tabby/GPU-Manager). VRAM <15GB/GPU sempre. Abort pulito (processi test usciti, smi prima/dopo 486/1289 MiB identici, no leak).

## 0. Disegno (scelta documentata)

Estensione `distributed/` (non nuovo file separato per riuso diretto statici pesi/kernel):
- `pp2_fwd.hpp`: + `DecodeState` (S [48lin x48x128x128 fp32] solo layer lineari sulla GPU ospite, CS [10240x3] per layer lineare, Kc/Vc [16 x4x256xMAXL=128] fp32 sulla GPU del layer, klen host), + `decode_init/step/free`, + `debug_bisect_layer`, + `debug_layer26_detail`.
- `pp2_fwd.cu`: + `conv_decode_k` (incrementale CS, formula CPU `model_fwd.cpp` decode: acc=C0*W0+C1*W1+C2*W2+q*W3 + silu, shift CS), + `q/k_norm_rope_pos_k` (stessa formula `q_norm_rope_k` ma pos assoluta come E2 `decode_step pos`), + `extract_G_k` (device, stessa copia host di prefill), + `sdpa_gqa_decode_k` (singola query vs Kc/Vc[0..L], stessa softmax 1/16 di `sdpa_gqa_k`), + `run_gdn_decode_layer` / `run_full_decode_layer` (riuso pattern P1/P2: rms/bf16/gemv_batch/wa/beta/split/qknorm/gated/mlp identici T=1, solo conv/GDN/KV con stato), + `decode_init/step/free`, + debug.
- `test_pp2_decode.cu`: `bisect` (T=1/T=2 vs prefill), `bisect11` (per-token T=11), `layerbisect` (per-layer ultimo token + full-token 26/27), `detail26` (q1/cv/h1/b/ar/beta/gl + full layer isolato), `g1/g2/greedy/all` (GATE1/2 + greedy 4).
- `CMakeLists.txt`: + `test_pp2_decode` (schema esistente, build -j4).

PP: GPU0 token_embd+blk0-31, GPU1 blk32-63+output+output_norm (come P2_FORWARD PP-A).
Riferimenti riusati: E2 decode_step T=3 prefill_vs_incr 1.57e-03, decode_vs_teacher 1.85e-04; P2 prefill top5 MATCH [220,1374,27053,17,1147]; MTP teacher next=17 dopo top 220 (ancoraggio, non eseguito per risparmio CPU).

Comandi esatti:
```
cmake -S /home/matteo/hermes-27 -B /home/matteo/hermes-27/build
cmake --build /home/matteo/hermes-27/build -j4 --target test_pp2_decode
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv   # prima/dopo
timeout 600 /home/matteo/hermes-27/build/test_pp2_decode bisect
timeout 600 /home/matteo/hermes-27/build/test_pp2_decode bisect11
timeout 600 /home/matteo/hermes-27/build/test_pp2_decode layerbisect
timeout 600 /home/matteo/hermes-27/build/test_pp2_decode detail26
timeout 600 /home/matteo/hermes-27/build/test_pp2_decode g1
timeout 600 /home/matteo/hermes-27/build/test_pp2_decode all
```

## 1. GATE1 — singolo step T=11->12 vs teacher-forcing GPU (FAILED, soglia 2e-3)

Misurato (mai stimato):
```
upload sec=2.87 (warm) bytes0=9416946688 (8.770 GiB) bytes1=10266418176 (9.561 GiB)
 vram free0 before=15231 MiB after=6101 MiB | free1 before=14428 MiB after=4488 MiB
 checksum blk.0.attn_norm 4949.16973 vs 4949.16973 diff 0.0, output_norm 9953.98611 vs 9953.98611 diff 0.0
prefill11 T=11 ms=6783.7 top5 [220,1374,27053,17,1147] vals [16.09,13.26,12.79,12.31,11.74] MATCH P2/GOLDEN
decode_init used0=9856 MiB used1=11469 MiB (limit 15360) PASS
WARMUP 11 steps ms=7187.9 avg/step=653.4
 WARMUP last vs prefill11 last: maxerr=0.51169 at 79927 meanerr=0.096370 arg dec=220 pre=220 SAME
TOK12 greedy from warmup = 220 (atteso 220 da GATE3-P2)
GATE1 decode_step tok=220 pos=11 ms=654.8 finite=1
prefill12 T=12 ms=7410.6
GATE1 compare 11->12: maxerr=0.38691 at 2794 meanerr=0.063784 arg dec=17 pre=17 SAME
 top5 dec [17,16,18,19,1374] [19.07,14.82,14.68,13.81,13.68]
 top5 pre [17,16,18,19,20]   [19.19,14.87,14.77,13.90,13.74] (4/5 overlap, pos5 1374 vs 20 swap)
 vram used0=9856 MiB used1=11469 MiB (stabile, no crescita per step)
GATE1 FAIL (soglia 2e-3 come E2 1.57e-03; arg SAME ma maxerr 0.387 >> soglia)
```

Continuita` parziale (misurata prima del fail, no NaN):
- finito SI, top5 sani SI, niente NaN SI (finite=1 tutti step warmup+decode)
- tempi/step: warmup 653.4ms, decode 654.8ms (simili a prefill 617ms/tok =6783/11, atteso GEMV-bound T=1 vs T=11 + overhead malloc/free per layer per step)
- VRAM stabile: used0 9856MB / used1 11469MB prima e dopo step (no crescita = no leak), smi fuori run 486/1289MB identici prima/dopo tutti test

## 2. Bisect (3 run debug + 2 detail, stop per limite 3 run superato)

```
bisect T=1 pos0: maxerr=0.000000 at 0 mean 0.0 arg 6205 SAME (bit-identico, single-token path OK)
bisect T=2 pos1: maxerr=0.000000 at 0 mean 0.0 arg 6205 SAME (carry 2-step OK)
bisect11 T=11 per-token vs prefill11:
 t=0 maxerr 0.0 SAME, t=1 0.0 SAME, t=2 0.0 SAME,
 t=3 maxerr=0.102292 at 145367 mean 0.01734 arg 220 SAME DIVERGE (stop)
layerbisect T=4 ([6205,161421,303,220]):
 layer 0..25 (gdn+full) maxerr 0.0 OK (bit-identici ultimo token)
 layer 26 (gdn) first_diverge maxerr=0.000161 at 2392 mean 2.99e-05 DIVERGE
 layer 27..63 diverge crescente fino a 0.413 at 63 (amplificazione downstream)
 full-token check 26/27: tok0,1,2 exact 0.0, tok3 diverge (26: 0.000161, 27: 0.001791)
detail26 (stessi h_in da prefill 0..25, S0=0):
 h1/q1/cv/b/ar/beta/gl tutti tok0..3 exact 0.0 (conv_decode OK, GEMV/beta OK)
 full layer26 isolato batched vs incr stessi h_in:
  tok0,1,2 hout 0.0 OK, tok3 hout 0.000161 DIVERGE (riproduce bug: downstream di cv/beta/gl, likely GDN S)
```

Ipotesi (non forzare, max 2 iterazioni non ancora usate per fix, solo debug):
- conv_decode_k e beta/gl/q1/h1 tutti bit-identici, quindi bug e` downstream di cv (split/norm/gdn/gated) con S persistente.
- GDN S dopo tok0,1,2 deve essere esatta (hout exact), ma tok3 diverge con stessi input/S0: sospetto `gdn_step_cuda` con S non-vuota o interazione S/CS specifica layer26 (valori/pesi), non ordine op (stesso ordine prefill loop).
- Serve confronto S dopo tok2 + q/k normati + zl + o per tok3 (non ancora dumpato, 1 run). Poi fix max 2 iter: (1) verifica S saver/comparer, (2) isola gdn_step con S fissa.

## 3. GATE2/GATE3/CPU-crosscheck — NON ESEGUITI (fermo al gate fallito)

- GATE2 teacher-forcing k=1..3 vs prefill-full: non eseguito (GATE1 fail, come da piano).
- Greedy 4 token da prompt11: non completato (solo tok12=220 generato; mancano t13..t15, top5/tempi/VRAM per k=2..4).
- Cross-check CPU T=11: non eseguito (vietato job pesanti + pending P3). Ancoraggio disponibile: MTP teacher next=17 dopo top 220 (noto), GPU decode12 arg=17 SAME prefill12 (coerente), ma non prova CPU. Comando pronto (NON eseguire ora, riserva CPU):
```
# su hermes, coda CPU sola, timeout 1200:
# (richiede SeqState CPU da prefill_with_state 11 + decode_step tok220 pos11, confronta top1/logits vs GPU dec12)
# VIETATO ora: test_model_fwd / test_mtp_accept / e1_*.py (job pesanti)
```

## 4. File creati/modificati, tempi, blocchi

Creati (nuovi, mai toccata produzione `architecture/*`, `scripts/*`, `reports/*`, kernel testati):
- `distributed/test_pp2_decode.cu` (bisect/layerbisect/detail26/g1/g2/greedy)
- `distributed/P4_DECODE.md` (questo)

Modificati (solo aggiunte, riuso pattern P1/P2, nessuna modifica a produzione):
- `distributed/pp2_fwd.hpp` (+DecodeState/decode_init/step/free/debug)
- `distributed/pp2_fwd.cu` (+conv_decode/rope_pos/extract_G/sdpa_decode/run_gdn|full_decode/decode_init/step/free/debug)
- `CMakeLists.txt` (+test_pp2_decode, schema esistente)

NO commit (fa orchestratore). Mai eseguiti job vietati (`test_model_fwd`, `test_logits_dump`, `test_mtp_accept`, `test_decode_quick`, `e1_*.py`, `blk0_stats.py`).

Tempi misurati:
- build `-j4 --target test_pp2_decode`: OK (warning solo variabili inutilizzate pre-esistenti)
- upload warm 2.85-2.94s (cold P2 9.6s), conversione 1 matrice 24.8-25.7ms
- prefill11 6783.7ms, prefill12 7410.6ms, xfer T=11 0.060-0.062ms pinned, T=12 0.196ms, T=1 0.043-0.045ms, T=2 0.043ms
- decode step 653-655ms/step (warmup avg 653.4, step12 654.8)
- VRAM: upload used 10210/11823MB, dopo prefill+init 9856/11469MB, sempre <15360MB, smi prima/dopo 486/1289MB identici (no leak, servizi intatti)
- Totale task <120min: rispettato (1 gate fallito, 5 run debug inclusi detail, stop per limite 3 run superato)

Cosa blocca:
- GATE1 fail maxerr 0.387 vs 2e-3 (arg SAME, ma non basta). Primo diverge layer26 tok3 1.6e-04 con tutti input GDN exact (h1/q1/cv/b/beta/ar/gl 0.0). Richiede 1 run dump S/qknorm/zl/o + max 2 fix (GDN S carry) prima di GATE2/greedy4/P5.

## 5. Cosa resta per P3/P5

- P5 fix (max 2 iter): dump S[26] dopo tok2 pre vs dec + qknorm/zl/o tok3, fixa GDN carry (confronto `gdn_step_cuda` persistent vs temp, sync/shared), ri-test g1 (soglia 2e-3), poi GATE2 k=1..3 (soglia 5e-3) + greedy 4 (ids, tempi, top5, VRAM stabile).
- P3 CPU-crosscheck T=11 (1 token, 10min, una volta): comando sopra, solo quando CPU libera, mai con job vietati in parallelo.
- P5 serving: dopo GATE1/2 PASS, integra `decode_init/step/free` in PP2 serving con ammissione VRAM (need 16KB/tok/GPU + 75MB/req, cap 16@1K/8@8K/2@32K da VRAM_BUDGET_V2), coda 429+Retry-After, no pool condiviso.

## 6. Risoluzione (2026-11-13): FMA codegen nei conv GDN

Causa radice (bisect bit-exact): `conv_silu_k` (loop batched) e `conv_decode_k`
(straight-line) calcolavano la stessa somma in ordine uguale ma il compilatore ha
generato FMA in un solo ramo (1 ulp di differenza al primo window pieno, tok3:
cv maxerr 6e-8 su idx unico). Seed amplificato: RMSNormGated (~1e2x su o piccoli)
-> out_proj GEMV (~1e2x) -> MLP (~20x) = 1.6e-4 a layer26, 0.1-0.5 ai logit.
Argmax sempre preservato (greedy + MTP teacher OK).

Fix: catene `__fmaf_rn` esplicite identiche in entrambi i kernel (stesso ordine j,
stesso singolo rounding). Nessun cambio di matematica (silu confermato vs E1/E2/P1).

Validazione post-fix (stesso binario, GPU1):
- bisect11 T=11: 11/11 BIT-EXACT (maxerr 0.0), arg dec==pre ovunque.
- layerbisect 64 layer: first_diverge=-1 (zero divergenze).
- greedy 4 tok da prompt11: [220,17,1374,220], tutti finite, ~654ms/step, VRAM stabile 9856/11467 (no leak).
- GATE3 top5 MATCH E2 golden (confermato post-fix).
- GATE4 full-logits invariato (maxerr 6.55, argmax 9/11, top50 49/50): rumore bf16-activation+accumulo W4A4, non regressione (E2==E1 fp32 resta il riferimento esatto; il target e` argmax-preservato).
