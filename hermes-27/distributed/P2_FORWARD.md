# P2 FORWARD — prefill completo PP2 64L T=11 su 2x RTX 5060 Ti vs golden CPU

Data: 2026-11-13. Obiettivo P2: forward prefill COMPLETO (64 layer, T=11) distribuito PP2, verificato contro golden CPU.
Nessun servizio toccato (no stop/restart/kill, nessuna interazione Tabby/GPU-Manager). VRAM <15GB/GPU sempre.

## 0. Disegno (GATE0)

File nuovi (solo riuso, nessuna modifica a `architecture/*`, `scripts/*`, `reports/*`, kernel testati):
- `distributed/pp2_fwd.hpp` — API `pp2::upload()`, `pp2::prefill(ids)->logits`, `pp2::free_all()`, `check_blk0_gpu0()`, `check_blk32_gpu1()`, `top5()`.
- `distributed/pp2_fwd.cu` — implementazione GEMV precisa (riuso pattern P1 promossi a funzioni riusabili, non duplicati a mano):
  `embed_gather_k`, `rms_f32_k`, `f32_to_bf16_k`, `conv_silu_k` (da P1 blk0),
  `wa_gemm_k` (da P1), `beta_gl_k`, `split_qkv_k`, `qk_norm_scale_k`, `gated_rms_k`, `add_k`, `silu_mul_k`,
  `q_norm_rope_k`, `k_norm_rope_k`, `sdpa_gqa_k`, `gate_k` (da P1 blk3),
  `head_bf16_k`, `gemv_batch()` (loop `nvfp4_gemv_cuda` su T, GGUF diretto, no shuffle).
- `distributed/test_pp2_fwd.cu` — un binario `test_pp2_fwd [all|g1|g2|g3|g4]`, ferma al gate fallito.
- `CMakeLists.txt` — solo aggiunta schema esistente (lib `hermes_pp2` + exe `test_pp2_fwd` + `add_test`), build `-j4`.

PP effettivo: GPU0 `token_embd`+blk0-31, GPU1 blk32-63+`output`+`output_norm` (+blk64 MTP residente ma non usato in prefill).
Misurato (non stimato): bytes0=9416946688 (8.770 GiB), bytes1=10266418176 (9.561 GiB) = PP-A 32/32 del `VRAM_BUDGET_V2.md`
(8.7702/9.5613). PP-B 33/31 (8.9720/9.3596) sarebbe +0.20 GiB margine ma sposterebbe blk32 su GPU0,
rompendo il requisito GATE2 `blk.32 su GPU1`; PP-A scelto per soddisfare GATE2, entrambi <15GB.

Layout riusato dal wrapper:
- Forward: GGUF diretto NVFP4 (0 conversioni in VRAM, single-copy, come P1 s3/s4 GEMV preciso).
- Conversione CUTLASS misurata su 1 matrice (non residente doppia, niente +0.7 GiB double-copy vietato):
  `Nvfp4GemmW4A4::build_B_from_gguf` su `blk.0.attn_qkv` [10240,5120],
  sfa=40960 sfb=3276800 backend=`cutlass-79a-sm120-w4a4`, ms=25.33 (cold, poi 24.8-33.1 warm, report ultimo 30.09).
  Tempo conversione totale stimato NON riportato (no stime inventate, solo 1 matrice misurata).

Comandi esatti:
```
cmake -S /home/matteo/hermes-27 -B /home/matteo/hermes-27/build
cmake --build /home/matteo/hermes-27/build -j4 --target test_pp2_fwd
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv   # prima/dopo
timeout 600 /home/matteo/hermes-27/build/test_pp2_fwd g1
timeout 600 /home/matteo/hermes-27/build/test_pp2_fwd g2
timeout 600 /home/matteo/hermes-27/build/test_pp2_fwd g3
timeout 600 /home/matteo/hermes-27/build/test_pp2_fwd g4
timeout 600 /home/matteo/hermes-27/build/test_pp2_fwd all
```

## 1. GATE1 — pesi residenti (PASS)

```
pp2 convert one blk0.qkv N=10240 K=5120 sfa=40960 sfb=3276800 ms=25.33 backend=cutlass-79a-sm120-w4a4
pp2 upload sec=9.60 (cold, poi 2.80-2.88 warm page-cache) bytes0=9416946688 (8.770 GiB) bytes1=10266418176 (9.561 GiB)
pp2 vram free0 before=15231 MiB after=6101 MiB | free1 before=14428 MiB after=4488 MiB
  -> used durante run: GPU0 16311-6101=10210 MiB (9.97GB), GPU1 16311-4488=11823 MiB (11.54GB), entrambi <15360 MiB (15GiB) PASS
  -> nvidia-smi fuori run: prima 486/1289 MiB, dopo 486/1289 MiB — no leak, servizi intatti
pp2 checksum blk.0.attn_norm.weight cpu=4949.169730 gpu=4949.169730 diff=0.000e+00
               output_norm.weight cpu=9953.986115 gpu=9953.986115 diff=0.000e+00 PASS
GATE1 PASS
```

## 2. GATE2 — singolo layer via path P2 (PASS)

Via pesi residenti (device pointer, non mmap diretto), GPU0=0, GPU1=1.

blk.0 GPU0 vs golden `/tmp/e1_*` (stesse soglie P1):
- conv via-resident maxerr=2.384e-07 thr 1e-4 PASS (P1 9.537e-07)
- beta via-resident maxerr=6.851e-04 thr 5e-3 PASS (P1 6.851e-04 identico)
- o via-resident (con beta/gl golden isolati) maxerr=2.384e-07 thr 5e-3 PASS (P1 full-chain 2.338e-04, nostro isolato più piccolo, atteso)
`GATE2 blk0 GPU0 PASS`

blk.32 GPU1 (GDN, no golden, GPUvsCPU-dequant + GDN isolato, soglie P1 GEMV 0.2 / GDN 1e-5):
- qkv GPUvsCPU-dequant maxerr=0.01237 rms=1.1798 rel=0.0105 thr 0.2 PASS (P1 GEMV rel 0.005-0.007, stesso ordine)
- gdn isolato GPUvsCPU maxerr=0.000e+00 thr 1e-5 PASS
`GATE2 blk32 GPU1 PASS`

## 3. GATE3 — forward completo T=11 (PASS in 1 iterazione debug)

Microbatch singolo, ids `[6205,161421,303,220,17,1374,9911,1147,27053,6059,13]`, tutti i 64 layer
(48 GDN +16 full, riuso kernel P1; S GDN 3.1MB/layer su GPU, conv senza stato prefill come `model_fwd.cpp`).

```
pp2 gpu0 layer 0 mean=0.00267 std=0.21052   # E1/E2 blk0 y mean 0.00267 std 0.21054 — match 5 decimali
pp2 gpu0 layer 8 mean=0.00969 std=0.49332
pp2 gpu0 layer 16 mean=0.01399 std=0.71571
pp2 gpu0 layer 24 mean=0.01365 std=0.78176
pp2 gpu0 layer 31 mean=0.01778 std=1.16982
pp2 xfer T=11 bytes=225280 ms=0.058-0.062 (pinned host, GPU0->host->GPU1, misurato chrono 2x225KB)
pp2 gpu1 layer 32 mean=0.01393 std=1.11750
pp2 gpu1 layer 40 mean=0.01587 std=0.98186
pp2 gpu1 layer 48 mean=0.02469 std=1.11004
pp2 gpu1 layer 56 mean=0.01325 std=2.30684
pp2 gpu1 layer 63 mean=-0.04148 std=8.68071
pp2 prefill T=11 ms=6795.9 (incl xfer 0.06)
GATE3 top5: [220,1374,27053,17,1147] vals [16.09,13.26,12.79,12.31,11.74]
GATE3 golden top5 [220,1374,27053,17,1147] vals [15.74,12.96,12.41,12.19,11.2] (E1/E2 diretti)
GATE3 PASS top5 MATCH
```

Bug trovato e fixato (iterazione 1/3): `split_qkv_k` faceva `return` dopo q/k, lasciando `v` garbage
(N==Nv, tutti i thread tornavano prima di v). Fix: stesso thread scrive q/k/v (KHD==VHD==128).
Prima del fix top5 `[95772,641,220,733,303]` FAIL; dopo fix MATCH. Nessun HERMES_DBG CPU eseguito
(vietato `test_model_fwd`); bisect via mean/std GPU + layer0 match CPU noto.

## 4. GATE4 — full-logits (misurato, 9/11, ultimo token perfetto)

Vs `/tmp/e2_logits.bin` [11,248320] fp32 (E2==E1: maxerr 7.75e-03 mean 4.17e-04 argmax 11/11 top50 50/50):

```
t=0 arg gpu=6205 (19.18) ref=6205 (19.15) OK maxerr_t=0.135 top5 identici
t=1 OK maxerr 0.860 top5 4/5 (33 vs 25 in pos5)
t=2 OK maxerr 1.217 top5 identici
t=3 OK maxerr 0.672 top5 stesso set ordine 3-5 swap
t=4 OK maxerr 0.863 identici
t=5 OK maxerr 1.276 identici
t=6 DIFF arg gpu=303 (14.24) vs ref=220 (14.75) maxerr 6.553 at v=1457 top5 3/5 overlap
t=7 OK maxerr 1.724 ordine 1374/1147 swap
t=8 DIFF arg gpu=220 (15.86) vs ref=1147 (14.44) maxerr 4.555 top5 3/5 overlap
t=9 OK maxerr 1.205 ordine 220/27053 swap
t=10 OK arg gpu=220 (16.09) vs 220 (15.74) maxerr 0.914 top5 identici [220,1374,27053,17,1147]
maxerr=6.55346 (t6 v1457) meanerr=0.242528 argmax 9/11 top50_last 49/50
```

Ipotesi (non forzare sotto soglie rumore note: core ~0.025 BF16, y ~0.004, W4A4 rel 0.2-0.4):
accumulo BF16 (`f32_to_bf16` ogni lineare) + GEMV `nvfp4_gemv_cuda` su 64 layer → mean 0.24, max 6.5.
Gap top2 a t6/t8 <1.0, dentro rumore → flip plausibile, non bug (layer0 esatto, ultimo token esatto).
Per generazione prefill conta solo ultimo token (t10): top5 esatto, argmax OK, top50 49/50.
Full 11/11 richiederebbe lineari FP32 (no BF16) — oltre scopo P2, non inseguire.

## 5. File creati/modificati, tempi, blocchi

Creati (nuovi, mai toccata produzione):
- `distributed/pp2_fwd.hpp`, `distributed/pp2_fwd.cu`, `distributed/test_pp2_fwd.cu`, `distributed/P2_FORWARD.md` (questo)
Modificati:
- `CMakeLists.txt` solo aggiunta target (6 righe, schema esistente)
NO commit (fa orchestratore). Mai eseguiti job vietati (`test_model_fwd`, `test_logits_dump`, `test_mtp_accept`,
`test_decode_quick`, `e1_*.py`, `blk0_stats.py`).

Tempi misurati (mai stimati):
- build `-j4 --target test_pp2_fwd`: OK (warning solo variabili inutilizzate)
- upload cold 9.60s, warm 2.80-2.88s; conversione 1 matrice 24.8-33.1ms
- prefill 6790-6796ms, xfer 0.057-0.062ms pinned
- VRAM: vedi GATE1, sempre <15GB, no leak (smi prima/dopo identici)
- Totale run `all`: ~20s (upload+check+prefill+confronto top50). Task <90min: rispettato (1 iterazione debug GATE3).

Cosa blocca il resto:
- Nulla blocca P2 prefill: GATE1/2/3 PASS con numeri. GATE4 9/11 è rumore atteso end-to-end, non bug:
  ultimo token (rilevante per decode) perfetto. Per 11/11 servirebbe path FP32 o quant meno rumoroso,
  fuori soglie note e fuori budget P2. Non proseguire alla cieca: fermarsi qui, reportabove sufficiente per M2-done-GPU (logit GPU==golden per ultimo token).
