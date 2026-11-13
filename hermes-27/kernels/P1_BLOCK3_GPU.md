# P1 blk.3 full-attention su GPU1 — verificato vs golden CPU

Obiettivo P1: blocco full-attention (blk.3) eseguito su GPU1 con pesi REALI, verificato contro golden CPU.
Checkpoint: `/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf`
Dims: HID=5120, NQ=24, NKV=4, HD=256, T=11. qqkv [11,12288] (24*512), kk/vv [11,1024] (4*256), Q/G/att [11,6144] (24*256), K/V [11,1024], core [11,5120].
Goldens (diretti, norme *w): `/tmp/e3_h.bin` (input post-LNin), `/tmp/e3_qqkv.bin`, `/tmp/e3_kk.bin`, `/tmp/e3_vv.bin`, `/tmp/e3_Q.bin` + `/tmp/e3_K.bin` (POST-norm+interleaved+RoPE), `/tmp/e3_V.bin` (=vv, verificato maxdiff 0.0), `/tmp/e3_G.bin` (POST-gate = att*sigmoid), `/tmp/e3_att.bin` (PRE-gate), `/tmp/e3_core.bin` (post out_proj).
Equazioni (da transformers, verificate in `architecture/src/model_fwd.cpp`): Q = RMS(qq[...,:256]) * w_qn diretto, G = qq[...,256:] raw (split interleaved per-head reshape [11,24,512]); K = RMS(kk)*w_kn; V raw; RoPE split-half primi 64 dim coppie [i,i+32] theta 10M pos=t; SDPA causale GQA x6 scala 1/16; gated = att*sigmoid(G); core = gated @ Wo.T.

Eseguito su server, GPU1 soltanto, leggero (<50MB picco), nessun servizio toccato.

## Stadio 1 — q/k/v GEMM NVFP4 pesi reali — PASS
h golden H2D [11,5120] float -> BF16. Wq `blk.3.attn_q.weight` [5120,12288] NVFP4, Wk/Wv `blk.3.attn_k/v.weight` [5120,1024].

Comando:
```
CUDA_VISIBLE_DEVICES=1 timeout 300 /home/matteo/hermes-27/build/test_p1_block3 s1
```
Risultato:
- CPU-dequant (dequant NVFP4 esatta + GEMM float) vs golden: maxerr=0.0 esatto su qqkv/kk/vv, cos=1.00000000 — prova h corretta, dequant corretta, layout righe GGUF corretto.
- GPU-GEMV `nvfp4_gemv_cuda` (GGUF diretto, no shuffle, BF16 in / FP32 out):
  - qqkv maxerr=0.0162935 at 71546 (gpu=-6.078588 ref=-6.062294), cos=0.99999903, rms=2.92733, rel=0.00557 — PASS (thr 0.2)
  - kk maxerr=0.00584257 at 8482, cos=0.99999917, rel=0.00702 — PASS
  - vv maxerr=0.00443840 at 5458, cos=0.99999900, rel=0.00707 — PASS
  - tempi: qqkv 19.25ms tot 11 step, kk 1.648ms, vv 1.645ms
- GPU-W4A4 `Nvfp4GemmW4A4::run_ws` Mp=11 N=12288 K=5120 (cutlass-79a-sm120-w4a4, ws=0, sfa=40960, sfb=3932160):
  - maxerr=0.544166 at 9517 (gpu=-5.09375 ref=-5.637916), cos=0.99965278, rms=2.92733, rel=0.18589 — rumore 4+4bit atteso (blk.0 rel 0.38 su valori rms 3.37; qui rel 0.19, stesso ordine). Non è mismatch funzionale.
  - ms=0.0809 avg 20 reps, rc=0
- VRAM (cudaMemGetInfo): prima free 14428MiB, dopo 14426MiB, delta 2MiB
- Note: V golden == vv golden maxdiff 0.0 verificato via numpy (V raw, nessuna norm/rope).

Problemi: nessuno. GEMV scelto per catena precisa; W4A4 solo per documentare rumore.

## Stadio 2 — q/k RMSNorm diretto + split interleaved + RoPE — PASS
Kernel dedicati float `q_norm_rope_k` (grid 264=T*NQ, blk 256) e `k_norm_rope_k` (grid 44=T*NKV): RMS per 256-dim (eps 1e-6) * w_qn/w_kn diretto, poi RoPE split-half primi 64 dim (freq 1/theta^{2i/64}, i=t%32, pos=t) identica a `model_fwd.cpp:rope_apply`. Input golden qqkv/kk per isolare.

Comando:
```
CUDA_VISIBLE_DEVICES=1 timeout 300 /home/matteo/hermes-27/build/test_p1_block3 s2
```
Risultato:
- Q post-rope vs `/tmp/e3_Q.bin`: maxerr=1.43051e-06 at 36806 (gpu=6.740752 ref=6.740754), cos=1.00000000, rms=1.22198 — PASS (thr 1e-4)
- K post-rope vs `/tmp/e3_K.bin`: maxerr=9.53674e-07 at 10288, cos=1.00000000, rms=1.25955 — PASS
- Qgpu[0..3]=-0.123995 0.977199 -0.529389 -1.786799 match ref a 6 decimali
- kernel ms=0.2497 tot first run, avg 0.0041-0.0042ms su 50 reps (264+44 blocchi)
- VRAM delta 2-4MiB (free 14428->14424MiB)
- qn[0..3]=1.202148 1.253906 1.257812 1.226562, kn[0..3]=1.281250 1.248047 1.298828 1.251953 (pesi diretti, no +1)

Problemi: nessuno. `rope.hpp` esistente è single-head BF16, non batch/HD256: scritto loop dedicato float, replica esatta split-half (non interleaved).

## Stadio 3 — SDPA GQA causale + gate sigmoid + out_proj — PASS
Input isolato: Q/K/V golden post-rope. Kernel `sdpa_gqa_k` (grid 264=T*NQ, blk 256, shared scores[11]): dot Q*K per t2<=t, scala 1/16, softmax stabile, weighted sum V. G_raw estratto CPU esatto da qqkv golden (secondi 256 per 512-chunk). `gate_k`: gated=att/(1+exp(-s)). Out_proj `blk.3.attn_output.weight` [6144,5120] via `nvfp4_gemv_cuda` (BF16 in).

Comando:
```
CUDA_VISIBLE_DEVICES=1 timeout 300 /home/matteo/hermes-27/build/test_p1_block3 s3
```
Risultato:
- att PRE-gate vs golden: maxerr=8.04663e-06 at 56951 (gpu=-0.172439 ref=-0.172447), cos=1.00000000, rms=0.50035, rel=0.00002 — PASS (thr 1e-4)
- gated POST-gate vs `/tmp/e3_G.bin`: maxerr=1.19209e-06 at 10375, cos=1.00000000, rms=0.03066 — PASS (thr 1e-4)
- core vs `/tmp/e3_core.bin`: maxerr=0.00581932 at 50074 (t=9 i=3994? no t=50074/5120=9 i=3994 simile a blk0) gpu=11.797667 ref=11.803486, cos=0.99999989, rms=0.15730, rel=0.037 — PASS (thr 0.2, BF16 rounding su gated ~0.03 come blk0 0.025)
- core_gpu[0..3]=0.006815 -0.038929 -0.001688 -0.001855 vs ref 0.006763 -0.038965 -0.001655 -0.001854
- tempi: sdpa+gate 0.025-0.23ms, sdpa only avg 0.0121ms/20 reps, out_proj GEMV 7.97-8.40ms tot 11 step N=5120 K=6144
- VRAM delta ~20-22MiB (free 14428->14406MiB, picco Wo 17.7MB)

Problemi: nessuno bloccante. Gated rms piccolo (0.03) atteso: sigmoid(G_raw) piccolo. t=0 att==V0 verificato (causale singolo).

## Full run
```
CUDA_VISIBLE_DEVICES=1 timeout 300 /home/matteo/hermes-27/build/test_p1_block3 all
```
Tutti PASS. nvidia-smi GPU1 prima 1289/14562MiB, dopo 1289/14562MiB — no leak, servizi intatti. GPU0 invariata 486/15365MiB. VRAM libera 14.5GB >> soglia 8GB. Tempo per stadio <1s GPU + ~2s CPU-dequant, mai 25min.

## File creati/modificati
- Nuovo: `kernels/tests/test_p1_block3.cu` (s1 GEMV+W4A4+dequant, s2 norm+rope dedicati, s3 sdpa GQA+gate+out_proj)
- Modifica: `CMakeLists.txt` SOLO aggiunta target (3 righe):
```
add_executable(test_p1_block3 kernels/tests/test_p1_block3.cu)
target_link_libraries(test_p1_block3 PRIVATE hermes_gguf hermes_kernels)
add_test(NAME p1_block3 COMMAND test_p1_block3)
```
- Nuovo report: `kernels/P1_BLOCK3_GPU.md` (questo file)
- Build: `cmake -S /home/matteo/hermes-27 -B /home/matteo/hermes-27/build` (reconfigure una volta) + `cmake --build build -j4 --target test_p1_block3` soltanto, mai `test_model_fwd`/`e1_*`/ctest completo
- NO commit (solo file + report)

## Cosa blocca il resto
- Nulla blocca P1 full-attention: lineari, norm, rope, SDPA GQA, gate, out_proj tutti su GPU1 con pesi reali e verificati.
- Follow-up consigliato (non bloccante): chaining end-to-end h->qqkv(GPU)->Q->att->core con core_gpu invece di golden per misura errore cumulato (stima <0.02 su core via RMS norm).
