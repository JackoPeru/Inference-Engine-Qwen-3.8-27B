# P1 blk.0 lineare su GPU1 — verificato vs golden CPU

Obiettivo P1: primo blocco GDN (blk.0) eseguito su GPU1 con pesi REALI, verificato contro golden CPU.
Checkpoint: `/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf`
Dims: HID=5120, C=10240 (KD=2048,KD=2048,VD=6144), HV=48, KHD=VHD=128, T=11 token.
Goldens (diretti, norma *w): `/tmp/e1_cv_qkv.bin`, `/tmp/e1_cv.bin`, `/tmp/e1_beta.bin`, `/tmp/e1_gl.bin`, `/tmp/e1_o.bin`, `/tmp/e1_core.bin`, `/tmp/e1_y.bin`.
IDs: `[6205,161421,303,220,17,1374,9911,1147,27053,6059,13]`.

Eseguito su server, GPU1 soltanto, leggero (<200MB picco), nessun servizio toccato.

## Stadio 1 — conv1d depthwise k=4 + silu — PASS
Kernel nuovo `conv_silu_k` (1 thread per (t,c)), input qkv golden H2D + Wc reale F32 `blk.0.ssm_conv1d.weight` via `hermes_gguf`.
Confronto vs `/tmp/e1_cv.bin`.

Comando:
```
CUDA_VISIBLE_DEVICES=1 timeout 300 /home/matteo/hermes-27/build/test_p1_block0 s1
```
Risultato:
- maxerr=9.537e-07 at idx 56020 (t=5 c=4820), cos=1.00000000, meanabs=0.12194 — thr 1e-4 PASS
- kernel_ms=0.0028 avg 100 reps, blocks=440 threads=256
- VRAM (cudaMemGetInfo): prima free 14428MiB, dopo 14426MiB, delta 2MiB (2048 KiB)
- nvidia-smi prima/dopo intero run: GPU1 used 1289MiB free 14562MiB -> invariato dopo free
- golden qkv[0..4]=-1.084186 -0.825619 -0.207803 1.275512 2.144785, cv[0..4]=0.043570 -0.030456 -0.007152 0.041784 -0.057286

Problemi: nessuno. Wc layout `Wc[c*4+j]` confermato vs `model_fwd.cpp`.

## Stadio 2 — qkv GEMM NVFP4 — h esatta, W4A4 rumoroso (atteso)
h=RMS(x,attn_norm) diretto ricostruito su CPU da embed BF16 + norm F32 reali.
Verifica CPU-dequant (dequant NVFP4 esatta + h float) vs golden per isolare h.
Poi GPU W4A4 `Nvfp4GemmW4A4::run_ws` end-to-end da h BF16.

Comando:
```
CUDA_VISIBLE_DEVICES=1 timeout 300 /home/matteo/hermes-27/build/test_p1_block0 s2
```
Risultato:
- CPU-dequant vs golden: maxerr=2.289e-05 at 96522, cos=1.00000000 — prova h corretta, ids corretti, dequant corretta. qkv[0..4] match a 6 decimali.
- GPU-W4A4 vs golden: maxerr=1.28136 at 34412 (t=3 n=3692) gpu=-57.0 ref=-55.71864, rms=3.37115 rel=0.38010 cos=0.99957955 ms=0.0547 avg 20 reps Mp=11 N=10240 K=5120, backend=cutlass-79a-sm120-w4a4, ws=0, sfa=40960 sfb=3276800
- VRAM delta 34MiB (free 14428->14394MiB interno), nessun OOM

Analisi: 0.0038 documentato in `reports/VALIDATION.md`/`E1_E2_status.md` era vs-dequant su random tiny (T=256 N=128 K=512, valori ±0.15). Qui su pesi reali e valori rms 3.37, maxerr assoluto 1.28 e rel 0.38 sono dello stesso ordine del rumore 4+4bit (report: vs fp32 rel 0.46). Non è un mismatch funzionale: cos 0.99958, CPU-dequant 2.3e-05 prova il plumbing. Per lineari di precisione si usa GEMV (stadio 3/4) senza quant attivazioni.

## Stadio 3 — gate/beta/alpha + gdn_step + gated+out_proj — PASS
- Wz `blk.0.attn_gate.weight` NVFP4 [6144,5120] via `nvfp4_gemv_cuda` (GGUF diretto, no shuffle)
- Wb `blk.0.ssm_beta.weight` NVFP4 [48,5120] via GEMV
- Wa `blk.0.ssm_alpha.weight` F32 [48,5120] via kernel `wa_gemm_k` float su GPU
- beta=sigmoid, gl=-exp(A)*softplus(ar+DT) CPU elementwise (528 elem)
- q/k/v da cv golden, repeat×3, l2norm/scale CPU, poi `gdn_step_cuda` sequenziale T=11 con carry S su GPU1
- gated=RMS(o)*Wn*silu(z) CPU, out_proj `blk.0.ssm_out.weight` NVFP4 [5120,6144] via GEMV

Comando:
```
CUDA_VISIBLE_DEVICES=1 timeout 300 /home/matteo/hermes-27/build/test_p1_block0 s3
```
Risultato:
- Wz GEMV 9.39ms tot 11 step, Wb 0.308ms, Wa 0.796ms
- beta maxerr=6.851e-04 cos=0.99999998 PASS (<5e-3), gl maxerr=4.768e-06 cos=1.0 PASS
- gdn_step GPU 0.870ms tot T=11 H=48 K=128 V=128, o maxerr=2.338e-04 cos=1.0 PASS
- out_proj GEMV 7.996ms, core maxerr=2.517e-02 at t=9 i=3994 gpu=13.461571 ref=13.486743 cos=0.99999977 (report, BF16 rounding su valori ~13)
- VRAM free 14426->14384MiB delta 42MiB

Problemi: nessuno bloccante. Beta errore 6.8e-04 da BF16 input GEMV, atteso. Core 0.025 da BF16 gated, cos ~1.0.

## Stadio 4 — residuo + post_norm + MLP — PASS (isolato)
h1=x+core_golden (isola MLP), post_norm RMS diretto F32 `blk.0.post_attention_norm.weight` CPU, poi MLP NVFP4 via GEMV:
gate `blk.0.ffn_gate.weight` [17408,5120], up `blk.0.ffn_up.weight`, down `blk.0.ffn_down.weight` [5120,17408].

Comando:
```
CUDA_VISIBLE_DEVICES=1 timeout 300 /home/matteo/hermes-27/build/test_p1_block0 s4
```
Risultato:
- gate/up GEMV 53.2ms tot, down 15.6ms
- y maxerr=3.704e-03 at t=1 i=3994 gpu=19.353239 ref=19.349535 cos=0.99999998 PASS (thr 0.1, effettivo 0.0037)
- y[0..4] gpu=0.00293 0.08635 0.04720 -0.00886 0.08662 vs ref 0.00289 0.08630 0.04718 -0.00887 0.08664
- VRAM 14426->14278MiB delta 148MiB (picco gate/up/down, ancora <200MB)

Nota: usa core golden per isolare MLP. End-to-end con core_gpu (err 0.025) non ancora misurato — propagazione attesa piccola via RMS norm, da fare come follow-up (stima <0.05 su y).

## Full run
```
CUDA_VISIBLE_DEVICES=1 timeout 300 /home/matteo/hermes-27/build/test_p1_block0 all
```
Tutti PASS tranne note quant W4A4 sopra. nvidia-smi GPU1 prima 1289/14562MiB, dopo 1289/14562MiB — no leak, servizi intatti.

## File creati/modificati
- Nuovo: `kernels/tests/test_p1_block0.cu` (s1 conv kernel, s2 W4A4+dequant, s3 GEMV+gdn_step, s4 MLP GEMV)
- Modifica: `kernels/CMakeLists.txt` SOLO aggiunta target (3 righe):
```
add_executable(test_p1_block0 kernels/tests/test_p1_block0.cu)
target_link_libraries(test_p1_block0 PRIVATE hermes_gguf hermes_kernels)
add_test(NAME p1_block0 COMMAND test_p1_block0)
```
- Nuovo report: `kernels/P1_BLOCK0_GPU.md` (questo file)
- Build: `cmake --build build -j4 --target test_p1_block0` soltanto, mai `test_model_fwd`/`e2`/ctest completo
- NO commit (solo file + report)

## Cosa blocca il resto
- Nulla blocca P1 lineare: conv, lineari, ricorrenza, gated, out_proj, post_norm, MLP tutti su GPU1 con pesi reali e verificati.
- Follow-up consigliato (non bloccante, <20min):
  1. End-to-end GPU core->y (usa core_gpu invece di golden in s4) per chiudere catena completa.
  2. Full blk.0 con residui: y=x+core+MLP già fatto isolato, manca solo chaining.
  3. W4A4 vs GEMV: per prefill conviene W4A4 (0.05ms vs 9ms) ma con rumore attivazioni; documentare scelta.
- VRAM libera GPU1 14.5GB, picco nostro 150MB, soglia 8GB mai vicina. Tempo per stadio <5min, mai 20min.
