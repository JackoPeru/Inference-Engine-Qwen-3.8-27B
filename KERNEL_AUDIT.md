# HERMES-27 Kernel Audit — hot path NVFP4 + gdn_step sync

## 1. Cosa c'era nell'hot path (prima)

### `kernels/src/gdn_step.cu` — `gdn_step_k`
- `if (j>=V) return;` prima di `__syncthreads()` (riga 25 orig).
- Quando `V < blockDim (128)`, alcuni thread uscivano, altri attendevano in barriera = hang/UB.
- Secondo `__syncthreads()` dopo scale per-colonna (inutile per correttezza ma presente).

### `kernels/src/nvfp4_gemm_w4a4.cu` — `Nvfp4GemmW4A4::run()` (righe 191-197 orig)
- `cudaMalloc(&wptr, ws)` + `cudaFree(wptr)` per-call.
- `cudaDeviceSynchronize()` globale per-call.
- `quant_activation()` host-side: loop CPU per blocco da 16, `memcpy` BF16, encode E4M3 + nearest E2M1 via ricerca esaustiva 16 voci; richiede `Aq/SFA` host + H2D per-call.
- `build_B_from_gguf()` host-side: shuffle GGUF `[e_j,e_{j+8}]` -> nibble lineare + interleave SFB via layout CUTLASS. Gia` chiamata una volta per matrice (load-time), mai dentro `run()`, ma non documentato.

## 2. Cosa ho cambiato (file:righe)

### (TASK1) `kernels/src/gdn_step.cu` (49 righe totali)
- Riga 27: `bool active = (j < V);` — nessun early-return.
- Righe 31-33: `if(active){ scale }` poi `__syncthreads()` raggiunto da tutti.
- Righe 35-43: `if(active){ d/bv/update/a/store }`.
- Early `if(h>=H) return;` tenuto (block-level, tutti i thread del blocco escono insieme: sicuro).
- Ricostruito: `cmake --build ~/hermes-27/build --target test_gdn_step -j4` OK.

### (TASK3a+c) `kernels/include/hermes/nvfp4_gemm.hpp` (71 righe)
- Righe 32-36: `struct Workspace { void* ptr; size_t bytes; };`
- Riga 39: `workspace_bytes(M_T,N,K)` — 0 se shape non supportata o nessun workspace.
- Riga 41: `workspace_ensure(Workspace&,M_T,N,K)` — malloc una volta / cresce solo se serve.
- Riga 43: `workspace_release(Workspace&)` — free a shutdown.
- Righe 47-57: `run()` marcato COMPAT/COLD (tenuto invariato per retrocompatibilita`); `run_ws(...,void* workspace,size_t,cudaStream_t=0)` — nessun malloc/free/sync, enqueue su stream.
- Righe 61-68: `quant_activation_device()` dichiarato solo sotto `#ifdef HERMES_ENABLE_DEVICE_QUANT` (disabilitato di default).
- Nota (c): `build_B_from_gguf()` documentato come load-time; `run/run_ws` prendono `Bq/SFB` device-ready, nessun lavoro GGUF per-call.

### `kernels/src/nvfp4_gemm_w4a4.cu` (350 righe)
- Righe 174-200: `workspace_bytes()` — costruisce `Gemm::Arguments` con `nullptr` + `can_implement` + `get_workspace_size`.
- Righe 201-215: `workspace_ensure()` / `workspace_release()` — solo `cudaMalloc/Free` init-time.
- Righe 216-255: `run()` invariato (compat: malloc per-call + `cudaDeviceSynchronize()`).
- Righe 257-291: `run_ws()` — `can_implement` + check `need<=ws_bytes` + `gemm.run(args,workspace,stream)`, nessuna sync.
- Righe 292-349: `(b)` prototipo device quant sotto `#ifdef HERMES_ENABLE_DEVICE_QUANT` — un thread per blocco 16, scrive `Aq` lineare + `flat` scales; repack interleaved ancora host-side (`repack_sfa_cpu`). NON compilato di default.

### (TASK3b, disabilitato) `kernels/tests/test_nvfp4_quant_device.cu` (nuovo, NON in CMakeLists)
- Confronta `Aq` + `flat->repack->SFA` host-vs-device, tiny `T=32,K=128`.
- Richiede `-DHERMES_ENABLE_DEVICE_QUANT=1`; build manuale soltanto. Disabilitato per non rompere la build.

### (TASK2) `architecture/tests/test_mtp_accept.cpp` (51 righe, copiato da `/tmp/test_mtp_accept.cpp`)
- Exit: `0` = finito + `finite` + `dmx<1e4 && dmn>-1e4` + `dtop==tnext` (identita` greedy K=1); `1` = `!fin` o scala assurda (invalido numerico, include `open FAIL`); `2` = valido ma mismatch (wiring/acceptance da investigare, NON successo).
- `CMakeLists.txt` riga 77: `add_executable(test_mtp_accept ...)` + `target_link_libraries(... hermes_arch hermes_gguf)` + `add_test(NAME mtp_accept ...)` (stesso schema di `test_model_fwd` righe 72-74).
- Ricostruito per verifica build soltanto, NON eseguito.

## 3. Esiti test tiny (GPU1, `CUDA_VISIBLE_DEVICES=1 timeout 120`, build `-j4`)

- `test_gdn_step`: `H=4 K=16 V=16 T=8 out=5.59e-09 state=7.45e-09 PASS` + `H=48 K=128 V=128 T=2 PASS`. Nota: primo caso ha `V=16<128`, prima hangava.
- `test_nvfp4_const`: `expect 8192, got min=8192.0 max=8192.0 mean=8192.0 rc=0 PASS`.
- `test_nvfp4_order`: `expect 1408 (linear) or 1280 (swapped); got 1408.0 rc=0` (lineare confermato).
- `test_mtp_accept`: solo build (`cmake -S/-B` + `--target test_mtp_accept -j4` OK), esecuzione VIETATA (2 prefill ~20min, accodato altrove).
- Mai eseguiti: `test_model_fwd`, `test_e2_ext12`, `test_mtp_accept`, `test_decode_quick`, `e1_*.py`, `blk0_stats.py`, `ctest` completo (run 16GB attivo, rischio OOM).

## 4. Cosa resta (niente stime inventate)

- `run_ws()` rimuove solo malloc/free/sync. Resta per-call: H2D di `Aq/SFA` se si usa `quant_activation()` host + `cudaMemcpy` pesi se non persistenti. Misurare con benchmark prima di ottimizzare oltre.
- Batching, CUDA Graphs, split prefill/decode kernel: solo quando i benchmark lo giustificano (non ancora misurati qui).
- `(b)` device quant fused interleaved: prototipo attuale scrive `flat`, non SFA interleaved direttamente; serve repack device-side + verifica tiny host-vs-device su GPU1 prima di abilitare. Lasciato disabilitato di default con nota, mai rompere la build.
- Pesi `B`: `build_B_from_gguf()` + `sfb_bytes()` gia` load-time; mantenere `Bq/SFB` device persistenti a livello modello; nulla da fare per-call.
- Nessun dato inventato su speedup/memoria: servono benchmark su shape reali (prefill T,N,K) con stream/event timing.
