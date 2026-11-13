# M1 parziale — Loader GGUF + validazione reale (2026-11-12)

Stato: parser strict + dump + synth PASS, validazione GGUF reale PASS. Manca: mmap upload distribuito, confronto FP32/BF16 full, test forward completo (M2).

## Codice (indipendente, no runtime copiato)
- `model/include/hermes/gguf.hpp`, `model/src/gguf.cpp`: parser GGUF v3 strict (magic, version, n_tensors/n_kv, KV U8..F64/STR/ARR, tensor infos, alignment default 32, overlap/overflow/misalign/duplicati rifiutati). Tipi: F32/F16/BF16/I*/NVFP4(40)/MXFP4(39). `ggml_row_size`: NVFP4 36B/64elem.
- `model/include/hermes/weights.hpp`, `model/src/weights.cpp`: interfaccia `IWeightBackend`, `GgufWeightBackend` mmap (Linux) / stub Win. Isola futuro safetensors mixed.
- `tools/gguf_dump.cpp`: ispettore read-only.
- `model/tests/test_gguf_synth.cpp` + `model/tests/test_nvfp4.py`: PASS (`SYNTH OK`, `NVFP4 synth OK`).
- Build: `cmake -S . -B build && cmake --build build -j6` con gcc 13.3, C++20, OK.

## NVFP4 (verificato da formato, implementazione indipendente)
- Blocco 36B: 4x UE4M3 (1B/16elem) + 32B E2M1 packed. UE4M3: exp bias 7, `raw=ldexp(1+man/8,exp-7)` (subnorm `ldexp(man,-9)`), 0/0x7F→0, valore finale `raw`; convenzione codice `ue()*2*e2m1` = `raw*E2M1`. E2M1 LUT: [0,.5,1,1.5,2,3,4,6,-0,-.5,-1,-1.5,-2,-3,-4,-6].
- Test: 0x40→ue 1.0 (raw 2.0), nibble 1→0.5, blocco→1.0. PASS.

## Validazione file reale `Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf` (19694360544 B)
- `gguf_dump --strict`: PASS (version 3, 866 tensors, 50 KV, align 32, data_off 10995680).
- KV: arch `qwen35`, block_count 65, ctx 262144, embed 5120, FFN 17408, heads 24/KV 4, key/value len 256, rope sections [11,11,10]? (last 0), ssm inner 6144/group 16/state 128/conv 4/rank 48, full_interval 4, nextn 1, vocab 248320, merges 247587, file_type 7, quant v2.
- Tensori: NVFP4 448 (13690552320 B = 12.75 GiB), BF16 10 (2x 2.54GB embed/head + 8x MTP), F32 408, totale 19683364864 B = 18.33 GiB + header 11MB = 19.69GB. Checksum size OK.
- Per-blocco: 48x linear (14 tensori: attn_gate 5120x6144, attn_qkv 5120x10240, ssm_a/alpha/beta/conv1d 4x10240/dt/norm/out 6144x5120, FFN 3x, 2x norm) + 16x full (11 tensori: attn_q 5120x12288, k/v 5120x1024, q/k_norm 256, out 6144x5120, FFN, norms) ogni 4 (3,7,…,63) + blk.64 MTP BF16 (attn_q/k/v/out, FFN, nextn eh_proj 10240x5120 + 3x norm).
- Bug trovati e fixati: ARR STR early-break → desync (`key=bus`), limite array 100K→5M, arch `qwen35` accettata. Nessun mock: test su file reale.

## Decisioni
- Backend mmap, no upcast stabile BF16. TP2 shard pesi / PP2 shard layer, host-pinned (P2P False). KV paged FP8 + DeltaNet separato, prefix CoW solo se matematicamente identico.
- Prossimo M1→M2: upload distribuito test, dequant vs FP32/BF16 soglie, CUTLASS SM120 (richiede toolkit), kernel RMSNorm/RoPE/GEMV.

Riproduci: `cd ~/hermes-27 && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j6 && ./build/test_gguf_synth && python3 model/tests/test_nvfp4.py && ./build/gguf_dump ~/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf --strict | head`
