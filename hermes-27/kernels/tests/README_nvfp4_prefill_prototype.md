# HERMES-27: prototipo CUTLASS W4A4 per prefill batched

Stato: misurato sul GGUF reale, 13 novembre 2026. Questo codice non è integrato
in PP2. Non dimostra parità dei logit o della generazione completa.

## File nuovi

- kernels/include/hermes/nvfp4_prefill_prototype.hpp: interfaccia.
- kernels/src/nvfp4_prefill_prototype.cu: quantizzazione GPU e GEMV scalare.
- kernels/tests/bench_nvfp4_prefill_prototype.cu: benchmark e verifiche.

Il GEMM effettivo usa Nvfp4GemmW4A4::run_ws già implementato nel progetto.
La matrice dei pesi arriva dal GGUF tramite build_B_from_gguf.
La quantizzazione GPU genera E2M1 e scale UE4M3 nel layout CUTLASS.

## Compilazione isolata

Compila solo in una directory temporanea. Usa i binari statici esistenti del
progetto. Non modificare CMakeLists.txt.

    root=/home/matteo/hermes-27
    scratch=/tmp/hermes_prefill_worker2
    mkdir -p "$scratch/include/hermes" "$scratch/src" "$scratch/tests"
    cp "$root/kernels/include/hermes/nvfp4_prefill_prototype.hpp" "$scratch/include/hermes/"
    cp "$root/kernels/src/nvfp4_prefill_prototype.cu" "$scratch/src/"
    cp "$root/kernels/tests/bench_nvfp4_prefill_prototype.cu" "$scratch/tests/"
    NVCC=/usr/local/cuda-12.8/bin/nvcc
    "$NVCC" -std=c++20 -arch=sm_120a -O2 -I"$scratch/include" -I"$root/kernels/include" -c "$scratch/src/nvfp4_prefill_prototype.cu" -o "$scratch/proto.o"
    "$NVCC" -std=c++20 -arch=sm_120a -O2 -I"$scratch/include" -I"$root/kernels/include" -I"$root/model/include" -c "$scratch/tests/bench_nvfp4_prefill_prototype.cu" -o "$scratch/bench.o"
    "$NVCC" -std=c++20 -arch=sm_120a -O2 -o "$scratch/bench" "$scratch/bench.o" "$scratch/proto.o" "$root/build/libhermes_kernels.a" "$root/build/libhermes_gguf.a"

Esegui il controllo del layout senza usare la GPU:

    /tmp/hermes_prefill_worker2/bench --layout-only

Usa la GPU solo con accesso esclusivo. Prima verifica la memoria libera.
Poi esegui il test su T=64. Per gli altri batch imposta T=16 o T=256.

    model=/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf
    nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv
    timeout 35 /tmp/hermes_prefill_worker2/bench --gguf "$model" --device 1 --T 64 --repeats 2

## Metodo di confronto

Il benchmark usa il checkpoint reale:

- Pesi blk.0.attn_qkv.weight, NVFP4, N=10240, K=5120.
- Input dalla tabella BF16 token_embd.weight, con ID deterministici.
- RMSNorm diretta con blk.0.attn_norm.weight, seguita da BF16 rounding.

Questi input riproducono la prima proiezione lineare del blocco 0.
Gli stati dei layer successivi non sono inclusi.

Il baseline esegue un GEMV scalare GGUF per token. Il CUTLASS W4A4
esegue un GEMM per tutto il batch. Il benchmark registra:

- Tempi GPU con eventi CUDA: GEMV, GEMM, quantizzazione GPU e quant+GEMM.
- Tempi CPU del repack dei pesi, della quantizzazione e dei trasferimenti H2D.
- Delta di memoria GPU tra il momento prima e dopo le allocazioni.
- Errore W4A4 contro GEMV BF16 FP32 su tutti i valori di output.
- Errori contro un dot product CPU indipendente per 32 righe su ogni token.

Il test ricostruisce i pesi GGUF direttamente dai blocchi 36 byte,
secondo l'ordine nibble [j,j+8]. Ricostruisce inoltre la quantizzazione
di attivazione dal buffer Aq. La disposizione SFA su GPU è ricavata dal
repacker CUTLASS e controllata con una mappa biunivoca.

Il test si arresta se i byte Aq o SFA differiscono tra CPU e GPU,
se un risultato contiene NaN/Inf, oppure se un output campionato differisce
troppo dall'oracle CPU. La tolleranza campionata è:

    abs(output - reference) <= 0.05 + 0.02 * abs(reference)

Non si usa tale tolleranza per sostenere parità contro il modello.

## Risultati reali, RTX 5060 Ti, SM120, CUDA 12.8

Ripetizioni per le misure CUDA: 1/2/2. Tutti i controlli numerici
di conversione e layout sono PASS.

| Misura | T16 | T64 | T256 |
|---|---:|---:|---:|
| GEMV scalare ripetuto | 22.706 ms | 90.895 ms | 363.606 ms |
| CUTLASS W4A4 solo GEMM | 0.0562 ms | 0.0552 ms | 0.0963 ms |
| Quantizzazione GPU | 0.0052 ms | 0.0052 ms | 0.0138 ms |
| Quant GPU + GEMM | 0.0590 ms | 0.0591 ms | 0.1147 ms |
| Quantizzazione CPU | 1.54 ms | 6.09 ms | 24.32 ms |
| Repack pesi, una volta | 23.66 ms | 23.48 ms | 23.73 ms |
| Delta VRAM allocazioni | 65.01 MB | 71.30 MB | 85.98 MB |
| W4A4 vs GEMV: RMSE relativa | 4.33% | 4.52% | 4.57% |
| W4A4 vs GEMV: massimo errore | 0.935 | 1.063 | 1.209 |
| W4A4 vs CPU A4/W4: massimo errore | 0.0143 | 0.0149 | 0.0258 |

GEMV PP2 contro dequantizzazione CPU diretta: errore massimo 4.77e-7.
Quantizzazione CPU contro GPU: zero byte diversi per tutti i batch.
Gli output CUTLASS con quantizzazione CPU e GPU sono identici.

Il peso GGUF raw occupa 29,491,200 byte. Il peso ripacchettato
occupa 26,214,400 byte (Bq) + 3,276,800 byte (SFB): stesso totale.
Tenere le due rappresentazioni residenti raddoppia la VRAM di questa
matrice. Questo costo è un limite serio se esteso a tutti i layer.

La mappa SFA usa 40,960/40,960/81,920 byte per T16/64/256.
Il layout SFB richiede 3,276,800 byte per tutti i T: test CPU PASS.
Lo stato VRAM dopo l'esecuzione è tornato al baseline:
GPU0 486 MiB, GPU1 1289 MiB.

## Limiti e decisione

La quantizzazione A4 produce circa 4.5% di RMSE relativa nella
proiezione rispetto all'input BF16 originale. Il kernel è molto veloce,
ma la precisione della generazione finale è ancora sconosciuta.

Prima dell'integrazione verifica i logit finali, top-k e argmax del
modello completo. Usa gli stessi prompt e lo stesso GGUF in HERMES
GEMV originale, HERMES W4A4 e llama.cpp. Verifica più lunghezze.

Serve inoltre una politica per pesi raw e ripacchettati. Conservare
entrambe le versioni dell'intero modello supera il budget VRAM.

T=256 è solo un benchmark di proiezione. Il PP2 completo accetta
al massimo 64 token in prefill e usa un buffer SDPA di 64 posizioni.
Il miglioramento del solo GEMM non equivale a un miglioramento end-to-end.
