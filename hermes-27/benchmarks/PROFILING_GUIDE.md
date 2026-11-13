# HERMES-27 — profiling riproducibile PP2 (P7, preparato, NON ancora misurato)

Obiettivo: verificare **quali kernel CUDA consumano realmente i ~645 ms/token** del percorso decode dopo P6. Il tool `tools/nsys_profile_report.py` elabora una traccia Nsight Systems **offline, senza caricare GPU**. Non modifica engine, checkpoint, golden, serving o CMake.

## Stato, vincoli e dati esistenti

- Dati già misurati: `benchmarks/P3_BASELINE.md` (654 ms/token; prefill 6.79 s/11 tok; PP xfer 0.06 ms), `distributed/P5_BATCH.md` §2 (N4 par x1.02 vs N1), §P6 (645.1 ms/token dopo arene persistenti, x1.015). Queste sono misure **end-to-end**, non tempi per kernel.
- `nvfp4_gemv.cu:70-89` usa dequantizzazione per riga senza Tensor Core. `pp2_fwd.cu:983,1049` sincronizza una volta per layer; `:264-266` chiama GEMV una volta per token nel prefill. `:228-239` calcola logits con accumulazione FP64 scalare, candidata da misurare.
- Al momento della preparazione il comando `nsys` non risultava nel PATH di `matteo@hermes`. Eseguire i comandi di acquisizione **solo dopo coordinamento con il responsabile GPU** e disponibilità di Nsight Systems. Nessun benchmark GPU è stato eseguito per questo task.

## Profilazione, quando il server è libero

Eseguire sul server `/home/matteo/hermes-27`. Prima verificare `command -v nsys`, `nsys version` e che il binario `build/test_pp2_decode` esista. Per isolare la generazione greedy su un prompt di 11 token:

```bash
cd /home/matteo/hermes-27
nsys profile --trace=cuda --sample=none --force-overwrite=true --output=/tmp/h27_decode build/test_pp2_decode greedy
nsys export --type=sqlite --force-overwrite=true --output=/tmp/h27_decode.sqlite /tmp/h27_decode.nsys-rep
python3 tools/nsys_profile_report.py /tmp/h27_decode.sqlite --last-embeds 4 --json-out /tmp/h27_decode_summary.json
```

`--last-embeds 4` seleziona le ultime 4 occorrenze GPU0 di `embed_gather_k`, ciascuna all'inizio di un decode step. **Verificare manualmente nel log** che corrispondano ai quattro step greedy; in caso contrario usare `--start-ms` e `--end-ms` (coordinate relative alla prima attività GPU). Includere nel confronto gli stessi input, identica configurazione, nessun altro job GPU e **almeno 3 acquisizioni** per ciascuna variante. Misurare anche separatamente il tempo wall del test fuori dal profiler perché il tracciamento altera le prestazioni.

Per confrontare batch 4 e prefill, ripetere l'acquisizione con `build/test_pp2_batch g2` o `build/test_pp2_fwd g3` e filtrare l'intervallo esatto. Per il test batch che esegue più N in sequenza, `--last-embeds` deve corrispondere al numero di step per-request della *sezione finale*; verificare nel report i conteggi `embed_gather_k`.

L'analizzatore presenta, per GPU, busy time dei kernel, trasferimenti e sovrapposizione temporale delle due GPU; raggruppa GEMV, head, GDN, attention, elementwise e raccoglie le API `cudaDeviceSynchronize`, `cudaMemcpy`, `cudaLaunch*`, dove presenti nello schema Nsight. La somma dei tempi di invocazione GPU **non equivale al wall time** quando le GPU lavorano in parallelo; i tempi API sono tempo host e non vanno sommati a quelli GPU. La percentuale di GPU attiva è la copertura temporale degli eventi kernel registrati, non la percentuale di utilizzo degli SM da `nvidia-smi`.

Test del solo analizzatore, senza binari CUDA: `python3 tools/nsys_profile_report.py --self-test` (SQLite in memoria, nessun file creato).

## Decisioni dopo acquisizione

1. **Priorità multiutente:** misurare durata e sovrapposizione kernel GPU0/GPU1, conteggio e tempo di `cudaDeviceSynchronize`; solo dopo valutare stream espliciti, eventi e CUDA Graphs.
2. **Priorità prefill:** quantificare gli `nvfp4_gemv_k` in T=11 e T=64. Se dominanti, integrare W4A4 CUTLASS + quantizzazione attivazioni device e test golden.
3. **Priorità decode:** misurare quota `nvfp4_gemv_k`, `gdn_step_k` e `head_bf16_k` (e API host). Ottimizzare il kernel dominante per primo; conservare bit-exact decode-vs-prefill e regressioni P5.
4. **Concorrenza:** confronto N=1/2/4 a contesto uguale, throughput aggregato reale, TTFT e VRAM. Non interpretare il solo overlap GPU come speedup.

Nessun incremento prestazionale è dichiarato da questa preparazione: lo strumento consente di produrre le **prime misure per-kernel**, attualmente assenti nei report P3/P5.
