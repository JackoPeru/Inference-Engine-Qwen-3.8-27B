# Skip delle sincronizzazioni di fine layer PP2 (opt-in)

## Ambito

`HERMES_DECODE_SKIP_LAYER_SYNC=1` disabilita **solo** i due `cudaDeviceSynchronize()` al termine di `run_gdn_decode_layer` e `run_full_decode_layer`, nel file `distributed/pp2_fwd.cu`. Il valore deve essere esattamente `1`; la variabile viene letta una sola volta per processo. Quando e` assente, vuota o diversa da `1`, il comportamento rimane quello precedente. L'opt-in `HERMES_NVFP4_GEMV=warp4` e` indipendente.

Non sono stati modificati gli stream, i kernel, i buffer, i percorsi di debug o le sincronizzazioni di `decode_step`.

## Perche` l'ordine rimane valido

- Tutti i kernel di ciascun decode layer sono lanciati nello stream CUDA predefinito (`stream=0`), inclusi `nvfp4_gemv_cuda_variant` e `gdn_step_cuda`. Un singolo stream esegue i comandi nell'ordine di enqueue: il layer successivo puo` essere accodato senza attendere sulla CPU il completamento del precedente.
- I buffer workspace persistenti vengono riutilizzati dal layer successivo **nello stesso stream del medesimo device**, dopo l'ultimo `add_k`. Il riuso e` ordinato anche se `cudaDeviceSynchronize` e` omessa.
- La copia `cudaMemcpy(g_pinned, d_x0, ..., cudaMemcpyDeviceToHost)` al confine GPU0/GPU1 resta invariata: ritorna solo dopo il completamento della copia, e la copia nello stream predefinito aspetta i kernel GPU0 precedenti. Anche la successiva `cudaMemcpy(d_x1, g_pinned, ..., cudaMemcpyHostToDevice)` da memoria pinned e` sincrona rispetto all'host. Il mutex `g_xfer_mu` rimane attivo per entrambe le operazioni.
- Il kernel dell'output head, il successivo `cudaDeviceSynchronize()` e la copia bloccante D2H dei logits restano invariati. La funzione `decode_step` quindi restituisce logits pronti, non un puntatore a un risultato GPU incompleto.
- Dentro un layer full-attention, le due copie `cudaMemcpyDeviceToDevice` restano nello stream predefinito: il fatto che non attendano il completamento rispetto all'host non cambia l'ordine rispetto al successivo attention kernel.

Queste premesse dipendono dall'assenza di nuovi stream espliciti o kernel asincroni lanciati fuori dallo stream 0. Se in futuro si introducono stream personalizzati, eventi o CUDA Graphs, rivalutare la garanzia prima di usare l'opt-in.

Riferimenti NVIDIA: [CUDA Runtime API synchronization behavior](https://docs.nvidia.com/cuda/archive/12.8.2/cuda-runtime-api/api-sync-behavior.html) e [CUDA Programming Guide: streams](https://docs.nvidia.com/cuda/archive/13.4.3/cuda-programming-guide/02-basics/asynchronous-execution.html).

## Verifica

Compilazione isolata effettuata il 2026-11-13 con CUDA 12.8, `nvcc -std=c++20 -arch=sm_120a -O3 -c` su `/tmp/hermes-worker4-sync-20261113/pp2_fwd.cu`: **PASS (exit 0)**; i warning sono relativi a variabili inutilizzate in porzioni di codice non toccate. Nessun kernel CUDA eseguito in questa verifica.

La prova su GGUF reale richiede l'autorizzazione GPU esclusiva del coordinatore, dopo il lavoro del worker BF16 head. Usare lo **stesso eseguibile, lo stesso modello GGUF, gli stessi token di input e la stessa variante GEMV** per entrambe le condizioni:

1. Eseguire `scripts/check_generate.py` con `HERMES_DECODE_SKIP_LAYER_SYNC` assente; salvare output, exit code e log.
2. Ripetere con `HERMES_DECODE_SKIP_LAYER_SYNC=1`. Richiedere exit 0, token JSONL e motivi di terminazione identici, argmax uguali e `--verify-prefill` superato. Per confronto numerico completo, salvare e confrontare gli stessi logits FP32 (criterio fail-closed: `max_abs <= 1e-4` e nessuna differenza nell'argmax); se manca il dump logits non dichiarare parita` numerica dimostrata.
3. Eseguire `test_pp2_batch g1` con entrambe le impostazioni: richiedere che tutti i gate e il controllo dei logits N=2 passino. Confrontare inoltre gli ID generati tra A/B.
4. Eseguire `test_pp2_batch g2` con entrambe le impostazioni, nelle stesse condizioni hardware: leggere misure N=1 e N=4 (`seq`, `par`), ripetere almeno tre volte, registrare tempi, TPS aggregati e VRAM. I numeri da soli non costituiscono un test di correttezza.

**Fail closed:** disabilitare l'opt-in se uno qualsiasi dei controlli di correttezza, finitezza, integrita` del contesto, limite VRAM, CUDA o uscita del processo fallisce. La mancata disponibilita` della GPU lascia il risultato nello stato **compila, runtime non verificato**.

Il risparmio e` limitato al tempo di attesa/sincronizzazione dell'host: le operazioni GPU e le copie di confine restano necessarie. Non attribuire miglioramenti di throughput senza misure N=1/4 sul checkpoint reale.
