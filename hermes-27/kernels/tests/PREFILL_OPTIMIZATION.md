# HERMES-27: prefill NVFP4 con attivazioni BF16

**Stato: prototipo validato su una proiezione reale e compilato in CMake,
ma non abilitato nel percorso di produzione.** Il precedente W4A4 ha una RMSE relativa del 4.3-4.6%
sull'uscita della prima proiezione. Questi percorsi preservano le
attivazioni BF16 senza requantizzazione A4. La parita dei logit
del modello completo NON e ancora dimostrata.

## Percorso A: GGUF NVFP4 -> BF16 temporaneo -> Tensor Core

Il kernel CUDA trasforma una sola matrice raw GGUF in BF16 row-major.
cuBLAS GEMM calcola Y[T,N] FP32 = X[T,K] BF16 * W[N,K] BF16 trasposta,
con accumulo FP32. Il buffer BF16 si riutilizza tra matrici/layer.
L'ordine di somma GPU e differente da GEMV: misurare sempre la deriva
numerica su output e logit finali.

**Prova della precisione dei pesi:** i valori sono prodotti di una scala
UE4M3 con E2M1. La verifica CPU esaustiva controlla tutte le
4096 coppie codificate: ogni prodotto e rappresentabile esattamente in
BF16. Il test GPU confronta poi **tutti i N*K valori** della matrice
decompressa con una decodifica GGUF CPU, bit per bit.

Per N=10240 K=5120: raw 29,491,200 B, scratch BF16 104,857,600 B.
Non mantenere una seconda copia BF16 permanente dell'intero modello.
Misura decompressione+GEMM insieme: il GEMM puro esclude la lettura
GGUF e la scrittura di oltre 100MB per matrice.

### Follow-up: decoder vettoriale + scratch riusabile

- L'implementazione precedente usa una lane per byte E2M1, ma scrive
  due BF16 a distanza di 8 elementi nel gruppo di 16.
- Il nuovo decoder vector usa una warp per blocco GGUF da 64 valori,
  carica 32 byte E2M1, trasmette le quattro scale UE4M3 via shuffle,
  ricostruisce l'ordine dei nibble e scrive 32 coppie BF16 contigue,
  cioe 32 scritture a 32 bit. Una warp elabora quattro blocchi in loop.
- Il risultato deve corrispondere **bit per bit** al decoder CPU GGUF
  e al precedente decoder CUDA. L'ottimizzazione non cambia ne i
  pesi ne le attivazioni. Il miglioramento di latenza, se presente,
  va misurato sullo stesso GGUF e GPU.

La classe ReusableBF16WeightScratch riceve una sola allocazione esterna
BF16, riusabile su layer e richieste della **stessa GPU**. Possiede un
handle cuBLAS privato e un evento CUDA. Le chiamate enqueue da thread
diversi sono serializzate da mutex; se lo stream cambia, il nuovo stream
attende l'evento emesso **dopo** il GEMM precedente. Il buffer non viene
mai letto/scritto da operazioni concorrenti non coordinate.
Nessuna allocazione o sincronizzazione dispositivo nel percorso enqueue.
La chiusura sincronizza l'ultimo evento prima di restituire il buffer.
Se un enqueue fallisce, la classe non accetta altri lavori fino alla
chiusura (fail-closed). CUDA Graph capture non e supportato.

Il buffer puo essere inferiore a N*K*2: il wrapper dequantizza blocchi
di righe W, calcola GEMM e scrive direttamente nel risultato row-major
Y[T,N]. Il leading dimension cuBLAS rimane N; i tile non sono matrici
contigue di output da sovrascrivere. Esempio N=10240, K=5120:

| Scratch BF16 | Righe per tile | Numero di tile |
|---|---:|---:|
| 8 MiB | 768 | 14 |
| 32 MiB | 3200 | 4 |
| 105 MiB | 10240 | 1 |

I tile richiedono piu lanci di decoder+GEMM: usare meno memoria puo
**ridurre** le prestazioni. Non presumere una convenienza automatica.
Per carichi multiutente ad alta concorrenza e preferibile testare un
piccolo slot per stream attivo: un solo slot condiviso tra N stream
serializza intenzionalmente i prodotti e limita il throughput. Non
riutilizzare la stessa scratch fuori dalla classe fino alla shutdown.

La forma matematica e la stessa del GEMV PP2 solo se X e gia BF16:
il percorso prefill mantiene lo stato hidden in FP32 e lo converte
esplicitamente in BF16 con f32_to_bf16_k prima di gemv_batch.
L'API GEMM riceve il relativo puntatore BF16, **non** il float32
originale. W e' GGUF NVFP4 -> BF16 esatto; cuBLAS effettua
BF16 x BF16 con accumulo FP32, e restituisce FP32 [T,N].
Differenze nell'ordine di accumulo sono inevitabili e vanno
confrontate rispetto al baseline con logit finali.

## Percorso B: direct batch4 su GGUF raw

Ogni warp legge pesi GGUF di una riga e li riutilizza per quattro token
BF16. Il risultato e FP32. Nessuno scratch di pesi, nessuna A4/FP8.
La riduzione FP32 e diversa dal GEMV scalare e deve superare il gate.
Il benchmark lo mostra **piu lento del GEMV Warp4** su tutte le
dimensioni misurate. Non usarlo come ottimizzazione attuale.

## File

- kernels/src/nvfp4_prefill_exact.cu
- kernels/include/hermes/nvfp4_prefill_exact.hpp
- kernels/tests/prefill_exact_bench.cu

## Compila il prototipo in modo isolato

```bash
root=/home/matteo/hermes-27
s=/tmp/hermes_prefill_exact_worker2
mkdir -p "$s/src" "$s/include/hermes" "$s/tests"
cp "$root/kernels/src/nvfp4_prefill_exact.cu" "$s/src/"
cp "$root/kernels/include/hermes/nvfp4_prefill_exact.hpp" "$s/include/hermes/"
cp "$root/kernels/tests/prefill_exact_bench.cu" "$s/tests/"
nvcc=/usr/local/cuda-12.8/bin/nvcc
"$nvcc" -std=c++20 -arch=sm_120a -O3 -I"$s/include" \
  -c "$s/src/nvfp4_prefill_exact.cu" -o "$s/exact.o"
"$nvcc" -std=c++20 -arch=sm_120a -O3 \
  -I"$s/include" -I"$root/kernels/include" -I"$root/model/include" \
  -c "$s/tests/prefill_exact_bench.cu" -o "$s/bench.o"
"$nvcc" -std=c++20 -arch=sm_120a -O3 -o "$s/bench" \
  "$s/bench.o" "$s/exact.o" \
  "$root/build/libhermes_kernels.a" "$root/build/libhermes_gguf.a" -lcublas

"$s/bench" --cpu-only
# Test CPU del nuovo decoder vettoriale, dei tile e del formato BF16
python3 "$root/kernels/tests/prefill_unpack_cpu.py"
```

Esegui i benchmark GPU **soltanto con GPU esclusiva autorizzata**:

```bash
model=/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf
nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv
timeout 90 /tmp/hermes_prefill_exact_worker2/bench \
  --gguf "$model" --T 64 --device 1 --repeats 5
# Soltanto dopo aver verificato T64:
timeout 90 /tmp/hermes_prefill_exact_worker2/bench \
  --gguf "$model" --all --device 1 --repeats 5
# Solo quando la GPU e libera: vecchio vs nuovo unpack sulla stessa W
timeout 90 /tmp/hermes_prefill_exact_worker2/bench \
  --gguf "$model" --T 64 --device 1 --repeats 10
# Profilo reale dei buffer, senza seconda copia BF16 permanente:
for scratch in 8 32 105; do
  timeout 90 /tmp/hermes_prefill_exact_worker2/bench \
    --gguf "$model" --T 64 --device 1 --repeats 5 \
    --slot-only --scratch-mib "$scratch"
done
```

Input: veri embedding GGUF, normalizzazione del blocco 0, BF16.
Confronta GEMV scalare raw GGUF, cuBLAS BF16 e direct-batch4 su T16/64/128/256.
Oracle CPU FP64 su 32 righe di output per token.
Per ogni proiezione il gate richiede maxabs<=0.01 e RMSE relativa<=0.001
sia rispetto al GEMV sia all'oracle CPU; altrimenti esce con errore.
Eventi CUDA misurano GEMV, direct-batch4, decompressione, GEMM e pipeline.
Riporta separatamente delta VRAM e scratch della matrice BF16.
L'output UNPACK_AB_MS confronta il vecchio e il nuovo decoder; il
controllo WEIGHT_UNPACK_FULL_CHECK deve passare anche sulla versione
vettoriale. Con --slot-only il test usa *due stream CUDA*, alterna
GGUF originale e GGUF con pesi di segno invertito e confronta la
prima e l'ultima uscita contro Warp4: un ordine di eventi errato
non deve passare il gate.

### Misure del follow-up decoder vettoriale e riuso scratch (GPU1)

T=64, N=10240, K=5120. Input: BF16 derivati dagli embedding GGUF reali.
10 ripetizioni CUDA-event dopo warmup. Tutti i 52,428,800 pesi BF16
del nuovo decoder sono bit-identici alla decodifica CPU GGUF.
La comparazione di output per i due stream usa 655,360 valori ciascuno.
La scratch non mantiene copie dequantizzate persistenti dopo il GEMM.

| Misura | Tempo |
|---|---:|
| Vecchio unpack, stesso GGUF | 1.71469 ms |
| Nuovo unpack warp-vector + E2M1 bitcast | **0.36210 ms** |
| Riduzione del solo unpack | **4.735x** |
| Warp4 di riferimento, 64 lanci | 9.1103 ms |

| Scratch esterna | Tile seriali | Unpack+GEMM T64 | VRAM allocata nel test | Errore massimo vs Warp4 | Test due stream |
|---|---:|---:|---:|---:|---|
| **8 MiB** | 14 | **0.48368 ms** | 96,468,992 B | 0.00001145 | PASS |
| 32 MiB | 4 | 0.70335 ms | 121,634,816 B | 0.00016404 | PASS |
| 105 MiB | 1 | Non testato con decoder finale | Non misurata | Non misurato | Non testato |

In questo caso, 8 MiB sono sufficienti e piu veloci di 32 MiB
**nel microbenchmark isolato**, ma l'effetto puo dipendere dalla
taglia del GEMM, dall'occupazione, dal caching e dal traffico. Non
estrapolare il rapporto 9.1103/0.48368 al PP2 completo.
Il totale VRAM in tabella include due copie GGUF nel *test*
(una con segno invertito), gli input e piu output. Non e il consumo
addizionale di PP2. Il beneficio di scratch BF16, invece, e reale:
8,388,608 B allocati anziche 104,857,600 B per questa matrice.

Il test cross-stream accoda, senza sincronizzare dopo ogni chiamata:
W originale sullo stream A, W con i segni invertiti sullo stream B,
e nuovamente W originale su A. Gli output del primo e ultimo lancio
devono coincidere con il baseline (entro la tolleranza del GEMM),
quello centrale con il suo negativo. PASS con entrambi i buffer
8/32 MiB. Confronto sul caso 8MiB: RMSE relativa 1.11e-7 e
massimo errore assoluto 1.14e-5. Non equivale a una prova generale
di sicurezza concorrente su ogni scheduler.

I risultati numerici e le limitazioni sono anche in
kernels/tests/prefill_unpack_results.json. La GPU e stata rilasciata,
con nvidia-smi tornato ai valori prima del test (486/1289 MiB usati).

## GO / NO-GO per il sistema completo

### Misure su GGUF reale, 13 novembre 2026

RTX 5060 Ti 16 GB, SM120, CUDA 12.8, cuBLAS, GPU1. Pesi reali
blk.0.attn_qkv.weight: N=10240 e K=5120. Input T token da embedding
GGUF e RMSNorm del blocco 0. Warmup, cinque ripetizioni per kernel;
tempi CUDA event in millisecondi. **Questo non e un profilo PP2 completo.**

| Misura | T=16 | T=64 | T=128 | T=256 |
|---|---:|---:|---:|---:|
| GEMV scalare PP2, ripetuto | 22.640 | 90.732 | 181.483 | 362.858 |
| GEMV Warp4, stessa forma del kernel di produzione | 2.250 | 9.056 | 18.905 | 37.799 |
| Direct batch4 GGUF | 7.326 | 29.179 | 58.324 | 116.601 |
| Unpack raw GGUF -> BF16 GPU | 1.716 | 1.716 | 1.716 | 1.715 |
| cuBLAS GEMM BF16, accumulo FP32 | 0.262 | 0.604 | 0.425 | 0.610 |
| Unpack + cuBLAS, costo combinato | 1.964 | 2.320 | 2.132 | 2.328 |
| Speedup singola proiezione vs Warp4 | 1.15x | 3.90x | 8.87x | 16.24x |
| Delta memoria GPU, tutti i buffer di test | 138.4 MB | 153.1 MB | 161.5 MB | 182.5 MB |

Precisione: per tutte le 52,428,800 celle di W BF16 dequantizzato,
uguaglianza bit per bit rispetto all'oracle CPU su GGUF.
CuBLAS contro GGUF GEMV scalare su **tutte** le uscite:
errore massimo 0.00012970 (T16) e 0.00016785 (T64/128/256);
RMSE relativa 1.41e-6 / 1.29e-6 / 1.23e-6 / 1.24e-6.
Direct batch4 contro GEMV: errore massimo 3.815e-6 e RMSE relativa
circa 5.7e-8. Anche tutti i confronti CPU FP64 su 32 righe per token
hanno superato la soglia. **Non e una prova di identita dei logit.**

Il baseline Warp4 nel benchmark e una copia locale del percorso
nvfp4_gemv_warp4_k, con aritmetica e geometria di lancio equivalenti.
Non e un replay del motore completo. Warp4 rimane preferibile al
direct-batch4: a T64 e circa 3.2 volte piu veloce.

La decompressione deve avvenire ad ogni proiezione diversa se il
buffer BF16 viene riutilizzato. Per T16 il guadagno solo 1.15x
puo essere annullato dall'integrazione e dai costi di scheduling;
il dispatch deve mantenere Warp4 per batch piccoli o misurati
come non convenienti.

Il costo della decompressione e il principale limite della proposta:
a T64 occupa circa il 74% del tempo combinato misurato. Prima di
ottimizzare ancora il GEMM, valutare un decoder vettoriale che
scriva gruppi BF16 contigui, riusi le scale in warp e non allochi
per richiesta. Verificare nuovamente tutti i pesi bit per bit e il
tempo totale. Nessuna accelerazione futura e attribuita a queste idee.

**Attuale NO-GO per PP2 produzione:** mancano logit full-model,
benchmark end-to-end, verifiche cache/attenzione e conferma multiutente.

Per un futuro GO, dimostrare prima l'esattezza dei pesi e la precisione
delle proiezioni, poi eseguire il gate di qualita sui logit completi
in PREFILL_QUALITY_GATE.md, con stessa versione GGUF e attivazioni.
Serve un profilo PP2 completo comprensivo di decompressioni per layer,
GDN, attenzione, trasferimento GPU0->GPU1, output head, concorrenza
1/4/8 utenti, mediana e p95, e stabilita dei consumi VRAM. Confrontare
anche il GEMV Warp4 corrente, non soltanto quello scalare.

T128 e T256 sono soltanto forme di GEMM sperimentale: il prefill PP2
accetta fino a 64 token, l'attenzione ha score buffer T64 e ci sono
buffer pinned/ricorrenze da verificare. Non aumentare il limite PP2
senza test di causalita chunked, cache GDN/KV e logit end-to-end.

**Raccomandazione di integrazione:** percorso W4A16 temporaneo come
opt-in prefill per batch medi/grandi, con scratch riutilizzata e
fallback Warp4. Prima calibrare soglia dinamica del batch su misure
end-to-end con reali richieste concorrenti. Non adottare direct-batch4
nella sua forma attuale. Nessuna espansione automatica di T64.
