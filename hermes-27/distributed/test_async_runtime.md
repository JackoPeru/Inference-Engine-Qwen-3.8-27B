# Per-request PP2 CUDA streams: runtime, regression, rollback

## Attivazione

Per abilitare il nuovo decode concorrente (solo con due GPU CUDA):

```bash
HERMES_PP2_ASYNC_STREAMS=1 HERMES_NVFP4_GEMV=warp4 HERMES_HEAD_BF16=warp4 ./hermes_server ...
```

- `HERMES_PP2_ASYNC_STREAMS=1` va impostata **prima di `decode_init`**; solo la stringa esatta `1` abilita il nuovo percorso. Ogni `DecodeState` conserva la scelta per la propria vita. Il default rimane il decode sincrono originale. `HERMES_NVFP4_GEMV` e `HERMES_HEAD_BF16` sono flag indipendenti, non abilitati automaticamente dal nuovo percorso.
- Non combinare con `HERMES_L26` (debug): `decode_init` rifiuta esplicitamente la combinazione. Il percorso debug legacy non viene modificato.
- L'opzione `HERMES_DECODE_SKIP_LAYER_SYNC` (precedente ottimizzazione) non cambia il nuovo percorso: con stream privati non vengono chiamate sincronizzazioni device-wide di fine layer.

## Algoritmo effettivamente integrato

`decode_init` alloca per richiesta: due `cudaStreamNonBlocking` (device 0/1), due eventi, un intero pinned per input, 20.480 byte pinned di hidden e 993.280 byte pinned di logits. Mantiene S/CS/KV/workspace e pesi già previsti; ciascun `DecodeState` possiede il proprio staging. Le inizializzazioni GDN/KV sullo stream di default sono completate **prima** del primo launch sullo stream non bloccante.

`decode_step` esegue questi stadi per la richiesta:

1. Su stream GPU0: token H2D, embedding, blocchi 0–31, hidden D2H verso pinned buffer, record evento; il thread attende **l'evento di questa richiesta**, non l'intero device.
2. Dopo l'evento GPU0: su stream GPU1, hidden H2D da pinned buffer, blocchi 32–63, output norm e head (scalare o BF16 Warp4), logits D2H pinned, record evento; attende l'evento GPU1.
3. Dopo evento GPU1: copia i logits su `std::vector<float>`, controlla finitezza, restituisce il risultato. Il token successivo dello stesso stato non può partire prima del completamento precedente. Il mutex per `DecodeState` rifiuta una seconda chiamata concorrente allo stesso stato.

I kernel dei due tipi di blocco, GEMV NVFP4 scalar/Warp4/packed4, `gdn_step_cuda`, head BF16 e copie KV D2D sono accodati nel rispettivo stream per richiesta. Non esistono copie via `g_pinned` globale nel percorso async. Un errore di enqueue, di evento o di logits rende la richiesta **poisoned** e impedisce ulteriori decode finché il client non chiama `decode_free` e `decode_init` su un nuovo stato. `decode_free` attende gli stream per richiesta prima di liberare device buffers/pinned.

**Contratto di vita:** il chiamante non deve copiare un `DecodeState` inizializzato, chiamare `free_all()` mentre ci sono stati attivi, o distruggere `DecodeState` in concorrenza con `decode_step`. Il mutex impedisce un doppio token nello stesso stato, ma non rende sicura la distruzione concorrente dello stato. L'uso concorrente di stati *diversi* è supportato.

## Verifica sul GGUF reale (2026-11-13)

Test `distributed/test_async_multiuser.cu`, compilato con CUDA 12.8 `sm_120a` e linkato in scratch. Modello `Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf`. Parametri `HERMES_NVFP4_GEMV=warp4`, `HERMES_HEAD_BF16=warp4`. Ogni richiesta usa un prompt distinto di 11 token e 4 passi greedy; vengono confrontati **tutti i 248.320 logits di ciascuno dei 15 passi** tra percorso legacy sequenziale e nuovo percorso concorrente. La condizione di successo richiede `max_abs <= 1e-4`, argmax identici, finitezza e codice di uscita 0.

| Richieste | Token complessivi | Massimo errore assoluto vs legacy | Argmax | Tempo dei 15 passi in concorrenza | Exit |
| --- | ---: | ---: | --- | ---: | --- |
| N=1 | 15 | **0** | Tutti identici | 2252,655 ms | **0** |
| N=2 | 30 | **0** per entrambe | Tutti identici | 2876,563 ms | **0** |
| N=4 | 60 | **0** per tutte | Tutti identici | 4038,206 ms | **0** |

Queste sono misure di esecuzione del test **di correttezza**, non benchmark steady-state, TTFT o serving HTTP. Ogni esecuzione aveva un watchdog di 120 secondi e ha completato correttamente. Al massimo N=4, il test ha osservato 10.154 MiB GPU0 e 11.771 MiB GPU1 dopo `decode_init`, sotto il limite di 15.360 MiB. Dopo ciascuna esecuzione, VRAM di nuovo al livello iniziale (486 MiB GPU0, 1289 MiB GPU1). I log integrali risiedono sul server soltanto in `/tmp/hermes-worker4-real-async-20261113/run_n{1,2,4}.log`. Non sono stati modificati o fermati servizi in produzione.

**Retest finale indipendente dopo audit di tutti i lanci CUDA:** `test_async_stream_contract.py` verifica 29 kernel dei layer, 14 GEMV, 1 GDN e 3 kernel del percorso stage/head: tutti con stream esplicito (PASS). Ricompilato e linkato nuovamente l'intero test. N=2 PASS con `max_abs=0`, argmax identici, 2806,963 ms; N=4 PASS con `max_abs=0`, argmax identici, 4122,388 ms. Log `run_final_n2.log` e `run_final_n4.log` nello stesso scratch. SHA256 sorgente `pp2_fwd.cu`: `a3861bc2c5bbb6585a3f44454ef1e5ad94c45c317a43674ce51e29188bb8b711`; SHA256 eseguibile retest: `4d265ecc878da39d07d35421458872b6819719cb624513e6e569a79ff42460aa`. VRAM tornata esattamente a 486/1289 MiB dopo entrambi i retest. Non sono stati eseguiti altri test GPU dopo il rilascio esclusivo.

## Riproduzione e falsificabilità

Con binario `test_async_multiuser` costruito dal medesimo codice e modello reale:

```bash
HERMES_NVFP4_GEMV=warp4 HERMES_HEAD_BF16=warp4 timeout 120s \
  ./test_async_multiuser --n=1
HERMES_NVFP4_GEMV=warp4 HERMES_HEAD_BF16=warp4 timeout 120s \
  ./test_async_multiuser --n=2
HERMES_NVFP4_GEMV=warp4 HERMES_HEAD_BF16=warp4 timeout 120s \
  ./test_async_multiuser --n=4
```

Il test disabilita internamente il flag async per creare il riferimento e lo abilita per le richieste concorrenti; una divergenza produce exit non-zero e **non** viene etichettata PASS. Ulteriori test obbligatori prima dell'uso con utenti reali: richieste asincrone con contesti lunghi (128/129/1024/8192), variazioni di carico e cancellazione nel serving, N=2/4 via HTTP, throughput ripetuto con checkpoint invariato, simultaneità con prefill, gestione di driver reset e OOM. Non dedurre la capacità di throughput stabile dai soli tempi di correttezza.

## Rollback e integrazione

Rimuovere `HERMES_PP2_ASYNC_STREAMS` dall'ambiente di avvio e riavviare il processo: il vecchio decode rimane il default. Il flag è indipendente da Warp4/head. Il runtime async non cambia API pubbliche di `decode_step`; `pp2_batch::batch_step_parallel` e il serving esistenti possono utilizzarlo con richieste separate. Il coordinatore ha ownership di `CMakeLists.txt`, serving e integrazione del test nel build: questo task non modifica tali file.
