# MISSIONE: HERMES-27 — engine di inferenza indipendente, specializzato per Qwen3.8-27B su 2× RTX 5060 Ti

Sei un coding agent senior esperto in CUDA, C++20, Blackwell SM120, inferenza LLM e sistemi concorrenti. Devi progettare, implementare, misurare e ottimizzare un inference engine **realmente indipendente** (non un wrapper né un fork di llama.cpp, ExLlama, NInfer, vLLM, TensorRT-LLM o SGLang) per il server locale descritto sotto. Non limitarti a produrre una relazione: crea codice funzionante e testato, procedendo in milestone misurabili. Riutilizza liberamente librerie matematiche e primitive standard compatibili con le licenze, soprattutto CUDA/CUTLASS/cuBLAS, senza adottare un inference runtime preesistente come motore nascosto. Consulta codice altrui solo come riferimento e rispetta attribuzioni/licenze.

## 0. VINCOLI NON NEGOZIABILI

- **Modello unico:** Qwen3.8-27B, architettura checkpoint `qwen3_5`/`Qwen3_5ForConditionalGeneration`; non sviluppare un engine generale per altri LLM.
- **Modello target primario:** `BennyDaBall/Qwen3.8-Uncensored-NVFP4-MTP`, file `Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf` (~18,34 GiB dichiarati); GGUF con `GGML_TYPE_NVFP4=40`, MTP/NextN incorporato (blocco 64), embedding/output head/MTP in BF16. Vision projector separato, facoltativo e non richiesto per il primo rilascio. Verifica sul file reale e non fidarti ciecamente della scheda del modello.
- **Checkpoint alternativo di confronto:** `johnnyeric/Qwen3.8-27B-Uncensored-NVFP4`, safetensors `compressed-tensors`, mix NVFP4 W4A4 + FP8 W8A8 + BF16, MTP conservato. Non devi implementare subito il secondo loader: isola l'interfaccia per poterlo fare in seguito, senza compromettere i tempi della prima versione.
- **Hardware:** Ubuntu 24.04 headless; AMD Ryzen 5 7500F; DDR5 32 GiB; **2× RTX 5060 Ti 16 GiB**, Blackwell **SM120**, PCIe (nessun NVLink). Potrebbero esserci driver CUDA/PyTorch specifici già installati: rilevali prima di fare modifiche; non assumere che siano aggiornati.
- **Tecnologie:** core C++20, kernel CUDA C++ target SM120 e CUTLASS/CuTe per GEMM FP4; CMake/Ninja; Python solo per test, benchmark, conversioni offline, eventuale piccolo piano di controllo. Il decode caldo e lo scheduler devono restare nativi.
- **Ordine di ottimizzazione:** (1) **throughput aggregato e qualità di servizio multiutente**; (2) prefill e TTFT; (3) decode per singola richiesta. Non ottimizzare la singola chat a scapito del multiutente.
- **VRAM:** budget **PER SCHEDA** rigorosamente inferiore a 16 GiB con margine configurabile per runtime/CUDA, grafi, buffer, MTP, KV e stati Gated DeltaNet. I 32 GiB non sono memoria unificata. Vietato considerare la VRAM delle due schede come un singolo pool continuo.
- **Target:** inferenza full GPU TP=2 o PP=2, non CPU offload nel percorso critico, salvo fallback esplicitamente misurato. Non inventare velocità o promettere 150–200 tok/s senza misure replicabili.
- **Indipendenza e affidabilità:** non sostituire TabbyAPI/ComfyUI né alterare configurazioni di sistema o del GPU Manager senza rollback e autorizzazione esplicita. Lavora in una directory/repository isolati. Usa script non distruttivi. Non scaricare modelli duplicati se un checkpoint valido è già presente.

## 1. FASE ZERO — AUDIT DEL SERVER E BASELINE

Prima di progettare i kernel, ispeziona e documenta con output reali: GPU/VRAM/driver/PCIe topology (`nvidia-smi -L`, `nvidia-smi -q`, `nvidia-smi topo -m`, `lspci -tv`), compilatore/CUDA toolkit/compute capability, RAM e spazio libero, istanze e processi GPU esistenti, stato GPU Manager, modalità di rilascio delle GPU, versione ExLlamaV3/TabbyAPI attualmente in servizio. Verifica `cudaDeviceCanAccessPeer`, bandwith GPU↔GPU e GPU↔host, duplex, eventuale ACS/IOMMU rilevante. **Non abilitare P2P tramite hack di driver**.

Misura baseline riproducibile dell'attuale Qwen3.8-27B EXL3 3.07 bpw con le impostazioni effettive dell'utente; se possibile misura anche un runtime NVFP4 di riferimento separato (llama.cpp per GGUF, NInfer TP2 su 5060 Ti per comparazione, senza riutilizzarne il motore nella soluzione finale). Registra revisione dei software, prompt reali e sintetici, seed, warm-up, token effettivi, potenza, utilizzo GPU, VRAM per device e latenza. Salva `benchmarks/baseline.json` e un rapporto sintetico con i limiti/condizioni di confronto. Non cambiare checkpoint senza dichiararlo.

**Esperimenti minimi:** concorrenza 1/2/4/8/16 (non forzare OOM); prompt 1K/8K/32K/64K, dove il budget VRAM lo consente; output 128/512/1024; modalità reasoning on/off; campioni di codice, prosa e prefissi comuni. Misura throughput generato aggregato (tok/s totali), input prefill tok/s, TTFT p50/p95, TPOT p50/p95, tempo end-to-end, fairness/tail latency, cache hit rate, memoria e GPU utilization. Tratta i casi OOM come informazioni, non come successi.

## 2. ARCHITETTURA RICHIESTA

Struttura il progetto in moduli con API dichiarate e test unitari:

- `model/`: GGUF parser stretto per il formato reale (header, metadata, tokenizer/chat template, tensor offsets/layout, alignment, GGML type 40 NVFP4, BF16/F32, checksum/size validations); loader con mapping offline e upload distribuito GPU. Il loader deve rifiutare input incompatibili in modo esplicito. Non dequantizzare stabilmente tutti i pesi a BF16. Isola il backend dei pesi mediante un'interfaccia minimale per eventuale supporto futuro a safetensors mixed-precision.
- `kernels/`: embedding, RMSNorm, RoPE, NVFP4 W4A4 GEMM/GEMV, attivazioni e scaling, MLP/gated FFN, output head, attenzione causale, quantizzazione delle attivazioni e kernel FP8/KV. Sfrutta CUTLASS per **SM120**, non percorsi destinati unicamente a B100/B200 (SM100 e `tcgen05.mma` non sono intercambiabili con GeForce SM120). Prevedi kernel diversi per prefill con grande M e decode con piccolo M, e microbenchmark separati.
- `architecture/`: implementa **l'effettiva architettura Qwen3.8-27B**, non un Transformer fittizio. Verifica nel checkpoint numero/dimensione layer, pattern 48 layer Gated DeltaNet + 16 full attention (64 totali), hidden/FFN, tokenizer, normalization, gating, conv1d e stato ricorrente, possibilità vision e MTP. I parametri indicativi della scheda sono hidden 5120, FFN 17408, vocab 248320, context nativo 262144; prevale sempre il `config` verificato sul checkpoint. Implementa prefill chunked e decode matematicamente coerenti includendo **stato ricorrente e stato convolutivo per richiesta**. Verifica i logit contro un runtime noto sullo stesso identico checkpoint.
- `distributed/`: backend **TP2** e **PP2**, selezionabili a runtime; su TP partiziona pesi e comunicazioni per layer minimizzando all-reduce e traffico PCIe; su PP partiziona layer e pipeline microbatch. Inizia dal metodo più semplice che produce output corretto, ma implementa entrambi e confrontali sul throughput multiutente. Se P2P non è disponibile, usa un percorso host-pinned corretto e misurato. Non presumere velocità teoriche aggregate come bandwidth utilizzabile.
- `memory/`: memory pool e arena GPU per device; KV cache paginata per attenzione completa con FP8 o INT8 verificati; stati Gated DeltaNet gestiti separatamente; request lifecycle e reset/cancel; prefissi riutilizzabili tramite chiavi robuste e copy-on-write. Prefix caching deve riutilizzare **anche gli stati ricorrenti appropriati**, altrimenti disattivarlo per i casi non corretti. Definisci budget di memoria e backpressure prima di allocare, non affidarti all'OOM.
- `scheduler/`: **continuous batching**, separazione e interleaving prefill/decode, chunked prefill, admit/queue/drop con policy configurabili, limite di token attivi, fairness per richiesta, isolamento tra utenti, priorità opzionali e cancellazione immediata. Misura l'utilità di CUDA Graphs per batch shape comuni senza esaurire memoria. Scegli adattivamente batch size, numero di prefill tokens e strategia TP/PP solo se i benchmark giustificano la scelta; niente euristiche non validate.
- `speculation/`: per prima cosa decode **senza speculazione** corretto. Poi aggiungi MTP con il vero draft head incorporato nel GGUF: è una singola testa/layer riutilizzabile autoregressivamente, non tre teste indipendenti. Implementa proposal, verifica in blocco sul modello target, acceptance/rejection e rollback corretto di KV/cache/stato ricorrente. Per greedy, verifica identità dei token rispetto al percorso non speculativo; per sampling, preserva la distribuzione del target con corretto acceptance/rejection, senza affermare identità di sequenze in modalità stocastica. Profilare K=1/2/3/4/5 e abilitazione automatica solo dove aumenta il throughput (specialmente batch multiutente). Interfaccia futura per DFlash2 esterno, **non necessario al primo rilascio**; un draft esterno non è automaticamente compatibile o vantaggioso perché il modello è abliterato.
- `serving/`: API OpenAI-compatible minima (`/v1/models`, `/v1/chat/completions` streaming SSE e non-streaming, `/health`, `/metrics`) con template Qwen, streaming corretto, reasoning/thinking toggle, tool-calling e structured-output di base senza inventare sintassi. Validazione JSON, auth opzionale ma obbligatoria su binding di rete non-locale, rate limiting e limiti per utente. Evita server Python nel percorso caldo; lib HTTP nativa semplice.
- `ops/`: avvio/arresto ordinati, logging strutturato, GPU accounting per scheda, profiler NVTX, metriche Prometheus, script benchmark e meccanismo di handoff a GPU Manager senza interrompere altri servizi impropriamente.

## 3. PRECISIONE E CORRETTEZZA NUMERICA

Implementa NVFP4 autentico: FP4 E2M1, scale di blocco e scale globali secondo il **layout reale GGUF type 40**; controlla se il percorso GPU esegue **vere operazioni sui Tensor Core FP4**. Memorizzare FP4 nei pesi non basta: se espandi tutti i tensori a BF16 prima di ogni matmul, stai perdendo il vantaggio del formato. Gestisci attivazioni W4A4, allineamenti, padding, layout e scaling; fallback FP16/BF16 giustificato e misurato solo per operatori residui.

Test obbligatori: parser e tensori sintentici; kernel GEMM con confronto su reference FP32/BF16 e soglie esplicite; forward completo contro runtime esistente con lo **stesso checkpoint** (statistiche logit, token e qualità su prompt multipli); Gated DeltaNet full-prefill vs chunked-prefill vs token-by-token; isolamento richieste concorrenti e prefix cache; detokenizzazione Unicode; tool calls, EOS/stop, error handling; sampling temperature/top-k/top-p; MTP rollback e stati. Nessun benchmark prestazionale senza prima aver superato i test di correttezza per la stessa configurazione.

## 4. STRATEGIA DI OTTIMIZZAZIONE

Ottimizza in quest'ordine:

1. **Throughput multiutente:** continuous batching efficace, buona occupancy dei Tensor Core, gestione efficace di 2 GPU, riduzione dei lanci kernel e della comunicazione PCIe, KV/state memory efficienti e fairness p95. Analizza casi con contesti eterogenei e una richiesta lunga che arriva mentre altre generano.
2. **Prefill:** kernel W4A4/GEMM e Gated DeltaNet chunked, overlap compute/transfer, chunk-size tuning, prefix caching per prompt comuni, mitigazione head-of-line blocking.
3. **Decode:** GEMV/skinny GEMM, CUDA Graphs dove utili, fused ops, MTP con K adattivo, eventuale DFlash2 solo dopo una baseline MTP completa.

Profilazione: Nsight Systems/Compute quando disponibili; registra kernel latency, bandwidth effettiva, occupazione SM, tempo nelle comunicazioni, lanci per token e picchi VRAM. Ogni modifica ottimizzante deve avere benchmark A/B riproducibile e, se peggiora il carico prioritario multiutente, non entra come default. Non fare ottimizzazioni speculative senza profiler.

## 5. MILESTONE ED EVIDENZE DI COMPLETAMENTO

**M0 — Audit & baseline:** documento hardware/topologia + baseline misurata + rischi bloccanti + tabella memoria per GPU.
**M1 — Loader e CPU/reference validation:** parser GGUF, identificazione precisa dei tensori/MTP, checksum, test sintetici e design document; nessun kernel ancora spacciato per inferenza completa.
**M2 — Single-GPU math + TP2 inferenza corretta:** kernel essenziali, layer Qwen corretti, modello distribuito in 2×16 GiB, output affidabile non speculativo, confronto logits con reference. Non dichiarare M2 conclusa se manca Gated DeltaNet.
**M3 — Server multiutente:** continuous batching, chunked prefill, KV/state budget, API streaming, cancel, backpressure, metriche; benchmark concorrenza 1/2/4/8/16.
**M4 — PP2 e tuning:** comparazione TP2 vs PP2 su 1/2/4/8/16 e scelta del default con grafici; CUDA Graph, prefix caching e tuning SM120.
**M5 — Speculative decoding:** MTP 1–5, acceptance e corretta verifica/rollback; speedup netto per concurrency, K e classe prompt; DFlash2 solo se conviene e c'è un draft compatibile verificato.
**M6 — Integrazione stabile:** integrazione controllata GPU Manager, documentazione configurazione, recupero dopo OOM/crash, gestione di ComfyUI, strumenti per migrare la connessione di Hermes Hub/TabbyAPI, rollback al sistema precedente.

Dopo **ogni milestone**, produci `reports/Mx.md` con: commit hash, file cambiati, test pass/fail, ambiente/CUDA/revisioni, benchmark/raw JSON, grafici essenziali, problemi ancora aperti, decisioni motivate e piano per la fase seguente. È vietato segnare una fase completata sulla base di mock, test che non eseguono il vero modello o dati inventati. Se la realizzazione completa richiede più iterazioni, implementa il prossimo increment reale e documenta quanto manca, senza trasformare il lavoro in un semplice piano.

## 6. CRITERI DI SUCCESSO E GESTIONE DEL RISCHIO

- Il modello corretto è caricato **interamente sulle due GPU** e il sistema opera senza OOM con riserva VRAM per ciascuna scheda; i 262K nativi sono un obiettivo da provare, **non** la dimensione iniziale da preallocare per tutte le sessioni.
- Con 4/8 richieste, l'engine deve produrre throughput aggregato competitivo rispetto alla baseline, con TTFT e TPOT p95 riportati, non solo tok/s di una richiesta breve.
- Il modello mantiene il comportamento atteso di un checkpoint abliterato, il template di chat e il supporto per tool calling; non fare nuove modifiche ai pesi.
- MTP attivo solo quando i test mostrano vantaggi netti e non altera la distribuzione del target.
- Test di regressione, benchmark e comandi di riproduzione devono essere disponibili nel repository.
- Non dichiarare 150–200 tok/s come risultato se non verificato sul vero hardware. Tratta i report delle RTX 5090 solo come riferimenti esterni e non come stime.
- Prevedi licenze/attribuzioni di CUTLASS/altre dipendenze e avvertenze sul modello abliterato: se l'endpoint è esposto ad altri utenti, considera controlli e protezione da abuso separati.

## 7. PRIMO CICLO OPERATIVO — AGISCI ORA

1. Rileva la configurazione reale, verifica lo stato dei processi GPU e crea repository isolato, senza alterare lo stack esistente.
2. Controlla il checkpoint già presente, spazio disponibile e accessibilità su Hugging Face; verifica file, commit/revision, dimensioni e MTP. Non scaricare 20 GiB se il file identico è già disponibile; non presupporre l'accesso a modelli gated.
3. Disegna un diagramma sintetico dei moduli, il budget VRAM per GPU e un piano concreto M0→M2 che permetta una prima generazione corretta rapidamente.
4. Esegui subito M0 e inizia M1: produci codice, test e risultati ripetibili nello stesso ciclo di lavoro. Quando qualcosa blocca, proponi due alternative motivate e continua con il sotto-lavoro indipendente.

### Riferimenti tecnici da consultare prima di implementare
- Model card + files del modello target su Hugging Face: `BennyDaBall/Qwen3.8-Uncensored-NVFP4-MTP`.
- Modello alternativo W4A4 mixed e MTP: `johnnyeric/Qwen3.8-27B-Uncensored-NVFP4`.
- NVIDIA CUTLASS `examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu` per SM120.
- Specifiche GGUF/GGML `GGML_TYPE_NVFP4 = 40`; usa come riferimento il formato, non llama.cpp come runtime.
- `lynx-gt/ninfer-tp2-5060ti` per baseline e analisi della topologia 2×5060 Ti, non per costruire un fork.
- Documentazione vLLM MTP e speculators per meccanismo proposal/verification e criteri di correttezza.

**Regola finale:** parti da misure e correttezza, consegna un sistema indipendente. Ogni nuova funzionalità deve avere un test reale, un confronto quantitativo e una motivazione rispetto alla priorità multiutente > prefill > decode.
