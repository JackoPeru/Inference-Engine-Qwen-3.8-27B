# HERMES-27 — gate di qualita per prefill NVFP4 W4A4

**Stato al 2026-11-13: NO-GO per la produzione.** Il prototipo CUTLASS
ha verificato soltanto una proiezione del blocco 0. Su pesi reali GGUF,
la quantizzazione A4 cambia l'output rispetto al GEMV BF16: RMSE relativa
4.33%/4.52%/4.57% a T=16/64/256. Nessun logit finale o token generato
e stato confrontato dopo l'applicazione di W4A4 a tutto il modello.
Il guadagno del microkernel **non e un guadagno end-to-end**.

## Decisione minima

1. **Prima verifica il riferimento:** stesso SHA-256 del GGUF, stessa
   sequenza esplicita di token ID, stesso trattamento di BOS/EOS/posizioni,
   stessa modalita greedy (temperatura 0, nessuna penalty), MTP off.
   Conferma che PP2 GEMV originale corrisponde a llama.cpp almeno nella
   generazione greedy; dove disponibile confronta anche i logit completi.
   Una divergenza GEMV/llama.cpp blocca l'attribuzione delle differenze a W4A4.
2. **Poi confronta W4A4 e GEMV sul modello intero.** Non usare l'output
   della singola proiezione, top-k parziali, vecchi golden E1/E2 o solo
   l'argmax come sostituti dei logit finali. Salva tutti i 248320 logit
   FP32 per ogni posizione valutata, da entrambe le pipeline indipendenti.
3. **Blocca il rilascio** se fallisce una qualsiasi soglia di qualita,
   se si perde lo stato GDN/KV, se la memoria disponibile non basta
   oppure se il prefill end-to-end non migliora in modo misurabile.
   L'eventuale PASS del comparatore riguarda **solo la qualita**.

## Suite di valutazione (minimo obbligatorio)

- **Almeno 32 prompt distinti e 512 posizioni totali** con logit interi.
  Includi lunghezze T=1,4,16,32,64, e gruppi italiano, inglese,
  chat-template, codice, matematica/numeri. Includi BOS/EOS e stringhe
  Unicode nella fase di costruzione dei token ID; in fase di confronto
  usa **solo gli ID numerici** gia fissati e registrati.
- Registra i logit di **ogni posizione** 0..T-1, non soltanto dell'ultimo
  token. Registra almeno **16 token di continuazione greedy** per prompt
  (oppure fino al vero EOS), con stesso stop reason. La continuazione
  DEVE partire dallo **stato persistente prodotto dal prefill provato**,
  non da un nuovo seed token-by-token che aggira W4A4. Usa una decode-state
  nuova per ogni prompt. Non modificare il token di riferimento dopo una
  divergenza per nascondere la deriva autoregressiva.
- Esegui un test separato sulle transizioni di contesto e sulle cache:
  chunk 1+15, 16+16, 32+32, 64+1, continuita delle posizioni RoPE,
  equivalenza prefill/incrementale e isolamento di richieste concorrenti
  (1, 2, 4, 8 richieste). La via W4A4 e eleggibile solo dove supportata;
  T>64, incluso T=256, deve restare sul fallback validato finche PP2
  non dispone di un prefill causale chunked verificato.
- Esegui ciascuna configurazione con il **medesimo GGUF**, mantenendo
  identiche tutte le altre opzioni, il layer placement, la LM head,
  i kernel di attenzione e il formato dei logit. Per isolare il solo
  effetto A4, cambia **una variabile alla volta** (GEMV BF16 input vs
  W4A4 A4). Disabilita MTP, speculative sampling e test concorrenti
  non legati alla suite durante il confronto di correttezza.

### Soglie quantitative preliminari di qualita

Le soglie seguenti sono **criteri di ingresso conservativi definiti prima
del test**, non errori gia misurati a livello del modello. Il comparatore
CPU le applica alla coppia full-model GEMV/W4A4:

| Metrica sui 248320 logit FP32 | Condizione |
|---|---|
| Finitezza e dimensioni | 0 NaN/Inf, nessuna riga o posizione mancante |
| Argmax per posizione | 100% identico, inclusi casi di quasi parita |
| Top-5 come insieme | almeno 99% delle posizioni identico |
| Errore massimo assoluto globale | <= 0.10 logit |
| RMSE assoluta globale | <= 0.02 logit |
| RMSE relativa all'RMS del baseline | <= 1% |
| KL(softmax GEMV || softmax W4A4) | media <= 0.001 nat, massimo <= 0.005 nat |
| Sequenze greedy fino a 16 token/EOS | 100% identiche, stesso punto di arresto |

**Interpretazione:** un FAIL e NO-GO; dati incompleti o un comparatore
non eseguito sono **INCONCLUSIVE/NO-GO operativo**, mai PASS. Le
soglie non dimostrano equivalenza matematica o statistica universale:
servono come limite di regressione per questa implementazione.
Un PASS non sostituisce il riferimento indipendente, i test di stato,
la verifica della memoria o le prove di prestazioni.

## Contratto dei dump CPU verificabili

Due cartelle indipendenti, una GEMV e una W4A4, contengono un
manifest JSON e per ogni prompt un file binario little-endian
float32 senza header, **row-major [T,248320]**. La posizione
del logit e l'indice della riga coincidono. Esempio di struttura
del manifest (in produzione servono almeno 32 casi):

```json
{
  "schema": "hermes.prefill-quality.v1",
  "gguf_sha256": "<64 caratteri hex ottenuti dal GGUF effettivo>",
  "backend": "hermes-pp2",
  "variant": "gemv-raw",
  "kernel_mode": "gguf-gemv-bf16",
  "seed_mode": "full_prefill",
  "runner_sha256": "<64 caratteri hex del binario realmente usato>",
  "dtype": "float32_le",
  "vocab_size": 248320,
  "cases": [
    {
      "id": "it-T4-01",
      "group": "it",
      "prompt_ids": [6205, 161421, 303, 220],
      "positions": [0, 1, 2, 3],
      "logits_file": "it-T4-01.f32le",
      "continuation_limit": 16,
      "continuation_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
      "stop_reason": "length"
    }
  ]
}
```

Gli ID di continuazione nell'esempio sono **segnaposto**, non golden.
Il dump reale deve contenere gli ID generati dall'esecuzione.
Per W4A4 cambia variant in w4a4-gpu-quant, kernel_mode in
cutlass-w4a4-gpuquant, e usa i logit e gli output realmente prodotti.
Registra l'SHA-256 del runner realmente usato, anche se il binario
e lo stesso con flag diversi. Non riutilizzare i logit del baseline.
Conserva anche il log di dispatch che prova che CUTLASS W4A4 sia
stato realmente eseguito; il manifest da solo non lo dimostra.

Il comparatore standard-library Python:

- Verifica SHA-256 del file GGUF locale e coerenza tra manifest.
- Rifiuta file troncati, top-k anziche full-vocab, posizioni saltate,
  prompt duplicati, token non validi, input non finiti e manifest che
  dichiarano un seed incrementale anziche full_prefill.
- Riporta le metriche aggregate e il primo flip di argmax/continuazione.
- Ritorna 0 solo per **quality-only PASS**, 1 per regressione reale,
  2 per prova mancante/inconclusiva.

### Comandi

Controllo CPU subito disponibile, **senza GPU**:

```bash
python3 kernels/tests/prefill_quality_compare.py --self-test
```

Quando esisteranno i due dump full-model ottenuti con accesso GPU
esclusivo e la pipeline sperimentale sara realmente integrata:

```bash
model=models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf
sha256sum "$model"
python3 kernels/tests/prefill_quality_compare.py \
  --baseline /tmp/prefill_quality/gemv/manifest.json \
  --candidate /tmp/prefill_quality/w4a4/manifest.json \
  --gguf "$model" \
  --json /tmp/prefill_quality/quality_result.json
```

Una suite ridotta per diagnosi puo usare --diagnostic, ma il processo
restituisce comunque exit 2: non puo autorizzare il rilascio.

**Comandi esistenti, da eseguire solo in una sessione GPU autorizzata:**

```bash
# Controllo PP2 GEMV incrementale contro prefill (NON attiva W4A4).
./build/hermes_generate --model "$model" \
  --prompt-ids 6205,161421,303,220 --verify-prefill --max-new-tokens 16

# Confronto indipendente di token: richiede llama-server GIA avviato
# con lo stesso GGUF, sullo stesso host. Non avvia alcun servizio.
python3 tests/compare_llama.py --gguf "$model" \
  --engine ./build/hermes_generate \
  --llama-url http://127.0.0.1:8080 --tokens 16
```

**Blocco strumentale attuale:** hermes_generate non esporta i logit
completi della variante W4A4 e non abilita W4A4. Il prototipo
bench_nvfp4_prefill_prototype.cu calcola una sola proiezione.
Anche il seed per la generazione in hermes_generate e token-by-token:
non puo fornire una continuazione che testimoni il prefill W4A4.
Pertanto **i dump richiesti non esistono ancora**: non fingere
che uno dei comandi sopra produca il manifest. Quando il prime
integrera in futuro il percorso sperimentale, ogni runner dovra
emettere dump e manifest indipendenti, con provenienza verificabile.
Per una verifica esterna su tutti i logit occorre anche una vera
esportazione dei full-logits in llama.cpp; il confronto via
tests/compare_llama.py attualmente copre i token greedy di quattro
prompt, non tutti i logit o tutta la suite.

## Gate di VRAM e throughput, separato dalla qualita

Il peso blk.0.attn_qkv pesa **29,491,200 byte** sia raw-GGUF che
ripacchettato in Bq+SFB. Tenerli entrambi in VRAM costa **+29.49 MB
per matrice**, oltre ad Aq/SFA, mappa, output e workspace.
La duplicazione estesa all'intero modello non e un piano praticabile
su 2 x 16 GB. Misurare memoria peak del processo per GPU con pesi
residenti, 1/4/8 richieste, T=16/64, e mantenere almeno **1 GiB
libero per GPU** dopo il massimo carico previsto (budget provvisorio);
non e consentito OOM o eviction imprevedibile. Richiedere un piano
concreto di caching, repack a caricamento o fallback che non perda
il formato richiesto dal decode.

Solo dopo il quality gate, registrare almeno 30 ripetizioni per caso
con warmup, mediana e p95 per prefill end-to-end, first-token latency,
throughput aggregato multiutente e decode p95. Conteggiare quantizzazione,
repack ammortizzato, copie, GDN, attenzione, head e transito GPU0/GPU1.
**Criterio proposto:** >=20% migliore nella mediana di prefill T16/64,
nessuna regressione di p95 oltre 5% e nessuna diminuzione di
throughput aggregato/decode per 1/4/8 utenti. Nessun numero di
speedup del microbenchmark puo sostituire tali prove.

## Strade di riduzione dell'errore, da verificare separatamente

1. **Ibrido selettivo per layer/operatore:** conservare BF16 GEMV
   per i layer sensibili e usare W4A4 solo dove l'ablazione sul modello
   completo lascia margine alle soglie. Evita di quantizzare
   indiscriminatamente ogni attivazione; la decisione deve dipendere
   dai logit finali, non solo dalla MSE locale.
2. **Attivazioni FP8 oppure BF16, pesi NVFP4:** valutare un kernel
   W4A8 / W4A16 supportato realmente dalla toolchain SM120.
   Non assumere che CUTLASS 79a W4A4 accetti layout misti
   o offra lo stesso throughput; richiedono un kernel e prove proprie.
3. **Passaggio residuale:** dopo GEMM A4/W4, calcolare una
   correzione W4 * (X_BF16 - dequant(A4)) in precisione piu alta
   e accumularla in FP32. Il residuale elimina idealmente
   l'errore di quantizzazione di X, non quello dei pesi, ma richiede
   un altro percorso di prodotto: puo annullare ogni vantaggio.
4. **Outlier-aware per gruppi da 16:** identificare le attivazioni
   con saturazione/errore grande, mantenere un percorso BF16
   per quelle porzioni e misurare il costo di rilevamento/scatter.

**GO limitato:** solo dopo il superamento di tutti i gate, abilitare
W4A4 come opt-in per prefill T<=64, con fallback raw-GGUF/GEMV verificato
e rollback immediato. **NO-GO attuale:** nessuna prova di logit parity
end-to-end, nessuna soluzione residente per pesi doppi e nessuna
misura dell'intera pipeline.
