# HERMES-27 — gate integrato CPU/CUDA del 14 novembre 2026

**Decisione: NO-GO per release, deployment, inferenza reale e benchmark GPU.**
Questo rapporto descrive prove effettivamente eseguite, senza confondere test CPU/mock, compilazione CUDA e correttezza numerica del modello.

## Provenienza dell'integrazione

Checkout Git Ubuntu: `/home/matteo/hermes-27-gh/hermes-27`. La sessione è partita da `79e61625c6802983bcd4019f45c4a64fa1b94c5d` (HEAD=origin/main e status pulito). I contributi congelati sono stati copiati dal workspace Windows in archivi separati, verificati per SHA-256 prima dell'overlay; nessun file sotto ownership attiva è stato sovrascritto.

Commit autonomi pubblicati in sequenza su `main`, con corrispondenza verificata tra HEAD locale e remoto dopo ogni push:

| Commit | Intervento |
| --- | --- |
| `e5355f4d1d71de795100828391774de0c01a7c22` | Verificatore tracer v2 e mapping cross-engine fail-closed |
| `db063f14fd2d4d19ffa080f0001ef20d4c3767f4` | Metriche host scheduler ready opt-in |
| `4fc52608023e088c54f24cdc4a8c276e41710061` | Preflight private-stream per sliding KV |
| `79b8be6bd572861d98691fdbdedf99b8be5b10ef` | Audit read-only delle lacune ingress e ownership |
| `2b196cb85c06f22da3c49ac061978e6aeed240ab` | Notifica emergency OOM senza secondo ID-copy |
| `4968d6a62ab486dc80367e3ad5f86fba6054d91e` | Commit atomico del finale SSE e fallback esplicito |

I due vecchi sospetti di data race nella capability tokenizer e perdita di stato durante `finish()` erano già mitigati nello snapshot iniziale: non vengono presentati come nuove correzioni. Le nuove patch chiudono due gap specifici e verificati: unsafe sliding via flag mutabile senza `async_ctx`, e recupero terminale/SSE dopo allocation failure.

## Prove CPU e build effettive

| Gate | Esito | Limite |
| --- | --- | --- |
| Verificatore layer0 e shape/identity cross-engine | **14/14 PASS** su Ubuntu | Nessuna cattura HERMES v2 fisica; niente oracolo llama downstream due-token |
| Tracer patcher contro PP2 fwd aggiornato | **6/6 PASS**, patch scratch generata SHA `fafb82e1bd1d40127d174b353609a21fef96bdafac58b31daacb4a15b69ef73a` | Non eseguito contro il GGUF su GPU |
| KV ring sorgente/deque/softmax negativi | **16/16 PASS** | Non equivale a CUDA T8192+ numericamente validata |
| Scheduler ready-completion Linux GCC ASAN/UBSAN | **151/151 PASS** | Fast B/slow A è witness host simulato |
| Scheduler ready fault hardening Linux GCC ASAN/UBSAN | **21/21 PASS** | Fault injection CPU, non GPU |
| Metriche opt-in Linux GCC ASAN/UBSAN | **28/28 PASS** | Clock artificiale, nessuna accelerazione GPU misurata |
| Nuova notifica OOM terminale Linux GCC ASAN/UBSAN | **21/21 PASS** | Non dimostra recovery in OOM globale reale |
| HTTP CPU mock integrato `prod_sched_cpu_gate.sh` | **90/90 PASS** | Backend PP2 simulato, loopback isolato |
| Nuovi test dinamici errori terminali HTTP/SSE | **7/7 PASS** | Server C++ + PP2 mock, non inferenza reale |
| CUDA12.8 full build/link | **PASS** `hermes_generate` e `hermes_server` | Nessun kernel GPU eseguito |

La suite HTTP copre legacy, continuous, RAW/chunked prefill, ready-completion, cancellation, sliding KV mock, Unicode, autenticazione e limiti SSE. I sette test nuovi verificano assenza di falsa `finish_reason`/usage in caso di errore terminale, una sola `[DONE]`, e recupero della richiesta seguente. È una prova del contratto HTTP su CPU, non delle prestazioni del 27B.

Compilazione pulita da sorgenti coerenti con CMake e nvcc **CUDA 12.8.93**, `HERMES_REQUIRE_SERVER=ON`, `HERMES_REGISTER_GPU_TESTS=OFF`, `-j2`. Build scratch: `/tmp/h27-resume1-cuda128-post-sse-final`.

- `hermes_generate`: SHA256 `4b614312fa88f0632edf0a78167209a872adbe1923258c04380709f18c18fcb5`
- `hermes_server`: SHA256 `a8a15980ca400f436c209a74103e4d5a752637592c99604a7a75cf70136187c4`

La compilation ha prodotto warning preesistenti per variabili unused, ma target e CUDA device link sono terminati correttamente. Nessun benchmark CUDA/inferenza o modifiche al processo live.

## Gate fisico e numerico ancora negativo

Audit read-only delle 10:36:33 UTC: GPU0/GPU1 occupavano 21/26 MiB, 0% util, nessun compute app restituito da `nvidia-smi`. `hermes-tabby` e `hermes-gpu-manager` erano inattivi; erano però presenti listener su loopback, tutte le interfacce IPv4 e Tailscale. Nessuna prova autenticata di all-ingress drain, richieste SSE inflight zero, PID/starttime owner delle due GPU, lease anti-rollback e recovery/rollback fisici. Rapporto separato: `reports/PHYSICAL_INGRESS_GPU_OWNER_READONLY_PRIME_20261114.md`.

La sola cattura autentica llama layer0 esistente copre token 6205, posizione 0 e fino a `conv_output_silu`. Non esiste una coppia fisica HERMES v2 + llama.cpp dei token consecutivi `[6205,161421]` alle posizioni `[0,1]` che autentichi beta/GDN/attention/residual/FFN; Q/K richiede ancora prova di assi, ripetizione teste e scaling. Il confronto indipendente full-vocabulary, **248320 logits**, resta **FAIL 12/12**: nessuna correzione numerica reale full-model è stata dimostrata.

Sliding KV 8192 con finestra cronologica e RoPE assoluta resta sperimentale: la semantica windowed non coincide con la storia completa; rimangono 32 sincronizzazioni per token post-wrap nei 16 layer full-attention, non eliminate senza prova. MTP K>1 non promuovibile senza verifica target batched, rollback e vantaggio misurato.

## Sequenza per sbloccare il lavoro reale

1. Certificare ingress drain, owner grant per entrambe le GPU, PID/starttime, fencing anti-rollback e rollback sicuro, senza disturbare servizi live.
2. Solo dopo tale grant, compilare/acquisire tracce fisiche HERMES/llama consecutive e confrontare gli stadi da beta/GDN in avanti; correggere il primo errore CUDA effettivo.
3. Ripetere gate indipendente full-vocabulary 12/12 senza allentare soglie.
4. Misurare su checkpoint reale N1/N2/N4/N8, TTFT/TPOT verificabili, throughput aggregato, fairness, p50/p95, VRAM, backpressure e cancellazioni; poi continuation T8192+ contro oracolo sliding-equivalente.
5. Validare soak, crash/boot fence, live rollback e deployment reversibile prima di passare da NO-GO a GO.

**I risultati positivi sono limitati alle prove sopra elencate. Nessun servizio live è stato avviato/arrestato da questa integrazione, nessuna GPU è stata acquisita.**
