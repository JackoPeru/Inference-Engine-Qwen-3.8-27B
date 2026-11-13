# HERMES-27: deployment sul server, gate e rollback

Data verifica: **13 novembre 2026**. Stato: **NO-GO per sostituzione di TabbyAPI
in produzione**. È disponibile un percorso di manutenzione esplicito con
supervisione e rollback; non è stato eseguito alcun cutover reale.

## Inventario verificato in sola lettura

| Elemento | Riscontro diretto sul server `hermes` |
| --- | --- |
| GPU | 2 × NVIDIA GeForce RTX 5060 Ti, 16.311 MiB ciascuna (`nvidia-smi`) |
| Manager | `hermes-gpu-manager.service` **active**, avvia `/opt/hermes/gpu-manager/manager.py` |
| Tabby | `hermes-tabby.service` **active**, avvia TabbyAPI/ExLlamaV3 da `/opt/hermes/runtimes/tabbyapi` |
| Endpoint Tabby | `127.0.0.1:8000`, esistente; resta la destinazione di rollback |
| Manager HTTP | ascolta su `0.0.0.0:8643`; endpoint reale `GET /status` protetto da Bearer app/manager key |
| GGUF reale | `/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf`, **19.694.360.544 byte** |
| SHA256 GGUF | `db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1` |
| SHA256 tokenizer | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` (`/opt/hermes/models/Qwen3.8-27B-Uncensored-EXL3-3.07bpw/tokenizer.json`) |
| Binario GGUF | `~/hermes-27/build/gguf_dump` presente |
| Binario serving | **`~/hermes-27/build/hermes_server` assente** al momento dell'audit |

### Nuovo gate versioni distribuite (13 novembre 2026)

Il worker di quality assurance ha individuato un secondo blocco, riprodotto
in sola lettura sul server: il `gguf_dump` presente in `~/hermes-27/build/`
risponde alla richiesta `--chat-template` con il vecchio output descrittivo
(`version=3 n_tensors=866 ...`, `KV-S tokenizer.chat_template = ...`)
anziché il **template Jinja puro** atteso da
`serving/tokenizer_bridge.py`. Il suo exit code è 0, quindi controllare
soltanto che il processo termini senza errori non rileva il difetto.

Il nuovo gate `check_build_contract()` viene eseguito anche da
`dry-run --skip-hash`, prima di qualunque modifica ai servizi. Richiede
`hermes_server` e `gguf_dump` nella stessa directory assoluta `build_dir`,
senza symlink; verifica SHA256 esatti dei due binari e del bridge in
`build_dir/../serving/tokenizer_bridge.py`; esegue `hermes_server --help`
controllando le opzioni richieste, incluso `--api-key-file`; prova
`gguf_dump MODEL --chat-template` e ne valida la forma Jinja; infine esercita
il **bridge distribuito** su CPU con handshake `ready` ed `encode`.
Il modello non viene caricato su GPU e nessun servizio viene avviato.

Gli hash binari e bridge nella configurazione di esempio sono intenzionalmente
zero: devono essere sostituiti con i digest di un build approvato e verificato,
non con quelli del vecchio binario. La verifica di SHA256 controlla l'identità
dei file distribuiti rispetto agli hash dichiarati; non dimostra da sola che
la build provenga dal sorgente attuale, motivo per cui sono obbligatorie anche
le prove di comportamento.

L'hash del GGUF è stato calcolato sul checkpoint locale tramite `sha256sum`.
Questo prova l'identità delle successive letture rispetto alla copia corrente;
non costituisce una firma verificata del publisher originale.

Nel sorgente reale di GPU Manager, `/status` restituisce `desired_mode`,
`current_state`, `llm_serving`, `llm_online`, `queue_length`, `current_job`,
`worker_alive_s`, `direct_active`, `gaming_active` e altri campi. Le modalità
di controllo esistenti sono `LLM`, `AUTO`, `MEDIA`, `DIRECT`, `GAMING`.
**Non esiste una lease/maintenance mode per HERMES**. `llm_watchdog()` può
richiamare `restore_llm_with_retries()` e `transition_to_llm()`, che esegue
`systemctl restart` di Tabby. Arrestare soltanto Tabby mentre il manager è
attivo causa quindi una possibile contesa VRAM. Le chiamate di stato sono
autenticate; non sono state lette o pubblicate le credenziali.

## Componenti aggiunti

- `ops/production_deploy.py`: programma Python standard library per
  `dry-run`, `preflight`, `run --authorized`, `recover --authorized`, `health`.
- `ops/hermes-27.json.example`: configurazione dichiarativa con hash completi,
  server isolato, app token del manager, hook ingress/drain/restore, limiti.
- `ops/systemd/hermes-27-cutover.service.example`: unità **non installata**, senza
  sezione `[Install]`; `Restart=no`; `ExecStopPost` richiama recovery.
- `ops/tests/test_production_deploy.py`, `scripts/production_smoke.sh` e
  `ops/README.md`: smoke CPU-only e guida per l'operatore.

## Sequenza prevista per la finestra di manutenzione

1. Distribuire il binario HERMES e verificare staticamente file, eseguibili,
   hash GGUF e tokenizer, RAM CPU, 2 GPU, autenticazione `/status`, risposta
   Tabby e disponibilità della porta isolata **18127**. Qualunque errore blocca
   l'attivazione prima di mutare i servizi.
2. Bloccare l'ingresso di **tutti** i client verso Tabby con un hook esterno
   approvato dall'operatore. Un secondo hook deve restituire JSON con
   `{"blocked":true,"inflight":0}`; il supervisore non procede in sua assenza.
   La verifica deve comprendere client locali diretti e connessioni via proxy.
3. Ricontrollare che GPU Manager sia stabile in `LLM_READY`, modalità `AUTO`
   o `LLM`, senza job attivi o in coda. Solo a questo punto sospendere
   `hermes-gpu-manager.service`, poi `hermes-tabby.service` con
   `sudo -n systemctl stop` esplicitamente autorizzato.
4. Controllare che entrambe le GPU abbiano rilasciato la memoria sotto il
   limite configurato; avviare `hermes_server` su `127.0.0.1:18127`, con coda
   e contesto limitati. Non usare `127.0.0.1:8000` per il nuovo server.
5. Il supervisore richiede `/health` pronto e `/v1/models` con ID atteso;
   ripete i controlli ogni 5 s e verifica che i due servizi originari non
   siano stati riavviati. Dopo tre health probe falliti, uscita del processo
   o SIGTERM, arresta HERMES e avvia rollback.
6. Ripristinare Tabby, quindi GPU Manager; verificare risposte HTTP reali e
   lo stato autenticato del manager; **solo dopo** riaprire l'ingresso con
   l'hook di restore. Se una verifica fallisce, conservare il journal e
   mantenere l'ingresso chiuso. `recover --authorized` è idempotente.

La sequenza serializza le modifiche con `flock` e scrive un journal JSON 0600
prima della prima operazione esterna, aggiornato con `fsync` e rename atomico.
Il PID figlio viene confrontato con eseguibile e start time Linux prima di
inviare segnali; se HERMES risulta ancora vivo, il restore **non** riavvia
Tabby sulle medesime GPU. Il log applicativo passa da stdout/stderr al journal
systemd; il figlio scrive in `state_dir/hermes-server.log`.

## Sicurezza, Tailscale e autorizzazioni

Il bind imposto per HERMES è **solo `127.0.0.1`**. La porta 8000 di Tabby
non viene riutilizzata. Per accesso privato attraverso Tailscale è possibile
usare un tunnel SSH controllato dalle ACL di Tailscale, per esempio:

```bash
ssh -N -L 18127:127.0.0.1:18127 matteo@hermes
```

Non pubblicare 18127 tramite Funnel o reverse proxy pubblico senza TLS,
autenticazione e controllo dei client. L'HTTP server corrente accetta
`--api-key` via CLI, esponendo il valore negli argomenti di processo: finché
non è disponibile una modalità segreta non visibile in `ps`, la configurazione
di esempio evita questo parametro e impone solo loopback. Il manager è
attualmente in ascolto su `0.0.0.0:8643`: verificarne le regole firewall/
Tailscale indipendentemente da questo deployment.

Il processo di manutenzione gira come `matteo`; l'operatore deve predisporre
accesso **ristretto** e non interattivo a `systemctl start/stop` per le sole
due unità coinvolte, più un file segreto leggibile dal processo contenente
una app key valida per il `GET /status`. Nessuna credenziale va passata in
argomenti CLI. Gli hook devono essere file eseguibili assoluti, sotto il
controllo dell'operatore, e non sono forniti perché non è stato identificato
un contratto reale di drain sul gateway del sistema attuale.

## Verifiche eseguite e risultato

Prove eseguite su copia isolata in `/tmp/hermes_worker7_20261113` sul server,
senza carichi GPU e senza modifiche systemd:

| Prova | Esito |
| --- | --- |
| `python3 -m py_compile ops/production_deploy.py` | PASS |
| `python3 -m unittest discover -s ops/tests -p 'test_*.py' -v` | **20/20 PASS** (dopo modifica gate versioni distribuite) |
| `bash -n scripts/production_smoke.sh` | PASS |
| Negative cases: hash/model magic, build directory errata, GGUF dump descrittivo anche con hash ripinnato, server help incompatibile, bridge non pronto, bind/URL malformati, manager busy, drain non provato, rollback fallito, PID vivo, mancata autorizzazione | PASS |
| Test dry-run con fixture locale, `--skip-hash` | PASS: hash binari, controllo CLI e bridge CPU; nessuna chiamata a systemd, HTTP o GPU |
| Dry-run con percorsi reali e manifest di esempio | **FAIL previsto**: `missing or nonexecutable: server_path` |
| `shellcheck` | NON ESEGUITO: eseguibile assente sul server |
| Stato servizi dopo i test | `hermes-gpu-manager`: active; `hermes-tabby`: active |
| Cutover con stop/start reali, upload GGUF, throughput e accesso client | **NON ESEGUITI**: richiedono autorizzazione prime e preparazione ingress |

Il test del rollback usa mock dei servizi e degli endpoint: non certifica
effettiva ripartenza del modello sul server. Un crash tra creazione del
processo figlio e persistenza del suo PID, la perdita del disco di journal
oppure un guasto completo dell'host richiedono ancora procedure operative
indipendenti. L'unità systemd di esempio tenta `ExecStopPost` anche dopo
arresti inattesi del suo processo, ma non garantisce il recupero durante un
blackout. Al reboot restano le unità originarie; questa unità non è abilitata.

## Blocchi prima dell'uso reale continuativo

1. Compilare e validare `hermes_server` sul checkpoint reale; oggi il suo
   eseguibile di deployment non è presente.
2. Implementare un contratto di drain e ripristino reale dell'ingress,
   includendo connessioni dirette a Tabby e nuovi client. Se il port 8000
   deve rimanere disponibile senza interruzione, serve un gateway con handoff
   atomico e una strategia VRAM compatibile: questo supervisor è **a interruzione
   controllata** durante la finestra di manutenzione.
3. Introdurre in GPU Manager una lease HERMES esplicita e un backend gestito
   autonomamente, così da mantenere i servizi multimediali e la chat senza
   arrestare il manager. Le modalità oggi disponibili non lo consentono.
4. Preparare token app limitato, sudoers ristretto e ACL Tailscale;
   certificare HTTPS/autenticazione davanti a qualunque client remoto.
5. Completare quality gate indipendente, parallelismo multiutente, contesti
   lunghi, limiti VRAM e benchmark di throughput con test GPU esclusivi.

**Decisione:** gli script sono pronti per essere revisionati e usati come
guardrail in manutenzione; **HERMES-27 resta NO-GO per traffico di produzione**
finché i blocchi sopra non sono chiusi e il prime autorizza una prova
coordinata con rollback verificato sul server reale.
