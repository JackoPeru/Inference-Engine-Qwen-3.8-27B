# HERMES-27: deployment sul server, gate e rollback

Data verifica: **13 novembre 2026**. Stato: **NO-GO per sostituzione di TabbyAPI
in produzione**. Il nuovo percorso è **PREPARED/SIMULATED**: il contratto
GPU Manager e l'ingress non sono implementati sul server reale. Nessun
cutover e nessun comando systemd/GPU è stato eseguito in questa fase.

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
| Binario GGUF | `~/hermes-27/build/gguf_dump` presente all'audit precedente, ma obsoleto: verifica della nuova build necessaria |
| Binario serving | **`~/hermes-27/build/hermes_server` assente al momento dell'audit precedente**; non ancora ricontrollato su build distribuita |

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

La build pulita può usare il prototipo CUTLASS W4A4 soltanto come opzione
esplicita: il relativo percorso di prefill sperimentale resta **disabilitato
per impostazione predefinita** in CMake e non va abilitato dal deployment
senza un gate di qualità indipendente. Il preflight confronta i binari
effettivamente distribuiti, non il risultato di una compilazione isolata.

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

### Contratto di integrazione richiesto: PREPARED, non distribuito

La proposta completa si trova in `ops/MANAGER_LEASE_CONTRACT.md`. I controlli
aggiunti a `ops/production_deploy.py` richiedono nell'endpoint autentico
`/status` del manager: `watchdog_lease_contract_version: 1`,
`manager_instance_id` stabile durante il processo e `maintenance_lease`
esplicitamente `null` prima dell'acquisizione. Dopo l'acquisizione il manager
deve attestare `id`, owner `hermes-27`, scope
`tabby-restart-and-mode-transition`, `watchdog_restart_suppressed: true`,
fencing token e scadenza nell'intervallo 25–180 s. La lease viene rinnovata
ogni 8 s anche durante l'upload GPU e durante il rollback; il manager resta
attivo e **non** può riavviare Tabby durante la lease. In caso di scadenza,
il backend manager deve restare in recupero protetto senza riavviare Tabby,
finché HERMES non è stato arrestato con certezza.

Gli adapter di acquisizione/rinnovo/rilascio della lease accettano JSON su
stdin e inviano JSON su stdout. Non trasmettono credenziali negli argomenti
dei processi; ogni risposta viene verificata di nuovo contro `/status`.
Questi adapter e le modifiche al manager **non esistono ancora** in produzione:
il gate è intenzionalmente NO-GO.

### Prototipo manager-side preparato, non applicato

Nuovi file sotto `ops/`: `manager_lease_core.py` (journal fsync e lease fenced),
`manager_lease_runtime.py` (quattro endpoint autenticati su loopback e
protezione watchdog/drive/force/systemctl), `manager_lease_patch.py` (generatore
offline con controllo SHA256 del sorgente), `manager_lease_adapter.py`
(trasporto JSON stdin e localhost, senza credenziali nei parametri),
`manager_lease_v1_proposal.patch` (diff reviewable),
`manager_lease_REVIEW.md` (contratti, checkers e limitazioni). Sorgente reale
di riferimento letto via SSH senza scritture:
`/opt/hermes/gpu-manager/manager.py`, SHA256
`15744e3b769c0f6cd5ed26c2b85e3f282871f4a24681abd324c8cbc329a73eef`.

La patch è stata generata e compilata sintatticamente da una **copia locale**
del file reale e poi rimossa la copia temporanea. Non è stata applicata a
`/opt/hermes` né installata come unità systemd. Blocca il watchdog e le
transizioni manager mentre la lease è attiva o in recupero, e usa anche
`_submit_lock` e un gate nel `create_job()` centrale per impedire un job
media ammesso prima dell'acquisizione e inserito dopo il passaggio alla lease.
La scadenza della lease **non** dà il permesso di riavviare Tabby: occorre
una prova indipendente di ingress bloccato e di uscita completa di HERMES
dalle due GPU. Se Tabby è già ripartito in modo sicuro, il manager può soltanto
rilasciare il fence dopo aver provato l'ownership di Tabby sulle due GPU.

Mancano **ancora** i comandi privilegiati indipendenti in grado di certificare
il blocco di tutti i client e l'owner reale delle GPU. L'endpoint di test e
la patch non sostituiscono quei verificatori. Restano inoltre da verificare
attori esterni a GPU Manager (timer systemd, launcher Character ID e accesso
diretto alla porta Tabby), la persistenza del journal dopo reboot e il cutover
su hardware reale, con autorizzazione finale del prime/utente.

**Patch commit gate (13 novembre 2026):** corretti nel generatore i 17
contesti vuoti con uno spazio finale e la riga vuota finale della diff;
aggiornati i conteggi dell'ultimo hunk. La patch finale misura 7.083 byte,
si applica senza errori al sorgente manager SHA256 verificato e produce
un manager.py identico byte per byte al risultato generato. I controlli
Git apply dry-run e Git staged diff whitespace sono entrambi PASS.
Il runtime ora rifiuta un token operatore leggibile dal gruppo. La suite
offline aggiornata conta **115/115 test PASS** su Python 3.13 Windows
(48,897 s), con compilazione Python e sintassi Bash PASS; il test
113/113 svolto dal prime nel venv server si riferisce alla versione
precedente. Manifest e verifiche in ops/MANAGER_LEASE_PATCH_GATE.md.
**La produzione resta NO-GO** per mancanza di prove indipendenti di ingress,
owner/PID sulle due GPU e journal fencing persistente.

**Follow-up manager-side del 13 novembre 2026:** 113/113 test locali PASS
(40 del cutover preparato, 73 nuovi fra lease core, runtime, generatore patch
e adapter). Verificata la compilazione sintattica Python, la sintassi Bash e
la generazione della patch da una copia SHA256-identica al manager live. Il
generatore rifiuta il sorgente modificato e impedisce l'output in directory
preesistenti o di produzione. La diff finale `manager_lease_v1_proposal.patch`
contiene anche il gate su `_submit_lock` e `create_job()`, che evita la race
tra invio dei media e concessione della lease. Il manager installato **non è
stato modificato**; non sono stati eseguiti `systemctl`, carichi GPU o
attivazioni di rete durante questi test.

Il contratto di ingress richiede una prova fresca (massimo 10 s) con
`scope: "all-tabby-clients"`, `blocked: true`, `inflight: 0`, verificata
prima dell'arresto di Tabby e durante l'inferenza. Include client locali,
proxy e accessi Tailscale. Se il blocco cade, il supervisore arresta HERMES,
tenta di bloccare nuovamente l'ingresso e conserva journal e blocco per
ispezione manuale.

## Componenti aggiunti

- `ops/production_deploy.py`: programma Python standard library per
  `dry-run`, `preflight`, `run --authorized`, `recover --authorized`, `health`.
- `ops/hermes-27.json.example`: configurazione dichiarativa con hash completi,
  server isolato, app token del manager, hook ingress/drain/restore, limiti.
- `ops/systemd/hermes-27-cutover.service.example`: unità **non installata**, senza
  sezione `[Install]`; `Restart=no`; `ExecStopPost` richiama recovery.
- `ops/tests/test_production_deploy.py`, `scripts/production_smoke.sh` e
  `ops/README.md`: smoke CPU-only e guida per l'operatore.
- `ops/production_simulate.py`: simulazione completa isolata con 20 scenari,
  tra cui il flusso regolare, 17 guasti iniettati e 2 retry di recovery dopo
  la correzione dell'adapter che aveva fallito.
- `ops/MANAGER_LEASE_CONTRACT.md`: API, scadenza, fencing, watchdog, adapter
  privati e ingress completi da implementare prima del rilascio.

## Sequenza prevista per la finestra di manutenzione

1. Distribuire il binario HERMES e verificare staticamente file, eseguibili,
   hash GGUF e tokenizer, RAM CPU, 2 GPU, autenticazione `/status`, risposta
   Tabby e disponibilità della porta isolata **18127**. Qualunque errore blocca
   l'attivazione prima di mutare i servizi.
2. Il manager deve attestare il contratto lease v1; se assente si blocca prima
   del drain. Bloccare l'ingresso di **tutti** i client verso Tabby con un hook
   esterno approvato dall'operatore e con prova fresca `scope`, timestamp,
   `blocked=true` e `inflight=0`; controllare anche i client locali diretti.
3. Ricontrollare GPU Manager in `LLM_READY` con coda vuota; ottenere e
   verificare la lease manager inibendo watchdog e mode switch. Mantenere
   `hermes-gpu-manager.service` **attivo** e rinnovare la lease. Solo allora
   arrestare `hermes-tabby.service` con autorizzazione esplicita.
4. Controllare che entrambe le GPU abbiano rilasciato la memoria sotto il
   limite configurato; avviare `hermes_server` su `127.0.0.1:18127`, con coda
   e contesto limitati. Non usare `127.0.0.1:8000` per il nuovo server.
5. Il supervisore richiede `/health` e `/v1/models` attesi; durante l'upload
   e il servizio verifica lease, rinnovo, stato del manager, Tabby fermo e
   ingresso bloccato. La perdita di un controllo arresta HERMES e avvia
   rollback. Un riavvio di Tabby durante l'inferenza forza l'abbandono.
6. Arrestare HERMES con PID verificato; ripristinare Tabby; attendere una
   risposta valida e l'attestazione `LLM_READY` del manager, rilasciare la
   lease e verificare `maintenance_lease: null`; **solo dopo** riaprire e
   verificare ingress. Se un controllo fallisce, conservare il journal e il
   blocco. `recover --authorized` è disponibile per riprovare la procedura.

La sequenza serializza le modifiche con `flock` e scrive un journal JSON 0600
prima della prima operazione esterna, aggiornato con `fsync` e rename atomico.
Il PID figlio viene confrontato con eseguibile e start time Linux prima di
inviare segnali; se HERMES risulta ancora vivo, il restore **non** riavvia
Tabby sulle medesime GPU. Se il journal segna uno spawn ma non ha ancora il
PID, il rollback resta bloccato per evitare doppia allocazione. Il log
applicativo passa al journal systemd; HERMES scrive in
`state_dir/hermes-server.log`.

## Sicurezza, Tailscale e autorizzazioni

Il bind imposto per HERMES è **solo `127.0.0.1`**. La porta 8000 di Tabby
non viene riutilizzata. Per accesso privato attraverso Tailscale è possibile
usare un tunnel SSH controllato dalle ACL di Tailscale, per esempio:

```bash
ssh -N -L 18127:127.0.0.1:18127 matteo@hermes
```

Non pubblicare 18127 tramite Funnel o reverse proxy pubblico senza TLS,
autenticazione e controllo dei client. La versione recente dell'HTTP server
prevede `--api-key-file`, ma la configurazione di esempio espone solo il
loopback e non incorpora segreti. Il manager è
attualmente in ascolto su `0.0.0.0:8643`: verificarne le regole firewall/
Tailscale indipendentemente da questo deployment.

Il processo di manutenzione gira come `matteo`; l'operatore deve predisporre
accesso **ristretto** e non interattivo a `systemctl start/stop` per la sola
unità Tabby, più un file segreto leggibile dal processo contenente
una app key valida per il `GET /status`. Nessuna credenziale va passata in
argomenti CLI. Gli hook devono essere file eseguibili assoluti, sotto il
controllo dell'operatore, e non sono forniti perché non è stato identificato
un contratto reale di drain sul gateway del sistema attuale.

## Verifiche eseguite e risultato

La prima serie di 20 test era stata eseguita su copia isolata in
`/tmp/hermes_worker7_20261113` sul server, senza carichi GPU e senza modifiche
systemd. Il follow-up del contratto lease è stato sviluppato e verificato
**esclusivamente sul PC Windows locale**, con Python 3.13 e tutte le
dipendenze esterne (GPU, processi, systemd e rete) simulate: nessuna modifica
è stata effettuata sul server nel follow-up.

| Prova | Esito |
| --- | --- |
| `python3 -m py_compile ops/production_deploy.py` | PASS |
| `python -m unittest discover -s ops/tests -p 'test_*.py' -q` | **113/113 PASS** (40 cutover/client + 73 manager-side), 58,507 s sul PC Windows |
| `python -m py_compile ops/manager_lease_*.py` (file espliciti) | PASS, 4 nuovi moduli |
| `python ops/production_simulate.py --scenario all` | **20/20 PASS**, ingress/systemd/GPU/processi simulati |
| `bash -n scripts/production_smoke.sh` | PASS |
| Negative cases: hash/model magic, build directory errata, GGUF dump descrittivo anche con hash ripinnato, server help incompatibile, bridge non pronto, bind/URL malformati, manager busy, drain non provato, rollback fallito, PID vivo, mancata autorizzazione | PASS |
| Test dry-run con fixture locale, `--skip-hash` | PASS: hash binari, controllo CLI e bridge CPU; nessuna chiamata a systemd, HTTP o GPU |
| Dry-run con percorsi reali e manifest di esempio | **FAIL previsto**: `missing or nonexecutable: server_path` |
| `shellcheck` | NON ESEGUITO: eseguibile assente sul server |
| Stato servizi dopo i test precedenti sul server | `hermes-gpu-manager`: active; `hermes-tabby`: active; nessuna nuova verifica live nel follow-up |
| Cutover con stop/start reali, upload GGUF, throughput e accesso client | **NON ESEGUITI**: richiedono autorizzazione prime e preparazione ingress |

Il test del rollback usa mock dei servizi e degli endpoint: non certifica
effettiva ripartenza del modello sul server. Un crash tra creazione del
processo figlio e persistenza del suo PID provoca ora un blocco esplicito
di recovery in attesa di una verifica manuale del processo. La perdita del
disco di journal oppure un guasto completo dell'host richiedono ancora
procedure operative indipendenti. L'unità systemd di esempio tenta
`ExecStopPost` anche dopo arresti inattesi del suo processo, ma non garantisce
il recupero durante un blackout. Al reboot restano le unità originarie;
questa unità non è abilitata.

### Matrice dei guasti iniettati nel follow-up

| Esito simulato | Scenari | Comportamento verificato |
| --- | --- | --- |
| **COMPLETE** | `clean_shutdown`, `lease_acquire_denied`, `tabby_stop_denied`, `gpu_busy`, `hermes_health_failed`, `recover_after_lease_release_failure`, `recover_after_ingress_restore_failure` | Tabby ripristinato, lease rilasciata e ingress riaperto dopo verifica; i due recovery retry richiedono correzione dell'adapter |
| **DENIED PRE-MUTATION** | `manager_unsupported`, `manager_busy` | Nessun journal, nessun drain, nessun stop di servizi |
| **HELD_CLOSED** | `drain_unproven`, `lease_scope_invalid`, `ambiguous_spawn`, `lease_renew_denied`, `lease_expired`, `manager_restarted`, `ingress_reopened`, `child_stuck`, `tabby_restore_failed`, `lease_release_failed`, `ingress_restore_failed` | Journal mantenuto, ingresso bloccato, nessuna riapertura automatica durante stato non verificabile |

Ogni simulazione intercetta le interfacce esterne. La prova verifica
l'ordine `drain → lease → stop Tabby → avvio Hermes → stop Hermes → restore
Tabby → release lease → restore ingress` e vieta qualunque stop del manager.
La scadenza lease, il manager restart, i client che riaprono la connessione,
un processo figlio sconosciuto o un restore incompleto mantengono il fail
closed. Lo scenario `ingress_reopened` richiude l'ingresso e conserva il
journal per intervento manuale. **Sono prove di logica del supervisore,
non attestazioni dell'implementazione real-world dei contratti.**

## Blocchi prima dell'uso reale continuativo

1. Compilare e validare `hermes_server` e `gguf_dump` dalla medesima build
   approvata e distribuita; fissare gli SHA256 server/dumper/bridge e
   rieseguire la prova Jinja, tokenizer e versione. La toolchain live
   originale dell'audit era assente/obsoleta; una build isolata non la sostituisce.
2. Implementare un contratto di drain e ripristino reale dell'ingress,
   includendo connessioni dirette a Tabby e nuovi client. Se il port 8000
   deve rimanere disponibile senza interruzione, serve un gateway con handoff
   atomico e una strategia VRAM compatibile: questo supervisor è **a interruzione
   controllata** durante la finestra di manutenzione.
3. Implementare davvero `ops/MANAGER_LEASE_CONTRACT.md` nel manager, con
   modalità `HERMES_MAINTENANCE`, mantenimento del manager attivo, inhibition
   del watchdog, fencing token e semantica di scadenza senza restart di Tabby.
   Implementare adapter locali autenticati senza esporre segreti. Il manager
   attuale non espone alcun lease, perciò il gate blocca il cutover.
4. Preparare token app limitato, sudoers ristretto e ACL Tailscale;
   certificare HTTPS/autenticazione davanti a qualunque client remoto.
5. Completare quality gate indipendente, parallelismo multiutente, contesti
   lunghi, limiti VRAM e benchmark di throughput con test GPU esclusivi.

**Decisione:** percorso software **PREPARED**, verifica complessiva **SIMULATED
40/40 PASS** (20 guardrail + 20 scenari cutover), deployment reale **NO-GO**.
Sono ancora necessari manager con
lease, ingress verificabile, toolchain distribuita e autorizzazione finale
del prime/utente prima di qualsiasi cutover reale.
