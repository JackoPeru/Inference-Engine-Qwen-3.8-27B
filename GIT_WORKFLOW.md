# GIT_WORKFLOW — Inference-Engine-Qwen-3.8-27B (sicuro, read-only)

Data indagine: 2026-11-13. Nessuna operazione distruttiva eseguita. Nessun push.

## 1. Stato trovato

- Repo: https://github.com/JackoPeru/Inference-Engine-Qwen-3.8-27B , branch `main`.
- Commit remoto attuale (verificato da Windows `git ls-remote origin main` E da server `git ls-remote ... main`): `a657a4901a0d3d8a465f8366c9345a3a63e05e9f`
- Copia Git Windows (`...-Git`, HEAD `a657a4901a0d3d8a465f8366c9345a3a63e05e9f`): `git log --oneline -8` = 2 commit (`a657a49`, `036ba48`), `git status --short` = pulito, `git remote -v` = origin https fetch+push, `git ls-remote origin main` = OK stesso hash. Remoto NON avanzato.
- Server `/home/matteo/hermes-27`: `NO-GIT` (confermato, non spostato/cancellato), nessun `.gitignore` locale.
- Credenziali server: SI. `gh auth status` = Logged in `JackoPeru`, protocol `ssh`, token con scope `repo,workflow,...`. `git config --global -l` = credential helper `gh auth git-credential` + LFS. `ls ~/.ssh` = `id_ed25519`, `id_ed25519_pcmatteo`, `known_hosts` presenti. Nulla modificato.

## 2. Confronto 6 file chiave (SHA256)

Base Git: `...-Git/hermes-27/` — Base lavoro: `.../Inference Engine Qwen 3.8 27B/hermes-27/`. Attenzione: in copia-Git i file sono sotto `hermes-27/`, non in root.

| File | Git | Lavoro Win | Server | Esito |
|---|---|---|---|---|
| CMakeLists.txt | `10123323...CC2FBAB` | identico | `10123...` identico | UGUALI x3 |
| architecture/src/model_fwd.cpp | `6448A6A1...` | identico | identico | UGUALI x3 |
| kernels/src/gdn_step.cu | `68CAE2E2...` | identico | identico | UGUALI x3 |
| reports/M2.md | `B616CB5D...` | identico | identico | UGUALI x3 |
| reports/VALIDATION.md | `A5682601...` | identico | identico | UGUALI x3 |
| scripts/e1_runner.py | `FD04829F...` 11284 B | `D47711E9...` 23030 B | `1f6e5a41...` 11302 B | TRIPLA DIVERGENZA |

Da risolvere a mano: `e1_runner.py` lavoro-Windows è 2x più grande (probabile evoluzione locale), server e Git differiscono di ~18 B (forse newline). Non sovrascrivere alla cieca.

## 3. File solo-server (non in copia lavoro Windows, o da escludere)

Presenti su server, `ls ~/hermes-27/{scripts,reports,build}` + `ls -la ~/hermes-27`:

- `models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf` (19 GB) + `models/download.log` — RESTANO FUORI (`.gitignore`: `**/models/`, `*.gguf`, `*.log`).
- `venv/` (29 MB), `build/` (29 MB, con `CMakeCache.txt`, `*.a`, test binari) — RESTANO FUORI (`.gitignore`: `**/venv/`, `**/build/`, `*.o`, `*.a`, `*.so`).
- `scheduler/sim.py`, `scheduler/sched_result.json`, `serving/validate.py`, cartelle vuote `distributed/`, `memory/`, `ops/` — GIÀ TRACKATI in Git (`hermes-27/scheduler/*`, `hermes-27/serving/*`), ma MANCANTI in copia-lavoro Windows. Non sono "nuovi": vanno recuperati dal clone, non ricreati.
- `reports/` e `scripts/` server coincidono per nomi con lavoro-Windows (23 script, 12 report); contenuto da diffare solo per `e1_runner.py` noto + eventuali edit minori.
- `.gitignore` repo (verificato in copia-Git) copre: `**/models/`, `*.gguf/*.safetensors/*.pt/*.bin`, `**/build/`, `**/venv/`, `*.log`, secrets, `.vs/.vscode`. Adeguato. Non committare mai `models/`, `venv/`, `build/`.

## 4. Piano integrazione in 5 passi (NON eseguito, solo comandi pronti, sul server)

Principi: mai `push --force`, mai `reset --hard`, mai `rm -rf`, mai `kill`, mai test/build pesanti sul server.

Passo 1 — Clone separato piccolo, solo codice:
```
git clone --depth 1 https://github.com/JackoPeru/Inference-Engine-Qwen-3.8-27B ~/hermes-27-gh
cd ~/hermes-27-gh && git rev-parse HEAD && git ls-remote origin main
```

Passo 2 — Dry-run poi copia reale, solo codice (escludi artefatti):
```
rsync -av --dry-run --exclude='models/' --exclude='venv/' --exclude='build/' --exclude='*.gguf' --exclude='*.log' --exclude='__pycache__/' ~/hermes-27/ ~/hermes-27-gh/hermes-27/
# solo dopo aver letto il dry-run:
rsync -av --exclude='models/' --exclude='venv/' --exclude='build/' --exclude='*.gguf' --exclude='*.log' --exclude='__pycache__/' ~/hermes-27/ ~/hermes-27-gh/hermes-27/
```

Passo 3 — Ispeziona, gestisci solo-server e conflitto noto:
```
cd ~/hermes-27-gh && git status --short && git diff --stat
# e1_runner.py: tripla divergenza -> confronto manuale, NON scegliere a caso:
diff -u hermes-27/scripts/e1_runner.py ~/hermes-27/scripts/e1_runner.py | head -100
# scheduler/serving mancanti in copia Win ma già in Git: se rsync li ha toccati, ripristina con `git diff -- <file>` e tieni versione Git se server non l'ha modificata
git diff -- hermes-27/scheduler/sim.py hermes-27/serving/validate.py
```

Passo 4 — Commit (solo se diff volontario e .gitignore pulito):
```
cd ~/hermes-27-gh && git check-ignore -v hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf hermes-27/build/CMakeCache.txt || echo "ATTENZIONE gitignore"
git add hermes-27/scripts/e1_runner.py hermes-27/reports hermes-27/docs
git status --short
git commit -m "Sync hermes-27 da server $(date -u +%F)"
```

Passo 5 — Verifica remoto non avanzato + push (una sola volta, mai force):
```
git fetch origin main && git rev-parse HEAD && git rev-parse origin/main && git ls-remote origin main
# prosegui SOLO se origin/main == a657a4901a0d3d8a465f8366c9345a3a63e05e9f (o comunque == base del clone); se diverso -> `git pull --rebase` e rivaluta, MAI force
git push origin main
```

Se credenziali mancassero (oggi NON mancano): serve `gh auth login` interattivo oppure PAT con scope `repo` via `gh auth login --with-token`, senza inventare token; non committare token.

## 5. Rischi / blocchi

- `e1_runner.py` in 3 versioni: rischio sovrascrittura lavoro utile (23030 B Win vs 11302 B server). Blocco: merge manuale obbligatorio.
- `models/*.gguf` 19 GB + `venv/` + `build/`: se rsync senza `--exclude` li copia nel clone -> push enorme o leak. Blocco: usare sempre exclude + `git check-ignore`.
- Remoto potrebbe avanzare tra clone e push: ricontrollare `ls-remote` subito prima del push (Passo 5).
- Server con run CPU in corso: vietati `test_model_fwd`, `e1_*.py`, `cmake --build`, `ctest` (già rispettato).
- Copia lavoro Windows senza `scheduler/`/`serving/`: non interpretarla come delete; fonte verità = clone fresco + server.
