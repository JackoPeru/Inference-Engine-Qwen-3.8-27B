# Inferenza testuale PP2

`hermes_generate` carica il GGUF indicato sulle due GPU e genera in greedy. Ogni `--prompt-ids` usa uno stato separato e produce un record JSONL su stdout. La diagnostica va su stderr. Il valore EOS predefinito viene dal GGUF. `--verify-prefill` confronta i logit finali del prompt con il prefill completo, per prompt fino a 64 token.

## Riproduzione su Hermes

Da una copia isolata dei sorgenti su Linux, con `thirdparty/cutlass` disponibile e le due GPU libere:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc
cmake --build build --target hermes_generate test_pp2_decode -j4
model="$HOME/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf"
python3 scripts/check_generate.py build/hermes_generate "$model"
```

Per l'input testuale serve un `tokenizer.json` compatibile e il pacchetto Python `tokenizers`. `--chat` richiede anche `jinja2` e legge il template direttamente dal GGUF tramite `gguf_dump`.

```bash
python3 scripts/hermes_generate.py --executable build/hermes_generate \
  --model "$model" --tokenizer-json /path/to/tokenizer.json \
  --chat --no-thinking --max-new-tokens 8 'Rispondi con una parola.'
```

## Risultati su 2× RTX 5060 Ti, CUDA 12.8.93

- Il GGUF salva `ssm_a` già trasformato come `-exp(A_log)` e raggruppa le teste V per indice entro il gruppo K. Il runtime ora usa il valore diretto di `ssm_a` e associa Q/K con `h % 16`.
- Parità di logit tra prefill e decode (`max_abs=0`) su IT, EN e chat dopo la correzione. In chat i token `[66,21817,248046]` producono `ciao` e l'EOS del GGUF. Anche `llama-completion` b9704 in greedy su CPU, con lo stesso GGUF e gli stessi 21 token di prompt, produce `ciao`.
- Nei prompt senza template, IT `[271,248068,198,760]`, EN `[271,760,8323,21440]`. I testi prodotti coincidono col riferimento `llama-completion` per i quattro token. Le precedenti fixture E1 con `ssm_a` e raggruppamento errati sono storiche.
- Test eseguito con 127 token di input e un token generato. Valori fuori intervallo, lista con virgola finale e checkpoint inesistente rifiutati con exit code non zero.
- Tokenizzazione paritaria su quattro input, inclusi Unicode e chat. Confronto integrale delle voci: i primi 248077 ID coincidono, i 243 ID successivi del GGUF sono voci di padding assenti dal tokenizer esterno.
- Memoria post-run tornata a 486/1289 MiB. Log della verifica in `/tmp/hermes-advisor.hPxwCF` su Hermes.
