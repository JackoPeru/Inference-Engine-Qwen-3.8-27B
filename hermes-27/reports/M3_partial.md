# M3 parziale — Scheduler + serving (2026-11-12, in corso)

## Scheduler sim (`scheduler/sim.py`, calibrato su baseline EXL3)
- Calibrazione: single 65 tok/s, cap agg 130, prefill 550 tok/s,
  TP2 +18.6ms/tok (allreduce host), PP2 +0.6ms (xfer). Solo confronto policy.
- Risultati (`scheduler/sched_result.json`): conc4 tp2 70.7 / pp2 103.5 tok/s
  (misurato 121); conc8 tp2 84.5 / pp2 104.3 (misurato 101.7).
  PP2 > TP2 a ogni concorrenza per il costo allreduce senza P2P → **default PP2**.
- TTFT sim conc4 ~0.5s (misurato first-token 0.35s + coda). Chunk 512, MAXTOK 4096.

## Serving (`serving/validate.py`, PASS)
- Validazione OpenAI `/v1/chat/completions`: messages/ruoli, max_tokens 1..8192,
  temperature, stream bool, model allow-list, reasoning object. 7 casi negativi
  rifiutati. Formattazione SSE + `[DONE]`. Contratto per il layer HTTP nativo.

## Manca per M3-done
- Continuous batching reale nel runtime (code+admit/drop/backpressure/KV budget),
  CUDA Graphs per batch shape, prefix caching con stati ricorrenti (solo se
  matematicamente identico), API streaming `/health` `/metrics`, cancel,
  rate-limit. Richiede M2-done (inferenza full-model) + finestra GPU.
