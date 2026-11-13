# P3 — Baseline GPU PP2 misurata (2026-11-13)

Primo forward completo + decode su 2x RTX 5060 Ti. Dati in `gpu_pp2_baseline.json`.
Metodo: `test_pp2_fwd g3/g4` (prefill) + `test_pp2_decode greedy` (4 tok), con
`nvidia-smi dmon` in parallelo. Nessun servizio toccato; VRAM torna a 486/1289 MiB.

## Numeri (prompt 11 tok, greedy 4 tok)
- Upload pesi: 9.6s cold / 2.8s warm (8.77 + 9.56 GiB residenti, single-copy).
- Prefill T=11: 6.79s → 1.62 tok/s. TTFT ≈ 6.8s.
- Decode: ~654ms/tok → 1.53 tok/s, TPOT ≈ 654ms.
- Xfer confine PP2: 0.06ms (pinned host, 225KB).
- VRAM picco: 9869 / 11482 MiB; regime: 9856 / 11470 MiB (< 15 GiB, no leak).
- GPU util: burst 0/99 alternati per stadio (pipeline senza overlap di microbatch).

## Correttezza (stesso checkpoint del riferimento CPU)
- Prefill top5 = golden E1/E2 `[220,1374,27053,17,1147]`.
- Full-logits vs E2 fp32: maxerr 6.55, meanerr 0.24, argmax 9/11, top50 49/50
  (rumore bf16-activation + accumulo; E2==E1 resta il riferimento esatto).
- Decode-vs-prefill GPU: BIT-EXACT 11/11 (maxerr 0.0) — carry GDN/conv/KV corretti.
- Greedy `[220,17,1374,220]`, MTP K=1 ACCEPT (draft==teacher).

## Prossimo (P5, in ordine di throughput multiutente)
1. Microbatch overlap PP2 (utilizzo contemporaneo GPU), CUDA Graphs per decode step.
2. Batching continuo + backpressure (cap da VRAM_BUDGET_V2), prefix caching con stati GDN.
3. MTP K=1..5 acceptance sweep + sampling con distribuzione target.
4. CUTLASS W4A4 nel percorso prefill caldo (oggi GEMV diretto; wrapper gia` validato 0.0038).
