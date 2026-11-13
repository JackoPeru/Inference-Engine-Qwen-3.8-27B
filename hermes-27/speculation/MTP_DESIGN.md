# MTP/NextN speculative decoding — design (da llama.cpp qwen35.cpp + GGUF reale)

## Wiring (verificato su `qwen35.cpp::graph_mtp` + tensori GGUF blk.64)
- Input: `h` = ultimo hidden del target + `tok` = embedding del draft token.
- `h_norm = RMSNorm(h, nextn.hnorm)`; `e_norm = RMSNorm(emb[tok], nextn.enorm)`.
- `x = eh_proj(concat([e_norm; h_norm]))` — eh_proj BF16 [10240,5120].
- Blocco transformer BF16 completo (attn q/k/v/o + FFN) con **KV cache propria**
  del singolo layer MTP (attn_q/k/v BF16, out BF16, gate/up/down BF16).
  - q split [query 6144 | gate 6144], q/k norm [256], RoPE mRoPE-text,
    attn × sigmoid(gate), out_proj, +residuo, post_norm, FFN, +residuo.
- `shared_head_norm` (RMSNorm) + `output.weight` condiviso (lm head) → draft logits.

## Protocollo proposal/verification (greedy)
- **Stato implementazione 2026-11-13**: questo protocollo K>1 e` un design futuro, non una funzione gia` disponibile. La CLI verificata con checkpoint reale implementa K=1 iterative con `decode_step` target ad ogni token e nessun throughput gain (vedi `MTP_VERIFIED_PATH.md`). La verifica batch K>1 richiede un'API PP2 capace di costruire e committare/troncare gli stati GDN/conv/KV del prefisso accettato; il solo `prefill(P+draft)` non basta.
- Draft: K step autoregressivi del blocco MTP (ognuno usa h corrente + tok precedente).
- Verify: un forward target sui K token draft (single prefill chunk) → logit per posizione.
- Accept (greedy): finché draft_tok[i] == argmax(target_logits[i]); al primo mismatch
  scarta il resto, emetti argmax(target) e riparti (rollback = tronca KV/GDN/conv
  alla posizione accettata; gli stati MTP si rigenerano dal prefisso accettato).
- Invariante da testare: token greedy MTP == token greedy target-only (stesso checkpoint).

## Sampling (temperatura/top-k/top-p)
- Acceptance con probabilità min(1, p_target(t)/p_draft(t)) sul token draftato
  (speculative sampling standard); preserva esattamente la distribuzione target.
- Test: confronto distribuzionale (non identità di sequenza) vs target-only.

## Stato
- Attesa golden E1/E2 per il target; poi: MTP draft CPU vs greedy target (identità),
  acceptance rate K=1..5 su prompt campioni, rollback test, solo dopo innesto GPU.
- DFlash2 esterno: NON necessario al primo rilascio (draft esterno non compatibile
  con modello abliterato senza verifica; MTP incorporato è il draft primario).
