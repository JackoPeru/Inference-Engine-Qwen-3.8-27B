# MTP wiring audit — nostro vs llama.cpp `graph_mtp` (qwen35.cpp)

Scope: solo wiring del draft NextN blk.64. Nessun build/run in questo documento.

> Aggiornamento 2026-11-13: consultare **`speculation/MTP_VERIFIED_PATH.md`** per l'esecuzione GPU reale successiva a questo audit: due prompt, sequenze greedy target-only identiche per 5 nuovi token, 3/4 e 2/4 proposte K=1 accettate, rewind MTP verificato. Le righe V3–V6 qui sotto descrivono il piano storico; non rappresentano piu` lo stato dei test K=1. La generazione K>1 accelerata resta bloccata dall'assenza di una API di commit/rollback stateful PP2.
Sorgenti: `architecture/src/model_fwd.cpp` (`ModelFwd::mtp_hidden`, `ModelFwd::mtp_draft`, `MtpState`),
`architecture/include/hermes/model_fwd.hpp`, `speculation/MTP_DESIGN.md`, tensori GGUF reali blk.64.
Reference: `llama.cpp` `qwen35.cpp::graph_mtp`.

## 1. Tensori blk.64 (GGUF reale, check leggero PASS)

- `blk.64.nextn.eh_proj.weight` BF16 `[10240, 5120]` (in 10240 = 2x5120, out 5120).
- `blk.64.nextn.enorm.weight` F32 `[5120]`.
- `blk.64.nextn.hnorm.weight` F32 `[5120]`.
- `blk.64.nextn.shared_head_norm.weight` F32 `[5120]`.
- Blocco decoder full-attention BF16 completo: `attn_q/k/v/output`, `attn_q_norm`, `attn_k_norm`,
  `attn_norm`, `post_attention_norm`, `ffn_gate/up/down` (stesse dims del target: HID 5120, FFN 17408,
  NQ 24, NKV 4, HD 256).
- Assenti (verificato): `nextn.embed_tokens`, `nextn.shared_head_head` -> si riusano
  `token_embd.weight` e `output.weight`.
- Metadata: `KV 65 blocchi totali`, `nextn_predict_layers = 1` (64 target + 1 draft).

## 2. Tabella wiring riga per riga

Legenda stato: CONFERMATO = combacia con `graph_mtp`; EMPIRICO = assunto da resto modello,
da chiudere con golden/run; DIVERGENTE-NOTA = differenza intenzionale documentata.

| # | Voce | llama.cpp `graph_mtp` | Nostro (`model_fwd.cpp`) | Stato |
|---|------|----------------------|--------------------------|-------|
| 1 | Input h | ultimo hidden target `h` | `mtp_hidden(h_last, ...)` / `mtp_draft(h_last, ...)`; `h_last` = hidden pre-`output_norm` da `prefill(h_last_out)` / `decode_step` | CONFERMATO (plumbing) |
| 2 | Input tok | embedding del draft token | `e = token_embd.weight[tok]` via `bf16()` + `bf2f` | CONFERMATO (reuse embed, nessun `nextn.embed_tokens`) |
| 3 | h_norm | `RMSNorm(h, nextn.hnorm)` | `rms(h_last, f32("blk.64.nextn.hnorm.weight"), hn, ...)` | CONFERMATO formula; peso EMPIRICO vedi sez. 3 (convenzione diretta) |
| 4 | e_norm | `RMSNorm(tok_embd, nextn.enorm)` | `rms(e, f32("blk.64.nextn.enorm.weight"), en, ...)` | Come sopra |
| 5 | Concat order | `concat([e_norm; h_norm])`, embedding prima | `eh[i]=en[i]; eh[HID+i]=hn[i]` (e prima) | CONFERMATO |
| 6 | eh_proj | `x = eh_proj(concat)`, `[10240,5120]` | `linear_nvfp4_rows("blk.64.nextn.eh_proj.weight", eh, x_out, 1, 0, HID)` | CONFERMATO (dims + direzione) |
| 7 | attn_norm | norm ingresso blocco draft | `rms(x, f32("blk.64.attn_norm.weight"), h, ...)` | CONFERMATO |
| 8 | q split | `q_proj` split interleaved per-head `[q_h(256) \| g_h(256)]`, non halves | loop per-head `qq[hd*512+d]` = q, `qq[hd*512+256+d]` = gate | CONFERMATO (stessa trappola del target, cfr `VALIDATION.md`) |
| 9 | q/k RMSNorm | q/k norm con pesi dedicati `[256]` | `Q = RMS(q)*qn[d]`, `K = RMS(k)*kn[d]` con `attn_q_norm` / `attn_k_norm` | CONFERMATO struttura; peso EMPIRICO (diretto) |
| 10 | RoPE | mRoPE-text, split-half GPT-J, partial 0.25 (prime 64 dim di HD 256) | `rope1` split-half su 32+32, `theta 10M`, `pos` assoluta | CONFERMATO (stesso nucleo del target) |
| 11 | Scale | `scale = 1/sqrt(HD)` = `1/16` (HD 256) | `sc = dd*(1.f/16.f)` sia prefill che draft | CONFERMATO |
| 12 | Gate | `attn_out * sigmoid(gate)` (transformers `Qwen3_5Attention`, non SiLU) | `att[i] /= (1+exp(-G[i]))` | CONFERMATO (fix E2 SiLU->sigmoid applicato anche al draft) |
| 13 | out_proj + residuo 1 | `out_proj(att_gated)`, `x += core` | `linear_nvfp4_rows(attn_output)`, `x[i]+=core[i]` | CONFERMATO |
| 14 | post_norm | `post_attention_norm` prima di FFN | `rms(x, post_attention_norm, h, ...)` | CONFERMATO struttura; peso EMPIRICO (diretto) |
| 15 | FFN | SiLU-gated: `silu(gate)*up`, poi `down` | `gg=silu(gg)*uu`, `ffn_down` | CONFERMATO |
| 16 | Residuo 2 | `x += FFN(post_norm(x+attn))` | `x[i]+=down[i]` | CONFERMATO |
| 17 | shared_head_norm | RMS su draft hidden; fallback `output_norm` se tensore mancante in llama | `rms(x, f32("blk.64.nextn.shared_head_norm.weight"), xn, ...)` diretto, senza fallback | PARZIALE: tensore presente nel nostro GGUF quindi path primario identico; fallback `output_norm` NON implementato (da aggiungere per robustezza, vedi sez. 4) |
| 18 | LM head | head condivisa col target; fallback `model.output` se `nextn.shared_head_head` mancante | `linear_nvfp4_rows("output.weight", ...)` in 8 shard, nessun fallback | PARZIALE: come sopra; riuso `output.weight` corretto per questo checkpoint; fallback `model.output` NON implementato |
| 19 | KV draft | KV propria del singolo layer MTP, solo storia draft (non condivide KV target) | `MtpState{kcache, vcache, len}`, append post-norm/post-RoPE, `len++` per step, GQA x6 come target | CONFERMATO architettura; semantica rollback EMPIRICA (sez. 3) |
| 20 | Dtype | matmul BF16, norm F32 | draft legge BF16 via `bf2f` + accumulo fp32 CPU; NVFP4 via `deq_row` on-the-fly | CONFERMATO per path CPU; path GPU ancora da validare (M2-done-GPU) |

## 3. Cosa resta empirico (non chiudere senza run)

- E1. Convenzione norme diretta `rms(x)*w` (parametro `plus_one` ignorato in `ModelFwd::rms`).
  Assunta dal resto del modello (GGUF contiene pesi scala diretta, medie 0.5..2.3, nessun cluster ~0;
  `llama.cpp build_norm` fa `rms(x)*w` senza `+1`). Vale per `hnorm`, `enorm`, `attn_q/k_norm`,
  `attn_norm`, `post_attention_norm`, `shared_head_norm`. Se il golden E1/E2 del draft divergesse
  con bias uniforme, rivalutare voce per voce prima di toccare il target (che e` E1==E2 PASS).
- E2. KV solo-draft: `MtpState` separato da `SeqState` target. L'invariante "draft KV contiene solo
  storia draft" e` assunta da `graph_mtp`; va provata dal rollback test (troncare `MtpState.len`
  e rigenerare dal prefisso accettato deve ridare gli stessi logits).
- E3. Fallback `shared_head_norm -> output_norm` e `head -> model.output`: non implementati.
  Non bloccano questo checkpoint (tensori presenti) ma vanno aggiunti prima di dichiarare il wiring
  portabile.

## 4. Matrice validazione

| Voce | Criterio pass/fail | Stato |
|------|--------------------|-------|
| V1 check tensori leggeri (nomi, dtype, shape blk.64: eh_proj [10240,5120] BF16, enorm/hnorm/shared_head_norm [5120] F32; assenza embed/head dedicate; KV 65 blocchi, predict_layers 1) | PASS se tutti presenti con shape/dtype attesi, zero mancanti/spuri | PASS |
| V2 build `architecture` con `mtp_hidden` + `mtp_draft` + `MtpState` | PASS se `cmake --build` OK senza warning nuovi sul path MTP | PASS (compila OK) |
| V3 K=1 accept greedy in convenzione diretta (`draft == argmax(target)` su prompt campione, teacher-forcing) | PASS se match su sequenza fissata; FAIL altrimenti con dump `draft_tok` vs `teacher_tok` | PENDING re-run. Precedente run in convenzione `(1+w)` dava REJECT (`draft 1268 vs teacher 88`): risultato INVALIDATO dal cambio norme, non utilizzabile come evidenza |
| V4 acceptance sweep K=1..5 (prompt campioni, greedy) | PASS se curva accept-rate monotona decrescente riportata con N token e prompt hash; FAIL se NaN/0% ovunque o mismatch di protocollo (verify non single-chunk) | DA FARE |
| V5 rollback test (tronca `SeqState` target + `MtpState` alla posizione accettata, rigenera, confronta logits) | PASS se logits post-rollback == logits teacher-forcing entro tolleranza fp32 fissata nel test; FAIL se divergenza oltre tolleranza | DA FARE |
| V6 sampling distributivo (temperatura/top-k/top-p, `min(1, p_target/p_draft))`) | PASS se distribuzione output == distribuzione target-only entro test statistico fissato (non identita` di sequenza); FAIL se test mock o senza seed fissato | DA FARE |
| V7 fallback head (solo se si implementa) | PASS se con `shared_head_norm` rimosso il draft usa `output_norm`, e con head mancante usa `model.output`, senza crash | DA FARE dopo implementazione |

Regole: niente mock; ogni voce riporta comando, seed/prompt hash, tolleranze e output grezzo.
Il vecchio REJECT K=1 non deve comparire come stato corrente: e` pre-convenzione-diretta.
