# HERMES-27 VRAM budget per-GPU (2x 16GiB, NO unified pool)

- Per-GPU usable: 16311 MiB (nvidia-smi) / 15849 MiB (torch). Budget rigoroso <16GiB con margine runtime.
- P2P: NO (can_device_access_peer 0->1 False, 1->0 False, topo PHB). TP richiede path host-pinned, PP favorito.
- PCIe: Gen5 max, idle Gen1; H2D/D2H misurati ~26 GB/s su GPU1 (pinned, 5x256MB). GPU0 piena, non testabile senza fermare servizi.

## Modello target (da verificare su GGUF reale, download in corso 19.69GB)
- Arch qwen3_5 verificata su EXL3 config: hidden 5120, FFN 17408, 64 layer (48 Gated DeltaNet + 16 full_attn ogni 4), vocab 248320, ctx 262144, head 24x256? KV 4 head, MTP 1 layer.
- GGUF target: NVFP4 (36B/64elem = 0.5625 B/elem) per matmul + BF16 per embed/head/norm/conv/state.

### Stima pesi (upper bound, da confermare con gguf_dump)
- Param totali ~27B. Se 90% NVFP4 (~24B * 0.5625B) ~13.5GB + 10% BF16 (~2.7B*2B)=5.4GB -> totale ~18.9GB combaciante con 18.34GiB file (header+KV esclusi).
- TP2 split: ~9.5GB pesi per GPU. PP2 split 32+32 layer: ~9.5GB per GPU. Entrambi <16GB, OK.
- KV full-attn: 16 layer * 4 KV head * 256 dim? head_dim 256? config: head_dim 256, num_kv 4 -> per token per layer KV = 2*4*256*2B(FP8?1B) = 4KB FP16 / 2KB FP8. 16 layer -> 64KB/tok FP16, 32KB FP8. 32K ctx * 32KB = 1GB per seq FP8. 8 seq concorrenti 32K = 8GB -> troppo. Serve paged + cap + backpressure. Target iniziale: max 8K-32K per seq con FP8, pool 2-3GB per GPU.
- Gated DeltaNet state: per layer lineare: state ~ (value_dim 128 * 48 head?) + conv1d state (kernel 4). Da quantificare su checkpoint: in_proj + A_log + dt_bias. Stima <100MB per seq 32K? Da misurare, stati separati da KV, copy-on-write per prefix caching solo se matematicamente identici.
- MTP draft head: 1 layer riutilizzabile, BF16, <500MB.
- Runtime/CUDA/graph: riserva 96MB per GPU (come autosplit_reserve attuale) + 1GB margine.

### Budget per GPU (TP2 esempio, PP simile)
- Pesi: 9.5GB
- KV pool FP8: 2.5GB
- DeltaNet+conv state pool: 0.5GB
- Attivazioni/transient prefill chunk 8K: 1.0GB
- MTP draft + graph: 0.5GB
- Riserva CUDA/margine: 1.0GB
- Totale: ~15.0GB < 16GB OK con backpressure. 262K nativi NON preallocare: obiettivo, non dimensione iniziale.

Azioni: gguf_dump su file reale -> conteggi esatti per tipo/tensore; poi memory pool arena per device con limiti configurabili.
