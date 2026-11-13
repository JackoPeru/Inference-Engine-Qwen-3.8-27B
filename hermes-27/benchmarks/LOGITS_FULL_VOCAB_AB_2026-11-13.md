# Full-vocabulary logit numerical A/B, 2026-11-13

## Scope and verdict

Same checkpoint, same 11 input token IDs, three autoregressive greedy
steps, **248320 float32 logits per step**. The following is a *numerical
regression investigation*, not a quality certification. It does not
compare against independent llama.cpp logits or cover long contexts.

The variants keep all 3 greedy token IDs equal: `271,248068,198`, and
all top-10 sets overlap, but default strict full-logit parity fails for
the GEMV variants. Do **not** promote an optimized GEMV as production
default until a real reference and longer-sequence gate are satisfied.

## Reproduction

Checkpoint:
`/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf`

Input IDs:
`6205,161421,303,220,17,1374,9911,1147,27053,6059,13`

`hermes_generate --model <GGUF> --prompt-ids <IDS> --max-new-tokens 3
--dump-logits-dir <EXISTING_EMPTY_DIR>` writes one `prompt_0_step_N.f32`
per step, always before token selection. Run with explicit environment:

| Trace directory | GEMV | BF16 head | Skip sync |
| --- | --- | --- | --- |
| `/tmp/h27_dump_scalar` | scalar | scalar | no |
| `/tmp/h27_dump_warp4` | warp4 | scalar | no |
| `/tmp/h27_dump_warp4head` | warp4 | warp4 | yes |
| `/tmp/h27_dump_packed_fp64` | packed4 | scalar | no |
| `/tmp/h27_dump_reference` | packed4 | warp4 | yes |

Use `tools/compare_fp32_logits.py BASELINE CANDIDATE
--max-abs 0.002 --max-rmse 0.0005 --report-softmax`.
Do not reuse directories containing stale traces.

## Measurements

| A/B | Worst max abs | Worst RMSE | Full-logit gate |
| --- | ---: | ---: | --- |
| scalar vs warp4, same head | 0.07307661 | 0.014054123 | FAIL 3/3 |
| warp4 vs packed4, same head | 0.08813429 | 0.015980206 | FAIL 3/3 |
| warp4 head scalar vs head warp4 + skip | 0.000002861 | 0.000000358 | PASS 3/3 |
| packed4 head scalar vs head warp4 + skip | 0.000002384 | 0.000000358 | PASS 3/3 |

For scalar vs warp4, per generation step:

| Step | Max abs | RMSE | KL(softmax scalar || warp4) | Top-10 overlap | Top-1 margin, scalar |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.07307661 | 0.011702882 | 1.876424e-6 | 10/10 | 5.794383 |
| 1 | 0.06063616 | 0.011288044 | 8.907097e-10 | 10/10 | 11.949560 |
| 2 | 0.07097769 | 0.014054123 | 1.259366e-5 | 10/10 | 1.472837 |

For warp4 vs packed4 the KL values are 3.73649e-6,
5.73674e-9, and 1.04255e-5 with top-10 overlap 10/10.
These small distributional differences explain why the *short* greedy
outputs agree; they **do not** prove general model-quality equivalence
or distinguish altered reduction order from a bug. Full-vocabulary
absolute/RMSE parity still fails the strict comparator.

## Next gates

1. Compare each variant against independent llama.cpp from the *same
   GGUF*, with checkpoint checksum and shared token IDs.
2. Audit per-projection reduction/accumulation on *real* model inputs;
   microbench agreement on random small matrices is insufficient.
3. Test nontrivial mixed-length requests, Unicode/EOS and extended
   contexts, including the first generation divergence if any.
4. Treat `--report-softmax` diagnostics as supplementary only.
