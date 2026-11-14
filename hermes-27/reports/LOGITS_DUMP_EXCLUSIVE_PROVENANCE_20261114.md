# HERMES-27: exclusive, private logits dump for numerical evidence

**2026-11-14, CPU/offline and CUDA 12.8 compile-only.** This is a
diagnostic CLI evidence-integrity improvement, not model correctness,
throughput optimization or permission for GPU inference.

## Defect reproduced before modification

`tools/hermes_generate.cpp::dump_logits()` formerly opened
`prompt_N_step_M.f32.partial` with `fopen(...,"wb")` and published via
`std::rename(partial, final)`. POSIX rename **replaces an existing final
path** silently, even if it is the symlink of a previous file. A repeated
capture could therefore destroy independently hashed numeric evidence.
The partial file could also be truncated after a prior interrupted run.

A standalone C++ CPU-only harness compiled the *original* published
`hermes_generate.cpp` (SHA-256
`b03b04b4f679838c685c62812fa8ef6068daa22d2220afd6cf83fca14e056543`)
as an included translation unit while discarding the unused GPU entrypoint
at link time. It used only synthetic `248320`-element FP32 vectors and a
private `/tmp` directory; no GGUF, CUDA, network or service access.
Actual **RED**: exit 1, first unique dump succeeded and the next identical
target invocation was wrongly accepted, triggering:
`duplicate final must fail instead of silently replacing evidence`.

## Corrections to actual CLI

1. Create a private partial file exclusively with POSIX
   `O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC` and mode `0600`;
   Windows uses exclusive `_open`. No preexisting partial/symlink is
   truncated or adopted.
2. After the complete FP32 write and close, use an atomic, same-directory
   hard-link creation for the final name. If the final name already exists
   or the filesystem does not support no-replace hard-link publication,
   return an error and **do not overwrite**. Remove only the new partial
   after successful publication. Fail closed if cleanup fails.
3. Reject a noncanonical or symlink-aliased dump directory during option
   parsing. On Linux, require an existing directory owned by the process
   with no group/other permissions (`0700`). This prevents another user
   from substituting a partial file while evidence is published. These
   constraints apply only to explicit `--dump-logits-dir`, not ordinary
   generation.
4. Preserve the `248320`-logit shape, full FP32 bytes, greedy selection,
   normal output JSON and decode state management.

## Executed tests

The revised source, targeted test and CLI argument tests were transferred
to Ubuntu isolated scratch with SHA-256 checks. The new
`tools/tests/test_generate_logits_dump_atomic_cpu.cpp` was compiled
as C++20 with GCC `-Wall -Wextra -Werror`, ASAN and UBSAN:
**21/21 assertions PASS**, including full byte count, private file mode,
one-time final publication, no duplicate overwrite, preservation of a
preexisting partial and preservation of a preexisting final symlink.
No model runtime was executed.

An unchanged-CUDA-source, newly configured CUDA **12.8.93** release
build and device link of `hermes_generate` succeeded from the coherent
Git checkout, after staging the CLI patch. The resulting executable
SHA-256 was
`140e90db7ed54e245fac276d042eccad89184d6dce073a603696c04b5cdec642`.
The actual compiled CLI passed **7/7 CPU argument tests**:
the existing five strict startup cases plus refusal of a symlink-aliased
dump directory and of a group/world-readable directory **before GGUF
upload**. No hidden activations were produced or uploaded.

Remaining caveats: same-directory hard-links can be unsupported on unusual
filesystems, in which case dumps fail closed. Crash durability and
multi-process ownership of a malicious directory cannot be inferred from
these unit tests; production use requires a private scratch directory and
separate validated artifact provenance. The real independent full-vocabulary
gate stays FAIL 12/12. Live services and both GPUs were untouched.
