# HERMES-27 — long-context window CPU witness after CUDA prefill-pool merge

**14 November 2026. CPU reference/structural tests PASS; no CUDA inference or long-context production admission.**

The new, independently implemented deque/window oracle and source mutation
tests were rerun against a **fresh immutable Git archive** of
`727d9cdf7a418e3888fc4d33730a8779a9108e07`, including the newly
integrated request-local CUDA prefill-pool patch
(`distributed/pp2_fwd.cu` SHA-256
`87364d1f21062a6dd2c009ca15efe16cde7d92636a35a0c576b08f855edbf71e`).
The unchanged worker test SHA-256 is
`f6272d747b373457949d469fcd003da42f0bad8e20e5f4cc74466d9f9cc7de29`.

The isolated Ubuntu scratch is
`/tmp/h27-window-eq-published727-cpu-20261114`.
Executed command from its copied source tree:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover \
  -s distributed/tests -p test_sliding_window_equivalence_cpu.py -v
```

**Observed: 8/8 tests PASS, exit 0, 7.221 s.** The independent math
tests cover 24 query heads/4 KV groups, chronological eviction at
8191/8192/8193/16384, RoPE with absolute positions rather than modulo
slot positions, a reference 8192-token deque versus ring softmax,
48 persistent toy recurrent/conv states, GPU0/GPU1 logical layer
separation and a fail-closed partial-commit poison model.
Source mutations deliberately violate indexing, RoPE and persistence
contracts; the tests detect them.

This test demonstrates a bounded **window** reference; it explicitly
shows that results need not match full-history attention after eviction.
Existing HTTP serving already exposes a strict, single-user,
legacy-only `--sliding-kv` opt-in capable of advancing to absolute
positions beyond 8192; the normal CLI does not. Neither entrypoint
has passed an actual same-GGUF CUDA-vs-independent sliding-window
comparison, so **long-context release gate remains BLOCKED**.

No GGUF execution, GPU allocation, service request, ingress change,
deployment action or actual benchmark occurred.
