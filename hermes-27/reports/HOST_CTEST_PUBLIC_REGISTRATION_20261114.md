# HERMES-27: CTest registration of real CPU-only scheduler/CLI witnesses

**2026-11-14. CPU-only; model inference and production access NOT RUN.**

## Why this integration matters

The new ready-completion scheduler, terminal exception hardening and
`hermes_generate` evidence-integrity fixes had been verified in **isolated
standalone builds**, but several focused regressions were not yet
automatically addressable by standard CMake/CTest. These checks guard
against future source regressions without reserving either GPU.

`CMakeLists.txt` now defines separate host-only targets for the real
`ContinuousScheduler` source under `HERMES_ADMISSION_CPU_ONLY`, by
compiling a **dedicated C++ translation unit**
`distributed/tests/pp2_batch_cpu_only.cpp` which includes the actual
published `pp2_batch.cu`. The production CUDA target's language,
compile definitions and linking remain unchanged.

Additional host executables exercise `hermes_generate.cpp` linked against
an explicitly synthetic PP2 mock, and the unique/private logits dumper.
No hosted test invokes the real CUDA uploader. `--verify-prefill`
tests deliberately refuse all genuine decoder allocations.

## Source-anchored RED and corrected CMake dependency

The first immutable-source integration attempt over published Git commit
`5c74e211e19a7b4a6f3d05a7a62b63430aedb718`, with the candidate
CMake file alone, failed configuration because the referenced
`distributed/tests/pp2_batch_cpu_only.cpp` **was absent from Git**.
This is a genuine missing CMake source dependency, not a kernel error.

The new wrapper was added as an explicit separate file, with
SHA-256 `7b74bc5910fd85dbc0771dc73c6c61b5ecc2901df20d896e11fe2d600fba8d5e`.
The completed CMake source has SHA-256
`867a306d4f919b0c144305920b020a7bd3011f9712cad4c15a4f464433930eb1`.
The corrected pair was overlaid on a **second fresh archive** of the
same immutable commit, never in the production Git working tree while
testing.

## Executed Ubuntu CUDA 12.8.93 CMake + CPU results

Isolated snapshot and build directory:
`/tmp/h27-cmake-host-targets-green-candidate-20261114`.
CMake configuration succeeded with the CUDA compiler set to
`/usr/local/cuda-12.8/bin/nvcc`,
`HERMES_REGISTER_GPU_TESTS=OFF`, and
`HERMES_REGISTER_HEAVY_CPU_TESTS=OFF`.
All **seven host-only targets** compiled and linked successfully.
The following exact source-scoped CTest selection was run, exit 0:

| Host CTest | Observed |
| --- | --- |
| `host_test_ready_completion_optin_scheduler` | PASS |
| `host_test_ready_completion_fault_hardening` | PASS |
| `host_test_ready_observability_optin_scheduler` | PASS |
| `host_test_ready_terminal_exception_id_copy` | PASS |
| `host_test_ready_fairness_credits_cpu` | PASS |
| `generate_prefill_shape_host_cpu` | PASS |
| `generate_logits_dump_atomic_host_cpu` | PASS |

**CTest: 7/7 PASS, 0 failures, 0.74 seconds wall time.**
The tests validate original host scheduler source via CPU injection, not
an independently executed CUDA scheduler.

The new CMake hooks also register existing source-only checks where
appropriate. No duplicate unchanged full regressions were executed.
The existing GPU CTests remain behind the same strict opt-in switch.

## Gate

This closes a real test-registration/build integration gap and keeps
the repaired proof-contract behavior in future host CI. It does not
establish real GGUF full-vocabulary correctness, CUDA parity, GPU
throughput, sliding-window continuation or all-ingress ownership.
**HERMES-27 release remains NO-GO.**
