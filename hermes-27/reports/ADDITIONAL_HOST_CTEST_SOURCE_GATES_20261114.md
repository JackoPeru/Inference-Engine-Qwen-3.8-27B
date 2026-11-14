# Additional public CPU-only CTest source and staging contracts

Date: 2026-11-14. Release verdict **NO-GO**.

## Purpose

The prefill stream-narrowing dependency checks, async decode launch contracts
and external antirollback staging model were previously executable manually.
Register them in public CTest so subsequent integrated builds detect source
regressions without touching CUDA devices or asserting production readiness.

## Scope and provenance

- Source Git baseline:
  `3daac943c0e7ef8f46c5b24c94181c7a8ac1fbbf`.
- Only `CMakeLists.txt` modified; frozen candidate SHA256
  `de3cd1584170cb886cd2bd2c076e54a814cff61477738ff56416d5f43534d559`.
- Clean immutable Git archive with this one-file overlay in
  `/tmp/h27-public-host-gates-3daac943-20261114`.
- Configured CMake Release on Ubuntu using
  `/usr/local/cuda-12.8/bin/nvcc`,
  `HERMES_REQUIRE_SERVER=ON`,
  `HERMES_REGISTER_GPU_TESTS=OFF`.
- No kernel, model inference, live server or GPU context was run.

## New public test registrations

| CTest | Actual offline test | Ubuntu result |
| --- | --- | --- |
| `host_prefill_chunk_stream_barrier_cpu` | `distributed/tests/test_prefill_chunk_stream_barrier_cpu.py` | PASS 1/1 |
| `host_async_stream_contract_cpu` | `distributed/test_async_stream_contract.py` | PASS 1/1 |
| `host_external_authority_staging_cpu` | `python -B -m unittest -v ops.tests.test_external_monotonic_proof_staging` | PASS 1/1 |

Tests set `WORKING_DIRECTORY=${CMAKE_SOURCE_DIR}`, explicit 120-second
timeouts and labels `cpu-only;source-contract;no-gpu`.

The lease tests use **only synthetic** independent-issuer/store fixtures
and their positive cases always return
`production_admissible=False`. CTest success is **not** evidence of a
privileged antirollback authority, physical ingress drain, or CUDA owner.
The CUDA/source tests likewise do not prove full-model numerical parity or
measured N1/N2/N4/N8 performance. Physical deployment remains blocked.
