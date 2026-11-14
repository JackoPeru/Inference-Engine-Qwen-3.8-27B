# Two additional public CPU-only CTest gates

Date: 2026-11-14. **HERMES-27 physical release: NO-GO**.

## Scope and source

Baseline Git commit: `7b727136ed2894e29cb4025a5a181d5243fb60b6`.
The only production build-configuration change is `CMakeLists.txt`
(candidate SHA256 `5c03e054dabf43b904dcef5f6de4ef823a66707a385a25efcf86430c2b534e72`).
There are **no** changes to CUDA source, production serving, the GGUF,
the tracer patcher, the ingress verifier, or the external lease runtime.

On a clean immutable Git archive, two previously hand-run CPU-only
contracts are registered using `Python3 -B -m unittest -v`:

- `host_ingress_attestation_staging_cpu` runs
  `ops.tests.test_physical_ingress_attestation_contract_20261114`.
  This is a **synthetic** signed six-route/zero-inflight/dual-GPU
  owner/one-time nonce contract, not a real privileged attestation.
- `host_postffn_scratch_source_cpu` runs
  `experiments.tests.test_layer0_postffn_scratch_capture_source_cpu`.
  This checks the SHA-pinned **scratch-only** layer0 post-FFN
  instrumentation and fabricated negative sidecars, not genuine
  HERMES or llama.cpp tensors.

Both tests have `WORKING_DIRECTORY=${CMAKE_SOURCE_DIR}`, explicit
120-second timeouts and `cpu-only;source-contract;no-gpu` labels.

## Reproducible Ubuntu result

Scratch: `/tmp/h27-two-new-ctest-gates-20261114`. Used
`git archive` of the baseline commit and overlaid only the
SHA-verified `CMakeLists.txt`. CMake Release configuration:
`/usr/local/cuda-12.8/bin/nvcc`,
`HERMES_REQUIRE_SERVER=ON`,
`HERMES_REGISTER_GPU_TESTS=OFF`.
Both newly registered tests passed on the real checked-in source:

- `host_ingress_attestation_staging_cpu`: **1/1 PASS** (0.10 s).
- `host_postffn_scratch_source_cpu`: **1/1 PASS** (0.11 s).

No CUDA kernel, GGUF inference, physical GPU ownership, HTTP/SSE
service, network ingress, or production cutover was exercised.
In particular neither a trusted physical issuer nor a downstream
independent llama oracle exists here. The 12/12 248320-logit
numeric release gate and multiuser N1/N2/N4/N8 physical
benchmark gate remain **BLOCKED / NO-GO**.
