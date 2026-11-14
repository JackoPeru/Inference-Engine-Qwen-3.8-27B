# HTTP async callback opt-in: explicit stop and cancellation regressions

Date: 2026-11-14. **CPU-only positive witness, production NO-GO.**

## Scope

Test-only extension of the real-server/CPU PP2 mock integration published in
`3daac943c0e7ef8f46c5b24c94181c7a8ac1fbbf`; no production serving,
scheduler or CUDA code changed. `serving/tests/prod_sched_ready_callback_test.py`
now includes two additional deterministic HTTP paths:

1. A token-spanning `stop="good night"` with SSE, validates partial
   stop-prefix buffering, terminal reason `stop`, exactly one `[DONE]`,
   bounded completion usage, and exactly one PP2 state owner/free.
2. A long SSE generation cancelled with `POST /v1/cancel`, validates HTTP
   202, terminal reason `cancelled`, exactly one `[DONE]`, one usage
   emission, and exactly one backend state release.

Changed files:

- `serving/tests/prod_sched_ready_callback_test.py`, final SHA256
  `e9295178d770970de66606a9160fa7779f4b038767bf973edf662fa5ab649f44`.
- `serving/tests/prod_sched_cpu_gate.sh`: updates the expected ready
  callback test count to 7. The other existing gate invocations are unchanged.

## Independent Ubuntu receipt

- Reused previously compiled **unchanged** CPU mock server binary from
  `/tmp/h27-callback-http-integrated-3ca223f-20261114/host-server/build/hermes_server`
  (source `serving/server.cpp` published at `3daac943`, not the physical
  production service).
- Copied only the frozen test into the isolated host fixture.
- Ran Python from `/opt/hermes/runtimes/exllamav3-venv/bin/python -B`
  using the real project tokenizer, fake PP2 and actual cpp-httplib loopback.
- **7/7 HTTP test cases PASS**, including previous strict startup gates,
  N4 parallel nonstream, SSE Unicode/usage, malformed terminal allocation
  recovery, callback exception next-request recovery and both new stop/cancel
  cases. `Ran 7 tests in 5.485s; OK`.
- No need to repeat unchanged CUDA12.8 build or prior full HTTP regression:
  neither the production `.cu`/header nor `serving/server.cpp` changed.

**Limitations:** synthetic CPU backend does not model CUDA stream timing or
physical GPU OOM/cancellation latency. No real N1/N2/N4/N8 throughput,
248320-logit independent parity, long-context GPU continuation, privileged
ingress drain or external lease issuer was tested. Default callback dispatch
mode stays OFF. Production remains NO-GO.
