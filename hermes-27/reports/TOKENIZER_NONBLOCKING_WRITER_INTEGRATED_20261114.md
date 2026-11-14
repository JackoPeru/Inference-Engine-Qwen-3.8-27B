# Tokenizer pipe write deadline: verified legacy RED / patched GREEN

Date: 2026-11-14. **CPU source + CUDA12.8 compile gate PASS; physical release NO-GO.**

## Actual failure and production correction

Baseline published `serving/server.cpp` SHA256
`7482738e52cb1a560eb0a2c74ae070f6dc8b013bd4d3f742382ed335f9c34fe6`
used a **blocking** `write()` for outbound tokenizer requests while
holding the shared tokenizer mutex. The existing 15 s response-read deadline
started only after a complete write. A Python child that stopped consuming
standard input could fill the pipe and indefinitely wedge HTTP encode,
callback workers and graceful shutdown.

The corrected production serving source SHA256
`90cd825f4fea3990ffe7b61dc5ace3e0eb1a36dca0a8d378cd398a4375d22621`
sets only the parent's child-input FD `O_NONBLOCK`, uses
`poll(POLLOUT)` and a **single monotonic total write deadline**. Production
deadline is 15,000 ms; the short 150 ms override is accepted **only**
under `HERMES_ADMISSION_CPU_ONLY` for isolated tests. Failed
encode/decode can restart the tokenizer and retry **once**; a second
transport failure closes the child and produces HTTP **503**
`server_error/tokenizer_unavailable`, not client HTTP 400.
Streaming token state is not replayed across a child restart.
The existing per-transaction tokenizer mutex and normal request protocol
are preserved.

## Independent Linux CPU-only RED/GREEN

Tested from disposable scratch
`/tmp/h27-tokenizer-deadline-7b72713-20261114`.
The compiled test servers use the actual serving and scheduler sources with
`HERMES_ADMISSION_CPU_ONLY` and a fake PP2 backend. The test verifies
those fake-backend ELF markers before any launch. It runs only an
ephemeral 127.0.0.1 HTTP endpoint and a private fake tokenizer child which
prints a valid ready line, then never reads stdin.

The request JSON exceeds 320 KB and the measured kernel pipe capacity,
but stays under the HTTP 1 MiB body limit. A genuine pre-fix binary was
compiled from the **unmodified** earlier production serving source; it
does not contain the patched timeout override. A separate patched CPU binary
was built from the frozen candidate source.

Target test:
`serving/tests/test_tokenizer_write_deadline_cpu.py`.
Final SHA256:
`6fa44b61358ec12e46ddf5cb665fe6af297adf1e796495edbc51bd5a9b3754d8`.

Final `python3 -B` observed **3/3 PASS**:

1. Patched stuck tokenizer: HTTP **503**, exact
   `server_error/tokenizer_unavailable`, bounded duration, **exactly one**
   retry-child launch, no false success.
2. Two simultaneous blocked encode requests: concurrent SIGTERM exits the
   isolated HTTP server normally within the five-second gate; both callers
   are bounded and neither reports success.
3. Genuine earlier executable: request remains pending beyond 0.85 seconds
   and the baseline's SIGTERM does not drain within 0.45 seconds, proving
   the old stall; the test then cleans up *only its own isolated process
   group*. The fixture now uses SIGTERM for normal teardown and escalates
   to SIGKILL only when that intentionally blocked baseline fails to exit.

CPU binary SHA256:

- Earlier unmodified RED: `724511677dff2708a6841d6f5a5ac55fb615d68b7671f2a8b6a5ed783584e8fd`.
- Patched GREEN: `de91df16ee47749b6139e72fd6b567f1d5e209c7cd11d3c2c464934c1c035da6`.

The existing full offline HTTP suite was also run against the final candidate
source in scratch and passed, including legacy/continuous, multiuser
ready-callback, poison readiness, SSE, Unicode, long-context synthetic
ownership, and tokenizer-parity regressions. The suite entrypoint
`serving/tests/prod_sched_cpu_gate.sh` now includes the new CPU-only
deadline test (two mandatory positive cases, optional baseline SKIP if no
pre-fix executable was supplied).

## CUDA build evidence and remaining gates

Both genuine `hermes_server` and `hermes_generate` compiled and
device-linked under CUDA 12.8 Release in the same isolated project
snapshot; **no GPU kernel was executed**.

- CUDA-linked server SHA256:
  `16302c4afc05caab2377252a32386ae98fbe7dcae9054015d421fbefcbe04383`.
- CUDA-linked generator SHA256:
  `f37fff8704a404eef3f81b5bfdb56c58133fcfd323f5d613e759cf8abafda0db`.

No live Tabby/GPU Manager service was stopped, no installed HERMES runtime
was modified, no CUDA inference was launched, and no production cutover was
performed. The gate does not cover arbitrary kernel-driver hangs, physical
multiuser N1/N2/N4/N8 throughput, actual T8192+ continuation, or independent
12/12 full-vocabulary numerical comparison. External antirollback issuer,
all ingress drain, both GPU UUID/PID/starttime exclusive ownership and
real two-token post-convolution llama captures remain **BLOCKED**.
