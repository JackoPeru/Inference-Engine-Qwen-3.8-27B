# Tokenizer stalled-writer CPU witness — isolated host-only regression

Date: 2026-11-14. **Status: test harness authored and source-reviewed; execution against a Linux CPU mock build PENDING. Production/GPU deployment: NO-GO.** Neither a legacy CPU mock ELF nor a rebuilt patched CPU mock ELF was available in the shared Windows workspace; Linux Python/g++ and a WSL distribution are absent. SSH, GPU, live endpoints, services and Git were excluded by task scope. Do not interpret the authored test as a measured passing RED/GREEN runtime witness.

## Source finding, exact causal chain, and prime-owned correction

At frozen published `3daac943`, `serving/server.cpp` lines `259–266` implemented a blocking `write()` loop in `TokenizerBridge::write_all()`. `TokenizerBridge::call()` lines `292–299` held the one tokenizer mutex across that write, while `read_line()` lines `269–288` timed out only **after** the complete request had been written. With a tokenizer child that prints a valid `ready` response and stops consuming stdin, a JSON request larger than the kernel pipe capacity blocks `write()` indefinitely. A second callback, a concurrent HTTP encode or graceful shutdown can wait behind the same held mutex; in callback-delivery mode, callback workers cannot drain. The shutdown path `distributed/pp2_batch.cu:622–638` joins callbacks and therefore inherits the hang. The source guarantees the unbounded possibility; no physical original-binary hang has been executed during this witness assignment.

During this witness work, prime independently changed `serving/server.cpp` (read-only here), introducing an `O_NONBLOCK` tokenizer input FD, `poll(POLLOUT)` with one monotonic deadline, `HERMES_TEST_TOKENIZER_WRITE_TIMEOUT_MS` for `HERMES_ADMISSION_CPU_ONLY` **only** (strict integer range `25..1000`, normal production fixed at `15000` ms), and a `TokenizerTransportFailure` mapped to HTTP **503** `tokenizer_unavailable` for infrastructure failures while retaining HTTP 400 for invalid user templates. Source confirmed at lines `248–305`, `345–381`, `1769–1777` during inspection. Prime owns that source and its continued verification; the witness does not edit or publish it. The same `TokenBridge::call()` still correctly serializes each tokenizer pipe request/response pair under `mutex_`.

## Created exclusively inside assigned scope

New `serving/tests/test_tokenizer_write_deadline_cpu.py`, SHA256 **`cb2239aff8864de1ffaa2c25063ee58971ce3fd85e4f8ea314f9d308e35ce615`**. No prior tests, shell scripts, CMake, PP2, serving source or Git were modified. The test is standalone and requires only Python 3 stdlib. It uses the **actual CPU mock binary built from `server.cpp`, `pp2_batch.cu`, `prod_sched_mock_pp2.cpp`** and rejects binaries that lack the fake backend's `FAKE_TRACE_FILE`, `FAKE_IDS`, and `FAKE_INIT_FAIL` ELF markers. It rejects a fixed candidate lacking the test-only timeout marker and rejects a purported baseline that contains it.

Each case creates a temporary private executable tokenizer child using the public `--python` CLI option; this route is valid because the server executes `cfg_.python` directly and passes the tokenizer helper as its first argument. The stub prints a syntactically valid `{"ready": true, "incremental": false}` and then **never reads stdin**. Test model/tokenizer/GGUF-dump arguments are local disposable existence-only fixtures (the fake PP2 upload does not read real GGUF). The test probes `fcntl(F_GETPIPE_SZ)` on a real Linux pipe and constructs a JSON HTTP body exceeding both **200 KB** and the measured pipe capacity, yet staying below the server's **1 MiB** request limit. If the host's pipe capacity is too large to exceed under that limit, the test **SKIPs** explicitly rather than claiming saturation.

HTTP is forced to `127.0.0.1`, port `0`, with proxies disabled. Every helper/server is isolated in its own fresh process session; cleanup kills only that process group to prevent orphaned intentionally hung children. No public network, CUDA runtime, GPU allocation or actual model is involved. The CPU-only timeout override is `150` ms. Production cannot read the test override because the source encloses it in `#ifdef HERMES_ADMISSION_CPU_ONLY`.

The three meaningful cases are:

1. **Patched regression (mandatory):** a large encode causes the blocked stdin writer to time out and restart the fake tokenizer at most once; the HTTP caller receives a bounded **5xx JSON** error mentioning tokenizer, and a startup marker confirms the retry child was spawned. The test rejects HTTP 400 because loss of the tokenizer bridge is infrastructure failure.
2. **Concurrent shutdown regression (mandatory):** two large simultaneous HTTP encode requests are observed by `hermes_requests_total`; send `SIGTERM` while they contend on the same tokenizer mutex. The isolated server must exit with code 0 in **5 s**, and neither request may report success. Any connection cut by shutdown must be bounded. This tests the actual HTTP listener join path, not just the scheduler's mock callback workers.
3. **Legacy RED witness (optional, requires a physical pre-fix CPU mock ELF):** the same large request to a binary without the `HERMES_TEST_TOKENIZER_WRITE_TIMEOUT_MS` source remains pending for **at least 0.85 s**, then `SIGTERM` fails to drain in another **0.45 s**. The test forcibly kills its own test process group and releases the HTTP caller. It does not pretend a post-fix binary can serve as this old baseline.

### Execution gate (Linux CPU-only, run by the owner with a fresh isolated CPU mock)

```bash
python3 -B serving/tests/test_tokenizer_write_deadline_cpu.py \
  --server /tmp/hermes27-prod-sched-cpu/build/hermes_server

# Optional ONLY if an actual unmodified old CPU mock binary is available:
python3 -B serving/tests/test_tokenizer_write_deadline_cpu.py \
  --server /tmp/hermes27-prod-sched-cpu/build/hermes_server \
  --baseline-server /tmp/h27-old-cpu-only/build/hermes_server
```

The CPU server must be rebuilt from **the current prime-owned patched source** into disposable scratch prior to running this command, using the existing CPU-only fixture sources and compiler arguments in `serving/tests/prod_sched_cpu_gate.sh`. That existing script was read, **not edited or executed**. Avoid pointing to old binaries simply because their filename matches. Do not run a real PP2 CUDA server as the harness refuses non-fake ELF markers.

## Freeze, verdict and remaining evidence

| Evidence | Result |
|---|---|
| Actual source audited at published `3daac943` | **P1 confirmed from blocking write and global mutex** |
| Prime-owned candidate source O_NONBLOCK + bounded total writer deadline | **Inspected** in working source; actual HTTP gate not run by this worker |
| New standalone CPU witness created | **DONE**, SHA256 above |
| Actual legacy RED HTTP hang | **NOT RUN**: no original CPU-only ELF available locally |
| Actual post-fix 5xx HTTP and 2-client graceful SIGTERM | **NOT RUN**: Windows lacks Linux/Python/g++/WSL; remote commands forbidden |
| Real CUDA, GPU, model numerical parity/throughput, service restart | **NOT RUN / NO-GO** |

**No acceptance claim until** the new witness executes on the Linux CPU build and the prime verifies the source/binary identities, actual exit codes and elapsed bounds. This report covers the tokenizer write-stall P1 only; separate scheduler-poisoned readiness P1 has prime-owned source/test coverage.
