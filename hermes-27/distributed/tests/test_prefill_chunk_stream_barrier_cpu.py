#!/usr/bin/env python3
"""CPU-only dependency/negative oracle for PP2 pooled prefill stream barriers.

The model schedules independent per-device default and nonblocking streams.
It checks scratch lifetime, state ownership and deterministic dependences;
source checks ensure the real CUDA route matches those assumptions. This is
NOT a CUDA execution, real-GGUF parity test, or measured speedup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import re

SOURCE = (Path(__file__).resolve().parents[1] / "pp2_fwd.cu").read_text(encoding="utf-8")


class GateFailure(AssertionError):
    pass


def check(ok: bool, message: str) -> None:
    if not ok:
        raise GateFailure(message)


def between(src: str, before: str, after: str) -> str:
    check(before in src and after in src, f"missing source boundaries {before}/{after}")
    return src.split(before, 1)[1].split(after, 1)[0]


LAYER_STARTS = (
    ("static bool run_gdn_layer(", "static bool run_full_layer("),
    ("static bool run_full_layer(", "// ---------- API ----------"),
)


def audit_source(src: str) -> int:
    """Require all producers to be on stream 0 before each pooled sync."""
    total = 0
    for start, end in LAYER_STARTS:
        body = between(src, start, end)
        # Exactly one guarded barrier, after the final per-token FFN update.
        gates = re.findall(
            r"cudaError_t\s+layer_result\s*=\s*pool\s*\?\s*"
            r"cudaStreamSynchronize\(0\)\s*:\s*cudaDeviceSynchronize\(\)\s*;",
            body,
        )
        check(len(gates) == 1, f"{start} must synchronize only pooled stream 0")
        check(body.index("gemv_batch(d_hidb,") < body.index("cudaError_t layer_result="),
              f"{start} sync must follow FFN projection")
        check(body.index("add_k<<<", body.index("gemv_batch(d_hidb,")) <
              body.index("cudaError_t layer_result="),
              f"{start} must drain final residual update")
        check("if(layer_result!=cudaSuccess)" in body and "return false;" in body,
              f"{start} must fail-close on sync failure")
        check("PrefillLayerScratch scratch(pool);" in body,
              f"{start} must use same pooled ownership as barrier")
        launches = re.findall(r"<<<(.*?)>>>", body, re.DOTALL)
        check(len(launches) >= 10, f"{start} unexpectedly lost kernel launches")
        check(all("stream" not in launch for launch in launches),
              f"{start} spawned work in an untracked stream")
        check("cudaMemcpyAsync(" not in body and "cudaStreamCreate" not in body,
              f"{start} includes async work not ordered with stream 0")
        for call in re.findall(r"gemv_batch\([^;]*?\);", body, re.DOTALL):
            check(re.search(r",T\s*\);\s*$", call) is not None,
                  f"{start} GEMV must use default stream 0")
        if "run_gdn_layer" in start:
            calls = re.findall(r"gdn_step_cuda\([^;]*?\);", body, re.DOTALL)
            check(len(calls) == 1 and calls[0].strip().endswith("HV,KHD,VHD);"),
                  "GDN projection launched on unexpected stream")
            check("d_S=seed->S[li]" in body and "seed->CS[li]" in body,
                  "GDN state/conv tail lost request ownership")
        else:
            check("cudaMemcpy(seed->Kc[li]" in body and
                  "cudaMemcpy(seed->Vc[li]" in body,
                  "KV must be enqueued on the same default stream")
            check("cudaMemcpy(qq.data()" in body and
                  "cudaMemcpy(d_G,G.data()" in body,
                  "full attention host-gate copies must remain synchronous")
            check(body.index("cudaError_t layer_result=") <
                  body.index("if(seed) seed->klen[li]=pos0+T;"),
                  "KV host length committed before stream completion")
        total += 1 + len(launches)

    forward = between(src, "static std::vector<float> prefill_impl(",
                      "std::vector<float> prefill(")
    check("run_full_layer(li,d_x0,T,err,seed,pos0,pool0)" in forward and
          "run_gdn_layer(li,d_x0,T,err,seed,pos0,pool0)" in forward,
          "GPU0 layer source must receive GPU0 request pool")
    check("run_full_layer(li,d_x1,T,err,seed,pos0,pool1)" in forward and
          "run_gdn_layer(li,d_x1,T,err,seed,pos0,pool1)" in forward,
          "GPU1 layer source must receive GPU1 request pool")
    check(forward.index("allocations.alloc(0,d_ids") <
          forward.index("for(int li=0;li<=SPLIT_LAST0;++li)"),
          "GPU0 selection must precede GPU0 default-stream work")
    check(forward.index("allocations.alloc(1,d_x1") <
          forward.index("for(int li=SPLIT_LAST0+1;li<NL;++li)"),
          "GPU1 selection must precede GPU1 default-stream work")
    check("cudaSetDevice(0);" in forward and "cudaSetDevice(1);" in forward and
          "copy_err=cudaMemcpy(stage,d_x0" in forward and
          "copy_err=cudaMemcpy(d_x1,stage" in forward,
          "cross-device handoff must preserve synchronous pinned copies")
    check("cudaDeviceSynchronize();" in forward,
          "embedding/head fallback syncs not in scoped barrier change")
    check("return prefill_impl(ids,st,err,h_last_out,nullptr,false);" in src,
          "non-pooled raw prefill must keep prior behavior")

    chunk = between(src, "std::vector<float> prefill_chunks_into_decode_state(",
                    "std::vector<float> decode_step(") if False else src.split(
                        "std::vector<float> prefill_chunks_into_decode_state(", 1)[1]
    check("std::unique_lock<std::mutex>(ctx->step_mutex,std::try_to_lock)" in chunk,
          "request state must reject concurrent decoder")
    check("PrefillChunkScratchPool gpu0_pool(0),gpu1_pool(1);" in chunk and
          "(int)offset,true,&gpu0_pool,&gpu1_pool);" in chunk,
          "chunk loop must own both physical device pools")
    begin = chunk.index("PrefillChunkScratchPool gpu0_pool(0),gpu1_pool(1);")
    end = chunk.index("return fail(); // Both pool destructors ran while state_lock was held.")
    check("return fail();" not in chunk[begin:end],
          "fail must only execute after both pools leave their lexical scope")
    check("if(!complete)break;" in chunk[begin:end] and
          "}catch(const std::exception& e)" in chunk[end:],
          "failure/exception cleanup must unwind pools before releasing lock/state")
    check("chunked prefill BF16 disabled: full-model numeric gate failed" in chunk,
          "BF16 cuBLAS path is not validated for chunked stream narrowing")
    check("chunked prefill rejects experimental packed4 GEMV fail-stop path" in chunk,
          "packed4 must remain rejected")
    check("chunked prefill incompatible with HERMES_L26 debug" in chunk,
          "debug host copy path must remain rejected")
    check("cudaStreamCreateWithFlags(&ctx->stream0,cudaStreamNonBlocking)" in src and
          "cudaStreamCreateWithFlags(&ctx->stream1,cudaStreamNonBlocking)" in src,
          "other-request async decode streams must remain nonblocking")
    check("cudaGetDevice(&active)" in src and "active!=device" in src,
          "scratch pool must guard selected device")
    return total + 18


class StreamModel:
    """Queues execute in order only when their own stream is synchronized."""
    def __init__(self, device: int):
        self.device = device
        self.queues: dict[str, list] = {"default": [], "async": []}
        self.unrelated_completions = 0

    def enqueue(self, stream: str, callback) -> None:
        self.queues[stream].append(callback)

    def sync(self, stream: str) -> None:
        queue = self.queues[stream]
        while queue:
            queue.pop(0)()

    def sync_device(self) -> None:
        self.sync("default")
        self.sync("async")


@dataclass
class RequestModel:
    ident: int
    x: list[float]
    hidden_state: list[float] = field(default_factory=lambda: [0.0] * 64)
    kv_length: list[int] = field(default_factory=lambda: [0] * 64)
    busy: bool = False
    poisoned: bool = False
    freed: bool = False
    slots: list[dict] = field(default_factory=lambda: [
        {"owner": None, "generation": 0, "payload": []} for _ in range(2)
    ])


def run_request(
    length: int, chunk: int, sync_mode: str, ident: int = 1,
    fail_at: int | None = None,
) -> tuple[tuple, dict]:
    check(65 <= length <= 129 and 1 <= chunk <= 64,
          "CPU scheduler model only accepts bounded T65/T129")
    req = RequestModel(ident, [
        math.sin((i + 1) * 0.103 + ident * 0.21) for i in range(length)
    ])
    devices = [StreamModel(0), StreamModel(1)]
    trace: list[str] = []
    step = 0
    try:
        check(not req.busy and not req.freed, "state is already owned")
        req.busy = True
        trace.append("lock")
        for start in range(0, length, chunk):
            count = min(chunk, length - start)
            for li in range(64):
                device = 0 if li < 32 else 1
                dev = devices[device]
                slot = req.slots[device]
                gen = slot["generation"] + 1
                slot["generation"] = gen
                slot["owner"] = ident
                expected = gen
                values = list(req.x[start:start + count])
                # Independent decode client on a nonblocking stream, with
                # distinct scratch and state; this should remain runnable.
                dev.enqueue("async", lambda d=dev: setattr(
                    d, "unrelated_completions", d.unrelated_completions + 1))

                def producer(s=slot, generation=expected, src=values):
                    check(s["generation"] == generation,
                          "producer overwrote scratch after premature reuse")
                    s["payload"] = [v * 0.23 + 0.007 for v in src]

                def consumer(s=slot, generation=expected, r=req,
                             first=start, idx=li, dev_id=device):
                    check(s["generation"] == generation and s["owner"] == ident,
                          "consumer used wrong generation/request/GPU")
                    check(len(s["payload"]) == count,
                          "consumer ran before projection producer")
                    # Exact CPU oracle intentionally uses only sequential
                    # state and private buffers, no CUDA numeric emulation.
                    for token, val in enumerate(s["payload"]):
                        r.x[first + token] += val * 0.00037 + idx * 1e-7
                        r.hidden_state[idx] += val * 0.00021
                    if idx % 4 == 3:
                        check(r.kv_length[idx] == first,
                              "KV history was committed out of token order")
                        r.kv_length[idx] += count

                dev.enqueue("default", producer)
                dev.enqueue("default", consumer)
                step += 1
                if fail_at == step:
                    raise GateFailure("injected CUDA stream sync failure")
                if sync_mode == "stream":
                    dev.sync("default")
                elif sync_mode == "device":
                    dev.sync_device()
                elif sync_mode == "wrong_stream":
                    dev.sync("async")
                elif sync_mode == "none":
                    pass
                else:
                    raise ValueError(sync_mode)
        # An unrelated nonblocking stream can remain queued when only the
        # pooled request's default stream has been synchronized.
        pending = sum(len(d.queues["async"]) for d in devices)
        if sync_mode == "stream":
            check(pending > 0, "a stream-scoped sync drained unrelated work")
        if sync_mode == "device":
            check(pending == 0, "device-wide baseline did not drain all work")
        for dev in devices:
            dev.sync("default")
        check(all(not d.queues["default"] for d in devices),
              "scratch pools cannot be released with outstanding work")
        result = (tuple(round(v, 12) for v in req.x),
                  tuple(round(v, 12) for v in req.hidden_state),
                  tuple(req.kv_length))
        trace += ["pool1_destroy", "pool0_destroy", "unlock"]
        req.busy = False
        return result, {"pending_async": pending, "trace": trace}
    except (GateFailure, ValueError):
        req.poisoned = True
        trace += ["pool1_destroy", "pool0_destroy", "unlock", "decode_free"]
        req.busy = False
        req.freed = True
        check(trace.index("pool0_destroy") < trace.index("unlock") <
              trace.index("decode_free"), "poison cleanup ordering violated")
        raise


def negative(fn, fragment: str) -> None:
    try:
        fn()
    except GateFailure as e:
        check(fragment in str(e), f"unexpected fault: {e}")
    else:
        raise GateFailure(f"negative witness did not fail: {fragment}")


def model_tests() -> int:
    cases = 0
    for length in (65, 129):
        for chunk in (1, 3, 64):
            for req in (1, 2, 4):
                narrow, stream_info = run_request(length, chunk, "stream", req)
                broad, device_info = run_request(length, chunk, "device", req)
                check(narrow == broad, f"stream sync changed state T{length}/C{chunk}/R{req}")
                check(stream_info["pending_async"] > 0 and
                      device_info["pending_async"] == 0,
                      "independent decode stream should not be implicitly drained")
                cases += 1
    for mutant in ("none", "wrong_stream"):
        negative(lambda m=mutant: run_request(65, 64, m), "premature reuse")
        cases += 1
    negative(lambda: run_request(65, 64, "stream", fail_at=23),
             "injected CUDA stream sync failure")
    cases += 1
    return cases


def mutation_tests() -> int:
    audited = audit_source(SOURCE)
    check(audited > 40, "unexpectedly few source producer checks")
    changes = (
        ("pool ? cudaStreamSynchronize(0)", "pool ? cudaDeviceSynchronize()"),
        ("pool ? cudaStreamSynchronize(0)", "pool ? cudaStreamSynchronize(1)"),
        (": cudaDeviceSynchronize();", ": cudaStreamSynchronize(0);"),
        ("if(layer_result!=cudaSuccess)", "if(false)"),
        ("PrefillLayerScratch scratch(pool);", "PrefillLayerScratch scratch(nullptr);"),
        ("cudaMemcpy(seed->Kc[li]", "cudaMemcpyAsync(seed->Kc[li]"),
        ("cudaMemcpy(qq.data()", "cudaMemcpyAsync(qq.data()"),
        ("HV,KHD,VHD);", "HV,KHD,VHD,other_stream);"),
        ("cudaStreamCreateWithFlags(&ctx->stream0,cudaStreamNonBlocking)",
         "cudaStreamCreateWithFlags(&ctx->stream0,cudaStreamDefault)"),
        ("ctx->step_mutex,std::try_to_lock", "ctx->step_mutex,std::defer_lock"),
        ("run_full_layer(li,d_x1,T,err,seed,pos0,pool1)",
         "run_full_layer(li,d_x1,T,err,seed,pos0,pool0)"),
    )
    for old, new in changes:
        check(old in SOURCE, f"mutation anchor absent: {old}")
        modified = SOURCE
        if old == "ctx->step_mutex,std::try_to_lock":
            start = "std::vector<float> prefill_chunks_into_decode_state("
            prefix, tail = SOURCE.split(start, 1)
            modified = prefix + start + tail.replace(old, new, 1)
        else:
            modified = SOURCE.replace(old, new, 1)
        negative(lambda text=modified: audit_source(text), "")
    return len(changes)


def main() -> None:
    source_checks = audit_source(SOURCE)
    simulated = model_tests()
    mutants = mutation_tests()
    print(f"PREFILL_CHUNK_SYNC_BARRIER_CPU_PASS source={source_checks} "
          f"stream_model={simulated} source_mutants={mutants} "
          "GPU_CUDA_EXECUTION=NOT_TESTED")


if __name__ == "__main__":
    main()
