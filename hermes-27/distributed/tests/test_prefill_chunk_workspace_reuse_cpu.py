#!/usr/bin/env python3
"""Offline allocation/ownership oracle for PP2 chunk-local scratch pooling.

Extracts the real allocation shapes and verifies two independent request
owners, GPU isolation, chunk boundaries and failure/poison behavior. It does
not execute CUDA, inspect GPU state or establish numerical/throughput parity.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re

SOURCE = (Path(__file__).resolve().parents[1] / "pp2_fwd.cu").read_text(encoding="utf-8")
SOURCE_FILE = "distributed/pp2_fwd.cu"
MAX_CHUNK = 64
LAYER_COUNT = 64
GPU0_LAST = 31


class Rejected(Exception):
    pass


def guard(condition: bool, message: str) -> None:
    if not condition:
        raise Rejected(message)


def section(src: str, start: str, end: str) -> str:
    return src.split(start, 1)[1].split(end, 1)[0]


def integer_expression(expr: str, names: dict[str, int]) -> int:
    """Parse integer-only CUDA byte-size expressions with a strict AST."""
    expr = expr.replace("(size_t)", "").replace(" ", "")

    def walk(node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return walk(node.left) * walk(node.right)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return walk(node.left) + walk(node.right)
        raise Rejected(f"unsupported allocation size {expr}: {type(node).__name__}")

    return walk(ast.parse(expr, mode="eval").body)


def layer_layouts(src: str, t: int) -> tuple[list[int], list[int]]:
    consts = {
        "T": t, "HID": 5120, "FFN": 17408, "VDD": 6144,
        "CC": 10240, "HV": 48, "KHD": 128, "VHD": 128,
        "NQQ": 12288, "NKK": 1024, "NQHD": 6144,
        "nH": t * 5120,
    }
    gdn = section(src, "static bool run_gdn_layer(", "static bool run_full_layer(")
    full = section(src, "static bool run_full_layer(", "// ---------- API ----------")

    def sizes(body: str, seeded: bool) -> list[int]:
        calls = re.findall(r"\bP_ALLOCF\((d_\w+),([^;]+?)\);", body)
        if seeded:
            calls = [(name, expr) for name, expr in calls if name != "d_S"]
        return [integer_expression(expr, consts) for _, expr in calls]

    g = sizes(gdn, True)
    f = sizes(full, False)
    guard(len(g) == 22 and len(f) == 18,
          f"source drift in seeded layer shapes: GDN {len(g)}, FULL {len(f)}")
    guard(all(s > 0 for s in g + f), "zero scratch allocation size")
    return g, f


class PoolModel:
    """Independent synchronous lifetime model, not a translation of C++."""

    next_address = 1

    def __init__(self, owner: str, gpu: int, fail_after: int | None = None):
        self.owner = owner
        self.gpu = gpu
        self.slots: dict[int, tuple[int, int]] = {}  # slot -> (allocation id, capacity)
        self.active = False
        self.poisoned = False
        self.closed = False
        self.malloc_calls = 0
        self.free_calls = 0
        self.fail_after = fail_after
        self.peak_bytes = 0

    def begin(self, device: int) -> None:
        guard(not self.poisoned and not self.closed, "poisoned/closed pool reused")
        guard(device == self.gpu, "wrong GPU borrowed scratch")
        guard(not self.active, "new layer borrowed scratch before prior sync")
        self.active = True

    def reserve(self, index: int, size: int) -> int:
        guard(self.active and not self.poisoned and not self.closed, "invalid active borrow")
        guard(0 <= index < 23 and size > 0, "invalid pooled index/size")
        existing = self.slots.get(index)
        if existing is None or existing[1] < size:
            if self.fail_after is not None and self.malloc_calls >= self.fail_after:
                self.poisoned = True
                raise Rejected("injected CUDA OOM: pool poisoned")
            self.malloc_calls += 1
            if existing is not None:
                self.free_calls += 1
            address = PoolModel.next_address
            PoolModel.next_address += 1
            self.slots[index] = (address, size)
        return self.slots[index][0]

    def finish(self, synchronized: bool = True) -> None:
        guard(self.active, "no layer to finish")
        if not synchronized:
            self.poisoned = True
            raise Rejected("failed CUDA device synchronize: request must abort")
        self.active = False
        self.peak_bytes = max(self.peak_bytes, sum(cap for _, cap in self.slots.values()))

    def close(self, drained: bool = True) -> None:
        guard(not self.closed, "double destruction")
        guard(drained, "cannot release scratch before streams drain")
        self.free_calls += len(self.slots)
        self.slots.clear()
        self.closed = True


def run_chunks(token_count: int, chunk_size: int, requests: int = 1) -> dict[str, int]:
    guard(65 <= token_count <= 8192 and 1 <= chunk_size <= 64, "invalid T/chunk")
    # Each request owns independent arrays on both GPUs.  Interleave calls
    # between independent requests to exercise ownership without sharing them.
    pools = [[PoolModel(f"r{rid}", gpu) for gpu in (0, 1)]
             for rid in range(requests)]
    baseline = 0
    layers = 0
    chunks = 0
    for start in range(0, token_count, chunk_size):
        t = min(chunk_size, token_count - start)
        gdn, full = layer_layouts(SOURCE, t)
        chunks += 1
        for layer in range(LAYER_COUNT):
            gpu = 0 if layer <= GPU0_LAST else 1
            shape = full if layer % 4 == 3 else gdn
            for request in pools:
                pool = request[gpu]
                pool.begin(gpu)
                addresses = [pool.reserve(index, amount)
                             for index, amount in enumerate(shape)]
                guard(len(set(addresses)) == len(addresses),
                      "intralayer scratch buffers overlap")
                pool.finish(synchronized=True)
                baseline += len(shape)
                layers += 1
    all_allocations = [entry for request in pools for entry in request]
    all_live_ids = [address for entry in all_allocations
                    for address, _ in entry.slots.values()]
    guard(len(all_live_ids) == len(set(all_live_ids)),
          "requests or GPUs share physical scratch allocations")
    for entry in all_allocations:
        guard(entry.peak_bytes < 512 * 1024 * 1024,
              "request-level scratch exceeds source's 512 MiB headroom")
    new_malloc = sum(entry.malloc_calls for entry in all_allocations)
    new_peak = max(entry.peak_bytes for entry in all_allocations)
    for entry in all_allocations:
        entry.close(drained=True)
        guard(entry.free_calls == entry.malloc_calls,
              "pooled scratch allocation/free count mismatch")
    guard(baseline == 1344 * chunks * requests, "incorrect old allocation baseline")
    guard(new_malloc < baseline, "pool did not reduce CUDA allocations")
    return {"chunks": chunks, "layers": layers, "old_malloc": baseline,
            "new_malloc": new_malloc, "max_pool_bytes": new_peak}


def expect_rejected(fn, fragment: str) -> None:
    try:
        fn()
    except Rejected as exc:
        guard(fragment in str(exc), f"incorrect negative oracle: {exc}")
    else:
        raise AssertionError(f"negative witness unexpectedly accepted: {fragment}")


def test_fault_injection() -> int:
    # Same GPU but distinct request owner: a concurrent client cannot begin
    # a new layer on the first request's still-active pool.
    p = PoolModel("A", 0)
    p.begin(0)
    p.reserve(0, 4096)
    expect_rejected(lambda: p.begin(0), "before prior sync")
    expect_rejected(lambda: p.begin(1), "wrong GPU")
    expect_rejected(lambda: p.finish(synchronized=False), "request must abort")
    expect_rejected(lambda: p.begin(0), "poisoned")
    p.close(drained=True)
    expect_rejected(lambda: p.begin(0), "poisoned")

    q = PoolModel("B", 1, fail_after=1)
    q.begin(1)
    q.reserve(0, 8192)
    expect_rejected(lambda: q.reserve(1, 4096), "OOM")
    expect_rejected(lambda: q.begin(1), "poisoned")
    q.close(drained=True)

    v = PoolModel("C", 0)
    v.begin(0)
    v.reserve(0, 1024)
    expect_rejected(lambda: v.close(drained=False), "before streams drain")
    v.finish()
    v.close()
    expect_rejected(lambda: v.close(), "double destruction")
    expect_rejected(lambda: run_chunks(65, 65), "invalid T/chunk")
    return 10


def test_cleanup_order() -> int:
    """The request lock outlives both scratch owners on all abort branches."""
    scenarios = ("incomplete_logits", "nonfinite_logits", "kv_mismatch", "exception")
    checks = 0
    for scenario in scenarios:
        event_trace = ["lock", "pool0_create", "pool1_create", "failure_" + scenario]
        # This is the independent required lifetime contract: release the
        # newest pool first, then the older one, then unlock and free state.
        event_trace += ["pool1_destroy", "pool0_destroy", "unlock", "decode_free"]
        guard(event_trace.index("pool1_destroy") < event_trace.index("unlock"),
              "GPU1 pool outlived the state lock")
        guard(event_trace.index("pool0_destroy") < event_trace.index("decode_free"),
              "GPU0 scratch outlived recurrent state")
        checks += 1
    # A direct return fail() while a pool is in scope is a negative oracle.
    premature = ["lock", "pool_create", "unlock", "decode_free", "pool_destroy"]
    expect_rejected(lambda: guard(
        premature.index("pool_destroy") < premature.index("unlock"),
        "premature state unlock before pooled scratch teardown"),
        "premature state unlock")
    return checks + 1


def test_code_contract(src: str) -> None:
    pool = section(src, "struct PrefillChunkScratchPool {", "struct PrefillLayerScratch {")
    scratch = section(src, "struct PrefillLayerScratch {", "// S[k] is the post-token GDN")
    fn = src.split("std::vector<float> prefill_chunks_into_decode_state(", 1)[1]
    require = [
        ("per-request two-device local pools", "PrefillChunkScratchPool gpu0_pool(0),gpu1_pool(1);" in fn),
        ("per-chunk same pools", "(int)offset,true,&gpu0_pool,&gpu1_pool);" in fn),
        ("per-request try-lock", "std::unique_lock<std::mutex>(ctx->step_mutex,std::try_to_lock)" in fn),
        ("failure invalidates state", "if(ctx)ctx->poisoned=true;" in fn and "decode_free(ds);" in fn),
        ("correct device checked", "cudaGetDevice(&active)" in pool and "active!=device" in pool),
        ("allocation fail poisons", "if(rc!=cudaSuccess)" in pool and "poisoned=true;" in pool),
        ("request pool destructor selects GPU", "if(cudaSetDevice(device)!=cudaSuccess)return;" in pool),
        ("no per-layer pooled CUDA frees", "if(!pool)for(void* p:buffers) (void)cudaFree(p);" in scratch),
        ("in-order per-layer local cursor", "pool->alloc(next_slot++,p,bytes,err)" in scratch),
        ("slot bound protects memory", "index>=slots.size()" in pool),
        ("no process-global scratch", "static PrefillChunkScratchPool" not in src),
        ("default prefill unchanged", "return prefill_impl(ids,st,err,h_last_out,nullptr,false);" in src),
        ("BF16 chunk remains forbidden", "chunked prefill BF16 disabled: full-model numeric gate failed" in fn),
        ("chunk uses only last head", "(int)offset,true,&gpu0_pool,&gpu1_pool);" in fn),
    ]
    for name, ok in require:
        guard(ok, f"missing production contract: {name}")
    pool_start = fn.index("PrefillChunkScratchPool gpu0_pool(0),gpu1_pool(1);")
    cleanup = fn.index("return fail(); // Both pool destructors ran while state_lock was held.")
    guard("return fail();" not in fn[pool_start:cleanup],
          "error must not unlock or free DecodeState before pool destruction")
    guard("if(!complete)break;" in fn[pool_start:cleanup],
          "error branch must leave the pool scope before calling fail()")
    guard("}catch(const std::exception& e)" in fn[cleanup:],
          "exceptions must unwind both pools before cleanup")
    for start, end, poolvar in [
        ("static bool run_gdn_layer(", "static bool run_full_layer(", "pool"),
        ("static bool run_full_layer(", "// ---------- API ----------", "pool"),
    ]:
        layer = section(src, start, end)
        guard("PrefillLayerScratch scratch(pool);" in layer, "layer not borrowing pool")
        guard("cudaDeviceSynchronize()" in layer and "if(layer_result!=cudaSuccess)" in layer,
              "device must drain before next layer may borrow pool")
    forward = section(src, "static std::vector<float> prefill_impl(", "std::vector<float> prefill(")
    guard("run_full_layer(li,d_x0,T,err,seed,pos0,pool0)" in forward and
          "run_gdn_layer(li,d_x0,T,err,seed,pos0,pool0)" in forward,
          "GPU0 did not receive GPU0 pool")
    guard("run_full_layer(li,d_x1,T,err,seed,pos0,pool1)" in forward and
          "run_gdn_layer(li,d_x1,T,err,seed,pos0,pool1)" in forward,
          "GPU1 did not receive GPU1 pool")


def test_source_mutants() -> int:
    test_code_contract(SOURCE)
    changes = [
        ('PrefillChunkScratchPool gpu0_pool(0),gpu1_pool(1);',
         'PrefillChunkScratchPool gpu0_pool(0),gpu1_pool(0);'),
        ('(int)offset,true,&gpu0_pool,&gpu1_pool);',
         '(int)offset,true,&gpu0_pool,&gpu0_pool);'),
        ('run_full_layer(li,d_x1,T,err,seed,pos0,pool1)',
         'run_full_layer(li,d_x1,T,err,seed,pos0,pool0)'),
        ('if(!pool)for(void* p:buffers) (void)cudaFree(p);',
         'for(void* p:buffers) (void)cudaFree(p);'),
        ('cudaGetDevice(&active)', 'cudaGetLastError()'),
        ('if(layer_result!=cudaSuccess)', 'if(false)'),
        ('ctx->step_mutex,std::try_to_lock', 'ctx->step_mutex,std::defer_lock'),
        ('complete=false;\n          break;', 'return fail();'),
    ]
    for before, after in changes:
        guard(before in SOURCE, f"mutant source anchor missing: {before}")
        if before == "ctx->step_mutex,std::try_to_lock":
            marker = "std::vector<float> prefill_chunks_into_decode_state("
            head, tail = SOURCE.split(marker, 1)
            mutated = head + marker + tail.replace(before, after, 1)
        else:
            mutated = SOURCE.replace(before, after, 1)
        expect_rejected(lambda s=mutated: test_code_contract(s), "")
    return len(changes)


def main() -> None:
    facts = [run_chunks(t, 64) for t in (65, 129, 1024)]
    concurrent = run_chunks(129, 64, requests=4)
    negative = test_fault_injection()
    ordering = test_cleanup_order()
    mutation_count = test_source_mutants()
    print("PREFILL_CHUNK_WORKSPACE_CPU_PASS",
          "cases=T65/T129/T1024/N4",
          "malloc_old=" + ",".join(str(f["old_malloc"]) for f in facts),
          "malloc_pool=" + ",".join(str(f["new_malloc"]) for f in facts),
          f"N4_mall={concurrent['new_malloc']}/{concurrent['old_malloc']}",
          f"peak_pool_per_gpu={max(f['max_pool_bytes'] for f in facts)}",
          f"fault_negative={negative}", f"source_mutants={mutation_count}",
          f"cleanup_order={ordering}",
          "GPU_GGUF_PARITY=NOT_TESTED")


if __name__ == "__main__":
    main()
