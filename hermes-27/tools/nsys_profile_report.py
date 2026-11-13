#!/usr/bin/env python3
"""Read an Nsight Systems SQLite export and summarize HERMES-27 CUDA work.

CPU-only; never launches a GPU workload. See benchmarks/PROFILING_GUIDE.md.
"""

import argparse
import json
import math
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path


GPU_TABLES = (
    ("CUPTI_ACTIVITY_KIND_KERNEL", "kernel"),
    ("CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL", "kernel"),
    ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"),
    ("CUPTI_ACTIVITY_KIND_MEMSET", "memset"),
)
RUNTIME_TABLES = ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUDA_API_TRACE")


def tables(con):
    return {name for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def rows(con, name):
    # Names originate from sqlite_master, not from user-supplied SQL.
    return con.execute('SELECT * FROM "' + name.replace('"', '""') + '"')


def strings(con, available):
    if "StringIds" not in available:
        return {}
    result = {}
    for row in rows(con, "StringIds"):
        r = dict(row)
        if "id" in r and "value" in r:
            result[r["id"]] = r["value"]
    return result


def first(r, *keys):
    for key in keys:
        if key in r and r[key] is not None:
            return r[key]
    return None


def event_name(r, lookup, kind):
    keys = ("demangledName", "shortName", "name", "mangledName", "nameId") if kind == "kernel" else ("nameId", "name", "shortName", "copyKind", "memKind")
    for key in keys:
        value = r.get(key)
        if value is not None:
            if isinstance(value, int):
                return str(lookup.get(value, f"{kind}#{value}"))
            return str(value)
    return kind


def collect(con):
    available = tables(con)
    lookup = strings(con, available)
    gpu = []
    runtime = []
    for table, kind in GPU_TABLES:
        if table not in available:
            continue
        for row in rows(con, table):
            r = dict(row)
            start = first(r, "start", "startNs")
            end = first(r, "end", "endNs")
            if start is None or end is None or end <= start:
                continue
            device = first(r, "deviceId", "gpuId", "device")
            gpu.append({"start": int(start), "end": int(end), "device": str(device if device is not None else "unknown"),
                        "kind": kind, "name": event_name(r, lookup, kind), "bytes": int(first(r, "bytes", "size") or 0)})
    for table in RUNTIME_TABLES:
        if table not in available:
            continue
        for row in rows(con, table):
            r = dict(row)
            start = first(r, "start", "startNs")
            end = first(r, "end", "endNs")
            if start is None or end is None or end <= start:
                continue
            runtime.append({"start": int(start), "end": int(end), "name": event_name(r, lookup, "runtime")})
    return gpu, runtime, available


def merged(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(end, out[-1][1]))
        else:
            out.append((start, end))
    return out


def duration(intervals):
    return sum(end - start for start, end in merged(intervals))


def intersection_ms(left, right):
    a, b = merged(left), merged(right)
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total / 1e6


def family(name):
    n = name.lower()
    if "nvfp4_gemv" in n or "gemv_k" in n:
        return "NVFP4 GEMV"
    if "head_bf16" in n:
        return "BF16 logits head"
    if "gdn_step" in n:
        return "GDN recurrent step"
    if "cutlass" in n or "gemm" in n:
        return "GEMM/other matrix"
    if "sdpa" in n or "rope" in n:
        return "attention/rope"
    if "rms" in n or "silu" in n or "conv" in n or "norm" in n or "gate" in n or "beta" in n or "split" in n or "add_k" in n or "f32_to_bf16" in n:
        return "elementwise/norm"
    return "other"


def clip_events(events, begin, end):
    return [{**e, "start": max(e["start"], begin), "end": min(e["end"], end)}
            for e in events if e["end"] > begin and e["start"] < end]


def percentile(values, p):
    seq = sorted(values)
    if not seq:
        return 0
    return seq[min(len(seq) - 1, max(0, math.ceil(len(seq) * p) - 1))]


def analyze(gpu, runtime, last_embeds=0, start_ms=None, end_ms=None, top=15):
    if not gpu:
        raise ValueError("No CUDA GPU events found. Export a CUDA-traced .nsys-rep to SQLite.")
    first_start = min(e["start"] for e in gpu)
    last_end = max(e["end"] for e in gpu)
    begin = first_start if start_ms is None else first_start + round(start_ms * 1e6)
    end = last_end if end_ms is None else first_start + round(end_ms * 1e6)
    if last_embeds:
        embeds = sorted(e["start"] for e in gpu if e["kind"] == "kernel" and "embed_gather_k" in e["name"])
        if len(embeds) < last_embeds:
            raise ValueError(f"Requested {last_embeds} embeds, found only {len(embeds)}; use --start-ms/--end-ms.")
        begin = max(begin, embeds[-last_embeds])
    if end <= begin:
        raise ValueError("Window is empty; adjust --start-ms/--end-ms/--last-embeds.")
    gpu = clip_events(gpu, begin, end)
    runtime = clip_events(runtime, begin, end)
    wall_ms = (end - begin) / 1e6
    device_groups = defaultdict(list)
    kernel_groups = defaultdict(list)
    by_family = defaultdict(list)
    for ev in gpu:
        device_groups[ev["device"]].append(ev)
        if ev["kind"] == "kernel":
            ms = (ev["end"] - ev["start"]) / 1e6
            kernel_groups[(ev["device"], ev["name"])].append(ms)
            by_family[family(ev["name"])].append(ms)
    devices = {}
    for dev, dev_events in sorted(device_groups.items()):
        kernels = [e for e in dev_events if e["kind"] == "kernel"]
        memcpy = [e for e in dev_events if e["kind"] == "memcpy"]
        kernel_busy_ms = duration((e["start"], e["end"]) for e in kernels) / 1e6
        devices[dev] = {"kernel_count": len(kernels), "kernel_busy_ms": round(kernel_busy_ms, 3),
                        "kernel_busy_pct_of_window": round(100 * kernel_busy_ms / wall_ms, 2),
                        "memcpy_count": len(memcpy), "memcpy_bytes": sum(e["bytes"] for e in memcpy),
                        "memcpy_activity_ms": round(duration((e["start"], e["end"]) for e in memcpy) / 1e6, 3)}
    overlap_ms = 0.0
    if len(devices) >= 2:
        left, right = list(devices)[:2]
        overlap_ms = intersection_ms([(e["start"], e["end"]) for e in device_groups[left] if e["kind"] == "kernel"],
                                     [(e["start"], e["end"]) for e in device_groups[right] if e["kind"] == "kernel"])
    api_groups = defaultdict(list)
    for ev in runtime:
        if re.search("(Synchronize|cudaLaunch|cudaMemcpy|cudaMemset|cudaMalloc|cudaFree|cudaGraph)", ev["name"], re.I):
            api_groups[ev["name"]].append((ev["end"] - ev["start"]) / 1e6)
    all_kernel_ms = sum(sum(samples) for samples in kernel_groups.values())
    ranked = sorted(kernel_groups.items(), key=lambda item: sum(item[1]), reverse=True)[:top]
    return {"window": {"start_offset_ms": round((begin - first_start) / 1e6, 3),
                       "end_offset_ms": round((end - first_start) / 1e6, 3), "wall_ms": round(wall_ms, 3)},
            "devices": devices, "gpu_overlap_ms": round(overlap_ms, 3),
            "gpu_overlap_pct_of_window": round(overlap_ms * 100 / wall_ms, 2),
            "all_kernel_gpu_ms_sum_across_devices": round(all_kernel_ms, 3),
            "kernel_families": [{"family": f, "calls": len(x), "gpu_ms": round(sum(x), 3),
                                  "pct_of_kernel_gpu_ms": round(100 * sum(x) / all_kernel_ms, 2) if all_kernel_ms else 0}
                                 for f, x in sorted(by_family.items(), key=lambda item: sum(item[1]), reverse=True)],
            "top_kernels": [{"device": dev, "name": name, "calls": len(x), "total_ms": round(sum(x), 3),
                             "mean_us": round(sum(x) * 1000 / len(x), 2), "p95_us": round(percentile([v * 1000 for v in x], .95), 2)}
                            for (dev, name), x in ranked],
            "cuda_runtime_apis": [{"api": name, "calls": len(x), "host_total_ms": round(sum(x), 3),
                                   "host_p95_us": round(percentile([v * 1000 for v in x], .95), 2)}
                                  for name, x in sorted(api_groups.items(), key=lambda item: sum(item[1]), reverse=True)]}


def human(report):
    w = report["window"]
    print(f"Window {w['start_offset_ms']:.3f}..{w['end_offset_ms']:.3f} ms (wall {w['wall_ms']:.3f} ms)")
    for gpu, info in report["devices"].items():
        print(f"GPU {gpu}: {info['kernel_count']} kernels, kernel-busy {info['kernel_busy_ms']} ms "
              f"({info['kernel_busy_pct_of_window']}% wall), memcpy {info['memcpy_count']} calls/"
              f"{info['memcpy_bytes']} B")
    print(f"2-GPU concurrent kernel activity {report['gpu_overlap_ms']} ms "
          f"({report['gpu_overlap_pct_of_window']}% wall); GPU kernel total "
          f"{report['all_kernel_gpu_ms_sum_across_devices']} ms (sum across devices)")
    print("GPU kernel families (sum of invocation durations):")
    for x in report["kernel_families"]:
        print(f"  {x['family']:24} {x['calls']:7} calls {x['gpu_ms']:10.3f} ms "
              f"{x['pct_of_kernel_gpu_ms']:5.1f}%")
    print("Top GPU kernels by summed device time:")
    for x in report["top_kernels"]:
        print(f"  GPU{x['device']} {x['total_ms']:10.3f} ms {x['calls']:6} calls "
              f"p95={x['p95_us']:.1f} us  {x['name'][:125]}")
    print("CUDA runtime APIs (host spans, NOT additional GPU time):")
    for x in report["cuda_runtime_apis"]:
        print(f"  {x['host_total_ms']:10.3f} ms {x['calls']:6} calls  {x['api']}")
    print("Note: profile instrumentation adds overhead; names are grouped by substring. "
          "GPU sums and host API spans must not be added to wall time.")


def self_test():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript("""
        CREATE TABLE StringIds (id INTEGER, value TEXT);
        INSERT INTO StringIds VALUES (1, 'void embed_gather_k(int)'), (2, 'void nvfp4_gemv_k(float)'),
                                     (3, 'cudaDeviceSynchronize');
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, deviceId INTEGER, demangledName INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (0, 10000000, 0, 1), (10000000, 30000000, 0, 2),
                                                      (15000000, 25000000, 1, 2), (40000000, 45000000, 0, 1),
                                                      (45000000, 55000000, 0, 2);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, nameId INTEGER);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (12000000, 17000000, 3);
    """)
    gpu, runtime, _ = collect(con)
    rep = analyze(gpu, runtime, top=5)
    assert rep["devices"]["0"]["kernel_count"] == 4
    assert rep["gpu_overlap_ms"] == 10
    assert rep["cuda_runtime_apis"][0]["calls"] == 1
    sliced = analyze(gpu, runtime, last_embeds=1)
    assert sliced["window"]["start_offset_ms"] == 40
    assert sliced["devices"]["0"]["kernel_count"] == 2
    assert sliced["gpu_overlap_ms"] == 0
    con.close()
    print("SELF-TEST PASS: SQLite schema, decode window, intervals, API classification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite_file", nargs="?", type=Path, help="SQLite exported from nsys")
    parser.add_argument("--last-embeds", type=int, default=0, metavar="N",
                        help="select time from N-th last embed_gather_k launch (typically last N decode steps)")
    parser.add_argument("--start-ms", type=float, help="window start relative to earliest GPU event")
    parser.add_argument("--end-ms", type=float, help="window end relative to earliest GPU event")
    parser.add_argument("--top", type=int, default=15, help="number of kernel names to list")
    parser.add_argument("--json-out", type=Path, help="also write a machine-readable summary")
    parser.add_argument("--self-test", action="store_true", help="CPU-only synthetic SQLite check")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.sqlite_file or not args.sqlite_file.is_file():
        parser.error("supply an existing nsys SQLite file or --self-test")
    if args.last_embeds < 0 or args.top <= 0:
        parser.error("--last-embeds must be >=0, --top must be >0")
    try:
        con = sqlite3.connect(f"file:{args.sqlite_file.resolve().as_posix()}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        gpu, runtime, available = collect(con)
        con.close()
        if not any(t in available for t, _ in GPU_TABLES):
            raise ValueError("Nsight SQLite contains no CUPTI GPU event tables; profile with --trace=cuda")
        report = analyze(gpu, runtime, args.last_embeds, args.start_ms, args.end_ms, args.top)
    except (sqlite3.Error, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    human(report)
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
