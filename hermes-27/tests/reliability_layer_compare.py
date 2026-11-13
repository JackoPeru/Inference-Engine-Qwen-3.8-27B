#!/usr/bin/env python3
"""Fail-closed offline comparator for independently captured layer tensors.

Never calls CUDA or launches llama/HERMES. Not a production release gate.
Compare matched checkpoint SHA256, token/position/layer, FP32 shapes and
content hashes before finding the first differing *captured* stage.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

STAGES = ("attn_norm", "linear_attn_qkv_mixed", "conv_output_raw",
          "conv_output_silu", "beta_sigmoid", "gate",
          "q_conv_predelta", "k_conv_predelta", "v_conv_predelta",
          "attn_output", "linear_attn_out", "attn_residual",
          "attn_post_norm", "ffn_out", "post_ffn")
MAX_FILE = 8 * 1024 * 1024


class Blocked(Exception):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Blocked(message)


def digest(file: Path) -> str:
    h = hashlib.sha256()
    with file.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path, backend: str) -> dict:
    require(path.is_file(), f"missing manifest: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict) and
            data.get("schema") in ("h27.reliability-layer-cb.v1",
                                   "h27.reliability-layer-hermes.v1"),
            f"unknown manifest format: {path}")
    require(data.get("backend") == backend,
            f"expected real {backend} source: {path}")
    require(data.get("status") == "CAPTURED_NOT_INDEPENDENTLY_COMPARED",
            f"capture status is not complete: {path}")
    require(len(data.get("model_sha256", "")) == 64 and
            data["model_sha256"] != "0" * 64,
            "capture must include real GGUF SHA256")
    require(isinstance(data.get("tensors"), list) and data["tensors"],
            "no captured tensor records")
    return data


def read_f32(manifest_path: Path, record: dict) -> array.array:
    require(isinstance(record, dict), "malformed tensor entry")
    filename = record.get("file")
    require(isinstance(filename, str) and filename.endswith(".f32") and
            Path(filename).name == filename, "unsafe raw tensor filename")
    path = manifest_path.parent / filename
    require(path.is_file(), f"missing captured file: {path}")
    n = record.get("elements")
    size = record.get("bytes")
    require(isinstance(n, int) and 0 < n <= MAX_FILE//4 and
            isinstance(size, int) and size == n*4 and
            path.stat().st_size == size, "missing/malformed bounded FP32 tensor")
    shape = record.get("shape")
    require(isinstance(shape, list) and len(shape) == 4 and
            all(isinstance(d, int) and d > 0 for d in shape) and
            math.prod(shape) == n, "tensor shape incompatible with flat FP32 data")
    require(digest(path) == record.get("sha256"), f"SHA256 changed: {path}")
    if sys.byteorder != "little":
        raise Blocked("raw FP32 little-endian requires little-endian platform")
    values = array.array("f")
    with path.open("rb") as stream:
        values.frombytes(stream.read(size))
    require(len(values) == n and all(math.isfinite(v) for v in values),
            f"nonfinite or malformed FP32 values: {path}")
    return values


def compare(llama_path: Path, hermes_path: Path, stages: tuple[str, ...],
            aliases: dict[str, str], max_abs: float, max_rmse: float) -> dict:
    llama = load(llama_path, "llama.cpp")
    hermes = load(hermes_path, "hermes-27")
    for key in ("model_sha256", "prompt_ids", "absolute_position", "layer"):
        require(llama.get(key) == hermes.get(key),
                f"mismatched same-checkpoint/token/position/layer: {key}")
    l_data = {item["name"]: item for item in llama["tensors"]}
    h_data = {item["name"]: item for item in hermes["tensors"]}
    require(len(l_data) == len(llama["tensors"]) and
            len(h_data) == len(hermes["tensors"]),
            "duplicate tensor names in provenance manifests")
    results = []
    failed = False
    for stage in stages:
        llama_name = stage + "-" + str(llama["layer"])
        hermes_name = aliases.get(stage, llama_name)
        require(llama_name in l_data and hermes_name in h_data,
                f"stage unavailable: llama={llama_name} HERMES={hermes_name}")
        la = l_data[llama_name]
        hb = h_data[hermes_name]
        require(la.get("shape") == hb.get("shape"),
                f"{stage}: captured shape differs {la.get('shape')} vs {hb.get('shape')}")
        x = read_f32(llama_path, la)
        y = read_f32(hermes_path, hb)
        n = len(x)
        mse = math.fsum((float(a)-float(b))**2 for a,b in zip(x,y))/n
        mx = max(abs(float(a)-float(b)) for a,b in zip(x,y))
        rmse = math.sqrt(mse)
        passed = mx <= max_abs and rmse <= max_rmse
        failed |= not passed
        results.append({"stage": stage, "llama_name": llama_name,
                        "hermes_name": hermes_name, "shape": la["shape"],
                        "elements": n, "max_abs": mx, "rmse": rmse,
                        "status": "MATCH" if passed else "DIFFERENCE"})
    first = next((v["stage"] for v in results if v["status"] == "DIFFERENCE"), None)
    return {
        "schema": "h27.reliability-layer-bisect.v1",
        "status": "DIFFERENCE_IN_CAPTURED_STAGES" if failed
                  else "CAPTURED_STAGES_MATCH_ONLY",
        "exit_code": 1 if failed else 0,
        "not_a_release_gate": True, "model_sha256": llama["model_sha256"],
        "prompt_ids": llama["prompt_ids"],
        "absolute_position": llama["absolute_position"], "layer": llama["layer"],
        "first_differing_captured_stage": first, "max_abs_gate": max_abs,
        "max_rmse_gate": max_rmse, "stages": results,
        "warning": "No claim about uninstrumented stages or model correctness."
    }


def selftest() -> None:
    with tempfile.TemporaryDirectory(prefix="h27-reliability-layer-cpu-") as d:
        root = Path(d)
        meta = {"schema": "h27.reliability-layer-cb.v1",
                "status": "CAPTURED_NOT_INDEPENDENTLY_COMPARED",
                "model_sha256": "a"*64, "prompt_ids": [6205],
                "absolute_position": 0, "layer": 0, "tensors": []}
        for backend, dirname in (("llama.cpp", "llama"), ("hermes-27", "hermes")):
            folder = root / dirname
            folder.mkdir()
            samples = array.array("f", [1.0, 2.0, 3.0, 4.0])
            if backend == "hermes-27":
                samples[2] += 0.1
            file = folder / "attn_norm-0.f32"
            file.write_bytes(samples.tobytes())
            obj = dict(meta, backend=backend, tensors=[{
                "name": "attn_norm-0", "file": file.name, "shape": [4,1,1,1],
                "elements": 4, "bytes": 16, "sha256": digest(file)}])
            (folder/"manifest.json").write_text(json.dumps(obj))
        l,h = root/"llama/manifest.json",root/"hermes/manifest.json"
        result = compare(l,h,("attn_norm",),{},1e-4,1e-5)
        assert result["exit_code"] == 1
        assert result["first_differing_captured_stage"] == "attn_norm"
        with (root/"hermes/attn_norm-0.f32").open("ab") as out:
            out.write(b"tamper")
        try:
            compare(l,h,("attn_norm",),{},1e-4,1e-5)
        except Blocked:
            pass
        else:
            raise AssertionError("tampered file should fail closed")
    print("PASS CPU synthetic first-difference, shape/hash and tamper fail-closed")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--llama-manifest", type=Path)
    p.add_argument("--hermes-manifest", type=Path)
    p.add_argument("--stages", default="attn_norm,linear_attn_qkv_mixed,conv_output_silu,post_ffn")
    p.add_argument("--aliases", type=Path,
                   help="JSON object mapping semantic stage to HERMES tensor name")
    p.add_argument("--max-abs", type=float, default=1e-3)
    p.add_argument("--max-rmse", type=float, default=2e-4)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.selftest:
        selftest()
        return 0
    if not args.llama_manifest or not args.hermes_manifest or not args.out:
        p.error("real comparison requires --llama-manifest --hermes-manifest --out")
    stages = tuple(args.stages.split(","))
    if not stages or len(set(stages)) != len(stages) or any(s not in STAGES for s in stages):
        p.error("--stages must be a nonempty unique ordered subset of known stage names")
    if not math.isfinite(args.max_abs) or not math.isfinite(args.max_rmse) or (
            args.max_abs <= 0 or args.max_rmse <= 0):
        p.error("numerical thresholds must be positive finite values")
    aliases = json.loads(args.aliases.read_text()) if args.aliases else {}
    require(isinstance(aliases, dict) and all(isinstance(v, str) and
            "/" not in v and "\\" not in v for v in aliases.values()),
            "invalid HERMES stage alias mapping")
    report = compare(args.llama_manifest,args.hermes_manifest,stages,aliases,
                     args.max_abs,args.max_rmse)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    for result in report["stages"]:
        print(f"{result['status']:10s} {result['stage']}: "
              f"max_abs={result['max_abs']:.8g} rmse={result['rmse']:.8g}")
    print(f"{report['status']}: {args.out}")
    return int(report["exit_code"])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Blocked, ValueError, KeyError, OSError) as exc:
        print(f"BLOCKED: {exc}",file=sys.stderr)
        raise SystemExit(2)
