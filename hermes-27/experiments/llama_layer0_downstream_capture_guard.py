#!/usr/bin/env python3
"""CPU-only, pinned preflight for the real llama layer-0 downstream tracer.

No model load, CUDA initialization, SSH, network, or mutation. This checks
physical identities and safe inspection modes before the prime's GPU grant.
Only --capture-manifest admits a completed trace, with an independent verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys


MODEL_SHA = "db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1"
MODEL_BYTES = 19_694_360_544
PRIOR_SHA = "49b0052027532e1e8572d9d865517b2beef92c0e8176a02adacd35efe46d540b"
TRACER_SOURCE_SHA = "ecaebd72739a2cc443147139b899e58717b532d9bea80da65cb8d11204e46429"
TRACER_HELPER_SHA = "eebe48b6c3518f84c8928b9613e13cb9f92bdb9fb7ceaae06a890c38bf2c37df"
VERIFIER_SOURCE_SHA = "2d9cc3da5fcb454672efcd63444f55e3da38ad57094f0a5f1bf3112e5087ae10"
SOURCE_SHA = {
    "src/models/qwen35.cpp": "ef973e9c789eb52ef27a77e0b1f36c6f41a378733a994d41b5924e3be012f671",
    "src/models/delta-net-base.cpp": "5e3b318953f854b3825dd7cfe4b483ca7935cd5f6680438a3ece4be9b4d5428f",
    "include/llama.h": "7279a2065cf7b7a081be9138dbe2be1bf8e0ce527e7df74429a7a982ae877a47",
}
LIB_SHA = {
    "libggml-base.so.0.25.3": "63be6a671ca0bdc97d5ceb4ba4406f9a267bff4ae9ad9a9b5ba4b84ace24c876",
    "libggml-cpu.so.0.25.3": "01e92fe889efdd9da239d43dd7154a9fbb66f52193d77c45144fb849dbb0a191",
    "libggml-cuda.so.0.25.3": "27dff5b6ded0e179b6fbe4504e7c1f0f5751598cdd35759000bbe4df30b20de3",
    "libggml.so.0.25.3": "cd3766cae482625d6e8a309a6536b00c8285c80aa986b04095f85b51af99f794",
    "libllama.so.0.5.0": "fd9e3e0810c6eca2e37b57abf443bcf15963671cbd0ee8489a37461a521094d4",
}
PRIOR_STAGES = {
    "attn_norm-0": (5120, "492ae3d622039e01dcce9c5d8bc32ff2986e1134d5c50fad47ff366e8561bfa3"),
    "linear_attn_qkv_mixed-0": (10240, "4b32da962e03f2481aa869955cea378b7921418d920459e9e62cf9300c08184f"),
    "conv_output_silu-0": (10240, "12513886591fd03cc1a28c2d671cb76217242339300037f1988da4031a9d56a2"),
}
CAPTURE_STAGES = {
    "attn_norm-0": ((5120, 2, 1, 1), 1),
    "linear_attn_qkv_mixed-0": ((10240, 2, 1, 1), 1),
    "conv_output_silu-0": ((10240, 2, 1, 1), 1),
    "beta_sigmoid-0": ((1, 48, 2, 1), 2),
    "gate-0": ((48, 2, 1, 1), 1),
    "q_conv_predelta-0": ((128, (16, 48), 2, 1), 2),
    "k_conv_predelta-0": ((128, (16, 48), 2, 1), 2),
    "attn_output-0": ((128, 48, 2, 1), 2),
}
SHA_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class Blocked(RuntimeError):
    pass


def need(condition: object, error: str) -> None:
    if not condition:
        raise Blocked(error)


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def checked_file(path: Path, sha: str, size: int | None = None) -> Path:
    need(bool(SHA_PATTERN.fullmatch(sha)), "invalid independently pinned SHA256")
    need(path.is_file(), f"physical file missing: {path}")
    actual_path = path.resolve(strict=True)
    need(actual_path.is_file(), f"physical file unresolved: {path}")
    if size is not None:
        need(actual_path.stat().st_size == size, f"physical size mismatch: {path}")
    need(sha256_file(actual_path) == sha, f"physical SHA256 mismatch: {path}")
    return actual_path


def json_no_duplicates(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        need(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def prior_manifest(path: Path, model: Path, lib: Path) -> dict:
    checked_file(path, PRIOR_SHA)
    document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=json_no_duplicates)
    need(isinstance(document, dict) and
         document.get("schema") == "h27.reliability-layer-cb.v1" and
         document.get("status") == "CAPTURED_NOT_INDEPENDENTLY_COMPARED" and
         document.get("backend") == "llama.cpp" and document.get("layer") == 0,
         "earlier pos0 trace has wrong schema/backend/layer")
    need(document.get("prompt_ids") == [6205] and
         document.get("absolute_position") == 0 and
         document.get("greedy_argmax") == 161421 and
         document.get("callback_ask_count") == 3 and
         document.get("callback_ready_count") == 3 and
         document.get("tensor_data_format") == "little-endian-fp32-ne0-fastest",
         "earlier pos0 trace has wrong identity or incomplete callbacks")
    need(document.get("model_sha256") == MODEL_SHA and
         document.get("libllama_sha256") == LIB_SHA["libllama.so.0.5.0"],
         "earlier pos0 model/runtime differs from pinned identities")
    need(Path(document.get("libllama_path", "")).resolve(strict=True) == lib,
         "earlier pos0 libllama path is different")
    if "model_path" in document:
        need(Path(document["model_path"]).resolve(strict=True) == model,
             "earlier pos0 model path is different")
    need(isinstance(document.get("tensors"), list) and len(document["tensors"]) == 3,
         "earlier pos0 must contain exactly three tensors")
    seen = set()
    for entry in document["tensors"]:
        need(isinstance(entry, dict), "malformed prior tensor")
        name = entry.get("name")
        need(name in PRIOR_STAGES and name not in seen, "unexpected/duplicate prior tensor")
        seen.add(name)
        count, expected_sha = PRIOR_STAGES[name]
        need(entry.get("shape") == [count, 1, 1, 1] and
             entry.get("elements") == count and entry.get("bytes") == count * 4 and
             entry.get("source_type") == "f32" and
             entry.get("file") == name + ".f32" and
             entry.get("sha256") == expected_sha,
             f"prior source tensor shape/bytes/identity wrong: {name}")
        checked_file(path.parent / entry["file"], expected_sha, count * 4)
    need(seen == set(PRIOR_STAGES), "earlier trace tensor set incomplete")
    # v1 did not record a physical exporter path; this cannot attest it.
    return document


def parse_lock(contents: str) -> dict[str, str]:
    mapping = {}
    for line in contents.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        need(len(parts) == 2, "runtime SHA lock has malformed row")
        name, digest = parts
        need(name in LIB_SHA and name not in mapping and
             bool(SHA_PATTERN.fullmatch(digest)),
             "runtime SHA lock has unknown/duplicate library")
        mapping[name] = digest
    need(mapping == LIB_SHA, "runtime SHA lock is not exact pinned installed set")
    return mapping


def parse_loaded_runtime(text: str) -> dict[str, str]:
    header, _, rows = text.partition("\n")
    need(header == "LOADED_RUNTIME_SHA_LOCK_TEMPLATE (no model initialized):",
         "unexpected installed tracer runtime inspection header")
    return parse_lock(rows)


def ensure_stage_layout(manifest: dict, stages: dict = CAPTURE_STAGES) -> None:
    """Validate runtime-reported GGML dimensions; never transpose by width."""
    need(manifest.get("schema") == "h27.reliability-layer0-downstream-cb.v1" and
         manifest.get("backend") == "llama.cpp" and
         manifest.get("status") == "CAPTURED_NOT_INDEPENDENTLY_COMPARED" and
         manifest.get("prompt_ids") == [6205, 161421] and
         manifest.get("absolute_positions") == [0, 1] and
         manifest.get("one_contiguous_decode_batch") is True and
         manifest.get("layer") == 0,
         "missing contiguous two-token llama batch provenance")
    argmax = manifest.get("greedy_argmax")
    need(isinstance(argmax, list) and len(argmax) == 2 and
         all(type(x) is int and 0 <= x < 248320 for x in argmax) and
         argmax[0] == 161421,
         "pos0 argmax differs from prior physically anchored token161421")
    records = manifest.get("tensors")
    need(isinstance(records, list) and len(records) == len(stages),
         "capture has missing or excess required callback stages")
    need(manifest.get("requested_stages") == sorted(stages) and
         manifest.get("callback_ready_count") == len(stages) and
         type(manifest.get("callback_ask_count")) is int and
         manifest["callback_ask_count"] >= len(stages),
         "requested/ready callbacks not complete")
    names = set()
    for record in records:
        need(isinstance(record, dict), "nonobject downstream tensor record")
        name = record.get("name")
        need(name in stages and name not in names, "unknown/duplicate downstream stage")
        names.add(name)
        dims = record.get("shape")
        spec, axis = stages[name]
        need(isinstance(dims, list) and len(dims) == 4 and
             all(type(v) is int and v > 0 for v in dims) and
             all(v in allowed if isinstance(allowed, tuple) else v == allowed
                 for v, allowed in zip(dims, spec)) and
             record.get("token_axis") == axis and dims[axis] == 2,
             f"GGML head/position layout mismatch: {name}")
        count = math.prod(dims)
        need(record.get("elements") == count and
             record.get("bytes") == count * 4 and
             record.get("file") == name + ".f32",
             f"FP32 length/file does not match GGML axes: {name}")
    need(names == set(stages), "missing downstream tensor")


def cpu_inspect(tracer: Path, mode: str, expected_line: str) -> str:
    need(mode in ("--inspect-plan", "--inspect-runtime", "--selftest"),
         "refusing model-loading CLI mode")
    run = subprocess.run([str(tracer), mode], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, timeout=25,
                         check=False, shell=False)
    need(run.returncode == 0 and expected_line in run.stdout and
         len(run.stdout) <= 16_384 and len(run.stderr) <= 16_384,
         f"CPU-only tracer mode {mode} failed")
    return run.stdout


def preflight(args: argparse.Namespace) -> dict:
    model = checked_file(args.model, MODEL_SHA, MODEL_BYTES)
    need(args.lib_dir.is_dir(), "installed llama runtime directory absent")
    physical_libs = {name: checked_file(args.lib_dir / name, digest)
                     for name, digest in LIB_SHA.items()}
    lib = physical_libs["libllama.so.0.5.0"]
    need(all(item.parent == lib.parent for item in physical_libs.values()),
         "installed runtime libraries resolved to different directories")
    for relpath, digest in SOURCE_SHA.items():
        checked_file(args.llama_source / relpath, digest)
    checked_file(args.tracer_source, TRACER_SOURCE_SHA)
    checked_file(args.tracer_helper, TRACER_HELPER_SHA)
    checked_file(args.verifier, VERIFIER_SOURCE_SHA)
    need(args.tracer.is_file() and args.tracer.resolve(strict=True) != args.tracer_source.resolve(),
         "compiled tracer executable missing")
    need(bool(SHA_PATTERN.fullmatch(args.trusted_exporter_sha)),
         "independently pinned compiled exporter SHA required")
    checked_file(args.tracer, args.trusted_exporter_sha)
    need(args.capture_dir.is_dir(), "capture directory missing")
    if args.capture_manifest is None:
        need(not any(args.capture_dir.iterdir()),
             "capture directory must exist and be empty before GPU run")
    else:
        need(args.capture_manifest.resolve().parent == args.capture_dir.resolve(),
             "post-capture manifest is outside selected capture directory")
    need(args.runtime_lock.is_file(), "runtime SHA lock missing")
    parse_lock(args.runtime_lock.read_text(encoding="ascii"))
    prior_manifest(args.prior_manifest, model, lib)
    plan = cpu_inspect(args.tracer, "--inspect-plan", "attn_output-0")
    for name in CAPTURE_STAGES:
        need(plan.splitlines().count(name) == 1,
             f"compiled tracer does not declare callback {name}")
    runtime = cpu_inspect(args.tracer, "--inspect-runtime",
                          "LOADED_RUNTIME_SHA_LOCK_TEMPLATE")
    parse_loaded_runtime(runtime)
    cpu_inspect(args.tracer, "--selftest", "PASS CPU callback selection")
    if args.capture_manifest is not None:
        need(args.capture_manifest.is_file(), "post-capture manifest absent")
        checked_capture = subprocess.run(
            [sys.executable, str(args.verifier), "--verify", str(args.capture_manifest),
             "--runtime-lock", str(args.runtime_lock),
             "--expected-model-sha", MODEL_SHA,
             "--expected-libllama-sha", LIB_SHA["libllama.so.0.5.0"],
             "--tracer-exe", str(args.tracer)], capture_output=True,
            text=True, timeout=240, check=False, shell=False)
        need(checked_capture.returncode == 0 and
             "VERIFIED_FULL_GGUF_LIBS_AND_TENSORS" in checked_capture.stdout,
             "independent completed-capture SHA/FP32 verification failed")
        parsed = json.loads(args.capture_manifest.read_text(encoding="utf-8"),
                            object_pairs_hook=json_no_duplicates)
        ensure_stage_layout(parsed)
    return {
        "status": ("VERIFIED_CAPTURE_STAGES_ONLY" if args.capture_manifest
                   else "PASS_CPU_PREFLIGHT_ONLY"),
        "capture_executed_by_guard": False,
        "physical_gguf_sha256": MODEL_SHA,
        "physical_libs_verified": sorted(LIB_SHA),
        "tracer_source_sha256": TRACER_SOURCE_SHA,
        "compiled_exporter_sha256": args.trusted_exporter_sha,
        "prior_pos0_manifest_sha256": PRIOR_SHA,
        "required_runtime_stages": {name: {"shape": list(shape), "token_axis": axis}
                                    for name, (shape, axis) in CAPTURE_STAGES.items()},
        "limitations": ["No GPU or model inference was run",
                        "Source SHA and exporter SHA do not prove a reproducible build",
                        "Q/K predelta are not assumed equal to HERMES q_norm_scaled/k_norm",
                        "Actual tensor axes proven only by independently verified post-capture manifest"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--lib-dir", type=Path, required=True)
    ap.add_argument("--llama-source", type=Path, required=True)
    ap.add_argument("--tracer-source", type=Path, required=True)
    ap.add_argument("--tracer-helper", type=Path, required=True)
    ap.add_argument("--verifier", type=Path, required=True)
    ap.add_argument("--tracer", type=Path, required=True)
    ap.add_argument("--trusted-exporter-sha", required=True)
    ap.add_argument("--prior-manifest", type=Path, required=True)
    ap.add_argument("--runtime-lock", type=Path, required=True)
    ap.add_argument("--capture-dir", type=Path, required=True)
    ap.add_argument("--capture-manifest", type=Path)
    args = ap.parse_args()
    try:
        print(json.dumps(preflight(args), indent=2))
        return 0
    except (Blocked, OSError, ValueError, subprocess.TimeoutExpired,
            json.JSONDecodeError) as e:
        print(f"BLOCKED_OFFLINE_LLAMA_DOWNSTREAM_PREFLIGHT: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
