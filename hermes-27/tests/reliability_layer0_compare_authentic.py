#!/usr/bin/env python3
"""Fail-closed CPU-only comparison of physical HERMES scratch v2 and llama callbacks.

This tool never initializes CUDA/llama, launches a model, or imports PP2 runtime.
It cannot assert parity for stages without independently captured llama data.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import sys

MODEL_SHA = "db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1"
MODEL_BYTES = 19694360544
LLAMA_LIB_SHA = "fd9e3e0810c6eca2e37b57abf443bcf15963671cbd0ee8489a37461a521094d4"
SCHEMA_LLAMA_SINGLE = "h27.reliability-layer-cb.v1"
SCHEMA_LLAMA_BATCH = "h27.reliability-layer0-downstream-cb.v1"
SCHEMA_HERMES_V2 = "h27.hermes-scratch-layer0-capture.v2"
SCRATCH_PATCHER_SHA = "80c0be0c1e29faf5db837da306af59ee729d4207c9bc21022163cfdef049c886"
HERMES_STATUS = "RUNTIME_DECODE_ARGUMENTS_RECORDED_NOT_LLAMA_COMPARED"
LLAMA_STATUS = "CAPTURED_NOT_INDEPENDENTLY_COMPARED"
HEX = re.compile(r"[0-9a-f]{64}\Z")
TOKENS = (6205, 161421)
HERMES_WIDTHS = {
    "attn_norm": 5120,
    "linear_attn_qkv_mixed": 10240,
    "conv_output_silu": 10240,
    "beta_sigmoid": 48,
    "gate": 48,
    "q_norm_scaled": 6144,
    "k_norm": 6144,
    "attn_output": 6144,
    "final_output": 6144,
    "linear_attn_out": 5120,
    "attn_residual": 5120,
}
# Order follows the causal layer0 graph, not alphabetic order. Only stages
# with a source-derived same-operator interpretation on BOTH backends.
COMPARABLE = (
    "attn_norm", "linear_attn_qkv_mixed", "conv_output_silu",
    "beta_sigmoid", "gate", "attn_output", "final_output",
    "linear_attn_out", "attn_residual",
)
LLAMA_EXTRA = {
    "conv_output_raw", "q_conv", "k_conv", "v_conv",
    "q_conv_predelta", "k_conv_predelta", "v_conv_predelta",
    "state_predelta", "new_state", "attn_post_norm", "ffn_out",
    "post_ffn", "z",
}
ALL_KNOWN = set(HERMES_WIDTHS) | LLAMA_EXTRA
AXIS2 = {"beta_sigmoid", "attn_output", "q_conv", "k_conv", "v_conv",
         "q_conv_predelta", "k_conv_predelta", "v_conv_predelta"}
NON_TOKEN = {"state_predelta", "new_state"}
HEAD_SHAPES = {
    "beta_sigmoid": (1, 48),
    "attn_output": (128, 48),
    "q_conv": (128, 16), "k_conv": (128, 16),
    "v_conv": (128, 48), "v_conv_predelta": (128, 48),
}
MAX_RAW = 8 * 1024 * 1024


class Blocked(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def need(condition: bool, message: str, code: str = "INVALID_PROVENANCE") -> None:
    if not condition:
        raise Blocked(code, message)


def duplicates_rejected(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        need(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict:
    need(path.is_file() and not path.is_symlink(), f"manifest absent/aliased: {path}")
    need(path.stat().st_size <= 512 * 1024, "oversized manifest")
    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           object_pairs_hook=duplicates_rejected)
    except (UnicodeError, json.JSONDecodeError) as e:
        raise Blocked("INVALID_PROVENANCE", f"cannot parse manifest: {e}") from e
    need(isinstance(value, dict), "manifest is not an object")
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for part in iter(lambda: file.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def expected_scratch_source(original: bytes) -> bytes:
    """Check the frozen scratch transform, independently of HERMES tensor data.

    Importing the SHA-pinned source-only patcher here is a provenance check;
    numeric differences are still computed solely from separate physical
    HERMES and llama tensors, never from the runtime's own verifier.
    """
    patcher = Path(__file__).resolve().parents[1] / (
        "experiments/pp2_layer0_trace_scratch_patch.py")
    need(patcher.is_file() and not patcher.is_symlink() and
         sha256_file(patcher) == SCRATCH_PATCHER_SHA,
         "missing or modified frozen scratch patcher")
    try:
        spec = importlib.util.spec_from_file_location(
            "h27_independent_compare_scratch_transform", patcher)
        need(spec is not None and spec.loader is not None,
             "frozen scratch patcher cannot be loaded")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.patched_source(original.decode("utf-8")).encode("utf-8")
    except Exception as exc:
        if isinstance(exc, Blocked):
            raise
        raise Blocked("INVALID_PROVENANCE",
                      "original PP2 does not support frozen scratch transform") from exc
    need(sha256_file(patcher) == SCRATCH_PATCHER_SHA,
         "frozen scratch patcher changed during verification")
    return result


def sha_field(value: object, name: str) -> str:
    need(isinstance(value, str) and HEX.fullmatch(value) is not None,
         f"missing/invalid SHA-256 of {name}")
    return value


def physical(path: Path, expected: str, *, limit: int | None = None,
             allow_symlink: bool = False) -> bytes | None:
    need(path.is_file(), f"missing physical file: {path}")
    need(allow_symlink or not path.is_symlink(), f"symlink capture rejected: {path}")
    if limit is not None:
        need(0 < path.stat().st_size <= limit,
             f"file size out of bounds: {path}")
    need(sha256_file(path) == sha_field(expected, path.name),
         f"physical SHA-256 mismatch: {path.name}", "HASH_MISMATCH")
    if limit is None:
        return None
    data = path.read_bytes()
    need(hashlib.sha256(data).hexdigest() == expected,
         f"physical capture changed between hash/read: {path.name}",
         "HASH_MISMATCH")
    return data


def verify_model(model: Path, sha: str = MODEL_SHA,
                 size: int = MODEL_BYTES) -> str:
    need(model.is_file() and model.stat().st_size == size,
         "physical model not present or size wrong")
    physical(model, sha, allow_symlink=True)
    return sha


def safe_tensor_path(folder: Path, filename: object) -> Path:
    need(isinstance(filename, str) and
         re.fullmatch(r"[a-zA-Z0-9_.-]+\.f32", filename) is not None and
         Path(filename).name == filename and filename not in (".", ".."),
         f"unsafe tensor filename {filename!r}")
    result = folder / filename
    need(result.is_file() and not result.is_symlink(),
         f"tensor file missing or aliased: {filename}")
    return result


def read_tensor(folder: Path, record: dict, *, expected: int | None = None,
                shape: tuple[int, ...] | None = None) -> tuple[float, ...]:
    need(isinstance(record, dict), "malformed tensor record")
    count, size = record.get("elements"), record.get("bytes")
    need(type(count) is int and type(size) is int and
         0 < count <= MAX_RAW // 4 and size == count * 4,
         "tensor elements/byte count invalid")
    if expected is not None:
        need(count == expected, "tensor element count differs from stage width")
    if shape is not None:
        need(record.get("shape") == list(shape), "tensor shape/head axes mismatch")
    filename = record.get("file")
    file = safe_tensor_path(folder, filename)
    need(file.stat().st_size == size, "tensor truncated or wrong byte count")
    data = physical(file, record.get("sha256"), limit=MAX_RAW)
    need(data is not None and len(data) == size, "tensor read length differs")
    numbers = tuple(x[0] for x in struct.iter_unpack("<f", data))
    need(len(numbers) == count and all(math.isfinite(x) for x in numbers),
         "nonfinite/truncated captured FP32 tensor")
    return numbers


def required_model_identity(manifest: dict, model: Path,
                            expected_sha: str, expected_bytes: int) -> None:
    need(manifest.get("model_sha256") == expected_sha,
         "model SHA in manifest not trusted", "HASH_MISMATCH")
    if "model_bytes" in manifest:
        need(manifest["model_bytes"] == expected_bytes,
             "model byte length differs from verified image")
    elif manifest.get("schema") != SCHEMA_LLAMA_SINGLE:
        need(False, "model byte length is missing")
    # Path identity is checked on any format declaring a model file.
    if "model_path" in manifest:
        declared = Path(manifest["model_path"])
        need(declared.is_file() and declared.resolve() == model.resolve(),
             "capture GGUF path differs from verified model")


def _llama_shape(stage: str, batch: bool) -> tuple[int, int, int, int] | None:
    tokens = 2 if batch else 1
    if stage in HEAD_SHAPES:
        a, b = HEAD_SHAPES[stage]
        return (a, b, tokens, 1)
    if stage in ("q_conv_predelta", "k_conv_predelta"):
        return None  # 16 or 48 heads, checked separately by caller.
    if stage in NON_TOKEN:
        return None  # State snapshots have no per-token axis.
    if stage in HERMES_WIDTHS:
        return (HERMES_WIDTHS[stage], tokens, 1, 1)
    if stage == "conv_output_raw":
        return (10240, tokens, 1, 1)
    if stage in ("attn_post_norm", "ffn_out", "post_ffn"):
        return (5120, tokens, 1, 1)
    return None


def _llama_entry(entry: dict, *, batch: bool) -> tuple[str, tuple[float, ...]]:
    need(isinstance(entry, dict) and isinstance(entry.get("name"), str),
         "llama tensor record not named")
    name = entry["name"]
    need(name.endswith("-0") and name[:-2] in ALL_KNOWN,
         f"unknown llama stage name: {name}")
    stage = name[:-2]
    dims = entry.get("shape")
    need(isinstance(dims, list) and len(dims) == 4 and
         all(type(x) is int and 0 < x <= 1_000_000 for x in dims),
         f"invalid llama GGML dims: {name}")
    need(math.prod(dims) == entry.get("elements"),
         f"GGML dimensions × elements disagree: {name}")
    ne = _llama_shape(stage, batch)
    if ne is not None:
        need(dims == list(ne), f"head/sequence layout invalid for {name}: {dims}")
    elif stage in ("q_conv_predelta", "k_conv_predelta"):
        need(dims in ([128, 16, 2 if batch else 1, 1],
                      [128, 48, 2 if batch else 1, 1]),
             f"unsupported Q/K head layout for {name}")
    # When no source-derived state shape is fixed, still ensure token_axis=-1.
    actual_axis = (-1 if stage in NON_TOKEN else 2 if stage in AXIS2 else 1)
    if batch:
        need(entry.get("token_axis") == actual_axis,
             f"token_axis wrong for {name}")
        if actual_axis >= 0:
            need(dims[actual_axis] == 2,
                 f"missing token position 1 for {name}")
    else:
        need(actual_axis == 1 and dims[1:] == [1, 1, 1],
             f"single-position v1 cannot represent {name}'s head axes")
    typ = entry.get("source_type")
    need(typ in (("f32", "f16", "bf16") if batch else ("f32",)),
         f"unexpected llama callback data type: {name}")
    strides = entry.get("source_strides_bytes")
    need(isinstance(strides, list) and len(strides) == 4 and
         all(type(s) is int and s > 0 for s in strides),
         f"missing GGML source strides: {name}")
    need(strides[0] == (4 if typ == "f32" else 2),
         f"GGML dimension0 stride does not match source scalar type: {name}")
    need(isinstance(entry.get("buffer"), str) and bool(entry["buffer"]),
         f"missing source backend buffer for {name}")
    return stage, tuple(dims)


def llama(manifest_path: Path, model: Path, *,
          expected_sha: str = MODEL_SHA, expected_bytes: int = MODEL_BYTES,
          lib_sha: str = LLAMA_LIB_SHA,
          exporter: Path | None = None,
          runtime_lock: Path | None = None) -> dict:
    m = read_json(manifest_path)
    schema = m.get("schema")
    need(schema in (SCHEMA_LLAMA_SINGLE, SCHEMA_LLAMA_BATCH),
         f"unrecognized llama capture schema {schema!r}", "UNSUPPORTED_SCHEMA")
    batch = schema == SCHEMA_LLAMA_BATCH
    need(m.get("backend") == "llama.cpp" and m.get("status") == LLAMA_STATUS,
         "llama backend/status not an authenticated capture")
    need(m.get("layer") == 0 and
         m.get("tensor_data_format") == "little-endian-fp32-ne0-fastest",
         "llama callback layer/format not supported")
    if batch:
        need(m.get("prompt_ids") == [6205, 161421] and
             m.get("absolute_positions") == [0, 1] and
             m.get("one_contiguous_decode_batch") is True,
             "llama batch does not prove same contiguous tokens [6205,161421]")
        need(type(m.get("callback_ready_count")) is int and
             type(m.get("callback_ask_count")) is int,
             "missing callback completeness counters")
        need(isinstance(m.get("greedy_argmax"), list) and
             len(m["greedy_argmax"]) == 2 and
             all(type(x) is int and 0 <= x < 248320 for x in m["greedy_argmax"]),
             "invalid two-token greedy output identity")
    else:
        need(m.get("prompt_ids") == [6205] and
             m.get("absolute_position") == 0 and m.get("greedy_argmax") == 161421,
             "llama single-token capture is not authentic pos0/token6205")
        need(m.get("callback_ask_count") == 3 and
             m.get("callback_ready_count") == 3,
             "incomplete v1 llama callback")
    required_model_identity(m, model, expected_sha, expected_bytes)
    libpath = Path(m.get("libllama_path", ""))
    need(libpath.is_file() and m.get("libllama_sha256") == lib_sha,
         "llama shared-library origin/hash invalid", "HASH_MISMATCH")
    physical(libpath, lib_sha, allow_symlink=True)
    # The older genuine v1 callback manifest records only an exporter hash,
    # not an exporter path. Flag it, never fabricate an authenticated path.
    exporter_verified = False
    if exporter is not None:
        need(exporter.is_file(), "supplied exporter file missing")
        physical(exporter, m.get("exporter_sha256"), allow_symlink=True)
        exporter_verified = True
    if batch:
        expath = Path(m.get("exporter_path", ""))
        need(exporter is not None and expath.is_file() and
             exporter.resolve() == expath.resolve(),
             "downstream llama exporter file not independently supplied")
        need(runtime_lock is not None and runtime_lock.is_file(),
             "downstream llama runtime lock not supplied")
        physical(runtime_lock, m.get("runtime_lock_sha256"))
        content = runtime_lock.read_text(encoding="ascii").splitlines()
        lock = {}
        for line in content:
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            need(len(cols) == 2 and (cols[0].startswith("libggml") or
                 cols[0].startswith("libllama.so")) and
                 cols[0] not in lock and HEX.fullmatch(cols[1]) is not None,
                 "unsafe/duplicate llama runtime-lock row")
            lock[cols[0]] = cols[1]
        entries = m.get("runtime_libraries")
        need(isinstance(entries, list) and entries and len(entries) <= 32,
             "missing runtime shared libraries")
        found = set()
        for library in entries:
            need(isinstance(library, dict) and
                 isinstance(library.get("name"), str) and
                 library["name"] not in found,
                 "malformed/duplicate runtime library")
            name = library["name"]
            found.add(name)
            need(lock.get(name) == library.get("sha256"),
                 "runtime library differs from lock")
            file = Path(library.get("path", ""))
            need(file.is_file() and file.resolve().name == name,
                 "runtime library physical filename absent or altered")
            physical(file, library["sha256"], allow_symlink=True)
        need(set(lock) == found, "runtime library set mismatch")
    entries = m.get("tensors")
    need(isinstance(entries, list) and 1 <= len(entries) <= 32,
         "llama callback tensors missing/excessive")
    if batch:
        requested = m.get("requested_stages")
        need(isinstance(requested, list) and len(requested) == len(entries) and
             len(set(requested)) == len(entries),
             "requested llama stages differ from captured tensors")
    captures = {}
    names = set()
    for rec in entries:
        stage, dims = _llama_entry(rec, batch=batch)
        need(stage not in captures, "duplicate stage in llama capture")
        names.add(rec["name"])
        values = read_tensor(manifest_path.parent, rec)
        captures[stage] = {"dims": dims,
                           "axis": rec.get("token_axis", 1),
                           "values": values, "file": rec["file"]}
    if batch:
        need(set(m["requested_stages"]) == names,
             "llama requested/ready callback names mismatch")
        need(m["callback_ready_count"] == len(entries) and
             m["callback_ask_count"] >= len(entries),
             "callback count incomplete")
    else:
        need(set(captures) == {
            "attn_norm", "linear_attn_qkv_mixed", "conv_output_silu"},
            "v1 callback missing/extra tensor")
    return {"schema": schema, "batch": batch, "position_tokens": (TOKENS if batch
            else TOKENS[:1]), "model_sha256": expected_sha,
            "exporter_verified": exporter_verified, "captures": captures,
            "manifest": str(manifest_path)}


def hermes(manifest_path: Path, capture_dir: Path, model: Path, *,
           source: Path, patched_source: Path, binary: Path,
           expected_sha: str = MODEL_SHA, expected_bytes: int = MODEL_BYTES) -> dict:
    m = read_json(manifest_path)
    need(m.get("schema") == SCHEMA_HERMES_V2,
         "HERMES runtime-token capture v2 required", "UNSUPPORTED_SCHEMA")
    need(m.get("backend") == "hermes-27-scratch" and
         m.get("status") == HERMES_STATUS and m.get("not_release_gate") is True,
         "HERMES manifest backend/status/provenance invalid")
    need(m.get("prompt_ids_runtime_reported") == [6205, 161421] and
         m.get("absolute_positions_runtime_reported") == [0, 1],
         "HERMES runtime token/absolute position not authenticated")
    required_model_identity(m, model, expected_sha, expected_bytes)
    need(capture_dir.is_dir() and not capture_dir.is_symlink(),
         "HERMES scratch capture directory absent/aliased")
    for path, field in ((source, "source_sha256"),
                        (patched_source, "patched_source_sha256"),
                        (binary, "binary_sha256")):
        physical(path, m.get(field), allow_symlink=False)
    original, patched = source.read_bytes(), patched_source.read_bytes()
    need(hashlib.sha256(original).hexdigest() == m["source_sha256"] and
         hashlib.sha256(patched).hexdigest() == m["patched_source_sha256"],
         "PP2 source changed between hash/read", "HASH_MISMATCH")
    need(source.name == patched_source.name == "pp2_fwd.cu" and
         source.resolve() != patched_source.resolve() and
         original != patched and
         b"HERMES_SCRATCH_LAYER0_TRACE_DIR" not in original,
         "scratch PP2 original/patched source identity invalid")
    need(patched.count(b"HERMES_SCRATCH_LAYER0_TRACE_DIR") == 3 and
         patched == expected_scratch_source(original),
         "scratch PP2 source differs from exact frozen three-marker patch")
    contexts = m.get("runtime_decode_contexts")
    need(isinstance(contexts, list) and len(contexts) == 2,
         "missing recorded runtime decode context")
    for pos, context in enumerate(contexts):
        need(isinstance(context, dict) and
             context.get("runtime_token_id") == TOKENS[pos] and
             context.get("runtime_absolute_position") == pos and
             context.get("decode_sequence") == pos,
             f"runtime token/position mismatch at {pos}")
        file = capture_dir / f"p{pos}_context.txt"
        data = physical(file, context.get("context_sha256"), limit=1024)
        need(data == (f"token_id={TOKENS[pos]}\nabsolute_position={pos}\n"
                      f"sequence={pos}\n").encode("ascii"),
             f"scratch runtime context bytes not exact at pos{pos}")
    recs = m.get("tensors")
    need(isinstance(recs, list) and len(recs) == 22,
         "HERMES layer0 v2 must record all 22 stage snapshots")
    seen = set()
    samples = {}
    for rec in recs:
        need(isinstance(rec, dict) and
             isinstance(rec.get("name"), str) and
             rec["name"] in HERMES_WIDTHS,
             "unexpected HERMES layer0 stage record")
        pos = rec.get("runtime_reported_position")
        name = rec["name"]
        need(type(pos) is int and pos in (0, 1) and (pos, name) not in seen,
             "missing/duplicate HERMES stage position")
        seen.add((pos, name))
        width = HERMES_WIDTHS[name]
        need(rec.get("dtype") == "little-endian-fp32" and
             rec.get("shape") == [width] and
             rec.get("file", None) is None,
             "HERMES v2 stage dtype/shape/record unsupported")
        filename = f"p{pos}_{name}.f32"
        record = dict(rec, file=filename, elements=width)
        samples[(pos, name)] = read_tensor(capture_dir, record,
                                            expected=width, shape=(width,))
    need(seen == {(p, stage) for p in (0, 1) for stage in HERMES_WIDTHS},
         "HERMES stage×position coverage incomplete")
    # The verifier's older metadata contract has hashes and shape but no
    # separately recorded file name; insist on exact expected filenames.
    contents = list(capture_dir.iterdir())
    need(all(p.is_file() and not p.is_symlink() for p in contents),
         "unaccounted or nonregular HERMES scratch files")
    actual_files = {p.name for p in contents}
    expected_files = {f"p{p}_{s}.f32" for p in (0, 1)
                      for s in HERMES_WIDTHS} | {
                      "p0_context.txt", "p1_context.txt"}
    need(actual_files == expected_files or
         actual_files == expected_files | {manifest_path.name},
         "unaccounted or missing HERMES scratch files")
    return {"schema": m["schema"], "model_sha256": expected_sha,
            "captures": samples, "manifest": str(manifest_path),
            "runtime_token_ids": list(TOKENS)}


def llama_token_vector(capture: dict, stage: str, position: int) -> tuple[float, ...]:
    need(position in (0, 1), "unsupported token position")
    item = capture["captures"].get(stage)
    if item is None:
        raise Blocked("MISSING_LLAMA_DOWNSTREAM",
                      f"no independent llama callback for {stage}-0")
    dims, axis = item["dims"], item["axis"]
    if not capture["batch"]:
        need(position == 0, "single-token v1 capture has no position1",
             "MISSING_LLAMA_DOWNSTREAM")
        need(axis == 1, "single-token tensor axis invalid")
        return item["values"]
    need(axis in (1, 2), "recurrent state has no position token axis",
         "UNSUPPORTED_STAGE")
    need(dims[axis] == 2 and all(d == 1 for d in dims[axis+1:]),
         "GGML time axis/tensor layout unsupported")
    stride = math.prod(dims[:axis])
    start = position * stride
    vec = item["values"][start:start + stride]
    need(len(vec) == stride, "token slice truncated")
    if stage in HERMES_WIDTHS:
        need(len(vec) == HERMES_WIDTHS[stage],
             "flattened callback heads differ from HERMES stage width")
    return vec


def compare(llama_data: dict, hermes_data: dict, position: int = 0,
            stages: tuple[str, ...] = COMPARABLE,
            max_abs: float = 1e-3, max_rmse: float = 2e-4) -> dict:
    need(position in (0, 1) and
         type(max_abs) in (int, float) and type(max_rmse) in (int, float) and
         math.isfinite(max_abs) and math.isfinite(max_rmse) and
         max_abs > 0 and max_rmse > 0,
         "invalid requested time position or numeric tolerances")
    need(llama_data["model_sha256"] == hermes_data["model_sha256"],
         "same-checkpoint provenance mismatch", "HASH_MISMATCH")
    need(position < len(llama_data["position_tokens"]) and
         llama_data["position_tokens"][position] ==
         hermes_data["runtime_token_ids"][position],
         "token ID/absolute-position differs between independent captures")
    need(len(stages) > 0 and len(set(stages)) == len(stages),
         "no stages or duplicate stages")
    selected_indices = [COMPARABLE.index(s) for s in stages if s in COMPARABLE]
    need(selected_indices == sorted(selected_indices),
         "requested comparable stages must be in causal graph order")
    entries = []
    blocked = None
    prefix_proven = True
    first_prefix = None
    first_observed = None
    previous_index = -1
    for stage in stages:
        if stage not in COMPARABLE:
            blocked = blocked or {
                "code": "UNSUPPORTED_STAGE", "stage": stage,
                "reason": "no source-grounded identical operator/head layout on both backends"}
            entries.append({"stage": stage, "status": "UNSUPPORTED_STAGE"})
            prefix_proven = False
            continue
        index = COMPARABLE.index(stage)
        if index != previous_index + 1:
            prefix_proven = False
        previous_index = index
        if stage not in llama_data["captures"]:
            blocked = blocked or {
                "code": "MISSING_LLAMA_DOWNSTREAM", "stage": stage,
                "reason": "llama callback for the same real stage was not captured"}
            entries.append({"stage": stage, "status": "MISSING_LLAMA_DOWNSTREAM"})
            prefix_proven = False
            continue
        if (position, stage) not in hermes_data["captures"]:
            blocked = blocked or {
                "code": "MISSING_HERMES_STAGE", "stage": stage,
                "reason": "HERMES runtime token stage was not captured"}
            entries.append({"stage": stage, "status": "MISSING_HERMES_STAGE"})
            prefix_proven = False
            continue
        left = llama_token_vector(llama_data, stage, position)
        right = hermes_data["captures"][(position, stage)]
        need(len(left) == len(right) == HERMES_WIDTHS[stage],
             f"{stage}: head/flattened slice dimensions differ",
             "INCOMPATIBLE_LAYOUT")
        diffs = [abs(a - b) for a, b in zip(left, right)]
        worst = max(range(len(diffs)), key=diffs.__getitem__)
        magnitude = max(diffs)
        rmse = math.sqrt(math.fsum(d * d for d in diffs) / len(diffs))
        match = magnitude <= max_abs and rmse <= max_rmse
        if not match and first_observed is None:
            first_observed = stage
        if not match and first_prefix is None and prefix_proven:
            first_prefix = stage
        entries.append({
            "stage": stage, "status": "MATCH" if match else "DIFFERENCE",
            "token_id": hermes_data["runtime_token_ids"][position],
            "absolute_position": position, "elements": len(left),
            "max_abs": magnitude, "rmse": rmse, "worst_index": worst,
            "llama_at_worst": left[worst], "hermes_at_worst": right[worst]})
    status = ("BLOCKED_" + blocked["code"] if blocked else
              "FIRST_DIFFERENCE" if first_observed else "CAPTURED_STAGES_MATCH_ONLY")
    return {
        "schema": "h27.reliability-layer0-authentic-compare.v1",
        "status": status, "not_release_gate": True,
        "model_sha256": llama_data["model_sha256"],
        "llama_capture_schema": llama_data["schema"],
        "hermes_capture_schema": hermes_data["schema"],
        "llama_exporter_verified": llama_data["exporter_verified"],
        "token_id": hermes_data["runtime_token_ids"][position],
        "absolute_position": position, "max_abs_gate": max_abs,
        "max_rmse_gate": max_rmse,
        "first_differing_captured_stage": first_observed,
        "first_differing_verified_prefix_stage": first_prefix,
        "blocker": blocked, "stages": entries,
        "warning": "No inference about missing downstream stages or llama Q8_1/BF16 operator parity."}


def inspect_llama(data: dict) -> dict:
    return {"schema": "h27.reliability-layer0-authentic-inspection.v1",
            "status": "LLAMA_CAPTURE_VERIFIED_NO_HERMES_COMPARE",
            "not_release_gate": True,
            "llama_schema": data["schema"],
            "model_sha256": data["model_sha256"],
            "exporter_verified": data["exporter_verified"],
            "token_ids": list(data["position_tokens"]),
            "captured_stages": sorted(data["captures"]),
            "missing_downstream": sorted(set(COMPARABLE) -
                                         set(data["captures"]))}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--llama-manifest", required=True, type=Path)
    p.add_argument("--hermes-manifest", type=Path)
    p.add_argument("--hermes-dir", type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--hermes-source", type=Path)
    p.add_argument("--hermes-patched-source", type=Path)
    p.add_argument("--hermes-binary", type=Path)
    p.add_argument("--llama-exporter", type=Path)
    p.add_argument("--llama-runtime-lock", type=Path)
    p.add_argument("--position", choices=(0, 1), type=int, default=0)
    p.add_argument("--stages", default=",".join(COMPARABLE))
    p.add_argument("--max-abs", type=float, default=1e-3)
    p.add_argument("--max-rmse", type=float, default=2e-4)
    p.add_argument("--inspect-llama", action="store_true")
    args = p.parse_args(argv)
    try:
        stages = tuple(args.stages.split(","))
        need(stages and all(stage in ALL_KNOWN and stage for stage in stages),
             "unknown/empty stage selection")
        verified_sha = verify_model(args.model)
        ll = llama(args.llama_manifest, args.model,
                   expected_sha=verified_sha, exporter=args.llama_exporter,
                   runtime_lock=args.llama_runtime_lock)
        if args.inspect_llama:
            need(args.hermes_manifest is None and args.hermes_dir is None,
                 "--inspect-llama must not present HERMES manifests")
            print(json.dumps(inspect_llama(ll), indent=2, allow_nan=False))
            return 0
        need(all(x is not None for x in (
            args.hermes_manifest, args.hermes_dir,
            args.hermes_source, args.hermes_patched_source, args.hermes_binary)),
            "full HERMES v2 manifest, capture dir, original/scratch source and binary required",
            "MISSING_HERMES_CAPTURE")
        hh = hermes(args.hermes_manifest, args.hermes_dir, args.model,
                    source=args.hermes_source, patched_source=args.hermes_patched_source,
                    binary=args.hermes_binary, expected_sha=verified_sha)
        result = compare(ll, hh, args.position, stages, args.max_abs,
                         args.max_rmse)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 2 if result["blocker"] else (
            1 if result["first_differing_captured_stage"] is not None else 0)
    except (Blocked, OSError, ValueError, KeyError, TypeError,
            UnicodeError, struct.error) as e:
        code = e.code if isinstance(e, Blocked) else "INVALID_PROVENANCE"
        print(json.dumps({"schema": "h27.reliability-layer0-authentic-compare.v1",
              "status": "BLOCKED_" + code, "not_release_gate": True,
              "reason": str(e)}, indent=2), file=sys.stdout)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
