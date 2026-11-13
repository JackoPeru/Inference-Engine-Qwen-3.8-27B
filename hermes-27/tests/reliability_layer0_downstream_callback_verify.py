#!/usr/bin/env python3
"""Offline independent verifier for llama.cpp downstream callback captures.

Never imports llama, CUDA or torch. --selftest only uses tiny synthetic
temporary files; --verify SHA256-hashes the COMPLETE physical GGUF and
every recorded runtime library and tensor, not file headings or samples.
This validator does NOT run model inference or infer correctness of logits.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import struct
import tempfile
from pathlib import Path

SCHEMA = "h27.reliability-layer0-downstream-cb.v1"
HEX = re.compile(r"[0-9a-f]{64}\Z")
STAGE = re.compile(r"[A-Za-z0-9_.]+-0\Z")
REAL_MODEL_HASH = "db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1"
REAL_LIBLLAMA_HASH = "fd9e3e0810c6eca2e37b57abf443bcf15963671cbd0ee8489a37461a521094d4"


def require(condition: object, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def canonical_file(path: object) -> Path:
    require(isinstance(path, str) and bool(path), "file path missing")
    p = Path(path)
    require(p.is_absolute(), "file path must be absolute")
    result = p.resolve(strict=True)
    require(result.is_file(), "expected regular file")
    return result


def sha_record(path: Path, expected: object) -> None:
    require(isinstance(expected, str) and HEX.fullmatch(expected),
            "malformed SHA256")
    require(sha256_file(path) == expected, f"SHA256 mismatch: {path.name}")


def parse_lock(path: Path) -> dict[str, str]:
    lock: dict[str, str] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln or ln.startswith("#"):
            continue
        cols = ln.split()
        require(len(cols) == 2, "invalid runtime-lock row")
        name, sha = cols
        require((name.startswith("libggml") or name.startswith("libllama.so"))
                and "/" not in name and "\\" not in name
                and bool(HEX.fullmatch(sha)), "bad runtime-lock identity")
        require(name not in lock, "duplicate runtime-lock entry")
        lock[name] = sha
    require(bool(lock), "runtime-lock empty")
    return lock


def verify_capture(manifest_path: Path, lock_path: Path,
                   trusted_model_sha: str, trusted_libllama_sha: str,
                   expected_exporter: Path) -> dict:
    require(HEX.fullmatch(trusted_model_sha) is not None, "invalid trusted model digest")
    require(HEX.fullmatch(trusted_libllama_sha) is not None, "invalid trusted lib digest")
    require(manifest_path.is_file(), "capture manifest missing")
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(isinstance(m, dict), "manifest not an object")
    require(m.get("schema") == SCHEMA, "unexpected capture schema")
    require(m.get("status") == "CAPTURED_NOT_INDEPENDENTLY_COMPARED",
            "capture status invalid")
    require(m.get("backend") == "llama.cpp", "wrong backend")
    require(m.get("prompt_ids") == [6205, 161421], "wrong actual token IDs")
    require(m.get("absolute_positions") == [0, 1] and
            m.get("one_contiguous_decode_batch") is True, "wrong causal batch")
    require(m.get("layer") == 0, "not layer0")
    require(m.get("n_gpu_layers") in range(129), "n_gpu_layers invalid")
    require(m.get("tensor_data_format") == "little-endian-fp32-ne0-fastest",
            "noncanonical tensor format")
    argmax = m.get("greedy_argmax")
    require(isinstance(argmax, list) and len(argmax) == 2
            and all(type(t) is int and 0 <= t < 248320 for t in argmax),
            "missing/nonfinite logits outputs")

    model = canonical_file(m.get("model_path"))
    require(m.get("model_bytes") == model.stat().st_size, "physical model size")
    require(m.get("model_sha256") == trusted_model_sha, "untrusted model identity")
    sha_record(model, trusted_model_sha)  # Full 19.7 GB, not first/last.
    lib = canonical_file(m.get("libllama_path"))
    require(m.get("libllama_sha256") == trusted_libllama_sha,
            "untrusted libllama identity")
    sha_record(lib, trusted_libllama_sha)
    exporter = canonical_file(m.get("exporter_path"))
    require(exporter == expected_exporter.resolve(strict=True), "unexpected exporter executable")
    sha_record(exporter, m.get("exporter_sha256"))

    lock_path = lock_path.resolve(strict=True)
    sha_record(lock_path, m.get("runtime_lock_sha256"))
    lock = parse_lock(lock_path)
    libraries = m.get("runtime_libraries")
    require(isinstance(libraries, list) and 1 <= len(libraries) <= 32,
            "invalid shared library manifest")
    seen_libraries: set[str] = set()
    for record in libraries:
        require(isinstance(record, dict), "malformed shared library record")
        name = record.get("name")
        require(isinstance(name, str) and name not in seen_libraries,
                "duplicate runtime library")
        seen_libraries.add(name)
        runtime = canonical_file(record.get("path"))
        require(runtime.name == name, "runtime library basename/canonical path mismatch")
        require(lock.get(name) == record.get("sha256"),
                "runtime library disagrees with external lock")
        sha_record(runtime, record["sha256"])
        if name.startswith("libllama.so"):
            require(runtime == lib, "loaded libllama is not the trusted one")
    require(set(lock) == seen_libraries, "runtime shared library set differs from lock")

    budget = m.get("max_total_bytes")
    single = m.get("max_single_bytes")
    require(type(budget) is int and 0 < budget <= 64 * 1024 * 1024,
            "total copy budget invalid")
    require(type(single) is int and 0 < single <= 8 * 1024 * 1024
            and single <= budget, "per-tensor budget invalid")
    captures = m.get("tensors")
    requested = m.get("requested_stages")
    require(isinstance(captures, list) and 1 <= len(captures) <= 32,
            "capture tensor count invalid")
    require(isinstance(requested, list) and 1 <= len(requested) <= 32
            and len(requested) == len(set(requested))
            and all(isinstance(s, str) and STAGE.fullmatch(s) for s in requested),
            "invalid requested stages")
    require(type(m.get("callback_ask_count")) is int
            and type(m.get("callback_ready_count")) is int
            and m["callback_ask_count"] >= len(captures)
            and m["callback_ready_count"] == len(captures),
            "missing or duplicate ready callbacks")

    folder = manifest_path.resolve().parent
    total = 0
    names: set[str] = set()
    files: set[str] = set()
    for record in captures:
        require(isinstance(record, dict), "malformed tensor record")
        name, filename = record.get("name"), record.get("file")
        require(isinstance(name, str) and bool(STAGE.fullmatch(name))
                and name not in names, "bad or duplicate stage label")
        names.add(name)
        require(isinstance(filename, str) and filename == name + ".f32"
                and Path(filename).name == filename and filename not in files,
                "unsafe/duplicate tensor file name")
        files.add(filename)
        shape = record.get("shape")
        strides = record.get("source_strides_bytes")
        require(isinstance(shape, list) and len(shape) == 4
                and all(type(v) is int and 1 <= v <= 1000000 for v in shape),
                "invalid tensor shape")
        require(isinstance(strides, list) and len(strides) == 4
                and all(type(v) is int and v >= 1 for v in strides),
                "invalid source strides")
        require(record.get("source_type") in ("f32", "f16", "bf16"),
                "unsupported source tensor type")
        count = math.prod(shape)
        size = count * 4
        require(record.get("elements") == count and record.get("bytes") == size
                and size <= single and total + size <= budget,
                "tensor shape/byte budget exceeded")
        total += size
        expected_axis = (-1 if name in ("state_predelta-0", "new_state-0")
                         else 2 if name in (
                             "q_conv-0", "k_conv-0", "v_conv-0",
                             "q_conv_predelta-0", "k_conv_predelta-0",
                             "v_conv_predelta-0", "beta_sigmoid-0",
                             "attn_output-0") else 1)
        require(record.get("token_axis") == expected_axis,
                "unexpected callback position axis")
        if expected_axis >= 0:
            require(shape[expected_axis] == 2, "tensor missing position1")
        path = folder / filename
        require(path.is_file() and not path.is_symlink() and
                path.stat().st_size == size, "missing/short/aliased tensor")
        sha_record(path, record.get("sha256"))
        with path.open("rb") as f:
            for (value,) in struct.iter_unpack("<f", f.read()):
                require(math.isfinite(value), "non-finite captured FP32")
    require(names == set(requested), "requested callback stage missing/extra")
    present = {x.name for x in folder.iterdir() if x.is_file()}
    require(present == files | {manifest_path.name}, "partial/unexpected capture file")
    return m


def selftest() -> None:
    # Exact same gate used on a real manifest, with tiny synthetic files.
    with tempfile.TemporaryDirectory(prefix="h27-downstream-tamper-") as tmp:
        d = Path(tmp)
        capture = d / "capture"
        capture.mkdir()
        model = d / "tiny.gguf"
        model.write_bytes(b"GGUF fixture - NOT a model\x00")
        llama = d / "libllama.so.0.5.0"
        llama.write_bytes(b"fixture llama shared library")
        ggml = d / "libggml-base.so.0.25.3"
        ggml.write_bytes(b"fixture ggml shared library")
        exporter = d / "tracer"
        exporter.write_bytes(b"fixture standalone tracer")
        lock = d / "runtime.lock"
        lock.write_text(
            f"{ggml.name} {sha256_file(ggml)}\n"
            f"{llama.name} {sha256_file(llama)}\n", encoding="ascii")
        tensor = capture / "attn_output-0.f32"
        tensor.write_bytes(struct.pack("<8f", *(float(i) for i in range(8))))
        m = {
            "schema": SCHEMA, "status": "CAPTURED_NOT_INDEPENDENTLY_COMPARED",
            "backend": "llama.cpp", "model_path": str(model),
            "model_sha256": sha256_file(model), "model_bytes": model.stat().st_size,
            "exporter_path": str(exporter), "exporter_sha256": sha256_file(exporter),
            "libllama_path": str(llama), "libllama_sha256": sha256_file(llama),
            "runtime_lock_sha256": sha256_file(lock),
            "prompt_ids": [6205, 161421], "absolute_positions": [0, 1],
            "one_contiguous_decode_batch": True, "layer": 0, "n_gpu_layers": 80,
            "greedy_argmax": [161421, 303],
            "tensor_data_format": "little-endian-fp32-ne0-fastest",
            "requested_stages": ["attn_output-0"], "callback_ask_count": 1,
            "callback_ready_count": 1, "max_total_bytes": 1024,
            "max_single_bytes": 512,
            "runtime_libraries": [
                {"name": p.name, "path": str(p), "sha256": sha256_file(p)}
                for p in (ggml, llama)],
            "tensors": [{
                "name": "attn_output-0", "file": tensor.name, "source_type": "f32",
                "buffer": "CPU", "shape": [2, 2, 2, 1],
                "source_strides_bytes": [4, 8, 16, 32],
                "token_axis": 2, "elements": 8, "bytes": 32,
                "sha256": sha256_file(tensor)}],
        }
        manifest = capture / "manifest.json"

        def write() -> None:
            manifest.write_text(json.dumps(m, indent=2), encoding="utf-8")

        def check() -> None:
            verify_capture(manifest, lock, sha256_file(model),
                           sha256_file(llama), exporter)

        write()
        check()
        failures = 0

        def tamper(mutator) -> None:
            nonlocal failures
            original_m = copy.deepcopy(m)
            original_files = {p: p.read_bytes() for p in (model, llama, ggml, exporter, lock, tensor)}
            try:
                mutator()
                write()
                try:
                    check()
                except (ValueError, OSError, json.JSONDecodeError):
                    failures += 1
                else:
                    raise AssertionError("tampered capture accepted")
            finally:
                m.clear()
                m.update(original_m)
                for p, content in original_files.items():
                    p.write_bytes(content)
                write()

        tamper(lambda: tensor.write_bytes(b"Z" + tensor.read_bytes()[1:]))
        tamper(lambda: tensor.write_bytes(tensor.read_bytes()[:-4]))
        tamper(lambda: tensor.write_bytes(struct.pack("<f", float("nan")) + tensor.read_bytes()[4:]))
        tamper(lambda: m["tensors"][0].update(shape=[2, 2, 1, 1]))
        tamper(lambda: m["tensors"][0].update(token_axis=1))
        tamper(lambda: m["tensors"][0].update(file="../attn_output-0.f32"))
        tamper(lambda: m["tensors"][0].update(sha256="0" * 64))
        tamper(lambda: m.update(prompt_ids=[6205, 6205]))
        tamper(lambda: m.update(absolute_positions=[1, 0]))
        tamper(lambda: m.update(callback_ready_count=0))
        tamper(lambda: m.update(requested_stages=["attn_residual-0"]))
        tamper(lambda: m["runtime_libraries"][0].update(sha256="0" * 64))
        tamper(lambda: model.write_bytes(b"tampered GGUF bytes"))
        tamper(lambda: llama.write_bytes(b"tampered libllama bytes"))
        tamper(lambda: exporter.write_bytes(b"tampered executable bytes"))
        tamper(lambda: lock.write_text("corrupted runtime lock", encoding="ascii"))
        tamper(lambda: m.update(model_bytes=2**63))
        print(f"PASS independent manifest/file/hash/bounds tamper tests: {failures}/17 rejected")
        require(failures == 17, "negative test count mismatch")
    print("CPU_ONLY_DOWNSTREAM_CAPTURE_VERIFY_PASS; no GPU/model executed")


def main() -> None:
    ap = argparse.ArgumentParser()
    modes = ap.add_mutually_exclusive_group(required=True)
    modes.add_argument("--selftest", action="store_true")
    modes.add_argument("--verify", type=Path, metavar="MANIFEST_JSON")
    ap.add_argument("--runtime-lock", type=Path)
    ap.add_argument("--expected-model-sha")
    ap.add_argument("--expected-libllama-sha")
    ap.add_argument("--tracer-exe", type=Path)
    args = ap.parse_args()
    if args.selftest:
        selftest()
    else:
        require(args.runtime_lock is not None
                and args.tracer_exe is not None
                and args.expected_model_sha is not None
                and args.expected_libllama_sha is not None,
                "--verify requires trusted lock/digests/exporter")
        result = verify_capture(args.verify, args.runtime_lock,
                                args.expected_model_sha,
                                args.expected_libllama_sha, args.tracer_exe)
        print(f"VERIFIED_FULL_GGUF_LIBS_AND_TENSORS stages={len(result['tensors'])} "
              "positions=0,1, not downstream quality parity")


if __name__ == "__main__":
    main()
