#!/usr/bin/env python3
"""Cheap, read-only HERMES-27 math/layout preflight for the actual GGUF.

This is a semantic/source guard, not a substitute for llama.cpp inference.
It prevents self-consistent E1/E2 references from certifying two known wrong
conventions: double-transformed ssm_a and GDN head order h//3.
No GPU, model materialization, or third-party Python libraries are needed.
"""

from __future__ import annotations

import argparse
import math
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass(frozen=True)
class Tensor:
    dims: tuple[int, ...]
    dtype: int
    offset: int


def take(fp: BinaryIO, fmt: str) -> tuple[int, ...]:
    size = struct.calcsize(fmt)
    b = fp.read(size)
    if len(b) != size:
        raise ValueError(f"truncated GGUF header at byte {fp.tell()}")
    return struct.unpack(fmt, b)


def skip(fp: BinaryIO, count: int) -> None:
    if count < 0 or count > 100_000_000_000:
        raise ValueError(f"unreasonable GGUF skip length: {count}")
    fp.seek(count, 1)


def read_str(fp: BinaryIO, materialize: bool = True) -> str:
    (length,) = take(fp, "<Q")
    if length > 100_000_000:
        raise ValueError(f"unreasonable GGUF string length: {length}")
    if not materialize:
        skip(fp, length)
        return ""
    b = fp.read(length)
    if len(b) != length:
        raise ValueError("truncated GGUF string")
    return b.decode("utf-8")


SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
         10: 8, 11: 8, 12: 8}


def skip_value(fp: BinaryIO, kind: int, depth: int = 0) -> None:
    if depth > 2:
        raise ValueError("nested GGUF metadata too deep")
    if kind == 8:
        read_str(fp, materialize=False)
    elif kind == 9:
        (elem_kind,) = take(fp, "<I")
        (count,) = take(fp, "<Q")
        if count > 10_000_000:
            raise ValueError(f"unreasonable GGUF array count {count}")
        if elem_kind in SIZES:
            skip(fp, count * SIZES[elem_kind])
        elif elem_kind == 8:
            for _ in range(count):
                read_str(fp, materialize=False)
        elif elem_kind == 9:
            for _ in range(count):
                skip_value(fp, elem_kind, depth + 1)
        else:
            raise ValueError(f"unknown GGUF array item type {elem_kind}")
    elif kind in SIZES:
        skip(fp, SIZES[kind])
    else:
        raise ValueError(f"unknown GGUF metadata type {kind}")


def gguf_tensor_index(path: Path) -> tuple[dict[str, Tensor], int]:
    with path.open("rb") as fp:
        if fp.read(4) != b"GGUF":
            raise ValueError("not a GGUF")
        (version,) = take(fp, "<I")
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version {version}")
        n_tensor, n_kv = take(fp, "<QQ")
        if n_tensor > 100_000 or n_kv > 100_000:
            raise ValueError("unreasonable GGUF counts")
        alignment = 32
        for _ in range(n_kv):
            name = read_str(fp)
            (kind,) = take(fp, "<I")
            if name == "general.alignment" and kind == 4:
                (alignment,) = take(fp, "<I")
            else:
                skip_value(fp, kind)
        if alignment < 1 or alignment & (alignment - 1):
            raise ValueError(f"invalid GGUF alignment {alignment}")
        tensors: dict[str, Tensor] = {}
        for _ in range(n_tensor):
            name = read_str(fp)
            (nd,) = take(fp, "<I")
            if nd < 1 or nd > 4:
                raise ValueError(f"invalid tensor rank {nd}")
            dims = tuple(take(fp, "<Q")[0] for _ in range(nd))
            dtype, off = take(fp, "<IQ")
            if name in tensors:
                raise ValueError(f"duplicate tensor {name}")
            tensors[name] = Tensor(dims, dtype, off)
        pos = fp.tell()
        offset = (pos + alignment - 1) // alignment * alignment
        return tensors, offset


def check_model_metadata(path: Path) -> list[str]:
    errors: list[str] = []
    tensors, data_offset = gguf_tensor_index(path)
    # Independent structural evidence: the checkpoint's ssm_a stores the
    # negative decay *itself*, not a positive log to be run through -exp().
    with path.open("rb") as fp:
        for layer in (0, 26, 62):
            name = f"blk.{layer}.ssm_a"
            t = tensors.get(name)
            if t is None or t.dtype != 0 or t.dims != (48,):
                errors.append(f"{name}: missing or expected F32[48], got {t}")
                continue
            fp.seek(data_offset + t.offset)
            values = take(fp, "<48f")
            if not all(math.isfinite(v) and v < 0 for v in values):
                errors.append(f"{name}: expected finite negative pretransformed decay values")
            else:
                print(f"PASS GGUF {name}: F32[48], all negative, "
                      f"range=[{min(values):.6g},{max(values):.6g}]")
    return errors


def check_source(root: Path) -> list[str]:
    errors: list[str] = []
    engine = root / "architecture/src/model_fwd.cpp"
    gpu = root / "distributed/pp2_fwd.cu"
    cpu = engine.read_text(encoding="utf-8")
    pp2 = gpu.read_text(encoding="utf-8")

    # For the GGUF conversion used here, the correct coefficient is
    # gl = ssm_a[h] * softplus(alpha[h] + dt[h]).
    if re.search(r"-\s*expf\s*\(\s*A\s*\[\s*hh\s*\]\s*\)", cpu):
        errors.append(f"{engine}: CPU still applies -exp(ssm_a); GGUF stores -exp(A_log)")
    if re.search(r"\b(?:int|auto)\s+g\s*=\s*hh\s*/\s*3\b", cpu):
        errors.append(f"{engine}: CPU GDN maps heads by h//3; GGUF maps by h%16")
    if not re.search(r"\bgl\s*\[\s*i\s*\]\s*=\s*A\s*\[\s*hh\s*\]\s*\*\s*sp\s*;", pp2):
        errors.append(f"{gpu}: direct ssm_a pretransform convention not recognized")
    if not re.search(r"\b(?:int|auto)\s+g\s*=\s*hh\s*%\s*\(\s*KD\s*/\s*KHD\s*\)", pp2):
        errors.append(f"{gpu}: GGUF GDN Q/K head order h%16 not recognized")

    # Mathematical sanity check independent of implementation:
    # .repeat(1,3,1) yields 0..15,0..15,0..15 (h % 16, CORRECT).
    # .repeat_interleave(3,dim=1) yields 0,0,0,1,1,1,.. (h // 3, WRONG).
    base_heads = list(range(16))
    repeat_heads = base_heads * 3
    interleave_heads = [h for h in base_heads for _ in range(3)]
    expected = [h % 16 for h in range(48)]
    if repeat_heads != expected or interleave_heads == expected:
        errors.append("synthetic GDN head-order invariant failed")
    else:
        print("PASS synthetic head-order proof: repeat(1,3,1) == h%16; "
              "repeat_interleave(3,dim=1) == h//3")

    # An E1/E2 golden can match another wrong forward exactly. Reject scripts
    # that interleave each K head three times (rather than repeating the set).
    for relative in ("scripts/e1_runner.py", "scripts/e1_teacher.py",
                     "scripts/blk0_stats.py"):
        file = root / relative
        source = file.read_text(encoding="utf-8")
        if re.search(r"\.repeat_interleave\s*\(\s*3\s*,\s*(?:dim\s*=\s*)?1\s*\)",
                     source):
            errors.append(f"{file}: GDN Q/K uses repeat_interleave(3,dim=1) "
                          "(h//3). Use repeat(1,3,1) for h%16.")
    # f1_block0.py is an older exploratory test; do not allow using it to
    # generate a new oracle while its direct RMSNorm convention is stale.
    f1 = (root / "scripts/f1_block0.py").read_text(encoding="utf-8")
    if re.search(r"\(\s*1\s*\+\s*w\s*\)", f1):
        print("NOTE old scripts/f1_block0.py uses obsolete (1+w) RMSNorm; "
              "exclude its historical outputs from every golden.")
    return errors


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--gguf", type=Path, help="Real GGUF path, unless --source-only")
    p.add_argument("--source-only", action="store_true",
                   help="No model read. Run a quick static check before GPU time.")
    args = p.parse_args()
    if not args.source_only and args.gguf is None:
        p.error("--gguf is required unless --source-only is set")
    try:
        errors = check_source(args.root.resolve())
        if not args.source_only:
            errors += check_model_metadata(args.gguf.resolve(strict=True))
    except (OSError, ValueError, struct.error) as exc:
        print(f"INCONCLUSIVE: cannot perform mathematical preflight: {exc}",
              file=sys.stderr)
        return 2
    for err in errors:
        print(f"FAIL {err}", file=sys.stderr)
    if errors:
        print(f"FAIL math/layout regression gate: {len(errors)} issue(s)", file=sys.stderr)
        return 1
    print("PASS GGUF math/layout preflight (not a substitute for llama.cpp token comparison)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
