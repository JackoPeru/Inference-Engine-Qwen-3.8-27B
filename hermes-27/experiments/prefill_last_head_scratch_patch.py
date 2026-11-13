#!/usr/bin/env python3
"""Offline, scratch-only opt-in last-head-only seeded PP2 experiment.

NEVER edits source repository or installed HERMES / Tabby. The original
T*VOC GGUF RAW prefill stays unchanged unless the resulting scratch binary
is explicitly run with HERMES_PP2_SEEDED_LAST_HEAD_ONLY=1.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

FILES = (
    "distributed/pp2_fwd.cu",
    "distributed/pp2_batch.cu",
    "serving/server.cpp",
    "tools/hermes_generate.cpp",
)


def replace_once(src: str, before: str, after: str, *, count: int = 1) -> str:
    number = src.count(before)
    if number != count:
        raise ValueError(
            f"fail-closed source contract: expected {count} exact anchors, "
            f"got {number}: {before[:140]!r}"
        )
    return src.replace(before, after)


def transform(files: dict[str, str]) -> dict[str, str]:
    pp2 = files[FILES[0]]
    # This seeded function already has bf16 and a state poison fail() hook,
    # so opt-in branch is introduced only at the exact preceding try.
    pp2 = replace_once(
        pp2,
        "  try {\n    PrefillBF16ProjectionScope scope;\n",
        '  const char* last_head_flag=std::getenv("HERMES_PP2_SEEDED_LAST_HEAD_ONLY");\n'
        '  const bool last_head_only=last_head_flag && '
        'std::strcmp(last_head_flag,"1")==0;\n'
        '  if(last_head_only && bf16){\n'
        '    err="last-head-only seeded prefill refuses BF16 quality prototype";\n'
        '    return fail();\n'
        '  }\n'
        "  try {\n    PrefillBF16ProjectionScope scope;\n",
    )
    pp2 = replace_once(
        pp2,
        "    std::vector<float> logits=prefill_impl(ids,st,err,h_last_out,&ds,true);\n"
        "    if(logits.size()!=T*(size_t)VOC){",
        "    std::vector<float> logits=prefill_impl("
        "ids,st,err,h_last_out,&ds,true,0,last_head_only);\n"
        "    if(logits.size()!=(last_head_only?(size_t)VOC:T*(size_t)VOC)){",
    )
    # The existing BF16 all-token comparator remains fully intact (and BF16
    # last-only is rejected). RAW only validates the remaining head row.
    pp2 = replace_once(
        pp2,
        "    for(size_t t=0;t<T;++t){\n"
        "      int winner_ref=-1,winner_bf16=-1;",
        "    for(size_t t=0;t<(last_head_only?1:T);++t){\n"
        "      int winner_ref=-1,winner_bf16=-1;",
    )

    batch = files[FILES[1]]
    batch = replace_once(
        batch,
        "  if (full.size() != prompt.size() * VOC) {\n"
        "    if (error.empty()) error = \"seeded prefill returned incomplete logits\";",
        '  const char* last_only_flag=std::getenv("HERMES_PP2_SEEDED_LAST_HEAD_ONLY");\n'
        '  const bool last_only=last_only_flag && '
        'std::string(last_only_flag)=="1";\n'
        "  if (full.size() != (last_only ? VOC : prompt.size() * VOC)) {\n"
        '    if (error.empty()) error = "seeded prefill returned incomplete logits";',
    )

    server = files[FILES[2]]
    server = replace_once(
        server,
        "    if (logits.size() != job->prompt.size() * static_cast<std::size_t>(kVocab)) {",
        '    const char* last_only_flag=std::getenv("HERMES_PP2_SEEDED_LAST_HEAD_ONLY");\n'
        '    const bool last_only=last_only_flag && '
        'std::strcmp(last_only_flag,"1")==0;\n'
        "    if (logits.size() != (last_only ? static_cast<std::size_t>(kVocab) "
        ": job->prompt.size() * static_cast<std::size_t>(kVocab))) {",
    )
    server = replace_once(
        server,
        "    const float* last = logits.data() +\n"
        "        (job->prompt.size() - 1) * static_cast<std::size_t>(kVocab);",
        "    const float* last = logits.data() + logits.size() - "
        "static_cast<std::size_t>(kVocab);",
    )

    cli = files[FILES[3]]
    cli = replace_once(
        cli,
        "      if(all.size() != prompt.size() * (std::size_t)kVocab) {",
        '      const char* last_only_flag=std::getenv("HERMES_PP2_SEEDED_LAST_HEAD_ONLY");\n'
        '      const bool last_only=last_only_flag && '
        'std::string_view(last_only_flag)=="1";\n'
        "      if(all.size() != (last_only ? (std::size_t)kVocab "
        ": prompt.size() * (std::size_t)kVocab)) {",
    )
    return dict(zip(FILES, (pp2, batch, server, cli)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    scratch = args.scratch.resolve(strict=True)
    if source == scratch or source in scratch.parents or scratch in source.parents:
        parser.error("source and scratch must be independent existing directories")
    if (scratch.parent != Path("/tmp") or
            not scratch.name.startswith("h27-prefill-lastonly-candidate-")):
        parser.error("output must be a disposable /tmp/h27-prefill-lastonly-candidate-*")
    inputs = {}
    for rel in FILES:
        src = source / rel
        dst = scratch / rel
        if not src.is_file() or not dst.is_file():
            parser.error(f"source or scratch missing {rel}")
        if src.read_bytes() != dst.read_bytes():
            parser.error(f"scratch {rel} differs from source; refusing edits")
        inputs[rel] = src.read_text(encoding="utf-8")
    outputs = transform(inputs)
    if args.check_only:
        print("PASS scratch last-only four-file source contract; 0 edits")
        return 0
    for rel, result in outputs.items():
        (scratch / rel).write_text(result, encoding="utf-8")
    print("SCRATCH_ONLY_SEEDED_LAST_HEAD_OPTIN_READY " + str(scratch))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"LAST_HEAD_PATCH_FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
