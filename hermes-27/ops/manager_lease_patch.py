#!/usr/bin/env python3
"""Create REVIEW-ONLY GPU Manager lease v1 source patch from a pinned snapshot.

This tool never edits the source input and refuses to write to `/opt/hermes`
or to overwrite an existing output. It does not import or execute manager.py.
The two runtime modules must be staged next to the generated `manager.py`.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
from pathlib import Path
import re
import shutil


AUDITED_SHA256 = "15744e3b769c0f6cd5ed26c2b85e3f282871f4a24681abd324c8cbc329a73eef"


class UnsupportedManagerSource(ValueError):
    pass


def replace_once(source: str, before: str, after: str, description: str) -> str:
    count = source.count(before)
    if count != 1:
        raise UnsupportedManagerSource(f"{description}: expected 1 exact source anchor, got {count}")
    return source.replace(before, after, 1)


def insert_guard(source: str, signature: str, body: str, reason: str) -> str:
    head = signature + "\n"
    count = source.count(head)
    if count != 1:
        raise UnsupportedManagerSource(f"{reason}: expected 1 exact source anchor, got {count}")
    insertion = source.index(head) + len(head)
    # Keep the audited function's actual docstring as the first statement.
    # Otherwise __doc__ changes silently and runtime introspection regresses.
    if source[insertion:].startswith('    """'):
        close = source.find('"""', insertion + len('    """'))
        if close < 0:
            raise UnsupportedManagerSource(f"{reason}: unterminated function docstring")
        newline = source.find("\n", close + 3)
        if newline < 0:
            raise UnsupportedManagerSource(f"{reason}: missing newline after docstring")
        insertion = newline + 1
    return source[:insertion] + body + source[insertion:]


def propose_patched_manager(source: str) -> str:
    """Strict manager.py transformation; rejects unfamiliar source variants."""
    if "_hermes_lease_v1" in source:
        raise UnsupportedManagerSource("manager already contains a lease integration")
    source = insert_guard(source, "async def llm_watchdog() -> None:",
                          "    if _hermes_lease_v1.fenced:\n        return\n",
                          "LLM watchdog restart suppression")
    source = replace_once(source,
        '    _state["last_drive_ts"] = now\n',
        '    _state["last_drive_ts"] = now\n'
        '    if await _hermes_lease_v1.heartbeat():\n        return\n',
        "worker auto-drive/restore gate")
    creation = ('def create_job(kind: str, workflow: dict, preset: str = "", backend: str = "",\n'
                '               model: str = "", params: dict | None = None) -> dict:\n'
                '    jid = uuid.uuid4().hex[:12]')
    source = replace_once(source, creation,
        creation.replace('    jid = uuid.uuid4().hex[:12]',
                         '    if _hermes_lease_v1.fenced:\n'
                         '        raise HTTPException(423, "GPU Manager lease fence active")\n'
                         '    jid = uuid.uuid4().hex[:12]'),
        "central media job creation fence")
    submit_anchor = '    async with _submit_lock:\n'
    if source.count(submit_anchor) != 2:
        raise UnsupportedManagerSource("expected exactly two media submit lock scopes")
    source = source.replace(submit_anchor,
        submit_anchor + '        if _hermes_lease_v1.fenced:\n'
                        '            raise HTTPException(423, "GPU Manager lease fence active")\n')
    for signature, denial in (
        ("async def systemctl(action: str, unit: str, timeout: int = 180) -> bool:",
         "    if _hermes_lease_v1.blocked_operation():\n        return False\n"),
        ("async def systemctl_user(action: str, unit: str, timeout: int = 180) -> bool:",
         "    if _hermes_lease_v1.blocked_operation():\n        return False\n"),
        ("async def unload_llm() -> bool:",
         "    if _hermes_lease_v1.blocked_operation():\n        return False\n"),
        ("async def cleanup_stray_cuda() -> int:",
         "    if _hermes_lease_v1.blocked_operation():\n        return 0\n"),
        ("async def transition_to_media() -> bool:",
         "    if _hermes_lease_v1.blocked_operation():\n        return False\n"),
        ("async def transition_to_llm() -> bool:",
         "    if _hermes_lease_v1.blocked_operation():\n        return False\n"),
        ("async def transition_to_direct() -> bool:",
         "    if _hermes_lease_v1.blocked_operation():\n        return False\n"),
        ("async def transition_to_gaming() -> bool:",
         "    if _hermes_lease_v1.blocked_operation():\n        return False\n"),
        ("async def restore_llm_with_retries(context: str) -> bool:",
         "    if _hermes_lease_v1.blocked_operation():\n        return False\n"),
    ):
        source = insert_guard(source, signature, denial, signature)
    source = insert_guard(source, "def _set_desired(mode: str) -> dict:",
                          '    if _hermes_lease_v1.fenced:\n'
                          '        raise HTTPException(423, "GPU Manager lease fence active")\n',
                          "manual mode switch rejection")
    source = insert_guard(source,
                          "async def force_mode(req: ForceRequest, _: None = Depends(require_laya)) -> dict:",
                          '    if _hermes_lease_v1.fenced:\n'
                          '        raise HTTPException(423, "GPU Manager lease fence active")\n',
                          "force-mode bypass rejection")
    source = replace_once(source,
                          '@app.get("/")\nasync def root()',
                          'from manager_lease_runtime import install_manager_lease\n'
                          '_hermes_lease_v1 = install_manager_lease(globals())\n\n\n'
                          '@app.get("/")\nasync def root()',
                          "lease integration installation")
    # Only adjust GET /status, not similarly-named dictionaries elsewhere.
    head, separator, remaining = source.partition('@app.get("/status")\n')
    if not separator:
        raise UnsupportedManagerSource("no /status route")
    route, boundary, after_route = remaining.partition('\ndef _set_desired(mode: str) -> dict:')
    if not boundary:
        raise UnsupportedManagerSource("status route boundary not found")
    route = replace_once(route,
                         '    return {\n        "desired_mode": desired,',
                         '    return {\n        **_hermes_lease_v1.status_fields(),\n'
                         '        "desired_mode": desired,',
                         "status lease contract attestation")
    route = replace_once(route,
                         '        "current_state": state,',
                         '        "current_state": await _hermes_lease_v1.current_state(state),',
                         "status maintenance state reporting")
    source = head + separator + route + boundary + after_route
    source = replace_once(source,
        '    _db_conn()\n    cleanup_old_artifacts()\n    _worker_task = asyncio.create_task(worker_loop())',
        '    _db_conn()\n    if not _hermes_lease_v1.fenced:\n'
        '        cleanup_old_artifacts()\n'
        '    _worker_task = asyncio.create_task(worker_loop())',
        "startup while crash-fenced")
    return source


def clean_unified_diff(patch_text: str) -> str:
    """Produce a committable patch without changing its application semantics.

    An empty unified-diff context line is normally encoded as `" \\n"`,
    which Git considers trailing whitespace when the *patch file itself* is
    committed. Git accepts an unprefixed empty context line as equivalent.
    A trailing empty context line must also be removed and BOTH last-hunk
    line counts corrected, or git apply reports a corrupt patch.
    """
    lines = patch_text.splitlines(keepends=True)
    if not lines or not lines[0].startswith("--- "):
        raise UnsupportedManagerSource("cannot sanitize a non-unified source diff")
    trailing_context = 0
    while lines and lines[-1] == " \n":
        lines.pop()
        trailing_context += 1
    if trailing_context:
        indices = [i for i, line in enumerate(lines) if line.startswith("@@ -")]
        if not indices:
            raise UnsupportedManagerSource("unified diff has no hunk header")
        index = indices[-1]
        header = lines[index]
        match = re.fullmatch(
            r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@(.*)\n", header)
        if match is None:
            raise UnsupportedManagerSource("unexpected last hunk header while trimming context")
        before_count = int(match.group(2)) - trailing_context
        after_count = int(match.group(4)) - trailing_context
        if min(before_count, after_count) < 1:
            raise UnsupportedManagerSource("trimming context would invalidate hunk")
        lines[index] = (f"@@ -{match.group(1)},{before_count} "
                        f"+{match.group(3)},{after_count} @@{match.group(5)}\n")
    lines = ["\n" if line == " \n" else line for line in lines]
    result = "".join(lines)
    if not result.endswith("\n") or result.endswith("\n\n"):
        raise UnsupportedManagerSource("sanitized patch has malformed EOF")
    if any(line.rstrip("\n").rstrip("\t ") != line.rstrip("\n")
           for line in result.splitlines(keepends=True)):
        raise UnsupportedManagerSource("patch still contains trailing whitespace")
    return result


def stage_patch(source_path: Path, output_dir: Path, expected_sha256: str = AUDITED_SHA256) -> tuple[Path, Path]:
    """Stage to a NEW offline directory, emitting an auditable unified diff."""
    source = source_path.resolve(strict=True)
    target = output_dir.resolve()
    if (source == target or target == source.parent or
            str(target).startswith("/opt/hermes") or
            str(target).startswith("/etc/systemd") or
            target.exists()):
        raise UnsupportedManagerSource("staging destination must be new and outside live system paths")
    body = source.read_bytes()
    if hashlib.sha256(body).hexdigest() != expected_sha256:
        raise UnsupportedManagerSource("manager.py SHA256 drift: re-audit before generating a patch")
    if b"\r\n" in body:
        raise UnsupportedManagerSource("manager source line endings unexpectedly changed")
    original = body.decode("utf-8")
    patched = propose_patched_manager(original)
    # Compile without executing server imports, FastAPI routes or GPU code.
    compile(patched, "staged_manager.py", "exec")
    target.mkdir(parents=True, mode=0o700)
    try:
        staged_manager = target / "manager.py"
        staged_manager.write_text(patched, encoding="utf-8", newline="\n")
        runtime_source = Path(__file__).resolve().parent
        for name in ("manager_lease_runtime.py", "manager_lease_core.py"):
            shutil.copyfile(runtime_source / name, target / name)
        patch = target / "manager-lease-v1.patch"
        delta = difflib.unified_diff(original.splitlines(keepends=True),
                                     patched.splitlines(keepends=True),
                                     fromfile="a/manager.py", tofile="b/manager.py")
        patch.write_text(clean_unified_diff("".join(delta)), encoding="utf-8", newline="\n")
        return staged_manager, patch
    except BaseException:
        shutil.rmtree(target)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="offline copy of audited manager.py")
    parser.add_argument("--output-dir", type=Path, required=True, help="new staging directory only")
    parser.add_argument("--expected-sha256", default=AUDITED_SHA256)
    args = parser.parse_args()
    try:
        manager, patch = stage_patch(args.source, args.output_dir, args.expected_sha256)
    except (OSError, ValueError, UnicodeError) as exc:
        print(f"REFUSED: {exc}")
        return 2
    print(f"PREPARED ONLY: staged manager at {manager}; review diff at {patch}")
    print("No live manager, service or GPU action performed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
