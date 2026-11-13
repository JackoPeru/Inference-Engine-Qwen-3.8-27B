"""Offline source transformation tests against exact reviewed anchor shapes."""

import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from ops import manager_lease_patch as patch


def manager_fixture() -> str:
    # Minimal valid Python containing audited integration anchors. Does not
    # import the actual GPU Manager or any installed service dependencies.
    return """def create_job(kind: str, workflow: dict, preset: str = "", backend: str = "",
               model: str = "", params: dict | None = None) -> dict:
    jid = uuid.uuid4().hex[:12]
    return {"job_id": jid}
async def submit_job():
    async with _submit_lock:
        return create_job("image", {})
async def submit_smart():
    async with _submit_lock:
        return create_job("video", {})
async def systemctl(action: str, unit: str, timeout: int = 180) -> bool:
    return True
async def systemctl_user(action: str, unit: str, timeout: int = 180) -> bool:
    return True
async def unload_llm() -> bool:
    return True
async def cleanup_stray_cuda() -> int:
    return 0
async def transition_to_media() -> bool:
    return True
async def transition_to_llm() -> bool:
    return True
async def transition_to_direct() -> bool:
    return True
async def transition_to_gaming() -> bool:
    return True
async def restore_llm_with_retries(context: str) -> bool:
    return True
async def drive_auto_once() -> None:
    now = time.monotonic()
    _state["last_drive_ts"] = now
async def llm_watchdog() -> None:
    return None
@app.get("/")
async def root():
    return {}
@app.get("/status")
async def status():
    return {
        "desired_mode": desired,
        "current_state": state,
    }

def _set_desired(mode: str) -> dict:
    return {}
async def force_mode(req: ForceRequest, _: None = Depends(require_laya)) -> dict:
    return {}
async def on_startup() -> None:
    _db_conn()
    cleanup_old_artifacts()
    _worker_task = asyncio.create_task(worker_loop())
"""


class ManagerLeaseSourcePatchTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix="hermes-lease-patch-")
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root / "manager.py"
        self.source.write_bytes(manager_fixture().encode("utf-8"))
        self.sha = hashlib.sha256(self.source.read_bytes()).hexdigest()

    def test_transform_compiles_and_contains_all_guards(self):
        text = patch.propose_patched_manager(manager_fixture())
        compile(text, "patched_manager.py", "exec")
        self.assertEqual(text.count("_hermes_lease_v1.blocked_operation()"), 9)
        self.assertIn("await _hermes_lease_v1.heartbeat()", text)
        self.assertIn("if _hermes_lease_v1.fenced", text)
        self.assertIn("install_manager_lease(globals())", text)
        self.assertIn("await _hermes_lease_v1.current_state(state)", text)
        self.assertIn("**_hermes_lease_v1.status_fields()", text)
        self.assertEqual(text.count('raise HTTPException(423, "GPU Manager lease fence active")'), 5)
        creation = text.index("def create_job(")
        self.assertLess(text.index("if _hermes_lease_v1.fenced", creation),
                        text.index("jid = uuid.uuid4().hex[:12]", creation))

    def test_source_preserved_and_diff_generated(self):
        before = self.source.read_bytes()
        manager, delta = patch.stage_patch(self.source, self.root / "stage", self.sha)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertTrue(manager.is_file())
        self.assertTrue(delta.is_file())
        self.assertTrue((manager.parent / "manager_lease_core.py").is_file())
        self.assertTrue((manager.parent / "manager_lease_runtime.py").is_file())
        self.assertIn("+++ b/manager.py", delta.read_text())
        raw = delta.read_bytes()
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.endswith(b"\n\n"))
        self.assertFalse(any(line.endswith((b" ", b"\t")) for line in raw.splitlines()))

    @unittest.skipUnless(shutil.which("git"), "git needed for offline patch apply validation")
    def test_git_apply_dry_run_and_exact_result_offline(self):
        expected = patch.propose_patched_manager(manager_fixture()).encode("utf-8")
        _, delta = patch.stage_patch(self.source, self.root / "stage", self.sha)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "core.autocrlf", "false"],
                       cwd=self.root, check=True)
        subprocess.run(["git", "apply", "--check", str(delta)],
                       cwd=self.root, check=True)
        subprocess.run(["git", "apply", str(delta)], cwd=self.root, check=True)
        self.assertEqual(self.source.read_bytes(), expected)
        subprocess.run(["git", "add", str(delta)], cwd=self.root, check=True)
        subprocess.run(["git", "diff", "--cached", "--check"],
                       cwd=self.root, check=True)

    def test_rejects_changed_source_hash(self):
        with self.assertRaisesRegex(patch.UnsupportedManagerSource, "SHA256 drift"):
            patch.stage_patch(self.source, self.root / "new-stage", "0" * 64)
        self.assertFalse((self.root / "new-stage").exists())

    def test_rejects_reusing_existing_output(self):
        output = self.root / "stage"
        output.mkdir()
        with self.assertRaisesRegex(patch.UnsupportedManagerSource, "must be new"):
            patch.stage_patch(self.source, output, self.sha)

    def test_rejects_same_directory_as_source(self):
        with self.assertRaisesRegex(patch.UnsupportedManagerSource, "must be new"):
            patch.stage_patch(self.source, self.root, self.sha)

    def test_rejects_missing_audited_anchor(self):
        changed = manager_fixture().replace("async def llm_watchdog() -> None:",
                                            "async def renamed_watchdog() -> None:")
        with self.assertRaisesRegex(patch.UnsupportedManagerSource, "LLM watchdog"):
            patch.propose_patched_manager(changed)

    def test_rejects_ambiguous_duplicate_anchor(self):
        changed = manager_fixture() + "\nasync def llm_watchdog() -> None:\n    pass\n"
        with self.assertRaisesRegex(patch.UnsupportedManagerSource, "got 2"):
            patch.propose_patched_manager(changed)

    def test_rejects_double_patching(self):
        transformed = patch.propose_patched_manager(manager_fixture())
        with self.assertRaisesRegex(patch.UnsupportedManagerSource, "already contains"):
            patch.propose_patched_manager(transformed)

    def test_rejects_status_route_shape_drift(self):
        changed = manager_fixture().replace('        "current_state": state,',
                                            '        "reported_state": state,')
        with self.assertRaisesRegex(patch.UnsupportedManagerSource, "status maintenance"):
            patch.propose_patched_manager(changed)

    def test_rejects_startup_shape_drift(self):
        changed = manager_fixture().replace("    cleanup_old_artifacts()",
                                            "    cleanup_new_artifacts()")
        with self.assertRaisesRegex(patch.UnsupportedManagerSource, "startup"):
            patch.propose_patched_manager(changed)

    def test_rejects_lost_media_submit_lock(self):
        changed = manager_fixture().replace("async with _submit_lock:",
                                            "async with custom_lock:", 1)
        with self.assertRaisesRegex(patch.UnsupportedManagerSource, "two media submit"):
            patch.propose_patched_manager(changed)

    def test_rejects_changed_central_create_job_function(self):
        changed = manager_fixture().replace("jid = uuid.uuid4().hex[:12]",
                                            "jid = some_new_id()")
        with self.assertRaisesRegex(patch.UnsupportedManagerSource, "central media"):
            patch.propose_patched_manager(changed)


if __name__ == "__main__":
    unittest.main()
