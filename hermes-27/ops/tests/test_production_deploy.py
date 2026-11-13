"""No GPU/network/service actions: production deployment guards and rollback."""

import importlib.util
import hashlib
import json
import subprocess
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock


MODULE = Path(__file__).resolve().parents[1] / "production_deploy.py"
spec = importlib.util.spec_from_file_location("hermes_production_deploy", MODULE)
prod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prod
spec.loader.exec_module(prod)


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        model = self.root / "model.gguf"
        with model.open("wb") as stream:
            stream.write(b"GGUF")
            stream.truncate(1_000_000_001)  # sparse; no expensive real-model hash
        tokenizer = self.root / "tokenizer.json"
        tokenizer.write_text("{}")
        executable = self.root / "executable"
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o700)
        def fixture_hook(label):
            path = self.root / label
            path.write_text("#!/bin/sh\nexit 0\n")
            path.chmod(0o700)
            return str(path)
        build_dir = self.root / "build"
        build_dir.mkdir()
        server = build_dir / "hermes_server"
        server.write_text(
            "#!/usr/bin/env python3\n"
            "print('Usage: hermes_server --model --gguf-dump --tokenizer --model-id --queue --context --api-key-file')\n"
        )
        server.chmod(0o700)
        dumper = build_dir / "gguf_dump"
        dumper.write_text(
            "#!/usr/bin/env python3\n"
            "print(\"{%- for message in messages %}{{ message['role'] }}: {{ message['content'] }} "
            "{%- endfor %}{%- if add_generation_prompt %}assistant: "
            "{%- else %}end-of-turn{% endif %}\")\n"
        )
        dumper.chmod(0o700)
        serving_dir = self.root / "serving"
        serving_dir.mkdir()
        bridge = serving_dir / "tokenizer_bridge.py"
        bridge.write_text(
            "import json, sys\n"
            "print(json.dumps({'ready': True, 'incremental': False}), flush=True)\n"
            "for line in sys.stdin:\n"
            "    print(json.dumps({'ids': [123, 124]}), flush=True)\n"
        )
        self.server, self.dumper, self.bridge = server, dumper, bridge
        token = self.root / "private-token"
        token.write_text("test-placeholder-not-secret")
        token.chmod(0o600)
        self.cfg = {
            "model_path": str(model), "model_sha256": "a" * 64,
            "tokenizer_path": str(tokenizer), "tokenizer_sha256": "b" * 64,
            "build_dir": str(build_dir),
            "server_path": str(server),
            "server_sha256": hashlib.sha256(server.read_bytes()).hexdigest(),
            "gguf_dump_path": str(dumper),
            "gguf_dump_sha256": hashlib.sha256(dumper.read_bytes()).hexdigest(),
            "tokenizer_bridge_sha256": hashlib.sha256(bridge.read_bytes()).hexdigest(),
            "python_path": sys.executable, "model_id": "hermes-27",
            "manager_status_url": "http://127.0.0.1:8643/status",
            "manager_token_file": str(token),
            "tabby_health_url": "http://127.0.0.1:8000/v1/models",
            "state_dir": str(self.root / "state"),
            "drain_command": [fixture_hook("ingress_drain")],
            "drain_check_command": [fixture_hook("ingress_check")],
            "restore_command": [fixture_hook("ingress_restore")], "port": 18127,
            "manager_lease_acquire_command": [fixture_hook("lease_acquire")],
            "manager_lease_renew_command": [fixture_hook("lease_renew")],
            "manager_lease_release_command": [fixture_hook("lease_release")],
            "tabby_restore_timeout_sec": 1,
        }
        self.config_path = self.root / "config.json"
        self.cfg["_path"] = str(self.config_path.resolve())
        self.write_config()
        if sys.platform == "win32":
            patcher = mock.patch.object(prod.subprocess, "run", side_effect=self.simulate_posix_executables)
            patcher.start()
            self.addCleanup(patcher.stop)
            permission_patcher = mock.patch.object(prod, "token_file_private", return_value=True)
            permission_patcher.start()
            self.addCleanup(permission_patcher.stop)

    def simulate_posix_executables(self, argv, **_kwargs):
        """Execute fake shebang fixtures as data on Windows, without a shell."""
        program = str(argv[0])
        if program == str(self.server) and "--help" in argv:
            text = self.server.read_text()
            output = ("Usage: unknown-old-server" if "unknown-old-server" in text else
                      "Usage: hermes_server --gguf-dump --tokenizer --model-id --queue --context --api-key-file")
        elif program == str(self.dumper) and "--chat-template" in argv:
            text = self.dumper.read_text()
            output = ("version=3 n_tensors=866\nKV-S tokenizer.chat_template = " if "version=3" in text else
                      "{%- for message in messages %}{{ message['role'] }}: {{ message['content'] }}"
                      "{%- endfor %}{%- if add_generation_prompt %}assistant{% endif %}")
        elif len(argv) > 1 and Path(argv[1]).resolve() == self.bridge.resolve():
            text = self.bridge.read_text()
            output = ("invalid protocol\n" if "invalid protocol" in text else
                      '{"ready": true}\n{"ids": [123, 124]}\n')
        else:
            raise AssertionError(f"offline test attempted an unmocked process: {program}")
        return subprocess.CompletedProcess(argv, 0, stdout=output, stderr="")

    def write_config(self):
        self.config_path.write_text(json.dumps({k: v for k, v in self.cfg.items() if k != "_path"}))

    def test_config_is_strict_and_valid(self):
        result = prod.read_config(self.config_path)
        self.assertEqual(result["port"], 18127)
        prod.check_static(result, verify_hash=False)

    def test_stale_gguf_dump_is_rejected_even_if_operator_re_pins_its_hash(self):
        self.dumper.write_text(
            "#!/usr/bin/env python3\n"
            "print('version=3 n_tensors=866 n_kv=50 align=32 data_off=10995680')\n"
            "print('KV-S tokenizer.chat_template = {%- for message in messages %}...')\n"
        )
        self.dumper.chmod(0o700)
        self.cfg["gguf_dump_sha256"] = hashlib.sha256(self.dumper.read_bytes()).hexdigest()
        with self.assertRaisesRegex(prod.DeploymentError, "raw Jinja"):
            prod.check_build_contract(self.cfg)

    def test_rejects_mixed_build_directories(self):
        other = self.root / "other-build"
        other.mkdir()
        dumper_copy = other / "gguf_dump"
        dumper_copy.write_bytes(self.dumper.read_bytes())
        dumper_copy.chmod(0o700)
        self.cfg["gguf_dump_path"] = str(dumper_copy)
        with self.assertRaisesRegex(prod.DeploymentError, "same configured build_dir"):
            prod.check_build_contract(self.cfg)

    def test_rejects_modified_binary_hash(self):
        self.dumper.write_text("#!/bin/sh\nexit 0\n")
        with self.assertRaisesRegex(prod.DeploymentError, "SHA256 mismatch"):
            prod.check_build_contract(self.cfg)

    def test_rejects_bridge_that_has_no_ready_handshake(self):
        self.bridge.write_text("print('invalid protocol')\n")
        self.cfg["tokenizer_bridge_sha256"] = hashlib.sha256(self.bridge.read_bytes()).hexdigest()
        with self.assertRaisesRegex(prod.DeploymentError, "tokenizer bridge"):
            prod.check_build_contract(self.cfg)

    def test_rejects_incompatible_server_help_even_with_matching_hash(self):
        self.server.write_text("#!/usr/bin/env python3\nprint('Usage: unknown-old-server')\n")
        self.server.chmod(0o700)
        self.cfg["server_sha256"] = hashlib.sha256(self.server.read_bytes()).hexdigest()
        with self.assertRaisesRegex(prod.DeploymentError, "serving options/version"):
            prod.check_build_contract(self.cfg)

    def test_dry_run_fails_closed_before_any_service_change_for_stale_dump(self):
        self.dumper.write_text("#!/usr/bin/env python3\nprint('version=3 n_tensors=866')\n")
        self.dumper.chmod(0o700)
        self.cfg["gguf_dump_sha256"] = hashlib.sha256(self.dumper.read_bytes()).hexdigest()
        self.write_config()
        with mock.patch.object(sys, "argv", ["production_deploy.py", "dry-run", "--config",
                                             str(self.config_path), "--skip-hash"]), \
             mock.patch.object(prod, "change_service") as change, \
             mock.patch.object(prod, "run_hook") as hook, \
             mock.patch.object(prod, "service_active") as service:
            self.assertEqual(prod.main(), 1)
            change.assert_not_called()
            hook.assert_not_called()
            service.assert_not_called()

    def test_wrong_model_magic_fails_closed(self):
        with Path(self.cfg["model_path"]).open("r+b") as stream:
            stream.write(b"NOPE")
        with self.assertRaisesRegex(prod.DeploymentError, "GGUF"):
            prod.check_static(self.cfg, verify_hash=False)

    def test_hash_mismatch_fails_closed(self):
        with self.assertRaisesRegex(prod.DeploymentError, "SHA256 mismatch"):
            prod.check_sha256(Path(self.cfg["tokenizer_path"]), "c" * 64)

    def test_rejects_remote_and_malicious_auth_urls(self):
        for url in (
            "http://example.org:8643/status", "http://127.0.0.1:8643@evil.org/status",
            "https://127.0.0.1:8643/status", "http://127.0.0.1:8643/not-status",
        ):
            with self.subTest(url=url):
                self.cfg["manager_status_url"] = url
                self.write_config()
                with self.assertRaises(prod.DeploymentError):
                    prod.read_config(self.config_path)

    def test_rejects_public_binding_and_tabby_port(self):
        self.cfg["host"] = "0.0.0.0"
        self.write_config()
        with self.assertRaisesRegex(prod.DeploymentError, "public binding"):
            prod.read_config(self.config_path)
        self.cfg["host"] = "127.0.0.1"
        self.cfg["port"] = 8000
        self.write_config()
        with self.assertRaisesRegex(prod.DeploymentError, "Tabby"):
            prod.read_config(self.config_path)

    @staticmethod
    def good_manager():
        return {"desired_mode": "AUTO", "current_state": "LLM_READY", "llm_serving": True,
                "llm_online": True, "queue_length": 0, "current_job": None,
                "direct_active": False, "gaming_active": False, "character_training": [],
                "worker_alive_s": 2.0, "watchdog_lease_contract_version": 1,
                "manager_instance_id": "a" * 32, "maintenance_lease": None}

    @staticmethod
    def ingress_proof(blocked=True, inflight=0, scope="all-tabby-clients", timestamp=None):
        return json.dumps({"blocked": blocked, "inflight": inflight, "scope": scope,
                           "observed_at_unix": time.time() if timestamp is None else timestamp})

    def test_manager_rejects_busy_or_unhealthy_states(self):
        for delta in ({"queue_length": 1}, {"current_job": "123"},
                      {"current_state": "MEDIA_BUSY"}, {"llm_serving": False},
                      {"worker_alive_s": 240}, {"desired_mode": "GAMING"}):
            status = {**self.good_manager(), **delta}
            with self.subTest(delta=delta), mock.patch.object(prod, "request_json", return_value=status):
                with self.assertRaises(prod.DeploymentError):
                    prod.manager_guard(self.cfg)

    def test_drain_proof_needs_closed_and_zero_inflight(self):
        for output in ("{}", '{"blocked":true,"inflight":"0"}', "not-json",
                       self.ingress_proof(scope="partial"),
                       self.ingress_proof(timestamp=time.time() - 90),
                       self.ingress_proof(inflight=-1)):
            with self.subTest(output=output), mock.patch.object(prod, "run_hook", return_value=output):
                with self.assertRaises(prod.DeploymentError):
                    prod.wait_ingress_closed(self.cfg)
        with mock.patch.object(prod, "run_hook", return_value=self.ingress_proof()):
            prod.wait_ingress_closed(self.cfg)

    def test_rollback_restores_tabby_then_manager_then_ingress(self):
        _, journal = prod.state_paths(self.cfg)
        prod.save_state(journal, {"config_path": self.cfg["_path"], "tabby_was_active": True,
                                  "manager_was_active": True, "ingress_pending": True})
        steps = []

        def change(action, name):
            steps.append((action, name))

        def fake_hook(argv, *args):
            if argv == self.cfg["drain_check_command"]:
                return self.ingress_proof(blocked=not ("ingress", "restore") in steps)
            steps.append(("ingress", "restore"))
            return ""

        with mock.patch.object(prod, "service_active", return_value=False), \
             mock.patch.object(prod, "change_service", side_effect=change), \
             mock.patch.object(prod, "request_json", return_value={}), \
             mock.patch.object(prod, "manager_status", return_value=self.good_manager()), \
             mock.patch.object(prod, "manager_guard", return_value=self.good_manager()), \
             mock.patch.object(prod, "run_hook", side_effect=fake_hook):
            prod.restore(self.cfg, journal)
        self.assertEqual(steps, [
            ("start", "hermes-tabby.service"),
            ("start", "hermes-gpu-manager.service"),
            ("ingress", "restore")])
        self.assertFalse(journal.exists())

    def test_drain_failure_reopens_ingress_without_stopping_services(self):
        hooks = []
        ingress_blocked = True

        def hook(command, timeout=30):
            hooks.append(command)
            nonlocal ingress_blocked
            if command == self.cfg["restore_command"]:
                ingress_blocked = False
            return self.ingress_proof(blocked=ingress_blocked)

        with mock.patch.object(prod, "service_active", return_value=True), \
             mock.patch.object(prod, "manager_guard", return_value=self.good_manager()), \
             mock.patch.object(prod, "request_json", return_value={}), \
             mock.patch.object(prod, "manager_status", return_value=self.good_manager()), \
             mock.patch.object(prod, "port_available"), \
             mock.patch.object(prod, "run_hook", side_effect=hook), \
             mock.patch.object(prod, "wait_ingress_closed", side_effect=prod.DeploymentError("unproven")), \
             mock.patch.object(prod, "change_service") as changes:
            _, journal = prod.state_paths(self.cfg)
            with self.assertRaisesRegex(prod.DeploymentError, "unproven"):
                prod.supervise(self.cfg, journal)
            changes.assert_not_called()
            self.assertEqual(hooks, [self.cfg["drain_command"], self.cfg["drain_check_command"],
                                     self.cfg["restore_command"], self.cfg["drain_check_command"]])
            self.assertFalse(journal.exists())

    def test_rollback_does_not_restart_tabby_if_hermes_still_running(self):
        _, journal = prod.state_paths(self.cfg)
        prod.save_state(journal, {"config_path": self.cfg["_path"], "server_pid": 987,
                                  "tabby_was_active": True, "manager_was_active": True,
                                  "ingress_pending": True})
        with mock.patch.object(prod, "stop_child"), \
             mock.patch.object(prod, "process_running", return_value=True), \
             mock.patch.object(prod, "change_service") as changes:
            with self.assertRaisesRegex(prod.DeploymentError, "Tabby restart withheld"):
                prod.restore(self.cfg, journal)
            changes.assert_not_called()
        self.assertTrue(journal.exists())

    def test_failed_restore_keeps_journal_and_ingress_closed(self):
        _, journal = prod.state_paths(self.cfg)
        prod.save_state(journal, {"config_path": self.cfg["_path"], "tabby_was_active": True,
                                  "manager_was_active": True, "ingress_pending": True})
        with mock.patch.object(prod, "service_active", return_value=False), \
             mock.patch.object(prod, "change_service", side_effect=RuntimeError("no sudo")), \
             mock.patch.object(prod, "run_hook") as reopen:
            with self.assertRaisesRegex(prod.DeploymentError, "ROLLBACK INCOMPLETE"):
                prod.restore(self.cfg, journal)
            reopen.assert_not_called()
        self.assertTrue(journal.exists())

    def test_no_mutation_without_explicit_authorization(self):
        with mock.patch.object(sys, "argv", ["production_deploy.py", "run", "--config", str(self.config_path)]), \
             mock.patch.object(prod, "change_service") as changes, \
             mock.patch.object(prod, "run_hook") as hooks:
            self.assertEqual(prod.main(), 1)
            changes.assert_not_called()
            hooks.assert_not_called()

    def test_successful_dry_run_never_checks_or_changes_services(self):
        with mock.patch.object(sys, "argv", ["production_deploy.py", "dry-run", "--config",
                                             str(self.config_path), "--skip-hash"]), \
             mock.patch.object(prod, "service_active") as service, \
             mock.patch.object(prod, "change_service") as changes, \
             mock.patch.object(prod, "request_json") as http, \
             mock.patch.object(prod, "run_hook") as hooks:
            self.assertEqual(prod.main(), 0)
            service.assert_not_called()
            changes.assert_not_called()
            http.assert_not_called()
            hooks.assert_not_called()

    def test_state_file_private(self):
        _, journal = prod.state_paths(self.cfg)
        prod.save_state(journal, {"phase": "draining"})
        if sys.platform != "win32":
            self.assertEqual(journal.stat().st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
