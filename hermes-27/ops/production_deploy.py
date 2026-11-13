#!/usr/bin/env python3
"""Opt-in HERMES-27 maintenance supervisor. No action without `run --authorized`.

The existing Tabby endpoint is a protected rollback target. GPU Manager does
not currently expose a Hermes maintenance lease, so this tool requires a
verified external ingress drain before it stops the manager and Tabby.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request


class DeploymentError(RuntimeError):
    pass


STOP_REQUESTED = False


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} hermes-prod: {message}", flush=True)


def on_stop(_signal: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def read_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise DeploymentError("deployment config must be a JSON object")
    required = (
        "model_path", "model_sha256", "tokenizer_path", "tokenizer_sha256",
        "build_dir", "server_path", "server_sha256", "gguf_dump_path",
        "gguf_dump_sha256", "tokenizer_bridge_sha256", "python_path", "model_id",
        "manager_status_url", "manager_token_file", "tabby_health_url",
        "drain_command", "drain_check_command", "restore_command", "state_dir",
    )
    for key in required:
        if key not in data or not data[key]:
            raise DeploymentError(f"missing required config field: {key}")
    for key in ("model_sha256", "tokenizer_sha256", "server_sha256",
                "gguf_dump_sha256", "tokenizer_bridge_sha256"):
        if not isinstance(data[key], str) or not re.fullmatch(r"[0-9a-fA-F]{64}", data[key]):
            raise DeploymentError(f"{key} must be a full 64-digit SHA256 digest")
    for key in ("drain_command", "drain_check_command", "restore_command"):
        command = data[key]
        if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
            raise DeploymentError(f"{key} must be a nonempty argv array (no shell)")
        if not Path(command[0]).is_absolute():
            raise DeploymentError(f"{key} executable must be an absolute path")
    for key in ("model_path", "tokenizer_path", "build_dir", "server_path",
                "gguf_dump_path", "python_path", "manager_token_file", "state_dir"):
        if not isinstance(data[key], str) or not Path(data[key]).is_absolute():
            raise DeploymentError(f"{key} must be an absolute path")
    for key in ("manager_status_url", "tabby_health_url"):
        parsed = urllib.parse.urlsplit(data[key])
        if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or
                parsed.username is not None or parsed.password is not None or
                parsed.fragment or not (parsed.port and 1 <= parsed.port <= 65535)):
            raise DeploymentError(f"{key} must use unambiguous loopback HTTP URL")
    if urllib.parse.urlsplit(data["manager_status_url"]).path != "/status":
        raise DeploymentError("manager_status_url must target the verified /status endpoint")
    if data.get("host", "127.0.0.1") != "127.0.0.1":
        raise DeploymentError("public binding forbidden: use loopback and a separately authenticated proxy")
    port = data.get("port", 18127)
    if type(port) is not int or not (1024 <= port <= 65535) or port == 8000:
        raise DeploymentError("port must be 1024..65535 and cannot replace Tabby port 8000")
    for key, low, high, default in (("active", 1, 16, 2), ("queue", 1, 32, 2), ("context", 1, 8192, 128)):
        n = data.get(key, default)
        if type(n) is not int or not low <= n <= high:
            raise DeploymentError(f"{key} outside supported range")
    for key, low, high, default in (("drain_timeout_sec", 1, 1800, 90),
                                   ("hermes_start_timeout_sec", 10, 1800, 240),
                                   ("tabby_restore_timeout_sec", 1, 1800, 240),
                                   ("manager_restore_timeout_sec", 1, 1800, 180),
                                   ("max_other_gpu_used_mib", 500, 8000, 3000)):
        n = data.get(key, default)
        if type(n) is not int or not low <= n <= high:
            raise DeploymentError(f"{key} outside supported range")
    for key in ("tabby_token_file", "http_token_file"):
        if key in data and (not isinstance(data[key], str) or not Path(data[key]).is_absolute()):
            raise DeploymentError(f"{key} must be an absolute file path")
    return data


def check_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest().lower() != expected.lower():
        raise DeploymentError(f"SHA256 mismatch for {path}")


def check_static(cfg: dict, verify_hash: bool) -> None:
    model = Path(cfg["model_path"])
    tokenizer = Path(cfg["tokenizer_path"])
    if not model.is_file() or model.stat().st_size < 1_000_000_000:
        raise DeploymentError("GGUF checkpoint missing or unexpectedly small")
    with model.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            raise DeploymentError("checkpoint magic is not GGUF")
    if not tokenizer.is_file():
        raise DeploymentError("tokenizer missing")
    for key in ("server_path", "gguf_dump_path", "python_path"):
        binary = Path(cfg[key])
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise DeploymentError(f"missing or nonexecutable: {key}")
    # Every mode that can precede cutover, including dry-run --skip-hash,
    # must check the *deployed* binary behavior; a successful build elsewhere
    # does not make stale artifacts under ~/hermes-27/build safe to start.
    check_build_contract(cfg)
    for key in ("drain_command", "drain_check_command", "restore_command"):
        executable = Path(cfg[key][0])
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise DeploymentError(f"missing or nonexecutable ingress hook: {key}")
    token = Path(cfg["manager_token_file"])
    if not token.is_file() or not token.read_text(encoding="utf-8").strip():
        raise DeploymentError("GPU Manager status token unavailable")
    if token.stat().st_mode & 0o007:
        raise DeploymentError("GPU Manager token file must not be accessible to other users")
    if verify_hash:
        log("validating full GGUF and tokenizer SHA256")
        check_sha256(model, cfg["model_sha256"])
        check_sha256(tokenizer, cfg["tokenizer_sha256"])


def chat_template_guard(template: str) -> None:
    """Fail on the legacy descriptive dump accepted by older gguf_dump builds."""
    text = template.lstrip("\ufeff \t\r\n")
    if (not 100 <= len(text) <= 1_000_000 or
            not text.startswith(("{%", "{{")) or
            "messages" not in text or "add_generation_prompt" not in text or
            not any(marker in text for marker in ("{%-", "{{", "{%")) or
            re.search(r"(?m)^(?:version=\d+ n_tensors=|KV-S |KV-I |tensors: NVFP4=|--- MTP/search ---)", text)):
        raise DeploymentError(
            "gguf_dump --chat-template did not return raw Jinja; stale or incompatible deployed binary"
        )


def check_build_contract(cfg: dict) -> None:
    """CPU-only verification of the exact deployed serving/metadata toolchain."""
    build = Path(cfg["build_dir"]).resolve(strict=True)
    server = Path(cfg["server_path"])
    dumper = Path(cfg["gguf_dump_path"])
    if (server.name != "hermes_server" or dumper.name != "gguf_dump" or
            server.parent.resolve(strict=True) != build or
            dumper.parent.resolve(strict=True) != build or
            server.is_symlink() or dumper.is_symlink()):
        raise DeploymentError("hermes_server and gguf_dump must be real binaries in the same configured build_dir")
    bridge = build.parent / "serving" / "tokenizer_bridge.py"
    if not bridge.is_file():
        raise DeploymentError(f"tokenizer bridge missing from server source tree: {bridge}")
    # Pins are generated from the reviewed deployed build and bridge. The
    # original checkpoint has its own (optionally skippable) large hash.
    for path, field in ((server, "server_sha256"), (dumper, "gguf_dump_sha256"),
                        (bridge, "tokenizer_bridge_sha256")):
        check_sha256(path, cfg[field])
    try:
        help_result = subprocess.run([str(server), "--help"], capture_output=True,
                                     text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise DeploymentError(f"deployed hermes_server --help failed: {exc}") from exc
    help_text = help_result.stdout + help_result.stderr
    if not all(flag in help_text for flag in (
        "Usage: hermes_server", "--gguf-dump", "--tokenizer", "--model-id",
        "--queue", "--context", "--api-key-file",
    )):
        raise DeploymentError("deployed hermes_server --help is missing required serving options/version")
    try:
        result = subprocess.run([str(dumper), cfg["model_path"], "--chat-template"],
                                capture_output=True, text=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise DeploymentError(f"gguf_dump --chat-template probe failed: {exc}") from exc
    chat_template_guard(result.stdout)
    # The server resolves its helper from the parent of its build directory.
    # Exercise exactly that helper with the configured Python/tokenizer/GGUF,
    # without importing CUDA or loading any model weights.
    probe = json.dumps({"op": "encode", "messages": [
        {"role": "user", "content": "HERMES preflight"}], "thinking": False}) + "\n"
    try:
        bridge_result = subprocess.run(
            [cfg["python_path"], str(bridge), "--tokenizer", cfg["tokenizer_path"],
             "--model", cfg["model_path"], "--gguf-dump", str(dumper)],
            input=probe, capture_output=True, text=True, timeout=45, check=True)
        lines = bridge_result.stdout.splitlines()
        if len(lines) != 2:
            raise DeploymentError("tokenizer bridge returned unexpected number of protocol records")
        ready, encoded = (json.loads(line) for line in lines)
        tokens = encoded.get("ids") if isinstance(encoded, dict) else None
        if (not isinstance(ready, dict) or ready.get("ready") is not True or
                not isinstance(tokens, list) or not 1 <= len(tokens) <= 4096 or
                not all(type(token) is int and 0 <= token < 248320 for token in tokens)):
            raise DeploymentError("tokenizer bridge failed ready/encode CPU handshake")
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
        raise DeploymentError(f"tokenizer bridge CPU handshake failed: {exc}") from exc
    log("deployed serving toolchain validated: hashes, same build_dir, raw Jinja, CPU tokenizer handshake")


def request_json(url: str, token_file: str = "", timeout: float = 5.0) -> dict:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.username or parsed.password:
        raise DeploymentError("authenticated probes may only target the local loopback")
    headers: dict[str, str] = {}
    if token_file:
        token = Path(token_file).read_text(encoding="utf-8").strip()
        if not token:
            raise DeploymentError("configured auth token is empty")
        headers["Authorization"] = f"Bearer {token}"
    class RejectRedirects(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args: object, **kwargs: object) -> None:
            return None

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), RejectRedirects())
    with opener.open(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
        if response.status != 200:
            raise DeploymentError(f"HTTP status {response.status} from {url}")
        payload = json.loads(response.read(512 * 1024))
        if not isinstance(payload, dict):
            raise DeploymentError("HTTP probe returned non-object JSON")
        return payload


def manager_guard(cfg: dict) -> dict:
    status = request_json(cfg["manager_status_url"], cfg["manager_token_file"], 25)
    if status.get("desired_mode") not in ("AUTO", "LLM"):
        raise DeploymentError("GPU Manager is not in AUTO/LLM mode")
    if status.get("current_state") != "LLM_READY":
        raise DeploymentError("GPU Manager is not stable in LLM_READY")
    if status.get("llm_serving") is not True or status.get("llm_online") is not True:
        raise DeploymentError("Tabby baseline not serving according to GPU Manager")
    if status.get("queue_length") != 0 or status.get("current_job") is not None:
        raise DeploymentError("GPU Manager has active/queued work")
    if status.get("direct_active") or status.get("gaming_active") or status.get("character_training"):
        raise DeploymentError("GPU Manager has competing workload")
    age = status.get("worker_alive_s")
    if not isinstance(age, (float, int)) or age > 120 or age < 0:
        raise DeploymentError("GPU Manager worker freshness unverified")
    return status


def command(argv: list[str], timeout: int = 45, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(argv, check=True, text=True, timeout=timeout,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None)


def service_active(name: str) -> bool:
    return subprocess.run(["systemctl", "is-active", "--quiet", name],
                          check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def change_service(action: str, name: str) -> None:
    if action not in ("start", "stop") or name not in ("hermes-gpu-manager.service", "hermes-tabby.service"):
        raise DeploymentError("unsupported service change")
    command(["sudo", "-n", "systemctl", action, name], 150)
    desired = action == "start"
    for _ in range(30):
        if service_active(name) == desired:
            return
        time.sleep(1)
    raise DeploymentError(f"{name} did not reach required state after {action}")


def gpu_snapshot() -> list[tuple[str, int, int]]:
    data = command(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"], 15).stdout
    lines = [line.strip() for line in data.splitlines() if line.strip()]
    if len(lines) != 2:
        raise DeploymentError(f"expected two NVIDIA GPUs; detected {len(lines)}")
    gpus: list[tuple[str, int, int]] = []
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise DeploymentError("cannot parse nvidia-smi GPU snapshot")
        name, total, used = parts[0], int(parts[1]), int(parts[2])
        if "5060 Ti" not in name or total < 15000 or used < 0:
            raise DeploymentError("GPU model/VRAM mismatch")
        gpus.append((name, total, used))
    return gpus


def machine_guard() -> None:
    if sys.platform != "linux":
        raise DeploymentError("cutover requires Linux")
    available_kib = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            available_kib = int(line.split()[1])
    if available_kib < 2 * 1024 * 1024:
        raise DeploymentError("insufficient available host RAM (<2 GiB)")
    gpus = gpu_snapshot()
    log(f"GPU baseline used MiB: {gpus[0][2]}, {gpus[1][2]}")


def free_gpu_guard(cfg: dict) -> None:
    limit = cfg.get("max_other_gpu_used_mib", 3000)
    if type(limit) is not int or not (500 <= limit <= 8000):
        raise DeploymentError("max_other_gpu_used_mib outside conservative bounds")
    for _ in range(30):
        usage = gpu_snapshot()
        if all(used <= limit for _, _, used in usage):
            return
        time.sleep(2)
    raise DeploymentError(f"GPU VRAM not released below {limit} MiB on both devices")


def port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise DeploymentError(f"isolated port {port} unavailable: {exc}") from exc


def run_hook(argv: list[str], timeout: int = 30) -> str:
    return command(argv, timeout).stdout.strip()


def wait_ingress_closed(cfg: dict) -> None:
    deadline = time.monotonic() + cfg.get("drain_timeout_sec", 90)
    while time.monotonic() < deadline:
        raw = run_hook(cfg["drain_check_command"])
        try:
            state = json.loads(raw)
        except ValueError as exc:
            raise DeploymentError("ingress drain checker emitted invalid JSON") from exc
        if not isinstance(state, dict) or type(state.get("blocked")) is not bool or type(state.get("inflight")) is not int:
            raise DeploymentError("ingress drain checker must emit blocked:bool and inflight:int")
        if state["blocked"] and state["inflight"] == 0:
            return
        time.sleep(2)
    raise DeploymentError("ingress drain timed out; Tabby left untouched")


def state_paths(cfg: dict) -> tuple[Path, Path]:
    directory = Path(cfg["state_dir"])
    return directory / "cutover.lock", directory / "cutover.json"


def save_state(journal: Path, state: dict) -> None:
    journal.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = journal.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, journal)
    finally:
        tmp.unlink(missing_ok=True)


def process_start_ticks(pid: int) -> str:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return fields[19]  # field 22: process start time since boot
    except (FileNotFoundError, PermissionError, IndexError):
        return ""


def process_running(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(") ", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def stop_child(state: dict, cfg: dict) -> None:
    pid = state.get("server_pid")
    if type(pid) is not int or pid <= 1:
        return
    proc = Path(f"/proc/{pid}/cmdline")
    try:
        cmdline = proc.read_bytes().split(b"\0")
        executable = os.fsencode(cfg["server_path"])
        if not cmdline or executable not in cmdline[0] or process_start_ticks(pid) != state.get("server_start_ticks"):
            raise DeploymentError("refusing to signal PID without matching executable and process start time")
        os.kill(pid, signal.SIGTERM)
        for _ in range(30):
            if not process_running(pid):
                return
            time.sleep(0.2)
        os.kill(pid, signal.SIGKILL)
    except FileNotFoundError:
        return


def restore(cfg: dict, journal: Path) -> None:
    if not journal.exists():
        log("no active cutover journal; nothing to recover")
        return
    state = json.loads(journal.read_text(encoding="utf-8"))
    if state.get("config_path") != str(Path(cfg["_path"]).resolve()):
        raise DeploymentError("recovery config does not match cutover journal")
    errors: list[str] = []
    try:
        stop_child(state, cfg)
    except Exception as exc:
        errors.append(f"Hermes child stop: {exc}")
    # Never restart Tabby while Hermes still owns the GPUs.
    pid = state.get("server_pid")
    if isinstance(pid, int) and process_running(pid):
        errors.append("Hermes process still exists; Tabby restart withheld")
    if errors:
        raise DeploymentError("; ".join(errors))
    if state.get("tabby_was_active"):
        try:
            if not service_active("hermes-tabby.service"):
                change_service("start", "hermes-tabby.service")
        except Exception as exc:
            errors.append(f"Tabby restore: {exc}")
    if state.get("manager_was_active"):
        try:
            if not service_active("hermes-gpu-manager.service"):
                change_service("start", "hermes-gpu-manager.service")
        except Exception as exc:
            errors.append(f"GPU Manager restore: {exc}")
    if not errors and state.get("tabby_was_active"):
        try:
            deadline = time.monotonic() + cfg.get("tabby_restore_timeout_sec", 240)
            while time.monotonic() < deadline:
                try:
                    healthy = request_json(cfg["tabby_health_url"], cfg.get("tabby_token_file", ""), 8)
                    if isinstance(healthy, dict):
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(3)
            else:
                errors.append("Tabby API did not return JSON HTTP 200 before timeout")
        except Exception as exc:
            errors.append(f"Tabby health: {exc}")
    if not errors and state.get("manager_was_active"):
        deadline = time.monotonic() + cfg.get("manager_restore_timeout_sec", 180)
        last_error = "unknown"
        while time.monotonic() < deadline:
            try:
                manager_guard(cfg)
                break
            except Exception as exc:
                last_error = str(exc)
                time.sleep(min(3, max(0, deadline - time.monotonic())))
        else:
            errors.append(f"GPU Manager health timeout: {last_error}")
    if not errors and state.get("ingress_pending"):
        try:
            run_hook(cfg["restore_command"], 45)
        except Exception as exc:
            errors.append(f"ingress restore: {exc}")
    if errors:
        # Retain the journal and block ingress. Operator can retry `recover`.
        raise DeploymentError("ROLLBACK INCOMPLETE; ingress remains closed: " + "; ".join(errors))
    journal.unlink()
    log("rollback complete: original Tabby service healthy and ingress restored")


def check_hermes(cfg: dict) -> None:
    url = f"http://127.0.0.1:{cfg.get('port', 18127)}"
    state = request_json(url + "/health", cfg.get("http_token_file", ""), 4)
    if state.get("ready") is not True or state.get("status") != "ok":
        raise DeploymentError("Hermes returned unhealthy /health")
    models = request_json(url + "/v1/models", cfg.get("http_token_file", ""), 4)
    if not any(item.get("id") == cfg["model_id"] for item in models.get("data", [])):
        raise DeploymentError("Hermes /v1/models reports unexpected model")


def supervise(cfg: dict, journal: Path) -> None:
    if not service_active("hermes-gpu-manager.service") or not service_active("hermes-tabby.service"):
        raise DeploymentError("both original services must be active before cutover")
    manager_guard(cfg)
    request_json(cfg["tabby_health_url"], cfg.get("tabby_token_file", ""), 8)
    port_available(cfg.get("port", 18127))
    state = {
        "config_path": str(Path(cfg["_path"]).resolve()),
        "manager_was_active": True, "tabby_was_active": True,
        "ingress_pending": True, "server_pid": None, "phase": "draining",
    }
    save_state(journal, state)  # write ahead of the first mutation
    try:
        log("requesting external ingress drain (no direct Tabby API drain exists)")
        run_hook(cfg["drain_command"], 45)
        wait_ingress_closed(cfg)
        manager_guard(cfg)
        log("ingress closed/inflight=0; suspending GPU Manager")
        state["phase"] = "manager-stop"
        save_state(journal, state)
        change_service("stop", "hermes-gpu-manager.service")
        state["phase"] = "tabby-stop"
        save_state(journal, state)
        change_service("stop", "hermes-tabby.service")
        free_gpu_guard(cfg)
        state["phase"] = "hermes-start"
        save_state(journal, state)
        args = [cfg["server_path"], "--model", cfg["model_path"],
                "--tokenizer", cfg["tokenizer_path"], "--gguf-dump", cfg["gguf_dump_path"],
                "--python", cfg["python_path"], "--model-id", cfg["model_id"],
                "--host", "127.0.0.1", "--port", str(cfg.get("port", 18127)),
                "--active", str(cfg.get("active", 2)), "--queue", str(cfg.get("queue", 2)),
                "--context", str(cfg.get("context", 128))]
        logfile = Path(cfg["state_dir"]) / "hermes-server.log"
        with logfile.open("ab", buffering=0) as out:
            child = subprocess.Popen(args, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            state["server_pid"] = child.pid
            state["server_start_ticks"] = process_start_ticks(child.pid)
            state["phase"] = "hermes-watch"
            save_state(journal, state)
            deadline = time.monotonic() + cfg.get("hermes_start_timeout_sec", 240)
            while time.monotonic() < deadline and child.poll() is None and not STOP_REQUESTED:
                try:
                    check_hermes(cfg)
                    break
                except (OSError, ValueError, DeploymentError):
                    time.sleep(2)
            else:
                raise DeploymentError("Hermes did not become healthy within startup deadline (or signal received)")
            log("Hermes serving on isolated loopback endpoint; health watchdog active")
            consecutive_failures = 0
            while not STOP_REQUESTED:
                if child.poll() is not None:
                    raise DeploymentError(f"Hermes exited with code {child.returncode}")
                if service_active("hermes-tabby.service") or service_active("hermes-gpu-manager.service"):
                    raise DeploymentError("original GPU services unexpectedly restarted during Hermes cutover")
                try:
                    check_hermes(cfg)
                    consecutive_failures = 0
                except (OSError, ValueError, DeploymentError):
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        raise DeploymentError("Hermes health failed 3 consecutive probes")
                time.sleep(5)
    finally:
        log("releasing Hermes allocation and restoring Tabby")
        restore(cfg, journal)


def with_lock(cfg: dict):
    lock, _ = state_paths(cfg)
    lock.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(descriptor)
        raise DeploymentError("another cutover/recovery owns the lock") from exc
    return descriptor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("dry-run", "preflight", "run", "recover", "health"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorized", action="store_true", help="required for service mutation")
    parser.add_argument("--skip-hash", action="store_true", help="dry-run ONLY")
    args = parser.parse_args()
    fd = None
    try:
        cfg = read_config(args.config.resolve())
        cfg["_path"] = str(args.config.resolve())
        if args.skip_hash and args.operation != "dry-run":
            raise DeploymentError("--skip-hash only allowed for dry-run")
        if args.operation == "health":
            check_hermes(cfg)
            log("Hermes health + model ID OK")
            return 0
        if args.operation == "recover" and not args.authorized:
            raise DeploymentError("recover changes services: pass --authorized")
        if args.operation == "run" and not args.authorized:
            raise DeploymentError("run changes services: pass --authorized")
        if args.operation in ("run", "recover"):
            fd = with_lock(cfg)
            _, journal = state_paths(cfg)
            if args.operation == "recover":
                restore(cfg, journal)
                return 0
            if journal.exists():
                raise DeploymentError("unfinished cutover journal exists: recover first")
        check_static(cfg, verify_hash=not args.skip_hash)
        if args.operation != "dry-run":
            machine_guard()
            if not service_active("hermes-gpu-manager.service") or not service_active("hermes-tabby.service"):
                raise DeploymentError("Tabby and GPU Manager must both be active")
            manager_guard(cfg)
            request_json(cfg["tabby_health_url"], cfg.get("tabby_token_file", ""), 8)
            port_available(cfg.get("port", 18127))
        if args.operation == "dry-run":
            log("DRY-RUN: files, hashes (unless --skip-hash) and hook executables verified; NO service/network/GPU changes")
            log("planned: verify authenticated manager + Tabby status; drain external ingress; wait inflight=0; stop manager; stop Tabby; verify GPU free; supervise Hermes; restore Tabby + manager; reopen ingress")
        elif args.operation == "preflight":
            log("PREFLIGHT PASS: hashes, CPU/GPU, authenticated service status, original endpoint and isolated port")
        else:
            signal.signal(signal.SIGINT, on_stop)
            signal.signal(signal.SIGTERM, on_stop)
            supervise(cfg, journal)
        return 0
    except (DeploymentError, OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
        log(f"ERROR: {exc}")
        return 1
    finally:
        if fd is not None:
            os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
