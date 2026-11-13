#!/usr/bin/env python3
"""Explicit opt-in loopback-only adapter: GPU Manager lease v1 JSON/stdin.

Prototype companion to manager_lease_patch.py. It neither creates listeners
nor mutates the server until invoked by an authorized maintenance supervisor.
Manager auth and separate lease token are read from private *files*, never
from CLI arguments or JSON output.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request


class AdapterRejected(RuntimeError):
    pass


def secret_file(variable: str) -> str:
    value = os.environ.get(variable, "")
    if not value or not Path(value).is_absolute():
        raise AdapterRejected(f"{variable} unavailable")
    path = Path(value)
    if (path.is_symlink() or not path.is_file() or
            path.stat().st_mode & 0o077 or path.stat().st_size > 512):
        raise AdapterRejected(f"{variable} must be a private regular credential file")
    secret = path.read_text(encoding="utf-8").strip()
    if not 32 <= len(secret) <= 256:
        raise AdapterRejected(f"{variable} has invalid credential length")
    return secret


def call_manager(action: str, payload: dict, *, manager_url: str | None = None) -> dict:
    if action not in ("acquire", "renew", "release"):
        raise AdapterRejected("unrecognized adapter operation")
    if not isinstance(payload, dict) or payload.get("action") != action or payload.get("protocol") != "hermes-manager-lease-v1":
        raise AdapterRejected("invalid JSON protocol/action in adapter stdin")
    target = manager_url or os.environ.get("HERMES_MANAGER_LEASE_URL", "http://127.0.0.1:8643")
    parsed = urllib.parse.urlsplit(target)
    if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or
            parsed.username is not None or parsed.password is not None or
            parsed.path not in ("", "/") or parsed.query or parsed.fragment or
            parsed.port is None or not 1 <= parsed.port <= 65535):
        raise AdapterRejected("manager lease API must use unambiguous 127.0.0.1 loopback only")
    key = secret_file("HERMES_MANAGER_API_KEY_FILE")
    lease_key = secret_file("HERMES_MANAGER_LEASE_TOKEN_FILE")
    url = f"http://127.0.0.1:{parsed.port}/maintenance-lease/v1/{action}"
    request = urllib.request.Request(
        url, method="POST", data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                 "X-Hermes-Manager-Lease-Token": lease_key})

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=8) as response:
            if response.status != 200:
                raise AdapterRejected("manager lease endpoint returned non-200")
            reply_raw = response.read(4097)
            if len(reply_raw) > 4096:
                raise AdapterRejected("oversized manager lease acknowledgement")
            reply = json.loads(reply_raw)
    except (OSError, urllib.error.HTTPError, ValueError):
        # Never log server response body, key, bearer header or raw exception.
        raise AdapterRejected("manager lease request refused or unavailable") from None
    if (not isinstance(reply, dict) or reply.get("protocol") != payload["protocol"] or
            reply.get("action") != action or reply.get("ok") is not True or
            reply.get("lease_id") != payload.get("lease_id")):
        raise AdapterRejected("invalid/fenced lease acknowledgement")
    return {"protocol": reply["protocol"], "action": action,
            "ok": True, "lease_id": reply["lease_id"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, choices=("acquire", "renew", "release"))
    args = parser.parse_args()
    try:
        raw = sys.stdin.buffer.read(4097)
        if len(raw) > 4096:
            raise AdapterRejected("oversized adapter stdin")
        payload = json.loads(raw)
        result = call_manager(args.action, payload)
    except (AdapterRejected, OSError, ValueError):
        print("manager lease adapter failed closed", file=sys.stderr)
        return 1
    print(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
