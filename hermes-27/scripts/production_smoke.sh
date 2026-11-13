#!/usr/bin/env bash
# CPU-only smoke and negative-case tests. Never alters systemd or GPU state.
set -Eeuo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python3 -m py_compile "$repo/ops/production_deploy.py"
python3 -m py_compile "$repo/ops/production_simulate.py"
python3 -m py_compile "$repo/ops/manager_lease_core.py" "$repo/ops/manager_lease_runtime.py" \
  "$repo/ops/manager_lease_patch.py" "$repo/ops/manager_lease_adapter.py"
python3 -m unittest discover -s "$repo/ops/tests" -p 'test_*.py' -v
bash -n "$repo/scripts/production_smoke.sh"
if command -v shellcheck >/dev/null 2>&1; then
  shellcheck "$repo/scripts/production_smoke.sh"
fi
printf 'PASS: offline production deployment smoke (no changes to running services)\n'
