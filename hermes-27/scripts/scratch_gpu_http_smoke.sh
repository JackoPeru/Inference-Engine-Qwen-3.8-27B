#!/usr/bin/env bash
# Isolated, opt-in GPU HTTP smoke for an already-built scratch HERMES-27 tree.
# Runs only on 127.0.0.1 and never stops, restarts or configures live services.
set -Eeuo pipefail

ROOT="${HERMES_TEST_ROOT:-/tmp/hermes27-prime-integ}"
SOURCE="${ROOT}/src"
BUILD="${ROOT}/build"
MODEL="${HERMES_TEST_GGUF:-/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf}"
TOKENIZER="${HERMES_TEST_TOKENIZER:-/opt/hermes/models/Qwen3.8-27B-Uncensored-EXL3-3.07bpw/tokenizer.json}"
PYTHON="${HERMES_TEST_PYTHON:-/opt/hermes/runtimes/exllamav3-venv/bin/python}"
PORT="${HERMES_TEST_PORT:-18127}"
MODE="${HERMES_TEST_MODE:-async}"
SCHEDULER="${HERMES_TEST_SCHEDULER:-legacy}"
PREFILL="${HERMES_TEST_BATCHED_PREFILL:-0}"
LOG="${ROOT}/http-smoke-${MODE}-${SCHEDULER}-seed${PREFILL}.log"
SERVER_PID=""

for file in "${BUILD}/hermes_server" "${BUILD}/gguf_dump" "${SOURCE}/serving/tokenizer_bridge.py" "${SOURCE}/serving/smoke.py" "${MODEL}" "${TOKENIZER}" "${PYTHON}"; do
  if [[ ! -f "$file" ]]; then
    echo "BLOCKED: missing required scratch artifact: $file" >&2
    exit 77
  fi
done
if [[ "$PORT" != "18127" ]]; then
  echo "BLOCKED: only isolated scratch port 18127 is permitted by this smoke" >&2
  exit 77
fi
if [[ "$MODE" != "async" && "$MODE" != "legacy" ]]; then
  echo "BLOCKED: HERMES_TEST_MODE must be async or legacy" >&2
  exit 77
fi
if [[ "$SCHEDULER" != "continuous" && "$SCHEDULER" != "legacy" ]]; then
  echo "BLOCKED: HERMES_TEST_SCHEDULER must be continuous or legacy" >&2
  exit 77
fi
if [[ "$PREFILL" != 0 && "$PREFILL" != 1 ]]; then
  echo "BLOCKED: HERMES_TEST_BATCHED_PREFILL must be 0 or 1" >&2
  exit 77
fi
if ss -ltnH "( sport = :${PORT} )" | grep -q .; then
  echo "BLOCKED: test port ${PORT} already occupied" >&2
  exit 77
fi
for service in hermes-tabby hermes-gpu-manager; do
  if [[ "$(systemctl is-active "$service")" != "active" ]]; then
    echo "BLOCKED: production service $service not active; refusing to proceed" >&2
    exit 77
  fi
done
# Conservative preflight: neither GPU may already host a full inference model.
mapfile -t used_mib < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if [[ "${#used_mib[@]}" != "2" ]] || (( used_mib[0] >= 1500 || used_mib[1] >= 2400 )); then
  echo "BLOCKED: GPUs not at idle baseline; no model upload attempted" >&2
  exit 77
fi

mkdir -p "${ROOT}/serving"
if [[ "$(readlink -f "${SOURCE}/serving/tokenizer_bridge.py")" != "$(readlink -f "${ROOT}/serving/tokenizer_bridge.py")" ]]; then
  cp "${SOURCE}/serving/tokenizer_bridge.py" "${ROOT}/serving/tokenizer_bridge.py"
fi
export HERMES_NVFP4_GEMV=packed4
export HERMES_HEAD_BF16=warp4
export HERMES_DECODE_SKIP_LAYER_SYNC=1
# BF16 full-model T64 quality is not approved. Keep this A/B strictly RAW.
unset HERMES_PP2_PREFILL_BF16 HERMES_PP2_CHUNKED_PREFILL || true
if [[ "$MODE" == "async" ]]; then
  export HERMES_PP2_ASYNC_STREAMS=1
else
  unset HERMES_PP2_ASYNC_STREAMS || true
fi
scheduler_args=()
if [[ "$SCHEDULER" == "continuous" ]]; then
  scheduler_args+=(--continuous-scheduler)
fi
if [[ "$PREFILL" == 1 ]]; then
  scheduler_args+=(--batched-prefill)
fi

cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  echo "POST-RUN GPU MEMORY:" >&2
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >&2 || true
}
trap cleanup EXIT

# The timeout is a second-line guard: TERM first, KILL only if the isolated
# scratch server ignores TERM for 10 seconds. Never operate on Tabby/Manager.
timeout --signal=TERM --kill-after=10 240 "${BUILD}/hermes_server" \
  --model "$MODEL" --tokenizer "$TOKENIZER" \
  --gguf-dump "${BUILD}/gguf_dump" --python "$PYTHON" \
  --host 127.0.0.1 --port "$PORT" --context 256 --active 4 --queue 32 \
  "${scheduler_args[@]}" \
  >"$LOG" 2>&1 &
SERVER_PID=$!

ready=0
for ((i=0; i<120; ++i)); do
  if curl --fail --silent --max-time 1 "http://127.0.0.1:${PORT}/health" >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 0.5
done
if [[ "$ready" != 1 ]]; then
  echo "FAIL: scratch HTTP server did not become healthy" >&2
  tail -n 40 "$LOG" >&2
  exit 1
fi
echo "SCRATCH SERVER READY mode=$MODE scheduler=$SCHEDULER batched_prefill=$PREFILL port=$PORT"
timeout --signal=TERM --kill-after=10 175 "$PYTHON" -B "${SOURCE}/serving/smoke.py" \
  --base-url "http://127.0.0.1:${PORT}" --model hermes-27
metrics="$(curl --fail --silent --max-time 10 "http://127.0.0.1:${PORT}/metrics")"
metric_value() {
  printf '%s\n' "$metrics" | awk -v name="$1" '$1 == name { print $2 }'
}
seeded="$(metric_value hermes_prefill_seeded_total)"
failed="$(metric_value hermes_prefill_failed_total)"
prompt_tokens="$(metric_value hermes_prefill_prompt_tokens_total)"
if [[ ! "$seeded" =~ ^[0-9]+$ || ! "$failed" =~ ^[0-9]+$ ||
      ! "$prompt_tokens" =~ ^[0-9]+$ ]]; then
  echo "FAIL: missing numeric prefill counters from real HTTP server" >&2
  exit 1
fi
if (( failed != 0 )) ||
   { [[ "$PREFILL" == 1 ]] && (( seeded < 1 || prompt_tokens < 1 )); } ||
   { [[ "$PREFILL" == 0 ]] && (( seeded != 0 || prompt_tokens != 0 )); }; then
  echo "FAIL: prefill counters do not match opt-in policy (seeded=$seeded failed=$failed tokens=$prompt_tokens flag=$PREFILL)" >&2
  exit 1
fi
echo "REAL HTTP PREFILL COUNTERS: PASS seeded=$seeded failed=$failed tokens=$prompt_tokens flag=$PREFILL"
echo "SCRATCH HTTP GPU SMOKE: PASS mode=$MODE scheduler=$SCHEDULER batched_prefill=$PREFILL"
if [[ "${HERMES_TEST_LOAD:-0}" == 1 ]]; then
  if [[ "${HERMES_TEST_LOAD_STYLE:-arithmetic}" == "explain" ]]; then
    timeout --signal=TERM --kill-after=10 175 "$PYTHON" -B "${SOURCE}/scripts/scratch_http_load.py" \
      --base-url "http://127.0.0.1:${PORT}" --clients 1 2 4 --rounds 2 \
      --max-tokens 16 --prompt-style explain
  else
    timeout --signal=TERM --kill-after=10 175 "$PYTHON" -B "${SOURCE}/scripts/scratch_http_load.py" \
      --base-url "http://127.0.0.1:${PORT}" --clients 1 2 4 8 --rounds 2 \
      --max-tokens 8 --prompt-style arithmetic
  fi
  echo "SCRATCH HTTP CONCURRENCY: PASS mode=$MODE scheduler=$SCHEDULER batched_prefill=$PREFILL"
fi
