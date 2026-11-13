#!/usr/bin/env bash
# Offline CPU-only serving regression. It copies sources to an isolated /tmp
# build, links the actual ContinuousScheduler, and never touches CUDA/GPUs.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$HERE/../.." && pwd)"
SCRATCH="${HERMES_CPU_SCRATCH:-/tmp/hermes27-prod-sched-cpu}"
VENDOR="${HERMES_CPP_HTTPLIB_ROOT:-/opt/hermes/src/llama-cpp-turboquant/vendor/cpp-httplib}"
TOKENIZER="${HERMES_TEST_TOKENIZER:-/opt/hermes/models/Qwen3.8-27B-Uncensored-EXL3-3.07bpw/tokenizer.json}"
PYTHON="${HERMES_TEST_PYTHON:-/opt/hermes/runtimes/exllamav3-venv/bin/python}"

mkdir -p "$SCRATCH/build" "$SCRATCH/serving/tests"
cp "$ROOT/serving/server.cpp" "$SCRATCH/server.cpp"
cp "$ROOT/serving/tokenizer_bridge.py" "$SCRATCH/serving/tokenizer_bridge.py"
cp "$ROOT/distributed/pp2_fwd.hpp" "$ROOT/distributed/pp2_batch.hpp" "$SCRATCH/"
cp "$ROOT/distributed/pp2_batch.cu" "$SCRATCH/pp2_batch.cu"
cp "$HERE/prod_sched_mock_pp2.cpp" "$SCRATCH/serving/tests/"
cp "$HERE/prod_sched_launcher.py" "$HERE/prod_sched_state_test.py" \
   "$HERE/test_production_http.py" "$HERE/test_stream_mock_http.py" \
   "$HERE/test_bytelevel_stream.py" "$SCRATCH/serving/tests/"
cp "$HERE/mock_gguf_dump.py" "$SCRATCH/gguf_dump_fake.py"
chmod 700 "$SCRATCH/gguf_dump_fake.py" "$SCRATCH/serving/tests/prod_sched_launcher.py"
touch "$SCRATCH/fake.gguf"

echo "Building real server.cpp + real pp2_batch.cu against fake CPU PP2"
g++ -std=c++20 -O0 -pthread -DHERMES_ADMISSION_CPU_ONLY -x c++ \
    -I"$SCRATCH" -I"$VENDOR" -I"$(dirname "$VENDOR")" \
    "$SCRATCH/server.cpp" "$SCRATCH/serving/tests/prod_sched_mock_pp2.cpp" \
    "$SCRATCH/pp2_batch.cu" "$VENDOR/httplib.cpp" \
    -o "$SCRATCH/build/hermes_server"

COMMON=(--model "$SCRATCH/fake.gguf" --gguf-dump "$SCRATCH/gguf_dump_fake.py"
        --tokenizer "$TOKENIZER" --python "$PYTHON")

echo "Legacy default HTTP contract (expected 19/19)"
"$PYTHON" "$SCRATCH/serving/tests/test_production_http.py" \
    --server "$SCRATCH/build/hermes_server" "${COMMON[@]}" --slow-reader

echo "Continuous opt-in HTTP contract (expected 19/19)"
HERMES_TEST_SERVER_BIN="$SCRATCH/build/hermes_server" \
    "$PYTHON" "$SCRATCH/serving/tests/test_production_http.py" \
    --server "$SCRATCH/serving/tests/prod_sched_launcher.py" "${COMMON[@]}" --slow-reader

echo "Continuous PP2 state ownership and cancellation (expected 7/7)"
"$PYTHON" "$SCRATCH/serving/tests/prod_sched_state_test.py" \
    --server "$SCRATCH/build/hermes_server" "${COMMON[@]}"

echo "Continuous SSE Unicode parity (expected 9/9 scenarios)"
HERMES_TEST_SERVER_BIN="$SCRATCH/build/hermes_server" \
    "$PYTHON" "$SCRATCH/serving/tests/test_stream_mock_http.py" \
    --server "$SCRATCH/serving/tests/prod_sched_launcher.py" "${COMMON[@]}"

echo "Real tokenizer parity (expected 5/5)"
HERMES_TEST_TOKENIZER="$TOKENIZER" \
    "$PYTHON" "$SCRATCH/serving/tests/test_bytelevel_stream.py" -v

echo "All CPU-only gates passed. No GPUs or live services used."
