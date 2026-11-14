#!/usr/bin/env python3
"""Produce a NEW disposable PP2 layer0 decode trace source; never edit live code.

The tracer writes bounded, first-two-token stage snapshots only when explicitly
enabled by HERMES_SCRATCH_LAYER0_TRACE_DIR. It is diagnostic, not a runtime mode
or an independent llama.cpp correctness oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile


class Blocked(ValueError):
    pass


INCLUDE = "#include <limits>\n"
BEGIN = "static bool run_gdn_decode_layer(int li, float* d_x, DecodeState& ds, std::string& err,"
POINT = (
    '  add_k<<<(int)((nH+255)/256),256,0,stream>>>(d_x,d_core,(int)nH);\n'
    '  if(getenv("HERMES_L26") && li==26){'
)
DECODE_POINT = (
    '  ds.ever_advanced=true;\n'
    '  cudaSetDevice(0);'
)
ASYNC_POINT = '  if(ds.async_ctx) return decode_step_async(tok,ds,pos,err);'
TAG = "HERMES_SCRATCH_LAYER0_TRACE_DIR"

HELPER = r'''
// SCRATCH-ONLY diagnostic: no default route change, no network, no model copy.
struct L0TraceField {
  const char* name;
  const float* device_data;
  size_t floats;
};
static std::atomic<unsigned> g_l0_trace_sequence{0};
static std::atomic<const DecodeState*> g_l0_trace_state{nullptr};
struct L0TraceInvocation {
  int token = -1;
  long position = -1;
  const DecodeState* state = nullptr;
};
static thread_local L0TraceInvocation g_l0_trace_invocation;
// Bind the snapshot to the actual public decode_step arguments, not to
// guessed CLI order or to the name of the output file.
struct L0TraceScope {
  explicit L0TraceScope(int token, long position, const DecodeState* state) {
    g_l0_trace_invocation = {token, position, state};
  }
  ~L0TraceScope() { g_l0_trace_invocation = {}; }
  L0TraceScope(const L0TraceScope&) = delete;
  L0TraceScope& operator=(const L0TraceScope&) = delete;
};
static bool l0_trace_scratch(
    const char* root, unsigned sequence, cudaStream_t stream,
    std::string& err, const L0TraceField* fields, size_t field_count) {
  if (!root || std::strncmp(root, "/tmp/h27-", 9) != 0 ||
      std::strstr(root, "..") || std::strlen(root) > 180) {
    err = "scratch layer0 path must be an existing /tmp/h27-* directory";
    return false;
  }
  // Refuse path aliases/symlinked scratch folders. No production path may be
  // reached by swapping a simple /tmp/h27-* prefix into another directory.
  // The caller must create a fresh canonical scratch directory beforehand.
  const std::filesystem::path root_path(root);
  std::error_code path_error;
  const auto canonical_root = std::filesystem::canonical(root_path, path_error);
  if (path_error || canonical_root != root_path.lexically_normal()) {
    err = "scratch layer0 directory is missing, noncanonical or symlinked";
    return false;
  }
  if (sequence > 1) {
    err = "scratch layer0 trace limited to exactly two decode tokens";
    return false;
  }
  const int expected_tokens[] = {6205, 161421};
  const auto& invocation = g_l0_trace_invocation;
  if (!invocation.state || invocation.position != (long)sequence ||
      invocation.token != expected_tokens[sequence]) {
    err = "scratch layer0 actual token/absolute position mismatch";
    return false;
  }
  if (sequence == 0) {
    const DecodeState* empty = nullptr;
    if (!g_l0_trace_state.compare_exchange_strong(empty, invocation.state)) {
      err = "scratch layer0 reused or concurrent request";
      return false;
    }
  } else if (g_l0_trace_state.load() != invocation.state) {
    err = "scratch layer0 tokens do not share one DecodeState";
    return false;
  }
  int current_device = -1;
  if (cudaGetDevice(&current_device) != cudaSuccess || current_device != 0) {
    err = "scratch layer0 requires GPU0";
    return false;
  }
  cudaError_t sync_error = cudaStreamSynchronize(stream);
  if (sync_error != cudaSuccess) {
    err = std::string("scratch layer0 stream sync: ") +
          cudaGetErrorString(sync_error);
    return false;
  }
  for (size_t i = 0; i < field_count; ++i) {
    const L0TraceField& stage = fields[i];
    if (!stage.device_data || !stage.name ||
        stage.floats == 0 || stage.floats > 10240) {
      err = "scratch layer0 stage exceeds 10240 F32 values";
      return false;
    }
    std::vector<float> host(stage.floats);
    cudaError_t copy_error = cudaMemcpy(
        host.data(), stage.device_data, stage.floats * sizeof(float),
        cudaMemcpyDeviceToHost);
    if (copy_error != cudaSuccess) {
      err = std::string("scratch layer0 stage copy: ") +
            cudaGetErrorString(copy_error);
      return false;
    }
    for (float value : host) {
      if (!std::isfinite(value)) {
        err = "scratch layer0 stage has nonfinite values";
        return false;
      }
    }
    char filename[256];
    int nch = std::snprintf(filename, sizeof(filename),
                            "%s/p%u_%s.f32", root, sequence, stage.name);
    if (nch < 0 || static_cast<size_t>(nch) >= sizeof(filename)) {
      err = "scratch layer0 stage path too long";
      return false;
    }
    FILE* output = std::fopen(filename, "wbx");
    if (!output) {
      err = std::string("scratch layer0 file already exists or not writable: ") +
            filename;
      return false;
    }
    const size_t written = std::fwrite(
        host.data(), sizeof(float), host.size(), output);
    const int close_result = std::fclose(output);
    if (written != host.size() || close_result != 0) {
      err = "scratch layer0 partial file; DO NOT use without external validation";
      return false;
    }
  }
  // Exclusive sidecar is produced from decode_step's actual arguments.
  // A missing, truncated or altered sidecar is a verifier error. Like the
  // tensor files, this scratch artifact must never be shipped.
  char context_path[256];
  int nc = std::snprintf(context_path, sizeof(context_path),
                         "%s/p%u_context.txt", root, sequence);
  if (nc < 0 || static_cast<size_t>(nc) >= sizeof(context_path)) {
    err = "scratch layer0 token context path too long";
    return false;
  }
  FILE* context = std::fopen(context_path, "wbx");
  if (!context) {
    err = "scratch layer0 token context already exists or not writable";
    return false;
  }
  int written = std::fprintf(context,
      "token_id=%d\nabsolute_position=%ld\nsequence=%u\n",
      invocation.token, invocation.position, sequence);
  int closed = std::fclose(context);
  if (written < 0 || closed != 0) {
    err = "scratch layer0 partial token context; discard capture";
    return false;
  }
  return true;
}
'''

CAPTURE = r'''  // Optional, strictly scratch-only, single-request two-token probe.
  // Must run through serial decode_step with prompt [6205,161421].
  if (li == 0 && std::getenv("HERMES_SCRATCH_LAYER0_TRACE_DIR")) {
    if (std::getenv("HERMES_PP2_ASYNC_STREAMS") ||
        ds.async_ctx || ds.sliding_kv_enabled) {
      err = "scratch layer0 capture refuses async/multiuser/sliding mode";
      return false;
    }
    const unsigned trace_sequence =
        g_l0_trace_sequence.fetch_add(1, std::memory_order_relaxed);
    const L0TraceField fields[] = {
      {"attn_norm", d_h, HID},
      {"linear_attn_qkv_mixed", d_q1, CC},
      {"conv_output_silu", d_cv, CC},
      {"beta_sigmoid", d_be, HV},
      {"gate", d_gl, HV},
      {"q_norm_scaled", d_q, VDD},
      {"k_norm", d_k, VDD},
      {"attn_output", d_o, VDD},
      {"final_output", d_g, VDD},
      {"linear_attn_out", d_core, HID},
      {"attn_residual", d_x, HID}
    };
    if (!l0_trace_scratch(
          std::getenv("HERMES_SCRATCH_LAYER0_TRACE_DIR"), trace_sequence,
          stream, err, fields, sizeof(fields)/sizeof(fields[0]))) {
      return false;
    }
  }
'''


def patched_source(source: str) -> str:
    if TAG in source:
        raise Blocked("source already contains scratch trace; refuse double patch")
    if (source.count(INCLUDE) != 1 or source.count(BEGIN) != 1 or
            source.count(POINT) != 1 or source.count(ASYNC_POINT) != 1 or
            source.count(DECODE_POINT) != 1):
        raise Blocked("PP2 source layout changed: exact anchors not verified")
    result = source.replace(INCLUDE,
                            INCLUDE + "#include <atomic>\n#include <filesystem>\n", 1)
    result = result.replace(BEGIN, HELPER + "\n" + BEGIN, 1)
    result = result.replace(POINT, POINT.split("\n")[0] + "\n" + CAPTURE +
                            "  if(getenv(\"HERMES_L26\") && li==26){", 1)
    # Fail at the public dispatch boundary. In particular, sliding KV
    # automatically creates an async context even without ASYNC_STREAMS=1:
    # otherwise a scratch request silently skips layer0 instrumentation.
    result = result.replace(ASYNC_POINT,
                            '  if(std::getenv("HERMES_SCRATCH_LAYER0_TRACE_DIR") &&\n'
                            '     (ds.async_ctx || ds.sliding_kv_enabled ||\n'
                            '      std::getenv("HERMES_PP2_ASYNC_STREAMS"))){\n'
                            '    err="scratch layer0 trace refuses asynchronous/sliding dispatch";\n'
                            '    return {};\n'
                            '  }\n' + ASYNC_POINT, 1)
    # Sliding KV introduced a preflight block between the async dispatch
    # and ever_advanced assignment. Bind only after that block, inside the
    # original serial decode_step, never inside the async entrypoint.
    result = result.replace(DECODE_POINT,
                            "  L0TraceScope actual_decode_token(tok,pos,&ds);\n"
                            + DECODE_POINT, 1)
    if result.count(TAG) != 3:
        raise Blocked("scratch trace insertion incomplete")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="unchanged original distributed/pp2_fwd.cu")
    parser.add_argument("--output", type=Path, required=True,
                        help="NEW /tmp scratch pp2_fwd.cu (must not exist)")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    temp = Path(tempfile.gettempdir()).resolve()
    if source.name != "pp2_fwd.cu" or output.name != "pp2_fwd.cu":
        raise Blocked("source/output must be named pp2_fwd.cu")
    if source == output or output.exists() or not output.parent.is_dir():
        raise Blocked("output must be a new scratch file; never overwrite")
    if not output.is_relative_to(temp):
        raise Blocked("output must reside under system temp; never edit project")
    if output.is_symlink() or output.parent.is_symlink():
        raise Blocked("scratch symlinks are forbidden")
    before = source.read_text(encoding="utf-8")
    result = patched_source(before)
    output.write_text(result, encoding="utf-8", newline="\n")
    if hashlib.sha256(source.read_bytes()).hexdigest() != hashlib.sha256(
            before.encode("utf-8")).hexdigest():
        output.unlink(missing_ok=True)
        raise Blocked("original source changed during trace preparation")
    print("PASS scratch PP2 layer0 trace source:",
          hashlib.sha256(output.read_bytes()).hexdigest(), output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Blocked, OSError) as exc:
        print("BLOCKED_L0_SCRATCH_TRACE:", exc)
        raise SystemExit(2) from exc
