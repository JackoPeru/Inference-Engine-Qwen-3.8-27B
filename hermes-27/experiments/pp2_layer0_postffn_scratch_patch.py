#!/usr/bin/env python3
"""Source-only extension of the FROZEN 3-marker PP2 layer0 scratch tracer.

Never runs CUDA/model code. It composes a NEW scratch source from an exact
published production SHA and exact original v2 patcher, then adds a bounded
post-FFN snapshot. Actual execution requires separate GPU authorization.
CPU verification of arbitrary raw sidecars is explicitly NOT an oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import struct
import sys
import tempfile

SOURCE_SHA = "6faccbda1aaf568deb7767393a53c5ea89a025a97fbc9746a0fc97a61e29469e"
BASE_PATCHER_SHA = "80c0be0c1e29faf5db837da306af59ee729d4207c9bc21022163cfdef049c886"
GGUF_SHA = "db17acbca53da5a7b0e861175b198cc6fd467865e99d1ad53d9dca584257a1a1"
GGUF_SIZE = 19_694_360_544
BASE_TAG = "HERMES_SCRATCH_LAYER0_TRACE_DIR"
POST_TAG = "HERMES_SCRATCH_LAYER0_POSTFFN_DIR"
POST_STAGES = {"attn_post_norm": 5120, "ffn_out": 5120, "post_ffn": 5120}
PRE_STAGES = {
    "attn_norm": 5120, "linear_attn_qkv_mixed": 10240,
    "conv_output_silu": 10240, "beta_sigmoid": 48, "gate": 48,
    "q_norm_scaled": 6144, "k_norm": 6144, "attn_output": 6144,
    "final_output": 6144, "linear_attn_out": 5120, "attn_residual": 5120,
}
TOKENS = (6205, 161421)
HELPER_ANCHOR = (
    "  L0TraceScope& operator=(const L0TraceScope&) = delete;\n"
    "};\n"
    "static bool l0_trace_scratch(\n"
)
CONTEXT_ANCHOR = (
    '    err = "scratch layer0 partial token context; discard capture";\n'
    "    return false;\n"
    "  }\n"
    "  return true;\n"
    "}\n"
)
FFN_ANCHOR = (
    "  gemv_batch(d_hidb,(uint8_t*)wFd->d,d_down,FFN,HID,T,stream);\n"
    "  add_k<<<(int)((nH+255)/256),256,0,stream>>>(d_x,d_down,(int)nH);\n"
    "  if(stream==0 && !decode_skip_layer_sync()) cudaDeviceSynchronize();\n"
)


class Blocked(ValueError):
    pass


def need(ok: bool, text: str) -> None:
    if not ok:
        raise Blocked(text)


def sha256_file(file: Path) -> str:
    h = hashlib.sha256()
    with file.open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def load_patcher(path: Path):
    need(path.is_file() and not path.is_symlink() and
         sha256_file(path) == BASE_PATCHER_SHA,
         "frozen 3-marker patcher SHA mismatch")
    spec = importlib.util.spec_from_file_location("h27_frozen_postffn_base", path)
    need(spec is not None and spec.loader is not None,
         "cannot import frozen 3-marker patcher")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    need(mod.TAG == BASE_TAG and callable(mod.patched_source),
         "frozen 3-marker patcher symbol contract changed")
    return mod


# Added only to an already instrumented, fully SHA-pinned v2 scratch source.
# No additional occurrence of the old BASE_TAG; its existing 3-marker guard
# remains exactly unchanged and the old v2 raw directory stays at 24 files.
POST_HELPER = r'''
// Scratch-only, SOURCE-DERIVED post-FFN witness. NEVER an independent oracle.
// Capture root is copied from the validated v2 layer0 pre-FFN hook. The
// *separate* root must exist as the literal canonical "<pre-root>-postffn".
static thread_local char g_l0_postffn_base[181] = {};
static thread_local unsigned g_l0_postffn_sequence = 2;
static bool l0_postffn_copy(const char* base, const char* post, unsigned seq,
                            cudaStream_t stream, std::string& err,
                            const L0TraceField* fields, size_t count) {
  try {
    const auto& invocation = g_l0_trace_invocation;
    const int expected_tokens[] = {6205, 161421};
    if (!base || !base[0] || !post || !*post || seq > 1 ||
        invocation.state == nullptr ||
        invocation.state != g_l0_trace_state.load() ||
        invocation.token != expected_tokens[seq] ||
        invocation.position != (long)seq ||
        g_l0_trace_sequence.load() != seq + 1 ||
        stream != 0 || count != 3 || !fields) {
      err = "postFFN wrong same-request token/absolute-position/sequence";
      return false;
    }
    char root_expected[192];
    const int n = std::snprintf(root_expected,sizeof(root_expected),
                                "%s-postffn",base);
    if (n < 0 || n > 180 || std::strcmp(post,root_expected) != 0) {
      err = "postFFN root must equal canonical pretrace root plus -postffn";
      return false;
    }
    const std::filesystem::path base_dir(base), post_dir(post);
    std::error_code ec;
    const auto canonical_base = std::filesystem::canonical(base_dir,ec);
    if (ec || canonical_base != base_dir.lexically_normal() ||
        std::filesystem::is_symlink(base_dir)) {
      err = "postFFN base root missing/aliased"; return false;
    }
    ec.clear();
    const auto canonical_post = std::filesystem::canonical(post_dir,ec);
    if (ec || canonical_post != post_dir.lexically_normal() ||
        std::filesystem::is_symlink(post_dir) ||
        !std::filesystem::is_directory(post_dir)) {
      err = "postFFN output root missing/aliased"; return false;
    }
    // Bind postFFN to the ACTUAL v2 runtime context bytes, not to an
    // independently guessed CLI token or just a p0/p1 filename.
    char ctx_filename[256], ctx_expected[128], ctx_buffer[128];
    int nc=std::snprintf(ctx_filename,sizeof(ctx_filename),
                        "%s/p%u_context.txt",base,seq);
    int ne=std::snprintf(ctx_expected,sizeof(ctx_expected),
        "token_id=%d\nabsolute_position=%u\nsequence=%u\n",
        expected_tokens[seq],seq,seq);
    if(nc<0 || (size_t)nc>=sizeof(ctx_filename) ||
       ne<0 || (size_t)ne>=sizeof(ctx_expected) ||
       std::filesystem::is_symlink(ctx_filename)) {
      err = "postFFN context filename invalid"; return false;
    }
    FILE* existing=std::fopen(ctx_filename,"rb");
    if (!existing){err="postFFN matching v2 context missing"; return false;}
    const size_t got=std::fread(ctx_buffer,1,sizeof(ctx_buffer),existing);
    const int read_error=std::ferror(existing);
    const int close_error=std::fclose(existing);
    if(read_error || close_error || got!=(size_t)ne ||
       std::memcmp(ctx_buffer,ctx_expected,(size_t)ne)!=0){
      err="postFFN v2 source context did not match actual token/position";
      return false;
    }
    unsigned prior=0;
    for(const auto& entry:std::filesystem::directory_iterator(post_dir)){
      const std::string filename=entry.path().filename().string();
      if(entry.is_symlink() || !entry.is_regular_file() || seq==0 ||
         (filename!="p0_attn_post_norm.f32" &&
          filename!="p0_ffn_out.f32" &&
          filename!="p0_post_ffn.f32" &&
          filename!="p0_context.txt")) {
        err="postFFN scratch directory stale/unknown/aliased file";
        return false;
      }
      ++prior;
    }
    if(seq==1 && prior!=4){
      err="postFFN pos0 outputs incomplete"; return false;
    }
    int gpu=-1;
    if(cudaGetDevice(&gpu)!=cudaSuccess || gpu!=0 ||
       cudaStreamSynchronize(stream)!=cudaSuccess) {
      err="postFFN GPU0 stream synchronization failed"; return false;
    }
    std::array<float,5120> host{};
    const char* expected_names[]={"attn_post_norm","ffn_out","post_ffn"};
    for(size_t i=0;i<count;++i){
      const L0TraceField& field=fields[i];
      if(!field.device_data || !field.name ||
         std::strcmp(field.name,expected_names[i])!=0 ||
         field.floats!=5120) {
        err="postFFN pointer/name/shape invalid"; return false;
      }
      if(cudaMemcpy(host.data(),field.device_data,
                    host.size()*sizeof(float),
                    cudaMemcpyDeviceToHost)!=cudaSuccess){
        err="postFFN CUDA copy failure"; return false;
      }
      for(float value:host)if(!std::isfinite(value)){
        err="postFFN nonfinite values"; return false;
      }
      char filename[256];
      const int nf=std::snprintf(filename,sizeof(filename),
                         "%s/p%u_%s.f32",post,seq,field.name);
      if(nf<0 || (size_t)nf>=sizeof(filename)){
        err="postFFN tensor path too long"; return false;
      }
      FILE* output=std::fopen(filename,"wbx");
      if(!output){err="postFFN file exists/unwritable"; return false;}
      const size_t written=std::fwrite(host.data(),sizeof(float),
                                      host.size(),output);
      const int closed=std::fclose(output);
      if(written!=host.size() || closed!=0){
        err="postFFN partial tensor write; discard entire trace"; return false;
      }
    }
    char sidecar[256];
    const int ns=std::snprintf(sidecar,sizeof(sidecar),
                              "%s/p%u_context.txt",post,seq);
    if(ns<0 || (size_t)ns>=sizeof(sidecar)){
      err="postFFN sidecar filename invalid"; return false;
    }
    FILE* side=std::fopen(sidecar,"wbx");
    if(!side){err="postFFN sidecar exists/unwritable"; return false;}
    const size_t written=std::fwrite(ctx_expected,1,(size_t)ne,side);
    const int closed=std::fclose(side);
    if(written!=(size_t)ne || closed!=0){
      err="postFFN partial sidecar; discard entire trace"; return false;
    }
    return true;
  }catch(const std::exception& e){
    err=std::string("postFFN scratch exception: ")+e.what();
    return false;
  }
}
'''

PRE_ARM = r'''  if(std::getenv("HERMES_SCRATCH_LAYER0_POSTFFN_DIR")){
    const size_t root_length=std::strlen(root);
    if(root_length==0 || root_length>180){
      err="postFFN invalid base root length"; return false;
    }
    std::memcpy(g_l0_postffn_base,root,root_length+1);
    g_l0_postffn_sequence=sequence;
  }
'''

POST_HOOK = r'''  // Exact layer0 serial context from the earlier v2 trace; no new
  // probe if POSTFFN_DIR is absent. This branch is scratch-only.
  if(li==0 && std::getenv("HERMES_SCRATCH_LAYER0_POSTFFN_DIR")){
    if(ds.async_ctx || ds.sliding_kv_enabled ||
       std::getenv("HERMES_PP2_ASYNC_STREAMS")){
      err="postFFN capture refuses async/sliding/multiuser"; return false;
    }
    const L0TraceField post_fields[]={
      {"attn_post_norm",d_h,HID},
      {"ffn_out",d_down,HID},
      {"post_ffn",d_x,HID}
    };
    if(!l0_postffn_copy(g_l0_postffn_base,
                        std::getenv("HERMES_SCRATCH_LAYER0_POSTFFN_DIR"),
                        g_l0_postffn_sequence,stream,err,post_fields,3)){
      return false;
    }
    g_l0_postffn_base[0]='\0';
    g_l0_postffn_sequence=2;
  }
'''


def extend_v2_source(v2: str) -> str:
    """Pure deterministic source transformation; no CUDA or filesystem writes."""
    need(v2.count(BASE_TAG) == 3 and POST_TAG not in v2,
         "requires unchanged 3-marker v2 tracer, not a re-patched source")
    need(v2.count(HELPER_ANCHOR) == 1 and
         v2.count(CONTEXT_ANCHOR) == 1,
         "frozen source anchor absent, duplicated, or changed")
    gdn_begin = v2.find("static bool run_gdn_decode_layer(")
    full_begin = v2.find("static bool run_full_decode_layer(")
    need(0 <= gdn_begin < full_begin and
         v2[gdn_begin:full_begin].count(FFN_ANCHOR) == 1,
         "postFFN hook must be inside serial layer0 GDN, not full-attn/prefill")
    need(v2.find("L0TraceScope actual_decode_token(tok,pos,&ds);") > full_begin,
         "source no longer binds v2 context to serial decode arguments")
    need("g_l0_trace_state.compare_exchange_strong" in v2 and
         "scratch layer0 trace refuses asynchronous/sliding dispatch" in v2,
         "v2 same-DecodeState or async preflight was removed")
    result = v2.replace(
        HELPER_ANCHOR,
        HELPER_ANCHOR.replace("static bool l0_trace_scratch(\n",
                              POST_HELPER + "\nstatic bool l0_trace_scratch(\n"), 1,
    )
    result = result.replace(
        CONTEXT_ANCHOR,
        CONTEXT_ANCHOR.replace("  return true;\n}\n",
                               PRE_ARM + "  return true;\n}\n", 1), 1,
    )
    gdn_begin = result.index("static bool run_gdn_decode_layer(")
    full_begin = result.index("static bool run_full_decode_layer(")
    gdn_region = result[gdn_begin:full_begin]
    inserted = FFN_ANCHOR.replace(
        "  if(stream==0 && !decode_skip_layer_sync()) cudaDeviceSynchronize();\n",
        POST_HOOK +
        "  if(stream==0 && !decode_skip_layer_sync()) cudaDeviceSynchronize();\n",
        1,
    )
    result = (result[:gdn_begin] +
              gdn_region.replace(FFN_ANCHOR, inserted, 1) +
              result[full_begin:])
    need(result.count(BASE_TAG) == 3 and
         result.count(POST_TAG) == 3 and
         result.count("l0_postffn_copy(") == 2 and
         result.count("L0TraceScope actual_decode_token(tok,pos,&ds);") == 1,
         "postFFN source composition did not preserve v2 guard and exact hook")
    return result


def compose(source: bytes, patcher_path: Path) -> bytes:
    need(hashlib.sha256(source).hexdigest() == SOURCE_SHA,
         "current production PP2 SHA differs from published frozen source")
    original = source.decode("utf-8")
    old = load_patcher(patcher_path)
    v2 = old.patched_source(original)
    need(v2.count(BASE_TAG) == 3, "published v2 patch is not three-marker")
    return extend_v2_source(v2).encode("utf-8")


def context_bytes(pos: int) -> bytes:
    need(type(pos) is int and pos in (0, 1), "only absolute positions 0,1")
    return (f"token_id={TOKENS[pos]}\nabsolute_position={pos}\n"
            f"sequence={pos}\n").encode("ascii")


def checked_inventory(folder: Path, expected: set[str]) -> None:
    need(folder.is_dir() and not folder.is_symlink(), "scratch directory unavailable/aliased")
    paths = list(folder.iterdir())
    need(len(paths) == len(expected) and {p.name for p in paths} == expected,
         "missing/extra stale scratch capture file")
    need(all(p.is_file() and not p.is_symlink() for p in paths),
         "scratch capture includes a symlink/nonregular file")


def validate_post_sidecars(base: Path, post: Path, *, model: Path | None = None,
                           synthetic_test: bool = False) -> dict:
    """Offline bytes check only. NO HERMES/llama oracle admission."""
    need(type(synthetic_test) is bool, "invalid synthetic_test flag")
    need(base.resolve() != post.resolve() and
         post.name == base.name + "-postffn" and base.name.startswith("h27-"),
         "postFFN directory must be distinct and paired with v2 base")
    pre_files = {f"p{pos}_{stage}.f32" for pos in (0, 1) for stage in PRE_STAGES}
    context_files = {f"p{pos}_context.txt" for pos in (0, 1)}
    post_files = {f"p{pos}_{stage}.f32" for pos in (0, 1) for stage in POST_STAGES}
    checked_inventory(base, pre_files | context_files)
    checked_inventory(post, post_files | context_files)
    if not synthetic_test:
        need(model is not None and model.is_file() and
             model.stat().st_size == GGUF_SIZE and
             sha256_file(model) == GGUF_SHA,
             "full pinned physical GGUF unavailable or wrong")
    tensors = []
    for pos in (0, 1):
        expected = context_bytes(pos)
        need((base / f"p{pos}_context.txt").read_bytes() == expected and
             (post / f"p{pos}_context.txt").read_bytes() == expected,
             "self-reported postFFN token/absolute-position identity mismatch")
        for stage, width in PRE_STAGES.items():
            raw = (base / f"p{pos}_{stage}.f32").read_bytes()
            need(len(raw) == width * 4 and all(
                 math.isfinite(value) for (value,) in struct.iter_unpack("<f", raw)),
                 "pre-FFN v2 raw tensor length or finite FP32 invalid")
        for stage, width in POST_STAGES.items():
            tensor = post / f"p{pos}_{stage}.f32"
            raw = tensor.read_bytes()
            need(len(raw) == width * 4 and all(
                 math.isfinite(value) for (value,) in struct.iter_unpack("<f", raw)),
                 "postFFN tensor not exactly 5120 finite little-endian F32")
            tensors.append({"name": stage, "runtime_reported_position": pos,
                            "shape": [width], "bytes": len(raw),
                            "sha256": hashlib.sha256(raw).hexdigest()})
    return {
        "schema": ("h27.postffn-synthetic-cpu-fixture.v1" if synthetic_test else
                   "h27.postffn-raw-sidecar-inspection.v1"),
        "status": ("SYNTHETIC_CPU_FIXTURE_NOT_RUNTIME" if synthetic_test else
                   "CPU_RAW_BYTES_CHECKED_NOT_INDEPENDENT_ORACLE"),
        "not_release_gate": True,
        "runtime_identity_scope": "INSTRUMENTED_HERMES_SELF_REPORT_ONLY",
        "model_sha256": (None if synthetic_test else GGUF_SHA),
        "independent_llama_numeric_oracle_verified": False,
        "postffn_tensors": tensors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="published original distributed/pp2_fwd.cu")
    parser.add_argument("--output", type=Path, required=True,
                        help="new disposable OS temp subtree pp2_fwd.cu")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()
    need(source.is_file() and not args.source.is_symlink() and
         source.name == "pp2_fwd.cu" and output.name == "pp2_fwd.cu",
         "source/output file names must be pp2_fwd.cu")
    need(not output.exists() and output.parent.is_dir() and
         output.parent != temp_root and
         output.is_relative_to(temp_root) and
         args.output.absolute() == output and
         not args.output.is_symlink() and not output.parent.is_symlink() and
         source != output,
         "output must be NEW canonical nonaliased temporary scratch file")
    original = source.read_bytes()
    patched = compose(original, Path(__file__).with_name(
        "pp2_layer0_trace_scratch_patch.py"))
    need(sha256_file(source) == SOURCE_SHA,
         "production source changed during scratch preparation")
    with output.open("xb") as stream:
        stream.write(patched)
    need(sha256_file(output) == hashlib.sha256(patched).hexdigest(),
         "scratch output SHA changed after write")
    print("PASS_CPU_SOURCE_ONLY_POSTFFN_NOT_ORACLE", hashlib.sha256(patched).hexdigest())
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (Blocked, OSError, UnicodeError) as exc:
        print("BLOCKED_POSTFFN_SCRATCH_SOURCE", str(exc), file=sys.stderr)
        sys.exit(2)
