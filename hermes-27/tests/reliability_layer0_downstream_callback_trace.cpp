// Independent llama.cpp C API layer0 Qwen35 downstream tracer.
// This is an ISOLATED source extension of the earlier layer tracer,
// not a production modification and has no HERMES runtime linkage.
// --inspect-plan, --selftest and --inspect-runtime NEVER load the GGUF
// or run model inference. --capture requires an explicit authorization flag,
// verified 19.7GB model SHA256 and verified loaded runtime-library lock.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "reliability_layer0_downstream_callback_sha256.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <link.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace fs=std::filesystem;
using reliability_layer::Sha256;
namespace {
constexpr int VOCAB=248320;
constexpr uint64_t MiB=1024u*1024u;
constexpr std::array<std::string_view,22> STAGES{{
  "attn_norm", "linear_attn_qkv_mixed", "conv_output_raw",
  "conv_output_silu", "beta_sigmoid", "gate", "q_conv",
  "k_conv", "v_conv", "q_conv_predelta", "k_conv_predelta",
  "v_conv_predelta", "state_predelta", "attn_output",
  "new_state", "final_output", "linear_attn_out",
  "attn_residual", "attn_post_norm", "ffn_out",
  "post_ffn", "z"
}};
struct Options {
  bool selftest=false, inspect=false, inspect_runtime=false;
  bool capture=false, exclusive=false;
  fs::path model, out_dir, llama_lib, runtime_lock;
  std::string expected_model_sha, expected_lib_sha;
  int token0=6205, token1=161421, gpu_layers=80, threads=4;
  uint64_t max_total=16*MiB, max_single=2*MiB;
  std::set<std::string> stages;
};
struct TensorCapture {
  std::string name, file, type, sha, buffer_name;
  std::array<int64_t,4> shape{};
  std::array<size_t,4> strides{};
  size_t elements=0, bytes=0;
  int token_axis=-1; // -1 for recurrent state (no per-position axis)
};
std::string json_string(std::string_view s) {
  std::ostringstream out;
  out << '"';
  const char* hex="0123456789abcdef";
  for(unsigned char c:s) {
    switch(c) {
      case '"':out<<"\\\"";break;
      case '\\':out<<"\\\\";break;
      case '\n':out<<"\\n";break;
      case '\r':out<<"\\r";break;
      case '\t':out<<"\\t";break;
      default:if(c<32)out<<"\\u00"<<hex[c>>4]<<hex[c&15];else out<<char(c);
    }
  }
  out<<'"';return out.str();
}
bool integer(std::string_view s,int& out) {
  if(s.empty())return false;
  const auto parsed=std::from_chars(s.data(),s.data()+s.size(),out);
  return parsed.ec==std::errc() && parsed.ptr==s.data()+s.size();
}
std::string tensor_name(std::string_view stage,int layer) {
  return std::string(stage)+"-"+std::to_string(layer);
}
bool valid_sha(std::string_view s) {
  if(s.size()!=64)return false;
  for(char c:s)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))return false;
  return true;
}
bool options(int argc,char** argv,Options& o) {
  for(int i=1;i<argc;++i) {
    const std::string_view key=argv[i];
    if(key=="--selftest"){o.selftest=true;continue;}
    if(key=="--inspect-plan"){o.inspect=true;continue;}
    if(key=="--inspect-runtime"){o.inspect_runtime=true;continue;}
    if(key=="--capture"){o.capture=true;continue;}
    if(key=="--gpu-exclusive"){o.exclusive=true;continue;}
    if(i+1>=argc)return false;
    const std::string v=argv[++i];
    if(key=="--model")o.model=v;
    else if(key=="--out-dir")o.out_dir=v;
    else if(key=="--llama-lib")o.llama_lib=v;
    else if(key=="--runtime-lock")o.runtime_lock=v;
    else if(key=="--expected-model-sha")o.expected_model_sha=v;
    else if(key=="--expected-libllama-sha")o.expected_lib_sha=v;
    else if(key=="--token0"){if(!integer(v,o.token0))return false;}
    else if(key=="--token1"){if(!integer(v,o.token1))return false;}
    else if(key=="--n-gpu-layers"){if(!integer(v,o.gpu_layers))return false;}
    else if(key=="--threads"){if(!integer(v,o.threads))return false;}
    else if(key=="--max-total-mib"){
      int x=0;if(!integer(v,x)||x<1||x>64)return false;o.max_total=uint64_t(x)*MiB;
    } else if(key=="--max-tensor-mib"){
      int x=0;if(!integer(v,x)||x<1||x>8)return false;o.max_single=uint64_t(x)*MiB;
    } else if(key=="--include") {
      std::string_view names(v);
      while(!names.empty()) {
        const auto pos=names.find(',');
        const std::string name(names.substr(0,pos));
        if(std::find(STAGES.begin(),STAGES.end(),name)==STAGES.end())return false;
        o.stages.insert(name);
        if(pos==std::string_view::npos)break;
        names.remove_prefix(pos+1);
        if(names.empty())return false;
      }
    } else return false;
  }
  if(int(o.selftest)+int(o.inspect)+int(o.inspect_runtime)+int(o.capture)!=1)return false;
  if(!o.capture)return true;
  if(!o.exclusive || o.token0!=6205 || o.token1!=161421 ||
     !valid_sha(o.expected_model_sha) ||
     !valid_sha(o.expected_lib_sha) ||
     o.threads<1 || o.threads>64 ||
     o.gpu_layers<0 || o.gpu_layers>128 || o.max_single>o.max_total ||
     !fs::is_regular_file(o.model) || !fs::is_regular_file(o.llama_lib) ||
     !fs::is_regular_file(o.runtime_lock) ||
     !fs::is_directory(o.out_dir))return false;
  if(o.stages.empty()) {
    // Selected stages are directly tagged with cb() in qwen35.cpp or
    // delta-net-base.cpp. The recurrent cache state_predelta (~3 MiB)
    // and new_state (missing from fused GDN) are optional only.
    for(auto stage:{"attn_norm","linear_attn_qkv_mixed","conv_output_raw",
                    "conv_output_silu","beta_sigmoid","gate",
                    "q_conv_predelta","k_conv_predelta","v_conv_predelta",
                    "attn_output","final_output","linear_attn_out",
                    "attn_residual","attn_post_norm"})
      o.stages.insert(stage);
  }
  for(const auto& file:fs::directory_iterator(o.out_dir)) {
    if(file.is_regular_file())return false; // strict empty capture directory
  }
  return true;
}
std::string source_type(ggml_type t) {
  if(t==GGML_TYPE_F32)return "f32";
  if(t==GGML_TYPE_F16)return "f16";
  if(t==GGML_TYPE_BF16)return "bf16";
  return ggml_type_name(t);
}
struct RuntimeLib {
  std::string filename;
  std::string canonical_path;
  std::string sha256;
};
// This is a CPU-only inspection of libraries ALREADY linked into this
// tracer process. It never opens a GGUF, CUDA device or model context.
int library_visit(struct dl_phdr_info* info,size_t,void* user) {
  if(!info->dlpi_name || !*info->dlpi_name)return 0;
  const fs::path p(info->dlpi_name);
  const std::string base=p.filename().string();
  if(base.rfind("libllama.so",0)!=0 &&
     base.rfind("libggml",0)!=0)return 0;
  auto& result=*static_cast<std::map<std::string,RuntimeLib>*>(user);
  try {
    const fs::path real=fs::canonical(p);
    const std::string filename=real.filename().string();
    RuntimeLib entry{filename,real.string(),Sha256::file(real)};
    auto [it,added]=result.emplace(filename,entry);
    if(!added && it->second.sha256!=entry.sha256)
      throw std::runtime_error("ambiguous runtime library basename: "+filename);
  } catch(const std::exception& e) {
    // dl_iterate_phdr callback is C ABI, do not throw through glibc.
    std::fprintf(stderr,"LIBRARY_VISIT_BLOCKED: %s\n",e.what());
    return 1;
  }
  return 0;
}
std::map<std::string,RuntimeLib> runtime_libraries() {
  std::map<std::string,RuntimeLib> records;
  const int err=dl_iterate_phdr(&library_visit,&records);
  if(err!=0 || records.empty())
    throw std::runtime_error("loaded runtime library identity scan failed");
  bool found_llama=false;
  for(const auto& [name,rec]:records)
    if(name.rfind("libllama.so",0)==0)found_llama=true;
  if(!found_llama)throw std::runtime_error("libllama absent from runtime scan");
  return records;
}
std::map<std::string,std::string> parse_runtime_lock(const fs::path& path) {
  std::ifstream f(path);
  if(!f)throw std::runtime_error("cannot open runtime SHA lock");
  std::map<std::string,std::string> lock;
  std::string line;
  while(std::getline(f,line)) {
    if(line.empty() || line[0]=='#')continue;
    std::istringstream input(line);
    std::string name,sha,extra;
    if(!(input>>name>>sha) || input>>extra || !valid_sha(sha) ||
       name.find('/')!=std::string::npos ||
       (name.rfind("libllama.so",0)!=0 && name.rfind("libggml",0)!=0) ||
       !lock.emplace(name,sha).second)
      throw std::runtime_error("invalid/duplicate runtime lock entry");
  }
  if(!f.eof())throw std::runtime_error("runtime lock read failed");
  if(lock.empty())throw std::runtime_error("empty runtime lock");
  return lock;
}
void verify_runtime_lock(const std::map<std::string,RuntimeLib>& found,
                         const std::map<std::string,std::string>& lock) {
  if(found.size()!=lock.size())
    throw std::runtime_error("runtime library count differs from trusted lock");
  for(const auto& [name,entry]:found) {
    auto it=lock.find(name);
    if(it==lock.end() || it->second!=entry.sha256)
      throw std::runtime_error("runtime library hash mismatch: "+name);
  }
}
class Receiver {
 public:
  explicit Receiver(const Options& o):o_(o) {
    for(const auto& name:o_.stages)names_.insert(tensor_name(name,0));
  }
  static bool callback(ggml_tensor* tensor,bool ask,void* user) {
    auto& r=*static_cast<Receiver*>(user);
    if(!tensor || !r.names_.count(std::string(tensor->name)))return false;
    if(ask){++r.ask_;return true;}
    ++r.ready_;
    if(!r.error_.empty())return false;
    try {r.capture(tensor);return true;}
    catch(const std::exception& e){r.error_=e.what();return false;}
  }
  std::vector<std::string> missing()const {
    std::vector<std::string> out;
    for(const auto& name:names_)if(!seen_.count(name))out.push_back(name);
    return out;
  }
  const auto& captures()const{return captures_;}
  const auto& error()const{return error_;}
  size_t ask_count()const{return ask_;}
  size_t ready_count()const{return ready_;}
 private:
  void capture(const ggml_tensor* t);
  const Options& o_;
  std::set<std::string> names_,seen_;
  std::vector<TensorCapture> captures_;
  uint64_t total_=0;
  size_t ask_=0,ready_=0;
  std::string error_;
};

void Receiver::capture(const ggml_tensor* t) {
  const std::string name(t->name);
  if(!seen_.insert(name).second)
    throw std::runtime_error("duplicate callback: "+name);
  if(captures_.size()>=32)throw std::runtime_error("tensor count limit");
  if(t->type!=GGML_TYPE_F32 && t->type!=GGML_TYPE_F16 &&
     t->type!=GGML_TYPE_BF16)
    throw std::runtime_error("unsupported tensor type: "+name);
  const ggml_tensor* base=t;
  while(base->view_src)base=base->view_src;
  if(!t->data || !base->buffer)
    throw std::runtime_error("backend buffer/data absent: "+name);
  const size_t elem=ggml_type_size(t->type);
  if(t->nb[0]!=elem || (elem!=2 && elem!=4))
    throw std::runtime_error("noncontiguous scalar elements: "+name);
  const int64_t n=ggml_nelements(t);
  if(n<=0 || uint64_t(n)>o_.max_single/sizeof(float) ||
     uint64_t(n)> (o_.max_total-total_)/sizeof(float))
    throw std::runtime_error("bounded tensor capture over budget: "+name);
  const size_t bytes=static_cast<size_t>(n)*sizeof(float);
  std::vector<float> floats(static_cast<size_t>(n));
  std::vector<uint8_t> row(static_cast<size_t>(t->ne[0])*elem);
  size_t copied=0;
  const size_t extent=ggml_nbytes(t);
  // GGML dimension 0 is contiguous. Noncontiguous 2D/3D/4D views are
  // gathered one row at a time (no illegal bulk memcpy on GPU addresses).
  for(int64_t i3=0;i3<t->ne[3];++i3)
    for(int64_t i2=0;i2<t->ne[2];++i2)
      for(int64_t i1=0;i1<t->ne[1];++i1) {
        const size_t offset=static_cast<size_t>(i1)*t->nb[1]+
                            static_cast<size_t>(i2)*t->nb[2]+
                            static_cast<size_t>(i3)*t->nb[3];
        if(offset>extent || row.size()>extent-offset)
          throw std::runtime_error("strided tensor bounds invalid: "+name);
        ggml_backend_tensor_get(t,row.data(),offset,row.size());
        for(int64_t i0=0;i0<t->ne[0];++i0) {
          float value=0;
          if(t->type==GGML_TYPE_F32) {
            std::memcpy(&value,row.data()+static_cast<size_t>(i0)*4,4);
          } else if(t->type==GGML_TYPE_F16) {
            ggml_fp16_t h;
            std::memcpy(&h,row.data()+static_cast<size_t>(i0)*2,2);
            value=ggml_fp16_to_fp32(h);
          } else {
            ggml_bf16_t b;
            std::memcpy(&b,row.data()+static_cast<size_t>(i0)*2,2);
            value=ggml_bf16_to_fp32(b);
          }
          if(!std::isfinite(value))
            throw std::runtime_error("nonfinite tensor value: "+name);
          floats[copied++]=value;
        }
      }
  if(copied!=floats.size())throw std::runtime_error("truncated tensor copy");
  const std::string filename=name+".f32";
  const fs::path target=o_.out_dir/filename;
  const fs::path partial=target.string()+".partial";
  if(fs::exists(target)||fs::exists(partial))
    throw std::runtime_error("refusing stale file: "+filename);
  {
    std::ofstream output(partial,std::ios::binary|std::ios::trunc);
    if(!output)throw std::runtime_error("cannot write: "+partial.string());
    output.write(reinterpret_cast<const char*>(floats.data()),
                 static_cast<std::streamsize>(bytes));
    output.flush();
    if(!output)throw std::runtime_error("incomplete tensor output: "+filename);
  }
  fs::rename(partial,target);
  TensorCapture record;
  record.name=name;record.file=filename;
  record.type=source_type(t->type);
  record.sha=Sha256::digest(floats.data(),bytes);
  record.buffer_name=ggml_backend_buffer_name(base->buffer);
  record.elements=floats.size();record.bytes=bytes;
  for(int i=0;i<4;++i) {
    record.shape[i]=t->ne[i];
    record.strides[i]=t->nb[i];
  }
  // q/k/v views, beta_sigmoid and recurrent attention use [D,H,T,S].
  // Most other layer0 stage tensors use [D,T,S] or [D,T].
  // Recurrent state is per SEQUENCE and does not expose a token axis.
  const bool state=name=="state_predelta-0" || name=="new_state-0";
  const bool token_dim2=name=="q_conv-0" || name=="k_conv-0" ||
      name=="v_conv-0" || name=="q_conv_predelta-0" ||
      name=="k_conv_predelta-0" || name=="v_conv_predelta-0" ||
      name=="beta_sigmoid-0" || name=="attn_output-0";
  record.token_axis=state ? -1 : (token_dim2 ? 2 : 1);
  if(!state && record.shape[record.token_axis]!=2)
    throw std::runtime_error("callback did not contain exact 2-token sequence: "+name);
  captures_.push_back(record);
  total_+=bytes;
}

void manifest(const Options& o,const Receiver& rx,
              const std::string& model_sha,const std::string& binary_sha,
              const std::string& library_sha,
              const std::map<std::string,RuntimeLib>& libs,
              const std::array<int,2>& argmax) {
  const fs::path target=o.out_dir/"manifest.json";
  const fs::path part=target.string()+".partial";
  if(fs::exists(target)||fs::exists(part))
    throw std::runtime_error("refusing stale manifest");
  std::ofstream f(part);
  if(!f)throw std::runtime_error("cannot open manifest");
  f<<"{\n  \"schema\":\"h27.reliability-layer0-downstream-cb.v1\","
    <<"\n  \"status\":\"CAPTURED_NOT_INDEPENDENTLY_COMPARED\","
    <<"\n  \"backend\":\"llama.cpp\","
    <<"\n  \"model_path\":"<<json_string(fs::canonical(o.model).string())<<","
    <<"\n  \"model_sha256\":"<<json_string(model_sha)<<","
    <<"\n  \"model_bytes\":"<<fs::file_size(o.model)<<","
    <<"\n  \"exporter_path\":"<<json_string(fs::canonical("/proc/self/exe").string())<<","
    <<"\n  \"exporter_sha256\":"<<json_string(binary_sha)<<","
    <<"\n  \"libllama_path\":"<<json_string(fs::canonical(o.llama_lib).string())<<","
    <<"\n  \"libllama_sha256\":"<<json_string(library_sha)<<","
    <<"\n  \"runtime_lock_sha256\":"<<json_string(Sha256::file(o.runtime_lock))<<","
    <<"\n  \"llama_build\":"<<json_string(llama_print_system_info())<<","
    <<"\n  \"prompt_ids\":["<<o.token0<<","<<o.token1<<"],"
    <<"\n  \"absolute_positions\":[0,1],"
    <<"\n  \"one_contiguous_decode_batch\":true,"
    <<"\n  \"layer\":0,"
    <<"\n  \"n_gpu_layers\":"<<o.gpu_layers<<","
    <<"\n  \"greedy_argmax\":["<<argmax[0]<<","<<argmax[1]<<"],"
    <<"\n  \"callback_ask_count\":"<<rx.ask_count()<<","
    <<"\n  \"callback_ready_count\":"<<rx.ready_count()<<","
    <<"\n  \"max_total_bytes\":"<<o.max_total<<","
    <<"\n  \"max_single_bytes\":"<<o.max_single<<","
    <<"\n  \"tensor_data_format\":\"little-endian-fp32-ne0-fastest\","
    <<"\n  \"requested_stages\":[";
  size_t requested=0;
  for(const auto& stage:o.stages)
    f<<(requested++?",":"")<<json_string(tensor_name(stage,0));
  f<<"],"
    <<"\n  \"runtime_libraries\":[\n";
  size_t lib_index=0;
  for(const auto& [name,entry]:libs) {
    f<<"    {\"name\":"<<json_string(name)
     <<",\"path\":"<<json_string(entry.canonical_path)
     <<",\"sha256\":"<<json_string(entry.sha256)<<"}"
     <<(++lib_index<libs.size()?",":"")<<"\n";
  }
  f<<"  ],\n  \"tensors\":[\n";
  const auto& captures=rx.captures();
  for(size_t j=0;j<captures.size();++j) {
    const auto& c=captures[j];
    f<<"    {\"name\":"<<json_string(c.name)
     <<",\"file\":"<<json_string(c.file)
     <<",\"source_type\":"<<json_string(c.type)
     <<",\"buffer\":"<<json_string(c.buffer_name)
     <<",\"shape\":[";
    for(int i=0;i<4;++i)f<<(i?",":"")<<c.shape[i];
    f<<"],\"source_strides_bytes\":[";
    for(int i=0;i<4;++i)f<<(i?",":"")<<c.strides[i];
    f<<"],\"token_axis\":"<<c.token_axis
     <<",\"elements\":"<<c.elements<<",\"bytes\":"<<c.bytes
     <<",\"sha256\":"<<json_string(c.sha)<<"}"
     <<(j+1<captures.size()?",":"")<<"\n";
  }
  f<<"  ]\n}\n";
  f.flush();
  if(!f)throw std::runtime_error("failed to write manifest");
  f.close();
  fs::rename(part,target);
}

int selftest() {
  constexpr std::string_view abc=
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
  if(Sha256::digest("abc",3)!=abc)
    throw std::runtime_error("SHA256 CPU test vector mismatch");
  // Verify multi-block SHA streaming, needed for a multi-gigabyte real GGUF.
  std::string million(1000000,'a');
  Sha256 rolling;
  for(size_t offset=0;offset<million.size();offset+=63) {
    const size_t n=std::min<size_t>(63,million.size()-offset);
    rolling.update(million.data()+offset,n);
  }
  if(rolling.finish()!="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0")
    throw std::runtime_error("SHA256 1M-byte streaming test mismatch");
  const fs::path dir=fs::temp_directory_path()/
    ("h27-reliability-layer-cb-synthetic-"+std::to_string(getpid()));
  // This must not silently accept prior results.
  if(fs::exists(dir))throw std::runtime_error("selftest scratch already exists");
  fs::create_directory(dir);
  Options o;o.out_dir=dir;o.stages.insert("attn_norm");
  Receiver rx(o);
  ggml_init_params params{};
  params.mem_size=4096;params.no_alloc=true;
  ggml_context* ctx=ggml_init(params);
  if(!ctx)throw std::runtime_error("cannot create GGML synthetic context");
  alignas(256) std::array<float,16> input{};
  for(size_t i=0;i<input.size();++i)input[i]=float(i);
  ggml_tensor* base=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,4,4);
  base->data=input.data();
  base->buffer=ggml_backend_cpu_buffer_from_ptr(input.data(),sizeof(input));
  if(!base->buffer)throw std::runtime_error("CPU backend buffer unavailable");
  ggml_tensor* view=ggml_view_2d(ctx,base,2,2,4*sizeof(float),0);
  view->data=input.data();
  ggml_set_name(view,"attn_norm-0");
  ggml_tensor* other=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,2);
  ggml_set_name(other,"attn_norm-1");
  if(Receiver::callback(other,true,&rx) || Receiver::callback(other,false,&rx) ||
     !Receiver::callback(view,true,&rx) || !Receiver::callback(view,false,&rx))
    throw std::runtime_error("callback ask/ready selection or capture failed: "+rx.error());
  if(rx.captures().size()!=1 || rx.captures()[0].shape[0]!=2 ||
     rx.captures()[0].shape[1]!=2 || rx.captures()[0].bytes!=16 ||
     rx.captures()[0].token_axis!=1)
    throw std::runtime_error("synthetic shape or FP32 bytes wrong");
  std::array<float,4> output{};
  {
    std::ifstream file(dir/"attn_norm-0.f32",std::ios::binary);
    file.read(reinterpret_cast<char*>(output.data()),sizeof(output));
    if(!file)throw std::runtime_error("synthetic FP32 read incomplete");
  }
  for(int r=0;r<2;++r)for(int col=0;col<2;++col)
    if(output[size_t(r*2+col)]!=input[size_t(r*4+col)])
      throw std::runtime_error("strided view gather has wrong layout");
  if(Sha256::file(dir/"attn_norm-0.f32")!=rx.captures()[0].sha)
    throw std::runtime_error("synthetic digest mismatch");
  Options bounded=o;
  bounded.max_single=8; // view is 16 FP32 bytes; reject before backend copy
  Receiver budget_check(bounded);
  if(!Receiver::callback(view,true,&budget_check) ||
     Receiver::callback(view,false,&budget_check) ||
     budget_check.error().find("budget")==std::string::npos)
    throw std::runtime_error("oversize tensor must fail before host copy");
  if(!Receiver::callback(view,true,&rx) || Receiver::callback(view,false,&rx) ||
     rx.error().find("duplicate")==std::string::npos)
    throw std::runtime_error("duplicate capture must fail closed");
  // Reproduce the downstream GDN per-head layout [D,H,2 tokens,1 seq],
  // including the position axis at index 2 and a 4D strided view.
  Options heads_opt=o;
  heads_opt.stages.clear();heads_opt.stages.insert("attn_output");
  Receiver heads_rx(heads_opt);
  ggml_tensor* heads=ggml_view_4d(ctx,base,2,2,2,1,
      4*sizeof(float),8*sizeof(float),16*sizeof(float),0);
  heads->data=input.data();
  ggml_set_name(heads,"attn_output-0");
  if(!Receiver::callback(heads,true,&heads_rx) ||
     !Receiver::callback(heads,false,&heads_rx) ||
     heads_rx.captures().size()!=1 ||
     heads_rx.captures()[0].token_axis!=2 ||
     heads_rx.captures()[0].shape[2]!=2 ||
     heads_rx.captures()[0].bytes!=32)
    throw std::runtime_error("4D two-token GDN-like callback failed: "+heads_rx.error());
  std::array<float,8> head_output{};
  {
    std::ifstream file(dir/"attn_output-0.f32",std::ios::binary);
    file.read(reinterpret_cast<char*>(head_output.data()),sizeof(head_output));
    if(!file)throw std::runtime_error("4D synthetic output incomplete");
  }
  const std::array<float,8> head_expected{{0,1,4,5,8,9,12,13}};
  if(head_output!=head_expected)
    throw std::runtime_error("4D GDN token/head gather mapping incorrect");
  const fs::path lock_file=dir/"test-runtime-lock.txt";
  {
    std::ofstream lock(lock_file);
    lock<<"libllama.so.0.5.0 "<<std::string(64,'0')<<"\n";
  }
  auto known=parse_runtime_lock(lock_file);
  if(known.size()!=1 || known.begin()->first!="libllama.so.0.5.0")
    throw std::runtime_error("CPU runtime lock parser rejected valid fixture");
  std::map<std::string,RuntimeLib> found{
      {"libllama.so.0.5.0",
       {"libllama.so.0.5.0","/tmp/placeholder",std::string(64,'0')}}};
  verify_runtime_lock(found,known);
  found.begin()->second.sha256[0]='a';
  bool mismatch_failed=false;
  try{verify_runtime_lock(found,known);}
  catch(const std::exception&){mismatch_failed=true;}
  if(!mismatch_failed)
    throw std::runtime_error("runtime hash mutation accepted");
  {
    std::ofstream lock(lock_file,std::ios::app);
    lock<<"libllama.so.0.5.0 "<<std::string(64,'0')<<"\n";
  }
  bool duplicate_failed=false;
  try{(void)parse_runtime_lock(lock_file);}
  catch(const std::exception&){duplicate_failed=true;}
  if(!duplicate_failed)
    throw std::runtime_error("duplicate runtime entry accepted");
  ggml_backend_buffer_free(base->buffer);
  ggml_free(ctx);
  fs::remove(dir/"attn_norm-0.f32");
  fs::remove(dir/"attn_output-0.f32");
  fs::remove(lock_file);
  fs::remove(dir);
  std::puts("PASS CPU callback selection, 2D and 4D token axes, "
            "noncontiguous GGML gather, FP32/SHA256, bounds and duplicate fail-closed");
  return 0;
}

int capture_real(const Options& o) {
  // Validate identity of *loaded* libllama, not merely a supplied filename.
  Dl_info info{};
  if(!dladdr(reinterpret_cast<const void*>(&llama_backend_init),&info) ||
     !info.dli_fname || fs::canonical(info.dli_fname)!=fs::canonical(o.llama_lib))
    throw std::runtime_error("loaded libllama differs from expected library");
  // Model identity is checked by STREAMING SHA256 of the entire physical
  // 19.7GB GGUF BEFORE initializing llama, even for a mmap-backed runtime.
  const std::string model_sha=Sha256::file(o.model);
  const std::string lib_sha=Sha256::file(o.llama_lib);
  const std::string binary_sha=Sha256::file("/proc/self/exe");
  if(model_sha!=o.expected_model_sha || lib_sha!=o.expected_lib_sha)
    throw std::runtime_error("physical GGUF / libllama SHA256 lock mismatch");
  const auto lock=parse_runtime_lock(o.runtime_lock);
  // Inspect already-loaded shared libraries and fail on any unrecognized
  // binary even BEFORE model initialization; after model-load compare the
  // full resolved set to the lock (CUDA backends can be loaded lazily).
  for(const auto& [name,lib]:runtime_libraries()) {
    const auto it=lock.find(name);
    if(it==lock.end() || it->second!=lib.sha256)
      throw std::runtime_error("pre-model runtime lib lock mismatch: "+name);
  }
  std::fprintf(stderr,"model SHA256 %s; loading 27B weights only because "
    "--capture --gpu-exclusive was supplied\n",model_sha.c_str());
  llama_backend_init();
  llama_model_params mp=llama_model_default_params();
  mp.n_gpu_layers=o.gpu_layers;
  llama_model* model=llama_model_load_from_file(o.model.string().c_str(),mp);
  if(!model) {
    llama_backend_free();
    throw std::runtime_error("independent llama.cpp GGUF model load failed");
  }
  if(llama_vocab_n_tokens(llama_model_get_vocab(model))!=VOCAB) {
    llama_model_free(model);llama_backend_free();
    throw std::runtime_error("real model vocabulary mismatch");
  }
  Receiver rx(o);
  llama_context_params cp=llama_context_default_params();
  // A SINGLE, contiguous two-token decode preserves causal/convolution/GDN
  // state semantics of token6205 at pos0 followed by token161421 at pos1.
  // A second independent single-token context would NOT be a valid pos1.
  cp.n_ctx=128;cp.n_batch=8;cp.n_ubatch=8;cp.n_seq_max=1;
  cp.n_outputs_max=2;cp.n_outputs_max_per_seq=2;
  cp.n_threads=o.threads;cp.n_threads_batch=o.threads;
  cp.cb_eval=&Receiver::callback;cp.cb_eval_user_data=&rx;
  llama_context* ctx=llama_init_from_model(model,cp);
  if(!ctx) {
    llama_model_free(model);llama_backend_free();
    throw std::runtime_error("llama context creation failed");
  }
  const auto libs_before=runtime_libraries();
  verify_runtime_lock(libs_before,lock);
  llama_batch batch=llama_batch_init(2,0,1);
  if(!batch.token || !batch.pos || !batch.logits || !batch.seq_id ||
     !batch.n_seq_id) {
    llama_batch_free(batch);
    llama_free(ctx);llama_model_free(model);llama_backend_free();
    throw std::runtime_error("cannot allocate exact two-token llama batch");
  }
  batch.n_tokens=2;
  for(int i=0;i<2;++i) {
    batch.token[i]=static_cast<llama_token>(i==0?o.token0:o.token1);
    batch.pos[i]=i;
    batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;
    batch.logits[i]=1;
  }
  const int decode_result=llama_decode(ctx,batch);
  llama_batch_free(batch);
  std::array<int,2> best_ids{{-1,-1}};
  if(decode_result==0 && rx.error().empty()) {
    for(int pos=0;pos<2;++pos) {
      const float* logits=llama_get_logits_ith(ctx,pos);
      if(!logits)continue;
      float best=-std::numeric_limits<float>::infinity();
      for(int id=0;id<VOCAB;++id) {
        if(!std::isfinite(logits[id])){best_ids[pos]=-1;break;}
        if(logits[id]>best){best=logits[id];best_ids[pos]=id;}
      }
    }
  }
  const auto missing=rx.missing();
  const bool passed=decode_result==0 && rx.error().empty() &&
                    missing.empty() && best_ids[0]>=0 && best_ids[1]>=0;
  // Record the actually loaded shared-library identities while llama
  // remains in process (the backends may be dlopened after initialization).
  const auto runtime_after=runtime_libraries();
  verify_runtime_lock(runtime_after,lock);
  llama_free(ctx);llama_model_free(model);llama_backend_free();
  if(!passed) {
    std::ostringstream error;
    error<<"llama layer trace blocked: decode="<<decode_result
         <<" callback="<<rx.error()<<" captured="<<rx.captures().size()
         <<" missing=";
    for(const auto& name:missing)error<<name<<",";
    throw std::runtime_error(error.str());
  }
  manifest(o,rx,model_sha,binary_sha,lib_sha,runtime_after,best_ids);
  std::cout<<"CAPTURED_NOT_COMPARED tensors="<<rx.captures().size()
           <<" prompt_ids=["<<o.token0<<","<<o.token1
           <<"] absolute_positions=[0,1] layer=0"
           <<" manifest="<<(o.out_dir/"manifest.json").string()<<"\n";
  return 0;
}
} // namespace

int main(int argc,char** argv) {
  Options o;
  if(!options(argc,argv,o)) {
    std::fputs(
      "USAGE: downstream_callback_trace --inspect-plan | --selftest | --inspect-runtime |\n"
      "  --capture --gpu-exclusive --model FILE.gguf --llama-lib libllama.so "
      "--runtime-lock LOCK.txt --expected-model-sha 64hex "
      "--expected-libllama-sha 64hex --out-dir EMPTY_DIR "
      "[--token0 6205] [--token1 161421] [--n-gpu-layers 80] "
      "[--threads 4] [--max-total-mib 16] [--max-tensor-mib 2] "
      "[--include attn_norm,conv_output_silu,attn_output,attn_residual,...]\n"
      "Real GGUF inference is never run without --capture and --gpu-exclusive.\n",
      stderr);
    return 2;
  }
  try {
    if(o.inspect) {
      std::cout<<"Installed callback uses ask=true to select and ask=false "
        "after backend synchronization; ggml_backend_tensor_get reads bounded "
        "row slices into host memory.\n";
      for(auto stage:STAGES)std::cout<<tensor_name(stage,0)<<"\n";
      return 0;
    }
    if(o.inspect_runtime) {
      std::cout<<"LOADED_RUNTIME_SHA_LOCK_TEMPLATE (no model initialized):\n";
      for(const auto& [name,lib]:runtime_libraries())
        std::cout<<name<<" "<<lib.sha256<<"\n";
      return 0;
    }
    if(o.selftest)return selftest();
    return capture_real(o);
  } catch(const std::exception& e) {
    std::fprintf(stderr,"RELIABILITY_LAYER_BLOCKED: %s\n",e.what());
    return 2;
  }
}
