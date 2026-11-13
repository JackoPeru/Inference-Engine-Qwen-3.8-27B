// Independent llama.cpp C API Qwen35 layer tracer. NO HERMES linkage.
// --inspect-plan and --selftest do not load the real GGUF or invoke GPUs.
// --capture requires explicit --gpu-exclusive, even if n_gpu_layers=0.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "reliability_layer_sha256.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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
constexpr std::array<std::string_view,23> STAGES{{
  "attn_norm", "linear_attn_qkv_mixed", "conv_output_raw",
  "conv_output_silu", "beta_sigmoid", "gate",
  "q_conv_predelta", "k_conv_predelta", "v_conv_predelta",
  "attn_output", "linear_attn_out", "attn_residual",
  "attn_post_norm", "ffn_out", "post_ffn",
  "Qcur_full", "Qcur_normed", "Kcur_normed", "Qcur",
  "Kcur", "attn_pregate", "attn_gated",
  "result_norm"
}};
struct Options {
  bool selftest=false, inspect=false, capture=false, exclusive=false;
  fs::path model, out_dir, llama_lib;
  int token=6205, layer=0, gpu_layers=80, threads=4;
  uint64_t max_total=32*MiB, max_single=2*MiB;
  std::set<std::string> stages;
};
struct TensorCapture {
  std::string name, file, type, sha, buffer_name;
  std::array<int64_t,4> shape{};
  std::array<size_t,4> strides{};
  size_t elements=0, bytes=0;
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
  return stage=="result_norm" ? "result_norm" : std::string(stage)+"-"+std::to_string(layer);
}
bool options(int argc,char** argv,Options& o) {
  for(int i=1;i<argc;++i) {
    const std::string_view key=argv[i];
    if(key=="--selftest"){o.selftest=true;continue;}
    if(key=="--inspect-plan"){o.inspect=true;continue;}
    if(key=="--capture"){o.capture=true;continue;}
    if(key=="--gpu-exclusive"){o.exclusive=true;continue;}
    if(i+1>=argc)return false;
    const std::string v=argv[++i];
    if(key=="--model")o.model=v;
    else if(key=="--out-dir")o.out_dir=v;
    else if(key=="--llama-lib")o.llama_lib=v;
    else if(key=="--token-id"){if(!integer(v,o.token))return false;}
    else if(key=="--layer"){if(!integer(v,o.layer))return false;}
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
  if(int(o.selftest)+int(o.inspect)+int(o.capture)!=1)return false;
  if(!o.capture)return true;
  if(!o.exclusive || o.token<0 || o.token>=VOCAB ||
     o.layer<0 || o.layer>63 || o.threads<1 || o.threads>64 ||
     o.gpu_layers<0 || o.gpu_layers>128 || o.max_single>o.max_total ||
     !fs::is_regular_file(o.model) || !fs::is_regular_file(o.llama_lib) ||
     !fs::is_directory(o.out_dir))return false;
  if(o.stages.empty()) {
    for(auto stage:STAGES) {
      if(stage=="result_norm")continue;
      const bool full=(o.layer%4)==3;
      if(full && (stage=="linear_attn_qkv_mixed" ||
                  stage=="conv_output_raw" || stage=="conv_output_silu" ||
                  stage=="beta_sigmoid" || stage=="gate" ||
                  stage=="q_conv_predelta" || stage=="k_conv_predelta" ||
                  stage=="v_conv_predelta" || stage=="linear_attn_out"))
        continue;
      if(!full && (stage=="Qcur_full" || stage=="Qcur_normed" ||
                   stage=="Kcur_normed" || stage=="Qcur" ||
                   stage=="Kcur" || stage=="attn_pregate" ||
                   stage=="attn_gated"))continue;
      o.stages.insert(std::string(stage));
    }
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
class Receiver {
 public:
  explicit Receiver(const Options& o):o_(o) {
    for(const auto& name:o_.stages)names_.insert(tensor_name(name,o_.layer));
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
  captures_.push_back(record);
  total_+=bytes;
}

void manifest(const Options& o,const Receiver& rx,
              const std::string& model_sha,const std::string& binary_sha,
              const std::string& library_sha,int argmax) {
  const fs::path target=o.out_dir/"manifest.json";
  const fs::path part=target.string()+".partial";
  if(fs::exists(target)||fs::exists(part))
    throw std::runtime_error("refusing stale manifest");
  std::ofstream f(part);
  if(!f)throw std::runtime_error("cannot open manifest");
  f<<"{\n  \"schema\":\"h27.reliability-layer-cb.v1\","
    <<"\n  \"status\":\"CAPTURED_NOT_INDEPENDENTLY_COMPARED\","
    <<"\n  \"backend\":\"llama.cpp\","
    <<"\n  \"model_path\":"<<json_string(fs::canonical(o.model).string())<<","
    <<"\n  \"model_sha256\":"<<json_string(model_sha)<<","
    <<"\n  \"exporter_sha256\":"<<json_string(binary_sha)<<","
    <<"\n  \"libllama_path\":"<<json_string(fs::canonical(o.llama_lib).string())<<","
    <<"\n  \"libllama_sha256\":"<<json_string(library_sha)<<","
    <<"\n  \"llama_build\":"<<json_string(llama_print_system_info())<<","
    <<"\n  \"prompt_ids\":["<<o.token<<"],"
    <<"\n  \"absolute_position\":0,"
    <<"\n  \"layer\":"<<o.layer<<","
    <<"\n  \"n_gpu_layers\":"<<o.gpu_layers<<","
    <<"\n  \"greedy_argmax\":"<<argmax<<","
    <<"\n  \"callback_ask_count\":"<<rx.ask_count()<<","
    <<"\n  \"callback_ready_count\":"<<rx.ready_count()<<","
    <<"\n  \"tensor_data_format\":\"little-endian-fp32-ne0-fastest\","
    <<"\n  \"tensors\":[\n";
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
    f<<"],\"elements\":"<<c.elements<<",\"bytes\":"<<c.bytes
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
  ggml_tensor* view=ggml_view_2d(ctx,base,2,4,4*sizeof(float),0);
  view->data=input.data();
  ggml_set_name(view,"attn_norm-0");
  ggml_tensor* other=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,2);
  ggml_set_name(other,"attn_norm-1");
  if(Receiver::callback(other,true,&rx) || Receiver::callback(other,false,&rx) ||
     !Receiver::callback(view,true,&rx) || !Receiver::callback(view,false,&rx))
    throw std::runtime_error("callback ask/ready selection or capture failed: "+rx.error());
  if(rx.captures().size()!=1 || rx.captures()[0].shape[0]!=2 ||
     rx.captures()[0].shape[1]!=4 || rx.captures()[0].bytes!=32)
    throw std::runtime_error("synthetic shape or FP32 bytes wrong");
  std::array<float,8> output{};
  {
    std::ifstream file(dir/"attn_norm-0.f32",std::ios::binary);
    file.read(reinterpret_cast<char*>(output.data()),sizeof(output));
    if(!file)throw std::runtime_error("synthetic FP32 read incomplete");
  }
  for(int r=0;r<4;++r)for(int col=0;col<2;++col)
    if(output[size_t(r*2+col)]!=input[size_t(r*4+col)])
      throw std::runtime_error("strided view gather has wrong layout");
  if(Sha256::file(dir/"attn_norm-0.f32")!=rx.captures()[0].sha)
    throw std::runtime_error("synthetic digest mismatch");
  Options bounded=o;
  bounded.max_single=16; // view is 32 FP32 bytes; reject before backend copy
  Receiver budget_check(bounded);
  if(!Receiver::callback(view,true,&budget_check) ||
     Receiver::callback(view,false,&budget_check) ||
     budget_check.error().find("budget")==std::string::npos)
    throw std::runtime_error("oversize tensor must fail before host copy");
  if(!Receiver::callback(view,true,&rx) || Receiver::callback(view,false,&rx) ||
     rx.error().find("duplicate")==std::string::npos)
    throw std::runtime_error("duplicate capture must fail closed");
  ggml_backend_buffer_free(base->buffer);
  ggml_free(ctx);
  fs::remove(dir/"attn_norm-0.f32");
  fs::remove(dir);
  std::puts("PASS CPU callback selection, synchronized tensor-get contract, "
            "noncontiguous GGML view, FP32 output, SHA256, duplicate fail-closed");
  return 0;
}

int capture_real(const Options& o) {
  // Validate identity of *loaded* libllama, not merely a supplied filename.
  Dl_info info{};
  if(!dladdr(reinterpret_cast<const void*>(&llama_backend_init),&info) ||
     !info.dli_fname || fs::canonical(info.dli_fname)!=fs::canonical(o.llama_lib))
    throw std::runtime_error("loaded libllama differs from expected library");
  const std::string model_sha=Sha256::file(o.model);
  const std::string lib_sha=Sha256::file(o.llama_lib);
  const std::string binary_sha=Sha256::file("/proc/self/exe");
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
  cp.n_ctx=128;cp.n_batch=128;cp.n_ubatch=128;cp.n_seq_max=1;
  cp.n_threads=o.threads;cp.n_threads_batch=o.threads;
  cp.cb_eval=&Receiver::callback;cp.cb_eval_user_data=&rx;
  llama_context* ctx=llama_init_from_model(model,cp);
  if(!ctx) {
    llama_model_free(model);llama_backend_free();
    throw std::runtime_error("llama context creation failed");
  }
  llama_token token=static_cast<llama_token>(o.token);
  const llama_batch batch=llama_batch_get_one(&token,1);
  const int decode_result=llama_decode(ctx,batch);
  int best_id=-1;
  if(decode_result==0 && rx.error().empty()) {
    const float* logits=llama_get_logits(ctx);
    if(logits) {
      float best=-std::numeric_limits<float>::infinity();
      for(int id=0;id<VOCAB;++id) {
        if(!std::isfinite(logits[id])){best_id=-1;break;}
        if(logits[id]>best){best=logits[id];best_id=id;}
      }
    }
  }
  const auto missing=rx.missing();
  const bool passed=decode_result==0 && rx.error().empty() &&
                    missing.empty() && best_id>=0;
  llama_free(ctx);llama_model_free(model);llama_backend_free();
  if(!passed) {
    std::ostringstream error;
    error<<"llama layer trace blocked: decode="<<decode_result
         <<" callback="<<rx.error()<<" captured="<<rx.captures().size()
         <<" missing=";
    for(const auto& name:missing)error<<name<<",";
    throw std::runtime_error(error.str());
  }
  manifest(o,rx,model_sha,binary_sha,lib_sha,best_id);
  std::cout<<"CAPTURED_NOT_COMPARED tensors="<<rx.captures().size()
           <<" token="<<o.token<<" absolute_pos=0 layer="<<o.layer
           <<" manifest="<<(o.out_dir/"manifest.json").string()<<"\n";
  return 0;
}
} // namespace

int main(int argc,char** argv) {
  Options o;
  if(!options(argc,argv,o)) {
    std::fputs(
      "USAGE: reliability_layer_llama_cb --inspect-plan | --selftest |\n"
      "  --capture --gpu-exclusive --model FILE.gguf --llama-lib libllama.so "
      "--out-dir EMPTY_DIR [--token-id 6205] [--layer 0] "
      "[--n-gpu-layers 80] [--threads 4] [--max-total-mib 32] "
      "[--max-tensor-mib 2] [--include attn_norm,linear_attn_qkv_mixed,...]\n"
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
    if(o.selftest)return selftest();
    return capture_real(o);
  } catch(const std::exception& e) {
    std::fprintf(stderr,"RELIABILITY_LAYER_BLOCKED: %s\n",e.what());
    return 2;
  }
}
