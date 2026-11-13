// gguf_dump: read-only inspector for real Qwen3.8 NVFP4-MTP GGUF.
#include "hermes/gguf.hpp"
#include <cstdio>
int main(int argc, char** argv){
  if(argc<2){printf("usage: gguf_dump <file.gguf> [--strict|--chat-template]\n");return 2;}
  bool strict = argc>2 && std::string(argv[2])=="--strict";
  hermes::GgufFile g;
  try{ g.open(argv[1], strict); }
  catch(std::exception& e){ printf("OPEN FAIL: %s\n", e.what()); return 1; }
  if(argc>2 && std::string(argv[2])=="--chat-template"){
    auto it=g.kv_str.find("tokenizer.chat_template");
    if(it==g.kv_str.end() || it->second.empty()){
      fprintf(stderr,"missing tokenizer.chat_template\n");
      return 1;
    }
    return fwrite(it->second.data(),1,it->second.size(),stdout)==it->second.size()?0:1;
  }
  printf("version=%u n_tensors=%llu n_kv=%llu align=%llu data_off=%llu\n",
    g.header.version,(unsigned long long)g.header.n_tensors,(unsigned long long)g.header.n_kv,
    (unsigned long long)g.alignment,(unsigned long long)g.data_offset);
  for(auto& kv: g.kv_str){ if(kv.first.size()<64) printf("KV-S %s = %.120s\n", kv.first.c_str(), kv.second.c_str()); }
  for(auto& kv: g.kv_u64) printf("KV-I %s = %llu\n", kv.first.c_str(), (unsigned long long)kv.second);
  uint64_t nv=0,bf=0,f32=0,other=0, bytes_nv=0;
  for(auto& t: g.tensors){
    if(t.type==hermes::GgmlType::NVFP4){nv++;bytes_nv+=t.n_bytes;}
    else if(t.type==hermes::GgmlType::BF16)bf++;
    else if(t.type==hermes::GgmlType::F32)f32++;
    else other++;
  }
  printf("tensors: NVFP4=%llu BF16=%llu F32=%llu other=%llu nvfp4_bytes=%llu (%.2f GiB)\n",
    (unsigned long long)nv,(unsigned long long)bf,(unsigned long long)f32,(unsigned long long)other,
    (unsigned long long)bytes_nv, bytes_nv/1024.0/1024/1024);
  // print first 20 + MTP-related
  int shown=0;
  for(auto& t: g.tensors){ if(shown<20){printf("T %-70s ne=[%llu,%llu] type=%s bytes=%llu off=%llu\n",t.name.c_str(),
    (unsigned long long)t.ne[0],(unsigned long long)t.ne[1],hermes::ggml_type_name(t.type).c_str(),
    (unsigned long long)t.n_bytes,(unsigned long long)t.offset); shown++;} }
  printf("--- MTP/search ---\n");
  for(auto& t: g.tensors){ std::string n=t.name; if(n.find("mtp")!=std::string::npos||n.find("MTP")!=std::string::npos||n.find("next")!=std::string::npos||n.find("drafter")!=std::string::npos)
    printf("MTP? %-80s type=%s ne=[%llu,%llu]\n",n.c_str(),hermes::ggml_type_name(t.type).c_str(),(unsigned long long)t.ne[0],(unsigned long long)t.ne[1]); }
  return 0;
}
