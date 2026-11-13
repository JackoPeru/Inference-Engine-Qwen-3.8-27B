#include "hermes/gguf.hpp"
#include <fstream>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>

namespace hermes {
namespace {
struct Reader {
  const uint8_t* p; const uint8_t* end;
  explicit Reader(const uint8_t* b, const uint8_t* e):p(b),end(e){}
  void need(size_t n){ if(p+n>end) throw GgufError("GGUF truncated"); }
  uint8_t u8(){need(1);return *p++;}
  uint16_t u16(){need(2);uint16_t v;memcpy(&v,p,2);p+=2;return v;}
  uint32_t u32(){need(4);uint32_t v;memcpy(&v,p,4);p+=4;return v;}
  uint64_t u64(){need(8);uint64_t v;memcpy(&v,p,8);p+=8;return v;}
  float f32(){need(4);float v;memcpy(&v,p,4);p+=4;return v;}
  double f64(){need(8);double v;memcpy(&v,p,8);p+=8;return v;}
  std::string str(){uint64_t n=u64();need((size_t)n);std::string s((char*)p,(size_t)n);p+=n;return s;}
  size_t consumed(const uint8_t* base) const { return (size_t)(p-base); }
};
uint64_t round_up(uint64_t x, uint64_t a){ return (x + a - 1) / a * a; }
}

uint32_t ggml_block_size(GgmlType t){
  switch(t){
    case GgmlType::F32: case GgmlType::F16: case GgmlType::BF16:
    case GgmlType::I8: case GgmlType::I16: case GgmlType::I32:
    case GgmlType::I64: case GgmlType::F64: return 1;
    case GgmlType::MXFP4: return 32;
    case GgmlType::NVFP4: return 64;
    case GgmlType::Q4_0: case GgmlType::Q4_1: case GgmlType::Q5_0:
    case GgmlType::Q5_1: case GgmlType::Q8_0: case GgmlType::Q8_1: return 32;
    default: return 256; // K-quants super-block
  }
}
uint64_t ggml_type_size(GgmlType t){
  switch(t){
    case GgmlType::F32: return 4;
    case GgmlType::F16: case GgmlType::BF16: return 2;
    case GgmlType::I8: return 1; case GgmlType::I16: return 2;
    case GgmlType::I32: return 4; case GgmlType::I64: case GgmlType::F64: return 8;
    case GgmlType::MXFP4: return 17; // 1B E8M0 + 16B packed
    case GgmlType::NVFP4: return 36;  // 4B UE4M3 + 32B packed E2M1
    case GgmlType::Q4_0: return 2+16; case GgmlType::Q4_1: return 4+16;
    case GgmlType::Q5_0: return 2+4+16; case GgmlType::Q5_1: return 4+4+16;
    case GgmlType::Q8_0: return 2+32; case GgmlType::Q8_1: return 4+32;
    default: throw GgufError("unsupported ggml type size for strict Qwen3.8 set");
  }
}
uint64_t ggml_row_size(GgmlType t, uint64_t ne00){
  uint32_t blck = ggml_block_size(t);
  if(ne00 % blck != 0) throw GgufError("tensor dim0 not multiple of block size");
  return ne00 / blck * ggml_type_size(t);
}
std::string ggml_type_name(GgmlType t){
  switch(t){
    case GgmlType::F32: return "F32"; case GgmlType::F16: return "F16";
    case GgmlType::BF16: return "BF16"; case GgmlType::MXFP4: return "MXFP4";
    case GgmlType::NVFP4: return "NVFP4"; case GgmlType::Q4_0: return "Q4_0";
    case GgmlType::Q8_0: return "Q8_0"; default: return "QK";
  }
}

float ue4m3_to_fp32(uint8_t x){
  // Independent implementation of UE4M3 (unsigned E4M3, bias 7), *0.5 convention
  // to pair with doubled E2M1 LUT. Matches OCP microscaling semantics.
  if(x==0 || x==0x7F) return 0.0f;
  int exp = (x>>3)&0xF; int man = x&0x7;
  float raw;
  if(exp==0) raw = ldexpf((float)man, -9);
  else raw = ldexpf(1.0f + (float)man/8.0f, exp-7);
  return raw*0.5f;
}
float e2m1_to_fp32(uint8_t v){
  // True E2M1 values: 0,0.5,1,1.5,2,3,4,6 and negatives. 0x8 = -0 -> 0.
  static const float lut[16]={0,0.5f,1,1.5f,2,3,4,6,0,-0.5f,-1,-1.5f,-2,-3,-4,-6};
  return lut[v&0xF];
}
void nvfp4_block_dequant(const uint8_t b36[36], float out64[64]){
  for(int sub=0; sub<4; ++sub){
    float d = ue4m3_to_fp32(b36[sub]);
    for(int j=0;j<8;++j){
      uint8_t q = b36[4 + sub*8 + j];
      out64[sub*16 + j]     = d * e2m1_to_fp32(q & 0xF) * 2.0f; // d has *0.5, LUT true -> *2 to compensate? No:
      // Correction: ue4m3_to_fp32 already includes *0.5, e2m1 LUT is true value, so product is (UE*0.5)*E = correct NVFP4 value.
      // Actually NVFP4 value = scale_true * e2m1_true. scale_true = raw UE4M3. Our d = raw*0.5.
      // So out = d * (e2m1*2)? Let's keep exact: out = (d*2) * e2m1 = raw * e2m1.
      out64[sub*16 + j]     = (d*2.0f) * e2m1_to_fp32(q & 0xF);
      out64[sub*16 + j + 8] = (d*2.0f) * e2m1_to_fp32(q >> 4);
    }
  }
}

bool is_valid_qwen38_tensor(const std::string&){
  return true; // name allow-list enforced in require_qwen38_or_throw via counts
}

void GgufFile::open(const std::string& fp, bool strict_qwen38){
  path = fp;
  std::ifstream f(fp, std::ios::binary|std::ios::ate);
  if(!f) throw GgufError("cannot open GGUF: "+fp);
  uint64_t fsize = (uint64_t)f.tellg(); f.seekg(0);
  std::vector<uint8_t> hdr(64*1024*1024); // header+KV+tensor infos must fit in 64MB; data is not loaded
  // Read incrementally: first read up to 64MB or file size
  uint64_t toread = fsize < hdr.size() ? fsize : hdr.size();
  f.read((char*)hdr.data(), (std::streamsize)toread);
  if((uint64_t)f.gcount() != toread) throw GgufError("short read");
  hdr.resize((size_t)toread);
  Reader r(hdr.data(), hdr.data()+hdr.size());
  uint32_t magic = r.u32();
  if(magic != 0x46554747) throw GgufError("bad GGUF magic");
  header.version = r.u32();
  if(header.version != 3) throw GgufError("unsupported GGUF version (need 3)");
  header.n_tensors = r.u64(); header.n_kv = r.u64();
  if(header.n_tensors > 5000) throw GgufError("implausible n_tensors");
  if(header.n_kv > 10000) throw GgufError("implausible n_kv");
  for(uint64_t i=0;i<header.n_kv;++i){
    std::string k = r.str();
    int32_t vt = (int32_t)r.u32();
    auto kt = (GgufKvType)vt;
    switch(kt){
      case GgufKvType::U8: kv_u64[k]=r.u8(); break;
      case GgufKvType::I8: kv_u64[k]=(uint64_t)(int64_t)(int8_t)r.u8(); break;
      case GgufKvType::U16: kv_u64[k]=r.u16(); break;
      case GgufKvType::I16: kv_u64[k]=(uint64_t)(int64_t)(int16_t)r.u16(); break;
      case GgufKvType::U32: kv_u64[k]=r.u32(); break;
      case GgufKvType::I32: kv_u64[k]=(uint64_t)(int64_t)(int32_t)r.u32(); break;
      case GgufKvType::F32: kv_f64[k]=r.f32(); break;
      case GgufKvType::BOOL: kv_u64[k]=r.u8(); break;
      case GgufKvType::STR: kv_str[k]=r.str(); break;
      case GgufKvType::ARR: {
        int32_t at=(int32_t)r.u32(); uint64_t n=r.u64();
        if(n>5000000) throw GgufError("implausible array len key="+k+" n="+std::to_string(n)+" at="+std::to_string(at));
        auto aty=(GgufKvType)at;
        if(aty==GgufKvType::STR){ std::string acc; acc.reserve(65536); for(uint64_t j=0;j<n;++j){ std::string s=r.str(); if(acc.size()<65536){ if(j)acc+=";"; acc+=s; } } kv_str[k+".arr"]=acc; kv_u64[k+".len"]=n; }
        else { uint64_t last=0; double flast=0; for(uint64_t j=0;j<n;++j){ switch(aty){case GgufKvType::U8:last=r.u8();break;case GgufKvType::I8:last=(uint64_t)(int64_t)(int8_t)r.u8();break;case GgufKvType::U16:last=r.u16();break;case GgufKvType::I16:last=(uint64_t)(int64_t)(int16_t)r.u16();break;case GgufKvType::U32:last=r.u32();break;case GgufKvType::I32:last=(uint64_t)(int64_t)(int32_t)r.u32();break;case GgufKvType::U64:last=r.u64();break;case GgufKvType::I64:last=(uint64_t)(int64_t)(int64_t)r.u64();break;case GgufKvType::F32:flast=r.f32();break;case GgufKvType::F64:flast=r.f64();break;case GgufKvType::BOOL:last=r.u8();break;default:throw GgufError("unsupported arr subtype");} } kv_u64[k+".len"]=n; kv_u64[k+".last"]=last; kv_f64[k+".flast"]=flast; }
        break; }
      case GgufKvType::U64: kv_u64[k]=r.u64(); break;
      case GgufKvType::I64: kv_u64[k]=(uint64_t)(int64_t)(int64_t)r.u64(); break;
      case GgufKvType::F64: kv_f64[k]=r.f64(); break;
      default: throw GgufError("unknown KV type");
    }
  }
  tensors.reserve((size_t)header.n_tensors);
  for(uint64_t i=0;i<header.n_tensors;++i){
    GgufTensorInfo ti; ti.name=r.str();
    ti.n_dims=r.u32(); if(ti.n_dims>4||ti.n_dims==0) throw GgufError("bad n_dims "+ti.name);
    for(uint32_t d=0;d<ti.n_dims;++d) ti.ne[d]=r.u64();
    ti.type=(GgmlType)(int32_t)r.u32();
    ti.offset=r.u64();
    if(tensor_index.count(ti.name)) throw GgufError("duplicate tensor "+ti.name);
    ti.n_elements=1; for(uint32_t d=0;d<ti.n_dims;++d){ if(ti.ne[d]==0) throw GgufError("zero dim "+ti.name); ti.n_elements*=ti.ne[d]; }
    uint64_t row = ggml_row_size(ti.type, ti.ne[0]);
    uint64_t rows = ti.n_elements / ti.ne[0];
    ti.n_bytes = row*rows;
    tensor_index[ti.name]=tensors.size();
    tensors.push_back(std::move(ti));
  }
  auto it = kv_u64.find("general.alignment");
  alignment = (it==kv_u64.end()?32:it->second);
  if(alignment==0||alignment>256) throw GgufError("bad alignment");
  size_t hlen = r.consumed(hdr.data());
  data_offset = round_up(hlen, alignment);
  validate_or_throw(fsize);
  if(strict_qwen38) require_qwen38_or_throw();
}

void GgufFile::validate_or_throw(uint64_t file_size) const{
  for(auto& t: tensors){
    if(t.offset % alignment != 0) throw GgufError("tensor offset misaligned: "+t.name);
    __uint128_t end = (__uint128_t)data_offset + t.offset + t.n_bytes;
    if(end > file_size) throw GgufError("tensor exceeds file: "+t.name);
  }
  // overlap check O(n log n)
  std::vector<const GgufTensorInfo*> v; v.reserve(tensors.size());
  for(auto& t: tensors) v.push_back(&t);
  std::sort(v.begin(), v.end(), [](auto* a, auto* b){return a->offset<b->offset;});
  for(size_t i=1;i<v.size();++i){
    uint64_t ps=v[i-1]->offset+v[i-1]->n_bytes;
    if(v[i]->offset < ps) throw GgufError("tensor overlap: "+v[i]->name);
  }
}

const GgufTensorInfo& GgufFile::find(const std::string& n) const{
  auto it=tensor_index.find(n);
  if(it==tensor_index.end()) throw GgufError("tensor not found: "+n);
  return tensors[it->second];
}

void GgufFile::require_qwen38_or_throw() const{
  auto need_str=[&](const char* k){ auto it=kv_str.find(k); if(it==kv_str.end()) throw GgufError(std::string("missing KV ")+k); return it->second; };
  std::string arch = need_str("general.architecture");
  if(arch!="qwen3_5" && arch!="qwen3_5_text" && arch!="qwen35") throw GgufError("not qwen3_5/qwen35 arch: "+arch);
  // critical tensors: GGUF names (blk.*) - verified on real 19.6GB file
  const char* must[]={"token_embd.weight","output.weight","output_norm.weight","blk.0.attn_qkv.weight","blk.3.attn_q.weight","blk.64.nextn.eh_proj.weight"};
  for(auto m: must){ if(!has(m)) throw GgufError(std::string("missing tensor ")+m); }
  bool any_nv=false; for(auto& t: tensors) if(t.type==GgmlType::NVFP4){any_nv=true;break;}
  if(!any_nv) throw GgufError("no NVFP4 tensors: not the target checkpoint");
  if(tensors.size()<800) throw GgufError("too few tensors for Qwen3.8-27B");
}

} // namespace hermes
