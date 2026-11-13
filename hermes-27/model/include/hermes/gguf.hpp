#pragma once
// HERMES-27 GGUF strict parser (independent implementation).
// Format reference: GGUF spec (magic "GGUF", v3), NOT llama.cpp runtime.
// Supports: metadata KV + tensor infos, alignment, offset/size validation,
// GGML types incl. NVFP4=40, MXFP4=39, BF16/F16/F32.
// No dequant-to-BF16 of full weights: parser only maps, decode is explicit.
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <stdexcept>

namespace hermes {

struct GgufError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

enum class GgmlType : int32_t {
  F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
  Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
  IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20,
  IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23, I8 = 24, I16 = 25, I32 = 26, I64 = 27,
  F64 = 28, IQ1_M = 29, BF16 = 30, MXFP4 = 39, NVFP4 = 40,
};

enum class GgufKvType : int32_t {
  U8=0,I8=1,U16=2,I16=3,U32=4,I32=5,F32=6,BOOL=7,STR=8,ARR=9,U64=10,I64=11,F64=12
};

struct GgufTensorInfo {
  std::string name;
  uint32_t n_dims = 0;
  uint64_t ne[4] = {1,1,1,1};
  GgmlType type = GgmlType::F32;
  uint64_t offset = 0; // relative to data section start
  uint64_t n_elements = 0;
  uint64_t n_bytes = 0;
};

struct GgufHeader {
  uint32_t version = 0;
  uint64_t n_tensors = 0;
  uint64_t n_kv = 0;
};

// Block sizes / type sizes (independent constants, verified vs GGML spec)
uint32_t ggml_block_size(GgmlType t);
uint64_t ggml_type_size(GgmlType t); // bytes per block
uint64_t ggml_row_size(GgmlType t, uint64_t ne00); // bytes for one row of ne00 elems
std::string ggml_type_name(GgmlType t);

// NVFP4 micro-scaling (independent implementation from OCP MX spec):
// block 64 elems: 4x UE4M3 scale (1 byte each, 16-elem sub-block) + 32B packed E2M1.
// UE4M3: unsigned, exp bias 7, 4E3M. E2M1 true values via LUT.
float ue4m3_to_fp32(uint8_t x);
float e2m1_to_fp32(uint8_t nibble4); // 0..15
void nvfp4_block_dequant(const uint8_t block36[36], float out64[64]);
bool is_valid_qwen38_tensor(const std::string& name);

class GgufFile {
 public:
  GgufHeader header{};
  std::unordered_map<std::string,std::string> kv_str;
  std::unordered_map<std::string,uint64_t> kv_u64;
  std::unordered_map<std::string,double> kv_f64;
  std::vector<GgufTensorInfo> tensors;
  std::unordered_map<std::string,size_t> tensor_index;
  uint64_t data_offset = 0; // file offset where tensor data starts
  uint64_t alignment = 32;
  std::string path;

  void open(const std::string& path, bool strict_qwen38 = false);
  const GgufTensorInfo& find(const std::string& name) const;
  bool has(const std::string& name) const { return tensor_index.count(name)!=0; }
  // validate: offsets, sizes, overflow, duplicates, alignment
  void validate_or_throw(uint64_t file_size) const;
  // minimal required keys for Qwen3.8-27B NVFP4-MTP
  void require_qwen38_or_throw() const;
};

} // namespace hermes
