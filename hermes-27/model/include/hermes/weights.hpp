#pragma once
// HERMES-27 weight backend interface: isolates GGUF NVFP4 loader from
// future safetensors mixed-precision loader (johnnyeric). No stable BF16 upcast.
#include "hermes/gguf.hpp"
#include <memory>
namespace hermes {
struct ITensorView { const GgufTensorInfo* info=nullptr; const uint8_t* data=nullptr; uint64_t size=0; };
class IWeightBackend {
 public:
  virtual ~IWeightBackend()=default;
  virtual void open(const std::string& path)=0;
  virtual ITensorView view(const std::string& name)=0;
  virtual std::string describe()=0;
};
// Mmap-backed GGUF: OS paging, no full copy, distributed upload later per-GPU.
class GgufWeightBackend : public IWeightBackend {
 public:
  void open(const std::string& path) override;
  ITensorView view(const std::string& name) override;
  std::string describe() override { return "gguf-nvfp4-mmap strict Qwen3.8-27B"; }
  const GgufFile& file() const { return gf_; }
 private:
  GgufFile gf_;
#ifdef _WIN32
  void* mh_=nullptr; const uint8_t* base_=nullptr; uint64_t len_=0;
#else
  int fd_=-1; const uint8_t* base_=nullptr; uint64_t len_=0;
#endif
};
}
