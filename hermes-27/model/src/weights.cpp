#include "hermes/weights.hpp"
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif
namespace hermes {
void GgufWeightBackend::open(const std::string& path){
  gf_.open(path, true);
#ifdef _WIN32
  mh_=nullptr; base_=nullptr; len_=0;
  // Windows: fallback to plain mapping via file size check (no persistent mmap yet)
  // Parser already validated; data access via re-read on demand in later milestone.
#else
  fd_ = ::open(path.c_str(), O_RDONLY);
  if(fd_<0) throw GgufError("mmap open failed");
  struct stat st{}; if(::fstat(fd_,&st)!=0) throw GgufError("fstat failed");
  len_=(uint64_t)st.st_size;
  void* m=::mmap(nullptr,len_,PROT_READ,MAP_PRIVATE,fd_,0);
  if(m==MAP_FAILED) throw GgufError("mmap failed");
  base_=(const uint8_t*)m;
#endif
}
ITensorView GgufWeightBackend::view(const std::string& name){
  const auto& ti = gf_.find(name);
#ifdef _WIN32
  return {&ti,nullptr,ti.n_bytes};
#else
  return {&ti, base_+gf_.data_offset+ti.offset, ti.n_bytes};
#endif
}
}
