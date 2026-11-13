// Standalone, small SHA-256 for the independent layer-capture harness.
// No GGUF parsing, CUDA calls, or external crypto package required.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace reliability_layer {

class Sha256 {
 public:
  void update(const void* bytes, size_t size) {
    const auto* p = static_cast<const uint8_t*>(bytes);
    if (size > UINT64_MAX / 8 - count_ / 8) {
      throw std::runtime_error("SHA256 byte count overflow");
    }
    count_ += static_cast<uint64_t>(size) * 8;
    while (size) {
      const size_t n = std::min(size, block_.size() - used_);
      std::memcpy(block_.data() + used_, p, n);
      used_ += n;
      p += n;
      size -= n;
      if (used_ == block_.size()) {
        compress();
        used_ = 0;
      }
    }
  }

  std::string finish() {
    const uint64_t bits = count_;
    uint8_t one = 0x80;
    update(&one, 1);
    uint8_t zero = 0;
    while (used_ != 56) update(&zero, 1);
    std::array<uint8_t, 8> length{};
    for (int i = 0; i < 8; ++i) length[7-i] = static_cast<uint8_t>(bits >> (8*i));
    update(length.data(), length.size());
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (uint32_t word : state_) out << std::setw(8) << word;
    return out.str();
  }

  static std::string digest(const void* p, size_t n) {
    Sha256 h;
    h.update(p, n);
    return h.finish();
  }

  static std::string file(const std::filesystem::path& p) {
    std::FILE* f = std::fopen(p.string().c_str(), "rb");
    if (!f) throw std::runtime_error("cannot read SHA256 file " + p.string());
    Sha256 h;
    std::array<uint8_t, 1 << 18> buf{};
    bool ok = true;
    for (;;) {
      const size_t n = std::fread(buf.data(), 1, buf.size(), f);
      if (n) h.update(buf.data(), n);
      if (n < buf.size()) { ok = std::ferror(f) == 0; break; }
    }
    if (std::fclose(f) != 0 || !ok) {
      throw std::runtime_error("incomplete SHA256 file read " + p.string());
    }
    return h.finish();
  }

 private:
  static uint32_t rotr(uint32_t x, unsigned r) { return (x >> r) | (x << (32-r)); }
  static uint32_t choose(uint32_t x,uint32_t y,uint32_t z) { return (x&y)^(~x&z); }
  static uint32_t majority(uint32_t x,uint32_t y,uint32_t z) { return (x&y)^(x&z)^(y&z); }
  void compress() {
    static constexpr std::array<uint32_t,64> k{{
      0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
      0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
      0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
      0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
      0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
      0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
      0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
      0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
    }};
    uint32_t w[64]{};
    for (size_t i=0; i<16; ++i) {
      const size_t j = i*4;
      w[i] = (uint32_t(block_[j])<<24) | (uint32_t(block_[j+1])<<16)
           | (uint32_t(block_[j+2])<<8) | uint32_t(block_[j+3]);
    }
    for (size_t i=16; i<64; ++i) {
      uint32_t s0=rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3);
      uint32_t s1=rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10);
      w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    uint32_t a=state_[0],b=state_[1],c=state_[2],d=state_[3];
    uint32_t e=state_[4],f=state_[5],g=state_[6],h=state_[7];
    for(size_t i=0;i<64;++i) {
      const uint32_t s1=rotr(e,6)^rotr(e,11)^rotr(e,25);
      const uint32_t t1=h+s1+choose(e,f,g)+k[i]+w[i];
      const uint32_t s0=rotr(a,2)^rotr(a,13)^rotr(a,22);
      const uint32_t t2=s0+majority(a,b,c);
      h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    state_[0]+=a;state_[1]+=b;state_[2]+=c;state_[3]+=d;
    state_[4]+=e;state_[5]+=f;state_[6]+=g;state_[7]+=h;
  }

  std::array<uint32_t,8> state_{{0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
                                 0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u}};
  std::array<uint8_t,64> block_{};
  size_t used_=0;
  uint64_t count_=0;
};

} // namespace reliability_layer
