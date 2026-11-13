// Synthetic tests: NVFP4 LUT, UE4M3, block dequant, row sizes, parser rejects.
#include "hermes/gguf.hpp"
#include <cstdio>
#include <cmath>
#include <cassert>
#include <fstream>
using namespace hermes;
int main(){
  // 1. E2M1 LUT
  assert(fabs(e2m1_to_fp32(0)-0.0f)<1e-6);
  assert(fabs(e2m1_to_fp32(1)-0.5f)<1e-6);
  assert(fabs(e2m1_to_fp32(6)-4.0f)<1e-6);
  assert(fabs(e2m1_to_fp32(7)-6.0f)<1e-6);
  assert(fabs(e2m1_to_fp32(14)+4.0f)<1e-6);
  // 2. UE4M3 zero/NaN -> 0
  assert(ue4m3_to_fp32(0)==0.0f);
  assert(ue4m3_to_fp32(0x7F)==0.0f);
  // UE4M3 0x40: exp=8,man=0 -> 2^(1)*0.5=1.0
  assert(fabs(ue4m3_to_fp32(0x40)-1.0f)<1e-6);
  // 3. row sizes
  assert(ggml_row_size(GgmlType::NVFP4,64)==36);
  assert(ggml_row_size(GgmlType::NVFP4,5120)==5120/64*36);
  assert(ggml_row_size(GgmlType::BF16,5120)==5120*2);
  // 4. block dequant: UE4M3 0x40 raw=2.0, ue()=1.0; E2M1 nibble 1=0.5 -> raw*E=1.0
  uint8_t b[36]={}; b[0]=b[1]=b[2]=b[3]=0x40;
  for(int i=4;i<36;++i)b[i]=0x11;
  float o[64]; nvfp4_block_dequant(b,o);
  for(int i=0;i<64;++i) assert(fabs(o[i]-1.0f)<1e-5);
  // 5. parser must reject bad magic
  { std::ofstream f("/tmp/bad.gguf",std::ios::binary); uint32_t m=0; f.write((char*)&m,4); f.close();
    GgufFile g; bool threw=false; try{g.open("/tmp/bad.gguf");}catch(...){threw=true;} assert(threw); }
  printf("SYNTH OK: E2M1/UE4M3/NVFP4/parser-reject pass\n");
  return 0;
}
