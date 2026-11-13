// test_gdn_block: tiny-config CPU vs transformers golden (/tmp/gdn_tiny.bin).
// Checks: prefill==golden (tol 1e-3), step-by-step==prefill (exact), conv carry ok.
#include "hermes/gdn_block.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#include <stdexcept>
using namespace hermes;
static std::vector<float> read_arr(FILE* f){
  int nd; if(fread(&nd,4,1,f)!=1) throw std::runtime_error("nd");
  std::vector<int> sh(nd); if(fread(sh.data(),4,nd,f)!=(size_t)nd) throw std::runtime_error("sh");
  size_t n=1; for(int s:sh)n*=s; std::vector<float> v(n);
  if(fread(v.data(),4,n,f)!=n) throw std::runtime_error("d"); return v;
}
int main(){
  GdnBlockF32 g = GdnBlockF32::load("/tmp/gdn_tiny.bin");
  FILE* f=fopen("/tmp/gdn_tiny.bin","rb"); if(!f) return 2;
  int hdr[8]; if(fread(hdr,4,8,f)!=8) return 2;
  for(int i=0;i<9;++i){ (void)read_arr(f); } // skip weights
  std::vector<float> x=read_arr(f), yg=read_arr(f); fclose(f);
  printf("cfg B=%d T=%d Hid=%d Hk=%d Khd=%d Hv=%d Vhd=%d convK=%d\n",g.B,g.T,g.Hid,g.Hk,g.Khd,g.Hv,g.Vhd,g.convK);
  std::vector<float> S;
  auto y = g.forward_prefill(x, S);
  float mx=0; for(size_t i=0;i<y.size();++i){ float e=fabsf(y[i]-yg[i]); if(e>mx)mx=e; }
  printf("prefill vs golden max_abs_err=%.2e %s\n",mx,mx<1e-3?"PASS":"FAIL");
  if(mx>=1e-3) return 1;
  // step-by-step from scratch == prefill
  std::vector<float> S2(g.Hv*g.Khd*g.Vhd,0.f), CS(128*(4-1),0.f);
  std::vector<float> ys; ys.reserve(y.size());
  for(int t=0;t<g.T;++t){
    std::vector<float> xt(x.begin()+t*g.Hid, x.begin()+(t+1)*g.Hid);
    auto yt = g.forward_step(xt, S2, CS);
    ys.insert(ys.end(), yt.begin(), yt.end());
  }
  float m2=0; for(size_t i=0;i<y.size();++i){ float e=fabsf(ys[i]-y[i]); if(e>m2)m2=e; }
  float m3=0; for(size_t i=0;i<S.size();++i){ float e=fabsf(S2[i]-S[i]); if(e>m3)m3=e; }
  printf("token-by-token vs prefill out=%.2e state=%.2e %s\n",m2,m3,(m2<1e-4&&m3<1e-4)?"PASS":"FAIL");
  return (m2<1e-4&&m3<1e-4)?0:1;
}
