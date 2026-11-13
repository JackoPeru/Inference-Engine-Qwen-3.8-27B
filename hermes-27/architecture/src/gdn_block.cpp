#include "hermes/gdn_block.hpp"
#include <cmath>
#include <cstdio>
#include <stdexcept>
namespace hermes {
namespace {
inline float silu(float v){ return v/(1.f+expf(-v)); }
inline float softplus(float v){ return v>20.f?v:logf(1.f+expf(v)); }
void linear(const std::vector<float>& x, const std::vector<float>& W, std::vector<float>& y,
            int M, int N){ // y[m,n] = sum_k x[m,k] W[n,k]; W [N,K] row-major
  for(int m=0;m<M;++m) for(int n=0;n<N;++n){ double a=0; for(int k=0;k<(int)x.size()/M;++k) a+=double(x[m*(x.size()/M)+k])*double(W[n*(x.size()/M)+k]); y[m*N+n]=(float)a; }
}
}
GdnBlockF32 GdnBlockF32::load(const char* path){
  FILE* f=fopen(path,"rb"); if(!f) throw std::runtime_error("open bin");
  GdnBlockF32 g; int hdr[8]; if(fread(hdr,4,8,f)!=8) throw std::runtime_error("hdr");
  g.B=hdr[0];g.T=hdr[1];g.Hid=hdr[2];g.Hk=hdr[3];g.Khd=hdr[4];g.Hv=hdr[5];g.Vhd=hdr[6];g.convK=hdr[7];
  std::vector<std::vector<float>*> dst={&g.Wqkv,&g.Wz,&g.Wb,&g.Wa,&g.Wconv,&g.Alog,&g.dt,&g.Wnorm,&g.Wo};
  for(auto d: dst){ int nd; if(fread(&nd,4,1,f)!=1) throw std::runtime_error("nd");
    std::vector<int> sh(nd); if(fread(sh.data(),4,nd,f)!=(size_t)nd) throw std::runtime_error("sh");
    size_t n=1; for(int s:sh)n*=s; d->resize(n);
    if(fread(d->data(),4,n,f)!=n) throw std::runtime_error("data"); }
  // skip x,y (test reads them separately)
  fclose(f); return g;
}
// core recurrence over prepared q,k,v,beta,glog; q,k already l2normed+scaled
static void core(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v,
                 const std::vector<float>& beta, const std::vector<float>& gl,
                 std::vector<float>& o, std::vector<float>& S, int T, int H, int K, int V){
  for(int t=0;t<T;++t) for(int h=0;h<H;++h){
    float dec=expf(gl[t*H+h]), be=beta[t*H+h];
    float* Sh=&S[h*K*V];
    for(int i=0;i<K*V;++i) Sh[i]*=dec;
    // bv = (v - S*k)*beta
    for(int j=0;j<V;++j){ double d=0; for(int i=0;i<K;++i) d+=double(Sh[i*V+j])*double(k[(t*H+h)*K+i]); double bv=(double(v[(t*H+h)*V+j])-d)*double(be);
      for(int i=0;i<K;++i) Sh[i*V+j]+=float(double(k[(t*H+h)*K+i])*bv); }
    for(int j=0;j<V;++j){ double a=0; for(int i=0;i<K;++i) a+=double(Sh[i*V+j])*double(q[(t*H+h)*K+i]); o[(t*H+h)*V+j]=(float)a; }
  }
}
std::vector<float> GdnBlockF32::forward_prefill(const std::vector<float>& x, std::vector<float>& state) const{
  int T=this->T,Hid=this->Hid,Hk=this->Hk,Khd=this->Khd,Hv=this->Hv,Vhd=this->Vhd,CK=convK;
  int Kd=Hk*Khd, Vd=Hv*Vhd, C=2*Kd+Vd;
  std::vector<float> qkv(T*C); linear(x,Wqkv,qkv,T,C);
  // causal depthwise conv + silu
  std::vector<float> cv(T*C,0.f);
  for(int t=0;t<T;++t) for(int c=0;c<C;++c){ double a=0; for(int j=0;j<CK;++j){ int tt=t-(CK-1-j); float xv=(tt<0)?0.f:qkv[tt*C+c]; a+=double(xv)*double(Wconv[c*CK+j]); } cv[t*C+c]=silu((float)a); }
  int rep=Hv/Hk;
  std::vector<float> q(T*Hv*Khd),k(T*Hv*Khd),v(T*Hv*Vhd);
  for(int t=0;t<T;++t){
    auto* qs=&cv[t*C]; auto* ks=&cv[t*C+Kd]; auto* vs=&cv[t*C+2*Kd];
    for(int h=0;h<Hv;++h){ int g=h/rep;
      for(int d=0;d<Khd;++d){ q[(t*Hv+h)*Khd+d]=qs[g*Khd+d]; k[(t*Hv+h)*Khd+d]=ks[g*Khd+d]; }
      for(int d=0;d<Vhd;++d) v[(t*Hv+h)*Vhd+d]=vs[h*Vhd+d]; }
  }
  std::vector<float> zl(T*Vd); linear(x,Wz,zl,T,Vd);
  std::vector<float> bl(T*Hv); linear(x,Wb,bl,T,Hv);
  std::vector<float> al(T*Hv); linear(x,Wa,al,T,Hv);
  std::vector<float> beta(T*Hv),gl(T*Hv);
  for(int i=0;i<T*Hv;++i){ int h=i%Hv; beta[i]=1.f/(1.f+expf(-bl[i])); gl[i]=-expf(Alog[h])*softplus(al[i]+dt[h]); }
  // l2norm q,k + scale
  float sc=1.f/sqrtf((float)Khd);
  for(int i=0;i<T*Hv;++i){ double nq=0,nk=0; for(int d=0;d<Khd;++d){ nq+=double(q[i*Khd+d])*double(q[i*Khd+d]); nk+=double(k[i*Khd+d])*double(k[i*Khd+d]); }
    nq=1.0/sqrt(nq+1e-6); nk=1.0/sqrt(nk+1e-6); // fla l2norm: true L2, x/sqrt(sum+eps)
    for(int d=0;d<Khd;++d){ q[i*Khd+d]=(float)(double(q[i*Khd+d])*nq*double(sc)); k[i*Khd+d]=(float)(double(k[i*Khd+d])*nk); } }
  state.assign(Hv*Khd*Vhd,0.f);
  std::vector<float> o(T*Hv*Vhd);
  core(q,k,v,beta,gl,o,state,T,Hv,Khd,Vhd);
  // gated norm + out
  std::vector<float> y(T*Hid,0.f);
  for(int t=0;t<T;++t) for(int h=0;h<Hv;++h){
    float* oo=&o[(t*Hv+h)*Vhd]; float* zz=&zl[(t*Hv+h)*Vhd];
    double va=0; for(int d=0;d<Vhd;++d) va+=double(oo[d])*double(oo[d]); float inv=1.f/sqrtf(float(va/Vhd)+1e-6f);
    for(int d=0;d<Vhd;++d){ float g2=oo[d]*inv*Wnorm[d]*silu(zz[d]); oo[d]=g2; }
  }
  linear(o,Wo,y,T,Hid);
  return y;
}
std::vector<float> GdnBlockF32::forward_step(const std::vector<float>& x, std::vector<float>& state,
                                             std::vector<float>& conv_state) const{
  int Hid=this->Hid,Hk=this->Hk,Khd=this->Khd,Hv=this->Hv,Vhd=this->Vhd,CK=convK;
  int Kd=Hk*Khd, Vd=Hv*Vhd, C=2*Kd+Vd;
  std::vector<float> qkv1(C); linear(x,Wqkv,qkv1,1,C);
  // streaming conv: conv_state [C,CK-1] holds previous inputs
  std::vector<float> cv1(C);
  for(int c=0;c<C;++c){ double a=0; for(int j=0;j<CK;++j){ float xv=(j<CK-1)?conv_state[c*(CK-1)+j]:qkv1[c]; a+=double(xv)*double(Wconv[c*CK+j]); } cv1[c]=silu((float)a); }
  for(int c=0;c<C;++c){ for(int j=0;j<CK-2;++j) conv_state[c*(CK-1)+j]=conv_state[c*(CK-1)+j+1]; if(CK>1) conv_state[c*(CK-1)+CK-2]=qkv1[c]; }
  int rep=Hv/Hk;
  std::vector<float> q(Hv*Khd),k(Hv*Khd),v(Hv*Vhd);
  for(int h=0;h<Hv;++h){ int g=h/rep;
    for(int d=0;d<Khd;++d){ q[h*Khd+d]=cv1[g*Khd+d]; k[h*Khd+d]=cv1[Kd+g*Khd+d]; }
    for(int d=0;d<Vhd;++d) v[h*Vhd+d]=cv1[2*Kd+h*Vhd+d]; }
  std::vector<float> zl(Vd); linear(x,Wz,zl,1,Vd);
  std::vector<float> bl(Hv); linear(x,Wb,bl,1,Hv);
  std::vector<float> al(Hv); linear(x,Wa,al,1,Hv);
  float sc=1.f/sqrtf((float)Khd);
  for(int h=0;h<Hv;++h){
    double nq=0,nk=0; for(int d=0;d<Khd;++d){ nq+=double(q[h*Khd+d])*double(q[h*Khd+d]); nk+=double(k[h*Khd+d])*double(k[h*Khd+d]); }
    nq=1.0/sqrt(nq+1e-6); nk=1.0/sqrt(nk+1e-6); // true L2
    float be=1.f/(1.f+expf(-bl[h])), dec=expf(-expf(Alog[h])*softplus(al[h]+dt[h]));
    float* Sh=&state[h*Khd*Vhd];
    for(int i=0;i<Khd*Vhd;++i) Sh[i]*=dec;
    for(int j=0;j<Vhd;++j){ double d=0; for(int i=0;i<Khd;++i) d+=double(Sh[i*Vhd+j])*double(k[h*Khd+i]*nk); double bv=(double(v[h*Vhd+j])-d)*double(be);
      for(int i=0;i<Khd;++i) Sh[i*Vhd+j]+=float(double(k[h*Khd+i]*nk)*bv); }
    for(int j=0;j<Vhd;++j){ double a=0; for(int i=0;i<Khd;++i) a+=double(Sh[i*Vhd+j])*double(q[h*Khd+i]*nq*double(sc)); zl[h*Vhd+j]=(float)a; }
  }
  // gated norm (reuse zl as o buffer)
  std::vector<float> hh(Vd);
  // zl currently holds o; need z separately:
  std::vector<float> z2(Vd); linear(x,Wz,z2,1,Vd);
  for(int h=0;h<Hv;++h){ double va=0; for(int d=0;d<Vhd;++d) va+=double(zl[h*Vhd+d])*double(zl[h*Vhd+d]); float inv=1.f/sqrtf(float(va/Vhd)+1e-6f);
    for(int d=0;d<Vhd;++d) hh[h*Vhd+d]=zl[h*Vhd+d]*inv*Wnorm[d]*silu(z2[h*Vhd+d]); }
  std::vector<float> y(Hid); linear(hh,Wo,y,1,Hid);
  return y;
}
}
