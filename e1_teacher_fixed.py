"""Teacher forcing: prefill [ids + gen_prefix], check next-token argmax.
Validates decode carry: incremental == full-prefill."""
import struct, mmap, math, torch, sys
import torch.nn.functional as F
HID=5120; FFN=17408; NL=64; VOC=248320; NQ=24; NKV=4; HD=256; RD=64
KD=16*128; VD=48*128; C=2*KD+VD; HV=48; KH=128; VH=128; TH=10000000.0
def rms(x,w):
    return x*torch.rsqrt((x.float()**2).mean(-1,keepdim=True)+1e-6)*(1+w)
path="/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf"
f=open(path,'rb'); f.read(4); struct.unpack('<I',f.read(4)); nt=struct.unpack('<Q',f.read(8))[0]; nkv=struct.unpack('<Q',f.read(8))[0]
def rs():
    n=struct.unpack('<Q',f.read(8))[0]; return f.read(n)
for _ in range(nkv):
    k=rs(); vt=struct.unpack('<I',f.read(4))[0]
    if vt in (0,1,7): f.read(1)
    elif vt in (2,3): f.read(2)
    elif vt in (4,5,6): f.read(4)
    elif vt==8: rs()
    elif vt==9:
        at=struct.unpack('<I',f.read(4))[0]; n=struct.unpack('<Q',f.read(8))[0]
        if at==8:
            for __ in range(n): rs()
        else: f.seek(n*{0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[at],1)
    else: f.read(8)
TENS={}
for _ in range(nt):
    k=rs().decode(); nd=struct.unpack('<I',f.read(4))[0]
    dims=[struct.unpack('<Q',f.read(8))[0] for __ in range(nd)]
    ty=struct.unpack('<I',f.read(4))[0]; off=struct.unpack('<Q',f.read(8))[0]
    TENS[k]=(dims,ty,off)
DO=((f.tell()+31)//32)*32
mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
import numpy as np
E2M1=np.array([0,0.5,1,1.5,2,3,4,6,0,-0.5,-1,-1.5,-2,-3,-4,-6],dtype=np.float32)
def ue(x):
    x=int(x)
    if x==0 or x==0x7F: return 0.0
    e=(x>>3)&0xF; m=x&0x7
    return math.ldexp(float(m),-9) if e==0 else math.ldexp(1.0+m/8.0,e-7)
def get(name):
    dims,ty,off=TENS[name]; ne0=dims[0]; ne1=dims[1] if len(dims)>1 else 1
    base=DO+off
    if ty==30:
        n=ne0*ne1
        return torch.frombuffer(bytearray(mm[base:base+n*2]),dtype=torch.bfloat16).clone().reshape(ne1,ne0).float()
    if ty==0:
        n=ne0*ne1
        return torch.frombuffer(bytearray(mm[base:base+n*4]),dtype=torch.float32).clone().reshape(ne1,ne0)
    assert ty==40
    nb=ne0//64
    out=np.empty((ne1,ne0),dtype=np.float32)
    for r in range(ne1):
        rb=mm[base+r*nb*36:base+(r+1)*nb*36]
        a=np.frombuffer(rb,dtype=np.uint8).reshape(nb,36)
        sc=np.array([[ue(v) for v in row[:4]] for row in a],dtype=np.float32)
        qs=a[:,4:]; lo=(qs&0xF).reshape(nb,4,8); hi=(qs>>4).reshape(nb,4,8)
        vals=np.empty((nb,4,16),dtype=np.float32); vals[:,:,:8]=E2M1[lo]; vals[:,:,8:]=E2M1[hi]
        out[r]=(sc[:,:,None]*vals).reshape(-1)
    return torch.from_numpy(out)
def l2n(x): return x*torch.rsqrt((x.float()**2).sum(-1,keepdim=True)+1e-6)
emb=get("token_embd.weight"); outnorm=get("output_norm.weight"); head=get("output.weight")
base_ids=[6205,161421,303,220,17,1374,9911,1147,27053,6059,13]
gen=[10,220,10,1076,10]
def forward(ids):
    T=len(ids)
    x=emb[ids]
    pos=torch.arange(T).float()
    freq=1.0/(TH**(torch.arange(0,64,2).float()/64))
    ang=pos[:,None]*freq[None,:]
    c=torch.cat([ang.cos(),ang.cos()],-1); s=torch.cat([ang.sin(),ang.sin()],-1)
    # FIX 2026-11-13: split-half RoPE with head broadcast (was x1*c shape mismatch).
    def rot(xx):
        r=xx[...,:64]; p=xx[...,64:]
        x1,x2=r[...,:32],r[...,32:]
        c32=c[...,:32]; s32=s[...,:32]
        while c32.dim()<x1.dim(): c32=c32.unsqueeze(-2); s32=s32.unsqueeze(-2)
        return torch.cat([torch.cat([x1*c32-x2*s32,x1*s32+x2*c32],-1),p],-1)
    for li in range(NL):
        P=f"blk.{li}."
        h=rms(x,get(P+"attn_norm.weight"))
        if li%4==3:
            Wq=get(P+"attn_q.weight"); Wk=get(P+"attn_k.weight"); Wv=get(P+"attn_v.weight")
            Wo=get(P+"attn_output.weight"); qn=get(P+"attn_q_norm.weight"); kn=get(P+"attn_k_norm.weight")
            qq4=(h@Wq.T).reshape(T,NQ,512); qq8=qq4[...,:256].reshape(T,NQ,HD); kk=(h@Wk.T).reshape(T,NKV,HD); vv=(h@Wv.T).reshape(T,NKV,HD)
            g8=qq4[...,256:].reshape(T,-1)
            qq8=rms(qq8,qn); kk=rms(kk,kn)
            qq8=rot(qq8); kk=rot(kk)
            qq8=qq8.transpose(0,1); kk=kk.transpose(0,1); vv=vv.transpose(0,1)
            kk6=kk.repeat_interleave(6,0); vv6=vv.repeat_interleave(6,0)
            att=torch.softmax((qq8*(HD**-0.5))@kk6.transpose(-1,-2)+torch.triu(torch.full((T,T),float('-inf')),1),-1)@vv6
            att=att.transpose(0,1).reshape(T,-1)*torch.sigmoid(g8)
            core=att@Wo.T
        else:
            Wqkv=get(P+"attn_qkv.weight"); Wz=get(P+"attn_gate.weight"); Wb=get(P+"ssm_beta.weight")
            Wa=get(P+"ssm_alpha.weight"); A=get(P+"ssm_a"); DT=get(P+"ssm_dt.bias")
            Wc=get(P+"ssm_conv1d.weight").reshape(C,1,4)
            Wn=get(P+"ssm_norm.weight"); Wout=get(P+"ssm_out.weight")
            qkv=h@Wqkv.T
            xp=torch.cat([torch.zeros(1,C,3),qkv.T.unsqueeze(0)],-1)
            conv=F.conv1d(xp,Wc,groups=C).squeeze(0).T
            conv=F.silu(conv)
            q,k,v=torch.split(conv,[KD,KD,VD],-1)
            q=q.reshape(T,16,128).repeat_interleave(3,1); k=k.reshape(T,16,128).repeat_interleave(3,1); v=v.reshape(T,48,128)
            z=(h@Wz.T).reshape(T,48,128)
            beta=torch.sigmoid(h@Wb.T); gg=(-A.exp()*F.softplus((h@Wa.T)+DT))
            qn=lqn(q)*(KH**-0.5); kn=lqn(k)
            S=torch.zeros(48,128,128); outs=[]
            for t in range(T):
                S=S*torch.exp(gg[t])[:,None,None]
                bv=(v[t]-(S*kn[t][:,:,None]).sum(1))*beta[t][:,None]
                S=S+kn[t][:,:,None]*bv[:,None,:]
                outs.append((S*qn[t][:,:,None]).sum(1))
            o=torch.stack(outs)
            n=o*torch.rsqrt((o**2).mean(-1,keepdim=True)+1e-6)
            core=(n*Wn*F.silu(z)).reshape(T,6144)@Wout.T
        x=x+core
        h2=rms(x,get(P+"post_attention_norm.weight"))
        gate=F.silu(h2@get(P+"ffn_gate.weight").T)*(h2@get(P+"ffn_up.weight").T)
        x=x+(gate@get(P+"ffn_down.weight").T)
    xn=rms(x,outnorm)
    return (xn@head.T)
def lqn(x): return l2n(x)
import sys
ok=True
for k in range(1,5):
    ids=base_ids+gen[:k]
    lg=forward(ids)
    pred=lg[-1].argmax().item()
    exp=gen[k]
    match = "OK" if pred==exp else "MISMATCH"
    if pred!=exp: ok=False
    print(f"step {k}: prefill {len(ids)} tok -> pred {pred} (decode said {exp}) {match}",flush=True)
print("TEACHER-FORCING", "PASS" if ok else "FAIL")
