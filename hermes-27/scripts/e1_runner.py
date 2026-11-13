"""E1: full-model Qwen3.8-27B NVFP4 reference forward (CPU fp32, exact equations).
Prefill T + greedy decode with full state carry (conv/GDN/KV). Saves golden logits.
Equations: transformers qwen3_5 (verified). Run: exllama-venv python e1_runner.py
"""
import struct, mmap, math, json, time, torch
import torch.nn.functional as F
torch.manual_seed(0)
HID=5120; FFN=17408; NL=64; VOC=248320; NQ=24; NKV=4; HD=256; RD=64
KD=16*128; VD=48*128; C=2*KD+VD; HV=48; KH=128; VH=128; TH=10000000.0
def rms(x,w): return x*torch.rsqrt((x.float()**2).mean(-1,keepdim=True)+1e-6)*w
def l2n(x): return x*torch.rsqrt((x.float()**2).sum(-1,keepdim=True)+1e-6)
path="/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf"
f=open(path,'rb'); f.read(4); struct.unpack('<I',f.read(4)); nt=struct.unpack('<Q',f.read(8))[0]; nkv=struct.unpack('<Q',f.read(8))[0]
def rs():
    n=struct.unpack('<Q',f.read(8))[0]; return f.read(n).decode()
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
    k=rs(); nd=struct.unpack('<I',f.read(4))[0]
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
        qs=a[:,4:]
        lo=(qs&0xF).reshape(nb,4,8); hi=(qs>>4).reshape(nb,4,8)
        vals=np.empty((nb,4,16),dtype=np.float32); vals[:,:,:8]=E2M1[lo]; vals[:,:,8:]=E2M1[hi]
        out[r]=(sc[:,:,None]*vals).reshape(-1)
    return torch.from_numpy(out)
print("loading embeddings + norms ...",flush=True)
emb=get("token_embd.weight").to(torch.bfloat16); outnorm=get("output_norm.weight"); head=get("output.weight").to(torch.bfloat16)
from tokenizers import Tokenizer
tok=Tokenizer.from_file("/opt/hermes/models/Qwen3.8-27B-Uncensored-EXL3-3.07bpw/tokenizer.json")
ids=tok.encode("Spiega in 2 frasi la gravita.").ids
print("prompt ids:",ids[:12],"...",len(ids))
T=len(ids)
x=emb[ids].float()
# mRoPE-text cos/sin [T,64]
pos=torch.arange(T).float()
freq=1.0/(TH**(torch.arange(0,64,2).float()/64))
ang=pos[:,None]*freq[None,:]  # [T,32]
c=torch.cat([ang.cos(),ang.cos()],-1); s=torch.cat([ang.sin(),ang.sin()],-1)
def rot(x):
    r=x[...,:64]; p=x[...,64:]
    x1,x2=r[...,:32],r[...,32:]
    return torch.cat([torch.cat([x1*c+x2*(-s),x1*s+x2*c],-1) if False else (r*c+torch.cat([-x2,x1],-1)*s),p],-1)
# GPT-J style: out=[x1*c-x2*s, x1*s+x2*c], c/s halves duplicated [c,c]
def rot_clean(x):
    r=x[...,:64]; p=x[...,64:]
    x1,x2=r[...,:32],r[...,32:]
    c32=c[...,:32]; s32=s[...,:32]
    while c32.dim()<x1.dim(): c32=c32.unsqueeze(-2); s32=s32.unsqueeze(-2)
    return torch.cat([torch.cat([x1*c32-x2*s32,x1*s32+x2*c32],-1),p],-1)
states=[]; convs=[]; kvcache=[]
t0=time.time()
for li in range(NL):
    P=f"blk.{li}."
    ln1=get(P+"attn_norm.weight"); ln2=get(P+"post_attention_norm.weight")
    h=rms(x,ln1)
    if li%4==3:
        Wq=get(P+"attn_q.weight"); Wk=get(P+"attn_k.weight"); Wv=get(P+"attn_v.weight")
        Wo=get(P+"attn_output.weight"); qn=get(P+"attn_q_norm.weight"); kn=get(P+"attn_k_norm.weight")
        qq4=(h@Wq.T).reshape(T,NQ,512); qq8=qq4[...,:256].reshape(T,NQ,HD); kk=(h@Wk.T).reshape(T,NKV,HD); vv=(h@Wv.T).reshape(T,NKV,HD)
        g8=qq4[...,256:].reshape(T,-1)
        qq8=rms(qq8,qn); kk=rms(kk,kn)
        qq8=rot_clean(qq8); kk=rot_clean(kk)
        qq8=qq8.transpose(0,1); kk=kk.transpose(0,1); vv=vv.transpose(0,1)
        kk6=kk.repeat_interleave(6,0); vv6=vv.repeat_interleave(6,0)
        att=torch.softmax((qq8*(HD**-0.5))@kk6.transpose(-1,-2)/1.0+torch.triu(torch.full((T,T),float('-inf')),1),-1)@vv6
        att=att.transpose(0,1).reshape(T,-1)*torch.sigmoid(g8)
        core=att@Wo.T
        del Wq,Wk,Wv,Wo,qn,kn
        states.append(None); convs.append(None);             kvcache.append((kk,vv))
    else:
        Wqkv=get(P+"attn_qkv.weight"); Wz=get(P+"attn_gate.weight"); Wb=get(P+"ssm_beta.weight")
        Wa=get(P+"ssm_alpha.weight"); A=get(P+"ssm_a"); DT=get(P+"ssm_dt.bias")
        Wc=get(P+"ssm_conv1d.weight"); Wn=get(P+"ssm_norm.weight"); Wout=get(P+"ssm_out.weight")
        qkv=h@Wqkv.T
        xp=torch.cat([torch.zeros(1,C,3),qkv.T.unsqueeze(0)],-1)
        conv=F.conv1d(xp,Wc.reshape(C,1,4),groups=C).squeeze(0).T
        conv=F.silu(conv)
        q,k,v=torch.split(conv,[KD,KD,VD],-1)
        q=q.reshape(T,16,128).repeat(1,3,1); k=k.reshape(T,16,128).repeat(1,3,1); v=v.reshape(T,48,128)
        z=(h@Wz.T).reshape(T,48,128)
        beta=torch.sigmoid(h@Wb.T); gg=A*F.softplus((h@Wa.T)+DT)
        qn=l2n(q)*(KH**-0.5); kn=l2n(k)
        S=torch.zeros(48,128,128); outs=[]
        for t in range(T):
            S=S*torch.exp(gg[t])[:,None,None]
            bv=(v[t]-(S*kn[t][:,:,None]).sum(1))*beta[t][:,None]
            S=S+kn[t][:,:,None]*bv[:,None,:]
            outs.append((S*qn[t][:,:,None]).sum(1))
        o=torch.stack(outs)
        n=o*torch.rsqrt((o**2).mean(-1,keepdim=True)+1e-6)
        core=(n*Wn*F.silu(z)).reshape(T,6144)@Wout.T
        del Wqkv,Wz,Wb,Wa,A,DT,Wc,Wn,Wout
        states.append(S); convs.append(conv[-3:].T.clone()); kvcache.append(None)
    x=x+core
    h2=rms(x,ln2)
    Wg=get(P+"ffn_gate.weight"); Wu=get(P+"ffn_up.weight"); Wd=get(P+"ffn_down.weight")
    y=(F.silu(h2@Wg.T)*(h2@Wu.T))@Wd.T
    del Wg,Wu,Wd
    x=x+y
    if li%8==0: print(f"layer {li} done {time.time()-t0:.0f}s",flush=True)
xn=rms(x,outnorm)
logits=torch.empty(T,VOC);
NSH=8; RSH=VOC//NSH
for _s in range(NSH):
    _h=head[_s*RSH:(_s+1)*RSH].float()
    logits[:,_s*RSH:(_s+1)*RSH]=xn.float()@_h.T
    del _h
print("logits:",tuple(logits.shape),"mean/std:",logits.mean().item(),logits.std().item())
assert torch.isfinite(logits).all()
top5=logits[-1].topk(5)
print("top5 ids:",top5.indices.tolist(),"vals:",[round(v,2) for v in top5.values.tolist()],flush=True)
import os as _os
if _os.environ.get("E1_SAVE_LOGITS"):
    torch.save({"ids":ids,"logits":logits.half().cpu(),"top5":top5.indices.tolist()},"/tmp/e1_logits.pt")
    print("saved /tmp/e1_logits.pt",flush=True)
import os
if os.environ.get("E1_PREFILL_ONLY"):
    print("E1 PREFILL DONE (decode skipped)",flush=True)
    raise SystemExit(0)
greedy=top5.indices[0].item()
print("greedy next:",repr(tok.decode([greedy])))
# decode 4 more with carry
gen=[greedy]
for step in range(4):
    xt=emb[[gen[-1]]].float(); T1=1
    pos1=torch.tensor([T+step]).float()
    ang1=pos1[:,None]*freq[None,:]
    c1=torch.cat([ang1.cos(),ang1.cos()],-1); s1=torch.cat([ang1.sin(),ang1.sin()],-1)
    def rot1(xx):
        r=xx[...,:64]; p=xx[...,64:]
        x1,x2=r[...,:32],r[...,32:]
        c32=c1[...,:32]; s32=s1[...,:32]
        while c32.dim()<x1.dim(): c32=c32.unsqueeze(-2); s32=s32.unsqueeze(-2)
        return torch.cat([torch.cat([x1*c32-x2*s32,x1*s32+x2*c32],-1),p],-1)
    for li in range(NL):
        P=f"blk.{li}."
        ln1=get(P+"attn_norm.weight"); ln2=get(P+"post_attention_norm.weight")
        h=rms(xt,ln1)
        if li%4==3:
            Wq=get(P+"attn_q.weight"); Wk=get(P+"attn_k.weight"); Wv=get(P+"attn_v.weight")
            Wo=get(P+"attn_output.weight"); qn=get(P+"attn_q_norm.weight"); kn=get(P+"attn_k_norm.weight")
            qq4=(h@Wq.T).reshape(1,NQ,512)
            g8=qq4[...,256:].reshape(1,-1)
            qq8=rms(qq4[...,:256].reshape(1,NQ,HD),qn); kk=rms((h@Wk.T).reshape(1,NKV,HD),kn); vv=(h@Wv.T).reshape(1,NKV,HD)
            qq8=rot1(qq8); kk=rot1(kk)
            qq8=qq8.transpose(0,1); kk=kk.transpose(0,1); vv=vv.transpose(0,1)
            Kk,Vv=kvcache[li]
            kk=torch.cat([Kk,kk],1); vv=torch.cat([Vv,vv],1); kvcache[li]=(kk,vv)
            kk6=kk.repeat_interleave(6,0); vv6=vv.repeat_interleave(6,0)
            L=kk6.shape[1]
            att=torch.softmax((qq8[:,-1:]*(HD**-0.5))@kk6.transpose(-1,-2),-1)@vv6
            att=att.transpose(0,1).reshape(1,-1)*torch.sigmoid(g8)
            core=att@Wo.T
            del Wq,Wk,Wv,Wo,qn,kn
        else:
            Wqkv=get(P+"attn_qkv.weight"); Wz=get(P+"ssm_gate.weight") if False else get(P+"attn_gate.weight")
            Wb=get(P+"ssm_beta.weight"); Wa=get(P+"ssm_alpha.weight"); A=get(P+"ssm_a"); DT=get(P+"ssm_dt.bias")
            Wc=get(P+"ssm_conv1d.weight"); Wn=get(P+"ssm_norm.weight"); Wout=get(P+"ssm_out.weight")
            qkv=h@Wqkv.T
            CS=convs[li]  # [C,3]
            full=torch.cat([CS,qkv.T],-1)  # [C,4]
            conv=F.conv1d(full.unsqueeze(0),Wc.reshape(C,1,4),groups=C).squeeze(0).T[-1:]
            convs[li]=full[:,-3:].clone()
            conv=F.silu(conv)
            q,k,v=torch.split(conv,[KD,KD,VD],-1)
            q=q.reshape(1,16,128).repeat(1,3,1); k=k.reshape(1,16,128).repeat(1,3,1); v=v.reshape(1,48,128)
            z=(h@Wz.T).reshape(1,48,128)
            beta=torch.sigmoid(h@Wb.T); gg=A*F.softplus((h@Wa.T)+DT)
            qn=l2n(q)*(KH**-0.5); kn=l2n(k)
            S=states[li]
            S=S*torch.exp(gg[0])[:,None,None]
            bv=(v[0]-(S*kn[0][:,:,None]).sum(1))*beta[0][:,None]
            S=S+kn[0][:,:,None]*bv[:,None,:]
            o=(S*qn[0][:,:,None]).sum(1)
            n=o*torch.rsqrt((o**2).mean(-1,keepdim=True)+1e-6)
            core=(n*Wn*F.silu(z)).reshape(1,6144)@Wout.T
            del Wqkv,Wz,Wb,Wa,A,DT,Wc,Wn,Wout
        xt=xt+core
        h2=rms(xt,ln2)
        Wg=get(P+"ffn_gate.weight"); Wu=get(P+"ffn_up.weight"); Wd=get(P+"ffn_down.weight")
        xt=xt+((F.silu(h2@Wg.T)*(h2@Wu.T))@Wd.T)
        del Wg,Wu,Wd
    _xn=rms(xt,outnorm).float()
    lg=torch.empty(1,VOC)
    for _s in range(8):
        _h=head[_s*(VOC//8):(_s+1)*(VOC//8)].float()
        lg[:,_s*(VOC//8):(_s+1)*(VOC//8)]=_xn@_h.T
        del _h
    nx=lg[0].argmax().item(); gen.append(nx)
print("generated:",gen)
print("decoded:",repr(tok.decode(gen)))
torch.save({"top5":top5.indices.tolist(),"gen":gen,"logit_mean":logits.mean().item(),"logit_std":logits.std().item()},"/tmp/e1_golden.pt")
print("E1 FULL-MODEL PASS")
