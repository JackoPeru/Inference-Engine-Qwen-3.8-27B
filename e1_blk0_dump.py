"""E1 blk0 dump for E1/E2 bisect (matches blk0_stats equations)."""
import struct, mmap, math, torch
import torch.nn.functional as F
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
    nb=ne0//64; out=np.empty((ne1,ne0),dtype=np.float32)
    for r in range(ne1):
        rb=mm[base+r*nb*36:base+(r+1)*nb*36]
        a=np.frombuffer(rb,dtype=np.uint8).reshape(nb,36)
        sc=np.array([[ue(v) for v in row[:4]] for row in a],dtype=np.float32)
        qs=a[:,4:]; lo=(qs&0xF).reshape(nb,4,8); hi=(qs>>4).reshape(nb,4,8)
        vals=np.empty((nb,4,16),dtype=np.float32); vals[:,:,:8]=E2M1[lo]; vals[:,:,8:]=E2M1[hi]
        out[r]=(sc[:,:,None]*vals).reshape(-1)
    return torch.from_numpy(out)
def rms(x,w,plus=True):
    return x*torch.rsqrt((x.float()**2).mean(-1,keepdim=True)+1e-6)*((1+w) if plus else w)
ids=[6205,161421,303,220,17,1374,9911,1147,27053,6059,13]
emb=get("token_embd.weight")
x=emb[ids]
HID=5120; KD=2048; VD=6144; C=2*KD+VD
P="blk.0."
h=rms(x,get(P+"attn_norm.weight"),True)
qkv=h@get(P+"attn_qkv.weight").T
xp=torch.cat([torch.zeros(1,C,3),qkv.T.unsqueeze(0)],-1)
Wc=get(P+"ssm_conv1d.weight")
conv=F.conv1d(xp,Wc.reshape(C,1,4),groups=C).squeeze(0).T
conv=F.silu(conv)
q,k,v=torch.split(conv,[KD,KD,VD],-1)
q=q.reshape(11,16,128).repeat_interleave(3,1); k=k.reshape(11,16,128).repeat_interleave(3,1); v=v.reshape(11,48,128)
z=(h@get(P+"attn_gate.weight").T).reshape(11,48,128)
beta=torch.sigmoid(h@get(P+"ssm_beta.weight").T)
gg=(-get(P+"ssm_a").exp()*F.softplus((h@get(P+"ssm_alpha.weight").T)+get(P+"ssm_dt.bias")))
def l2n(x): return x*torch.rsqrt((x.float()**2).sum(-1,keepdim=True)+1e-6)
qn=l2n(q)*(128**-0.5); kn=l2n(k)
S=torch.zeros(48,128,128); outs=[]
for t in range(11):
    S=S*torch.exp(gg[t])[:,None,None]
    bv=(v[t]-(S*kn[t][:,:,None]).sum(1))*beta[t][:,None]
    S=S+kn[t][:,:,None]*bv[:,None,:]
    outs.append((S*qn[t][:,:,None]).sum(1))
o=torch.stack(outs)
n=o*torch.rsqrt((o**2).mean(-1,keepdim=True)+1e-6)
gated=n*get(P+"ssm_norm.weight")*F.silu(z)
core=gated.reshape(11,6144)@get(P+"ssm_out.weight").T
h1=x+core
m=rms(h1,get(P+"post_attention_norm.weight"),True)
gate=F.silu(m@get(P+"ffn_gate.weight").T)*(m@get(P+"ffn_up.weight").T)
y=h1+(gate@get(P+"ffn_down.weight").T)
# save raw float32 row-major like E2 (T*C etc.)
qkv.float().contiguous().numpy().tofile("/tmp/e1_cv_qkv.bin")
conv.float().contiguous().numpy().tofile("/tmp/e1_cv.bin")
beta.float().contiguous().numpy().tofile("/tmp/e1_beta.bin")
gg.float().contiguous().numpy().tofile("/tmp/e1_gl.bin")
o.float().contiguous().numpy().tofile("/tmp/e1_o.bin")
core.float().contiguous().numpy().tofile("/tmp/e1_core.bin")
y.float().contiguous().numpy().tofile("/tmp/e1_y.bin")
print("e1 dumps saved")
print("e1 blk0: mean=%.5f std=%.5f min=%.2f max=%.2f"% (y.mean().item(),y.std().item(),y.min().item(),y.max().item()))
