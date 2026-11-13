"""F1: blk.0 real-weight forward (CPU fp32, T=4). Verifies layout+dequant+equations alive."""
import struct, mmap, math, torch, time
path="/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf"
f=open(path,'rb')
f.read(4); ver=struct.unpack('<I',f.read(4))[0]
nt=struct.unpack('<Q',f.read(8))[0]; nkv=struct.unpack('<Q',f.read(8))[0]
def rs():
    n=struct.unpack('<Q',f.read(8))[0]
    return f.read(n).decode()
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
        else:
            f.seek(n*{0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[at],1)
    else: f.read(8)
tensors={}
for _ in range(nt):
    k=rs(); nd=struct.unpack('<I',f.read(4))[0]
    dims=[struct.unpack('<Q',f.read(8))[0] for __ in range(nd)]
    ty=struct.unpack('<I',f.read(4))[0]; off=struct.unpack('<Q',f.read(8))[0]
    tensors[k]=(dims,ty,off)
data_off=((f.tell()+31)//32)*32
mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
E2M1=[0,0.5,1,1.5,2,3,4,6,0,-0.5,-1,-1.5,-2,-3,-4,-6]
def ue4m3(x):
    x=int(x)
    if x==0 or x==0x7F: return 0.0
    e=(x>>3)&0xF; m=x&0x7
    r=math.ldexp(float(m),-9) if e==0 else math.ldexp(1.0+m/8.0,int(e)-7)
    return r
import numpy as np
def read_nvfp4(name):
    dims,ty,off=tensors[name]; assert ty==40
    ne0,ne1=dims[0],dims[1] if len(dims)>1 else 1
    base=data_off+off; nb=ne0//64
    out=np.empty((ne1,ne0),dtype=np.float32)
    lut=np.array(E2M1,dtype=np.float32)
    for r in range(ne1):
        rb=mm[base+r*nb*36:base+(r+1)*nb*36]
        arr=np.frombuffer(rb,dtype=np.uint8).reshape(nb,36)
        sc=np.array([[ue4m3(v) for v in row[:4]] for row in arr],dtype=np.float32)
        qs=arr[:,4:]
        lo=(qs&0xF).reshape(nb,4,8); hi=(qs>>4).reshape(nb,4,8)
        vals=np.empty((nb,4,16),dtype=np.float32)
        vals[:,:,:8]=lut[lo]; vals[:,:,8:]=lut[hi]
        out[r]=(sc[:,:,None]*vals).reshape(-1)
    return torch.from_numpy(out)
def read_f32(name):
    dims,ty,off=tensors[name]; assert ty==0
    n=1
    for d in dims: n*=d
    return torch.frombuffer(bytearray(mm[data_off+off:data_off+off+n*4]),dtype=torch.float32).clone().reshape(tuple(dims[::-1]) if len(dims)>1 else (dims[0],))
def read_bf16(name):
    dims,ty,off=tensors[name]; assert ty==30
    n=1
    for d in dims: n*=d
    t=torch.frombuffer(bytearray(mm[data_off+off:data_off+off+n*2]),dtype=torch.bfloat16).clone()
    return t.reshape(tuple(dims[::-1]) if len(dims)>1 else (dims[0],)).float()
t0=time.time()
print("dequant blk.0 ...",flush=True)
Wqkv=read_nvfp4("blk.0.attn_qkv.weight"); Wz=read_nvfp4("blk.0.attn_gate.weight")
Wout=read_nvfp4("blk.0.ssm_out.weight"); Wb=read_nvfp4("blk.0.ssm_beta.weight")
Wa=read_f32("blk.0.ssm_alpha.weight"); A=read_f32("blk.0.ssm_a")
DT=read_f32("blk.0.ssm_dt.bias"); Wc=read_f32("blk.0.ssm_conv1d.weight")
Wn=read_f32("blk.0.ssm_norm.weight"); Wg=read_nvfp4("blk.0.ffn_gate.weight")
Wu=read_nvfp4("blk.0.ffn_up.weight"); Wd=read_nvfp4("blk.0.ffn_down.weight")
LNin=read_f32("blk.0.attn_norm.weight"); LNpa=read_f32("blk.0.post_attention_norm.weight")
AN=read_f32("blk.0.attn_norm.weight")
emb=read_bf16("token_embd.weight")
print(f"loaded in {time.time()-t0:.1f}s",flush=True)
Hid=5120; Kd=2048; Vd=6144; C=2*Kd+Vd; Hv=48; Khd=128; Vhd=128
x=emb[[0,1,2,3]]  # [4,5120]
def rms(x,w):
    return x*torch.rsqrt((x.float()**2).mean(-1,keepdim=True)+1e-6)*(1+w)
h=rms(x,LNin)
qkv=h@Wqkv.T  # [4,10240]
# causal depthwise conv k=4, no pad state (prefill start) + silu
qkv_t=qkv.T.unsqueeze(0)  # [1,C,T]
pad=torch.zeros(1,C,3)
xp=torch.cat([pad,qkv_t],dim=2)
wconv=Wc.reshape(C,1,4)
conv=torch.nn.functional.conv1d(xp,wconv,groups=C).squeeze(0).T
conv=torch.nn.functional.silu(conv)
q,k,v=torch.split(conv,[Kd,Kd,Vd],dim=-1)
q=q.reshape(4,16,128).repeat_interleave(3,dim=1); k=k.reshape(4,16,128).repeat_interleave(3,dim=1)
v=v.reshape(4,48,128)
z=(h@Wz.T).reshape(4,48,128)
beta=torch.sigmoid(h@Wb.T)
g=(-A.exp()*torch.nn.functional.softplus((h@Wa.T)+DT))
qn=q/((q**2).sum(-1,keepdim=True)+1e-6).sqrt(); kn=k/((k**2).sum(-1,keepdim=True)+1e-6).sqrt()
qn=qn*(Khd**-0.5)
S=torch.zeros(48,128,128)
outs=[]
for t in range(4):
    S=S*torch.exp(g[t])[:,None,None]
    bv=(v[t]-(S*kn[t][:,:,None]).sum(1))*beta[t][:,None]
    S=S+kn[t][:,:,None]*bv[:,None,:]
    outs.append((S*qn[t][:,:,None]).sum(1))
o=torch.stack(outs)  # [4,48,128]
n=o*torch.rsqrt((o**2).mean(-1,keepdim=True)+1e-6)
gated=n*Wn*torch.nn.functional.silu(z)
core=gated.reshape(4,6144)@Wout.T
h1=x+core  # residual over pre-norm input
m=rms(h1,LNpa)
gate=torch.nn.functional.silu(m@Wg.T)*(m@Wu.T)
y=h1+(gate@Wd.T)
print("blk.0 out mean/std/min/max:", y.mean().item(), y.std().item(), y.min().item(), y.max().item())
print("finite:", torch.isfinite(y).all().item())
assert torch.isfinite(y).all() and y.std().item()>1e-6
print("F1 BLOCK0 REAL-WEIGHT PASS")
