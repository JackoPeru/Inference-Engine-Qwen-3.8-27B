import struct, mmap, math
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
            sz={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[at]
            f.seek(n*sz,1)
    else: f.read(8)
data_off=None; tensors={}
for _ in range(nt):
    k=rs(); nd=struct.unpack('<I',f.read(4))[0]
    dims=[struct.unpack('<Q',f.read(8))[0] for __ in range(nd)]
    ty=struct.unpack('<I',f.read(4))[0]; off=struct.unpack('<Q',f.read(8))[0]
    tensors[k]=(dims,ty,off)
    if data_off is None:
        cur=f.tell()
        # alignment 32
        data_off=((cur+31)//32)*32
emb=tensors['token_embd.weight']
print("token_embd:",emb)
assert emb[1]==30  # BF16
ne0,ne1=emb[0]
mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
base=data_off+emb[2]
def bf16_row(tok):
    o=base+tok*ne0*2
    raw=mm[o:o+ne0*2]
    assert len(raw)==ne0*2
    return raw
import numpy as np
for tok in [0,1,248044]:
    raw=bf16_row(tok)
    arr=np.frombuffer(raw,dtype=np.uint16).astype(np.uint32)
    # bf16 -> fp32: shift 16
    f32=(arr<<16).view(np.uint8)
    vals=np.frombuffer(raw,dtype='>u2')
    # use struct for proper conversion
    import struct as st
    vals=[st.unpack('<e',st.pack('<H',int(x)))[0] if False else 0 for x in []]
    # simpler: torch if available
    try:
        import torch
        t=torch.frombuffer(bytearray(raw),dtype=torch.bfloat16).float()
        print(f"tok {tok}: mean={t.mean():.5f} std={t.std():.5f} min={t.min():.5f} max={t.max():.5f} finite={torch.isfinite(t).all().item()}")
        assert torch.isfinite(t).all()
    except ImportError:
        print(f"tok {tok}: {len(raw)} bytes OK (torch missing)")
print("EMBED GATHER PASS")
