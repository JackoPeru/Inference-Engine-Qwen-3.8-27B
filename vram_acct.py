"""Leggero: solo header GGUF, come tp2_shard.py. Nessun peso caricato."""
import struct
path = "/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf"
f = open(path, 'rb')
f.read(4); struct.unpack('<I', f.read(4))
nt = struct.unpack('<Q', f.read(8))[0]
nkv = struct.unpack('<Q', f.read(8))[0]
def rs():
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode()
for _ in range(nkv):
    k = rs(); vt = struct.unpack('<I', f.read(4))[0]
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
TS={0:4,1:2,30:2,39:17,40:36}
BL={0:1,1:1,30:1,39:32,40:64}
def row_bytes(ty,ne0): return ne0//BL[ty]*TS[ty]
def nbytes(ty,dims):
    ne0=dims[0]; ne1=dims[1] if len(dims)>1 else 1
    return row_bytes(ty,ne0)*ne1
tensors=[]
total=0
for _ in range(nt):
    k=rs(); nd=struct.unpack('<I',f.read(4))[0]
    dims=[struct.unpack('<Q',f.read(8))[0] for __ in range(nd)]
    ty=struct.unpack('<I',f.read(4))[0]; off=struct.unpack('<Q',f.read(8))[0]
    nb=nbytes(ty,dims)
    total+=nb
    tensors.append((k,ty,dims,nb))
print(f"TOTAL tensors={len(tensors)} bytes={total} GiB={total/1024**3:.4f} GB={total/1e9:.4f}")
# breakdown per tipo
from collections import Counter, defaultdict
by_type=defaultdict(int)
for k,ty,dims,nb in tensors:
    by_type[ty]+=nb
TN={0:'F32',1:'F16',30:'BF16',40:'NVFP4'}
for ty,nb in sorted(by_type.items()):
    print(f" type {TN.get(ty,ty)}: {nb} bytes = {nb/1024**3:.4f} GiB")
# TP2 logic (copia tp2_shard.py)
per_gpu=0; n_col=n_row=n_rep=0; rep_bytes=0
rep_list=[]
for k,ty,dims,nb in tensors:
    ne0=dims[0]; ne1=dims[1] if len(dims)>1 else 1
    is_norm_vec=(k.endswith("q_norm.weight") or k.endswith("k_norm.weight"))
    col_split=(not is_norm_vec) and ("ffn_gate" in k or "ffn_up" in k or "attn_qkv" in k or "attn_q" in k
               or "attn_k" in k or "attn_v" in k or "in_proj" in k or "attn_gate" in k
               or "ssm_beta" in k or "ssm_alpha" in k or "eh_proj" in k)
    row_split=("ffn_down" in k or "attn_output" in k or "ssm_out" in k or "output.weight" in k)
    if col_split: n_col+=1; per_gpu+=nb//2
    elif row_split: n_row+=1; per_gpu+=nb//2
    else: n_rep+=1; per_gpu+=nb; rep_bytes+=nb; rep_list.append((k,nb))
print(f"TP2 col={n_col} row={n_row} rep={n_rep}")
print(f"TP2 per_gpu={per_gpu} GiB={per_gpu/1024**3:.4f}")
print(f"TP2 replicated total={rep_bytes} GiB={rep_bytes/1024**3:.4f}")
rep_list_sorted=sorted(rep_list,key=lambda x:-x[1])[:12]
for k,nb in rep_list_sorted:
    print(f"  rep {k}: {nb} MiB={nb/1024**2:.2f}")
# PP splits
def blk_index(k):
    # ritorna 0-64, oppure -1 per token_embd, 99 per output*
    if k.startswith("token_embd"): return -1
    if k.startswith("output"): return 99
    if k.startswith("blk."):
        try: return int(k.split(".")[1])
        except: return None
    return None
def pp_sum(pred):
    s=sum(nb for k,ty,dims,nb in tensors if pred(k))
    return s
# Opzione A: GPU0 = embed + blk0-31, GPU1 = blk32-63 + blk64 + output
ppA0=pp_sum(lambda k: blk_index(k)==-1 or (blk_index(k) is not None and 0<=blk_index(k)<=31))
ppA1=pp_sum(lambda k: (blk_index(k) is not None and 32<=blk_index(k)<=64) or blk_index(k)==99)
# Opzione B: GPU0 = embed + blk0-32, GPU1 = blk33-63 + blk64 + output (sposta un layer per bilanciare MTP)
ppB0=pp_sum(lambda k: blk_index(k)==-1 or (blk_index(k) is not None and 0<=blk_index(k)<=32))
ppB1=pp_sum(lambda k: (blk_index(k) is not None and 33<=blk_index(k)<=64) or blk_index(k)==99)
# Opzione C: GPU0 = embed + blk0-30, GPU1 = resto (se GPU1 troppo piena con MTP+head)
ppC0=pp_sum(lambda k: blk_index(k)==-1 or (blk_index(k) is not None and 0<=blk_index(k)<=30))
ppC1=pp_sum(lambda k: (blk_index(k) is not None and 31<=blk_index(k)<=64) or blk_index(k)==99)
for name,a0,a1 in [("PP-A(0-31|32-63+MTP+head)",ppA0,ppA1),("PP-B(0-32|33-63+MTP+head)",ppB0,ppB1),("PP-C(0-30|31-63+MTP+head)",ppC0,ppC1)]:
    print(f"{name}: GPU0={a0} ({a0/1024**3:.4f} GiB) GPU1={a1} ({a1/1024**3:.4f} GiB) max={max(a0,a1)/1024**3:.4f} GiB")
# dettaglio embed/output/MTP/blocchi
emb=sum(nb for k,ty,dims,nb in tensors if k.startswith("token_embd"))
out=sum(nb for k,ty,dims,nb in tensors if k.startswith("output"))
mtp=sum(nb for k,ty,dims,nb in tensors if blk_index(k)==64)
print(f"embed={emb} ({emb/1024**3:.4f} GiB) output+norm={out} ({out/1024**3:.4f} GiB) mtp_blk64={mtp} ({mtp/1024**3:.4f} GiB)")
# media per layer 0-63
for i in [0,3,64]:
    s=sum(nb for k,ty,dims,nb in tensors if blk_index(k)==i)
    print(f" blk.{i} total={s} MiB={s/1024**2:.2f}")
# somma layer 0-63
sum64=sum(nb for k,ty,dims,nb in tensors if blk_index(k) is not None and 0<=blk_index(k)<=63)
print(f"sum blk0-63={sum64} GiB={sum64/1024**3:.4f} avg/layer MiB={sum64/64/1024**2:.2f}")
# file size check
import os
print(f"file size={os.path.getsize(path)} bytes GiB={os.path.getsize(path)/1024**3:.4f}")
