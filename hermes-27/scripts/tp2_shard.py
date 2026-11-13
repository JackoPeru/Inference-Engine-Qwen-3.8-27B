"""TP2 sharding validation over real GGUF: split rules, divisibility, per-GPU bytes."""
import struct
path="/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf"
f=open(path,'rb')
f.read(4); struct.unpack('<I',f.read(4)); nt=struct.unpack('<Q',f.read(8))[0]; nkv=struct.unpack('<Q',f.read(8))[0]
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
        else: f.seek(n*{0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[at],1)
    else: f.read(8)
TS={0:4,1:2,30:2,39:17,40:36}
BL={0:1,1:1,30:1,39:32,40:64}
def row_bytes(ty,ne0): return ne0//BL[ty]*TS[ty]
# TP2 rule: column-split (out dim = ne1) needs ne1%2==0 AND (nvfp4: block alignment on ne0 unaffected);
# row-split (in dim ne0) needs ne0%2==0 and for nvfp4 ne0/2 % 64 == 0.
per_gpu=0; total=0; bad=[]
n_col=n_row=n_rep=0
for _ in range(nt):
    k=rs(); nd=struct.unpack('<I',f.read(4))[0]
    dims=[struct.unpack('<Q',f.read(8))[0] for __ in range(nd)]
    ty=struct.unpack('<I',f.read(4))[0]; off=struct.unpack('<Q',f.read(8))[0]
    ne0=dims[0]; ne1=dims[1] if nd>1 else 1
    nb=row_bytes(ty,ne0)*ne1; total+=nb
    name=k
    is_norm_vec=(k.endswith("q_norm.weight") or k.endswith("k_norm.weight"))
    col_split=(not is_norm_vec) and ("ffn_gate" in k or "ffn_up" in k or "attn_qkv" in k or "attn_q" in k
               or "attn_k" in k or "attn_v" in k or "in_proj" in k or "attn_gate" in k
               or "ssm_beta" in k or "ssm_alpha" in k or "eh_proj" in k)
    row_split=("ffn_down" in k or "attn_output" in k or "ssm_out" in k or "output.weight" in k)
    if col_split:
        n_col+=1
        if ne1%2!=0: bad.append((k,"ne1 odd"))
        per_gpu+=nb//2
    elif row_split:
        n_row+=1
        if ne0%2!=0: bad.append((k,"ne0 odd"))
        if ty==40 and (ne0//2)%64!=0: bad.append((k,"nvfp4 row-split misaligned"))
        per_gpu+=nb//2
    else:
        n_rep+=1  # norms, biases, embeddings replicated
        per_gpu+=nb
print(f"tensors col={n_col} row={n_row} replicated={n_rep}")
print(f"total file tensors bytes={total} ({total/1024**3:.2f} GiB)")
print(f"per-GPU with TP2 col/row split = {per_gpu} ({per_gpu/1024**3:.2f} GiB)")
print("BAD:", bad if bad else "none")
assert not bad
# embedding/output replicated: 2x2.54GB=5.08GB dominates replicated. Note for PP alternative.
print("TP2 SHARD PASS")
