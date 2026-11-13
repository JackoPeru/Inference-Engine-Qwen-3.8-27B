import struct
from collections import defaultdict
path="/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf"
f=open(path,'rb')
f.read(4); ver=struct.unpack('<I',f.read(4))[0]
nt=struct.unpack('<Q',f.read(8))[0]; nkv=struct.unpack('<Q',f.read(8))[0]
def rs():
    n=struct.unpack('<Q',f.read(8))[0]
    return f.read(n).decode()
kv={}
for _ in range(nkv):
    k=rs(); vt=struct.unpack('<I',f.read(4))[0]
    if vt in (0,1,7): v=struct.unpack('B',f.read(1))[0]; kv[k]=v
    elif vt in (2,3): v=struct.unpack('H',f.read(2))[0]; kv[k]=v
    elif vt in (4,5): v=struct.unpack('I',f.read(4))[0]; kv[k]=v
    elif vt==6: v=struct.unpack('f',f.read(4))[0]; kv[k]=v
    elif vt==8: kv[k]=rs()
    elif vt==9:
        at=struct.unpack('<I',f.read(4))[0]; n=struct.unpack('<Q',f.read(8))[0]
        if at==8:
            for __ in range(n): rs()
        else:
            sz={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[at]
            f.seek(n*sz,1)
        kv[k+'.len']=n
    else: v=struct.unpack('Q',f.read(8))[0]; kv[k]=v
assert kv['general.architecture']=='qwen35', kv['general.architecture']
assert kv['qwen35.block_count']==65, kv['qwen35.block_count']
assert kv['qwen35.embedding_length']==5120
assert kv['qwen35.feed_forward_length']==17408
assert kv['qwen35.context_length']==262144
assert kv['qwen35.attention.head_count']==24
assert kv['qwen35.attention.head_count_kv']==4
assert kv['qwen35.nextn_predict_layers']==1
blocks=defaultdict(list)
for _ in range(nt):
    k=rs(); nd=struct.unpack('<I',f.read(4))[0]
    dims=[struct.unpack('<Q',f.read(8))[0] for __ in range(nd)]
    ty=struct.unpack('<I',f.read(4))[0]; off=struct.unpack('<Q',f.read(8))[0]
    p=k.split('.')
    if p[0]=='blk': blocks[int(p[1])].append(('.'.join(p[2:]),ty,tuple(dims)))
lin=[b for b in range(64) if any(n=='attn_qkv.weight' for n,_,_ in blocks[b])]
full=[b for b in range(64) if any(n=='attn_q.weight' for n,_,_ in blocks[b])]
print(f"linear blocks ({len(lin)}): {lin[:8]}...")
print(f"full blocks ({len(full)}): {full}")
assert len(lin)==48 and len(full)==16, (len(lin),len(full))
assert full==[b for b in range(64) if b%4==3], full
# spot dims
assert ('attn_qkv.weight',40,(5120,10240)) in [(n,t,d) for n,t,d in blocks[0]]
assert ('attn_q.weight',40,(5120,12288)) in [(n,t,d) for n,t,d in blocks[3]]
assert ('attn_k.weight',40,(5120,1024)) in [(n,t,d) for n,t,d in blocks[3]]
assert ('ffn_down.weight',40,(17408,5120)) in [(n,t,d) for n,t,d in blocks[0]]
assert ('ssm_conv1d.weight',0,(4,10240)) in [(n,t,d) for n,t,d in blocks[0]]
assert ('nextn.eh_proj.weight',30,(10240,5120)) in [(n,t,d) for n,t,d in blocks[64]]
print("ARCH VALIDATION PASS: 48 linear + 16 full (every 4th) + MTP blk64, dims match config")
