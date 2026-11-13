import struct, mmap
import numpy as np
path = "/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf"
f = open(path, 'rb'); f.read(4); struct.unpack('<I', f.read(4))
nt = struct.unpack('<Q', f.read(8))[0]; nkv = struct.unpack('<Q', f.read(8))[0]
def rs():
    n = struct.unpack('<Q', f.read(8))[0]; return f.read(n)
for _ in range(nkv):
    k = rs(); vt = struct.unpack('<I', f.read(4))[0]
    if vt in (0, 1, 7): f.read(1)
    elif vt in (2, 3): f.read(2)
    elif vt in (4, 5, 6): f.read(4)
    elif vt == 8: rs()
    elif vt == 9:
        at = struct.unpack('<I', f.read(4))[0]; n = struct.unpack('<Q', f.read(8))[0]
        if at == 8:
            for __ in range(n): rs()
        else: f.seek(n * {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[at], 1)
    else: f.read(8)
names = ["blk.0.attn_norm.weight", "blk.3.attn_q_norm.weight",
         "blk.64.attn_norm.weight", "blk.64.attn_q_norm.weight",
         "blk.64.nextn.hnorm.weight", "blk.64.nextn.enorm.weight",
         "blk.64.nextn.shared_head_norm.weight", "output_norm.weight",
         "blk.0.ssm_norm.weight"]
want = set(names)
found = {}
for _ in range(nt):
    k = rs().decode(); nd = struct.unpack('<I', f.read(4))[0]
    dims = [struct.unpack('<Q', f.read(8))[0] for __ in range(nd)]
    ty = struct.unpack('<I', f.read(4))[0]; off = struct.unpack('<Q', f.read(8))[0]
    if k in want:
        found[k] = (dims, ty, off)
DO = ((f.tell() + 31) // 32) * 32
mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
for k in names:
    dims, ty, off = found[k]
    n = 1
    for d in dims: n *= d
    assert ty == 0, (k, ty)
    a = np.frombuffer(mm[DO+off:DO+off+n*4], dtype=np.float32).copy()
    print(f"{k}: mean={a.mean():+.5f} std={a.std():.5f} min={a.min():+.4f} max={a.max():+.4f}")
