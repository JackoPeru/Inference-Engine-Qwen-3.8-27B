import struct, sys
path = "/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf"
filt = sys.argv[1] if len(sys.argv) > 1 else "blk.64"
f = open(path, 'rb')
f.read(4); struct.unpack('<I', f.read(4))
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
TN = {0:'F32',1:'F16',30:'BF16',40:'NVFP4'}
tot = 0
for _ in range(nt):
    k = rs().decode(); nd = struct.unpack('<I', f.read(4))[0]
    dims = [struct.unpack('<Q', f.read(8))[0] for __ in range(nd)]
    ty = struct.unpack('<I', f.read(4))[0]; off = struct.unpack('<Q', f.read(8))[0]
    if filt in k:
        print(f"{k} ne={dims} type={TN.get(ty,ty)} off={off}")
        tot += 1
print(f"match={tot} total={nt}")
