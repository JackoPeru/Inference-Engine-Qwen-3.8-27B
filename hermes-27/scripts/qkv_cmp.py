import struct, mmap, math, torch
import numpy as np
path = "/home/matteo/hermes-27/models/Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf"
f = open(path, 'rb')
f.read(4)
struct.unpack('<I', f.read(4))
nt = struct.unpack('<Q', f.read(8))[0]
nkv = struct.unpack('<Q', f.read(8))[0]
def rs():
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n)
for _ in range(nkv):
    k = rs()
    vt = struct.unpack('<I', f.read(4))[0]
    if vt in (0, 1, 7):
        f.read(1)
    elif vt in (2, 3):
        f.read(2)
    elif vt in (4, 5, 6):
        f.read(4)
    elif vt == 8:
        rs()
    elif vt == 9:
        at = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        if at == 8:
            for __ in range(n):
                rs()
        else:
            f.seek(n * {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[at], 1)
    else:
        f.read(8)
TENS = {}
for _ in range(nt):
    k = rs().decode()
    nd = struct.unpack('<I', f.read(4))[0]
    dims = [struct.unpack('<Q', f.read(8))[0] for __ in range(nd)]
    ty = struct.unpack('<I', f.read(4))[0]
    off = struct.unpack('<Q', f.read(8))[0]
    TENS[k] = (dims, ty, off)
DO = ((f.tell() + 31) // 32) * 32
mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
def ue(x):
    x = int(x)
    if x == 0 or x == 0x7F:
        return 0.0
    e = (x >> 3) & 0xF
    m = x & 0x7
    return math.ldexp(float(m), -9) if e == 0 else math.ldexp(1.0 + m / 8.0, e - 7)
def get(name):
    dims, ty, off = TENS[name][0], TENS[name][1], TENS[name][2]
    ne0 = dims[0]
    ne1 = dims[1] if len(dims) > 1 else 1
    base = DO + off
    if ty == 30:
        n = ne0 * ne1
        return torch.frombuffer(bytearray(mm[base:base + n * 2]), dtype=torch.bfloat16).clone().reshape(ne1, ne0).float().numpy()
    if ty == 0:
        n = ne0 * ne1
        return torch.frombuffer(bytearray(mm[base:base + n * 4]), dtype=torch.float32).clone().reshape(ne1, ne0).numpy()
    assert ty == 40
    nb = ne0 // 64
    out = np.empty((ne1, ne0), dtype=np.float32)
    for r in range(ne1):
        rb = mm[base + r * nb * 36:base + (r + 1) * nb * 36]
        a = np.frombuffer(rb, dtype=np.uint8).reshape(nb, 36)
        sc = np.array([[ue(v) for v in row[:4]] for row in a], dtype=np.float32)
        qs = a[:, 4:]
        lo = (qs & 0xF).reshape(nb, 4, 8)
        hi = (qs >> 4).reshape(nb, 4, 8)
        vals = np.empty((nb, 4, 16), dtype=np.float32)
        vals[:, :, :8] = E2M1[lo]
        vals[:, :, 8:] = E2M1[hi]
        out[r] = (sc[:, :, None] * vals).reshape(-1)
    return out
def rms(x, w):
    return x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-6) * w
ids = [6205, 161421, 303, 220, 17, 1374, 9911, 1147, 27053, 6059, 13]
emb = get("token_embd.weight")
x = emb[ids]
h = rms(x, get("blk.0.attn_norm.weight"))
Wq = get("blk.0.attn_qkv.weight")
qkv = h.astype(np.float64) @ Wq.T.astype(np.float64)
np.save("/tmp/e1_qkv.npy", qkv.astype(np.float32))
print("saved e1 qkv", qkv.shape, "mean/std:", qkv.mean(), qkv.std())
