#!/usr/bin/env python3
"""Test indipendente NVFP4 (no torch, no llama): verifica LUT E2M1 + UE4M3 + blocco."""
import struct, math
def ue4m3(x):
    if x==0 or x==0x7F: return 0.0
    exp=(x>>3)&0xF; man=x&0x7
    raw = math.ldexp(float(man),-9) if exp==0 else math.ldexp(1.0+man/8.0, exp-7)
    return raw*0.5
def e2m1(v):
    lut=[0,0.5,1,1.5,2,3,4,6,0,-0.5,-1,-1.5,-2,-3,-4,-6]
    return lut[v&0xF]
assert e2m1(1)==0.5 and e2m1(6)==4 and e2m1(14)==-4
assert ue4m3(0)==0.0 and abs(ue4m3(0x40)-1.0)<1e-6
# blocco: UE4M3 0x40 -> raw 2.0, ue4m3() ritorna raw*0.5=1.0; E2M1 nibble 1 = 0.5
# valore corretto NVFP4 = raw * E2M1 = 2.0*0.5 = 1.0 = (d*2)*e2m1
scales=bytes([0x40]*4); qs=bytes([0x11]*32)
out=[]
for sub in range(4):
    d=ue4m3(scales[sub])
    assert abs(d-1.0)<1e-6
    for j in range(8):
        q=qs[sub*8+j]
        out.append((d*2)*e2m1(q&0xF)); out.append((d*2)*e2m1(q>>4))
assert len(out)==64 and all(abs(v-1.0)<1e-6 for v in out), out[:8]
print("NVFP4 synth OK")
