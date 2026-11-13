#!/usr/bin/env python3
"""CPU-only Q8_1 x raw-GGUF NVFP4 DP4A lane-map contract.

No GPU, CUDA Python, torch, numpy or third-party dependencies.
Checks the precise mapping used by the experimental kernel:
  - one 32-lane warp reads four consecutive 64-value GGUF blocks;
  - lane>>3 selects a 64-block, (lane&7)>>1 selects a subscale;
  - lane&1 selects packed nibble bytes [0..3] or [4..7];
  - each lane packs 4 signed E2M1*2 values for DP4A twice;
  - the same raw GGUF words are reused for 8 Q8 token rows;
  - row-major T-by-N indexing and T-tail do not alias.

This is a source/mapping oracle only; no GPU performance/quality proof.
"""

import math
import random
import struct

from q8_1_cpu_test import quant_ref

E2_X2 = (0, 1, 2, 3, 4, 6, 8, 12,
         0, -1, -2, -3, -4, -6, -8, -12)
E2_FP32 = tuple(i * 0.5 for i in E2_X2)
RNG = random.Random(0x271E1)


def scale_ue4m3(c: int) -> float:
    if c in (0, 127):
        return 0.
    if c >> 3 == 0:
        return math.ldexp(float(c & 7), -9)
    return math.ldexp(1. + (c & 7) / 8., (c >> 3) - 7)


def f16_d(block: bytes) -> float:
    return struct.unpack_from("<e", block, 0)[0]


def as_uint32_le(raw: bytes) -> int:
    assert len(raw) == 4
    return int.from_bytes(raw, "little", signed=False)


def pack_fp4x2_twice_to_i8x4(packed_fp4: int, high_nibble: bool) -> int:
    packed = 0
    for i in range(4):
        code = (packed_fp4 >> (i * 8 + (4 if high_nibble else 0))) & 15
        packed |= (E2_X2[code] & 255) << (8*i)
    return packed


def dp4a_signed(a: int, b: int, acc: int) -> int:
    for i in range(4):
        av = (a >> (8*i)) & 255
        bv = (b >> (8*i)) & 255
        if av >= 128:
            av -= 256
        if bv >= 128:
            bv -= 256
        acc += av * bv
    return acc


def reference_output(q8_rows: list[list[bytes]], wr: bytes,
                     t: int, row: int, K: int) -> float:
    """Independent FP64 decoded direct dot across all 64 GGUF weights."""
    blocks_per_row=K//64
    q=q8_rows[t]
    answer=0.
    for b in range(blocks_per_row):
        blk=wr[(row*blocks_per_row+b)*36:
               (row*blocks_per_row+b+1)*36]
        for s in range(4):
            d=f16_d(q[b*2+s//2])
            sf=scale_ue4m3(blk[s])
            for j in range(16):
                packed=blk[4+s*8+j%8]
                code=(packed&15) if j<8 else packed>>4
                qval=q[b*2+s//2][4+(s%2)*16+j]
                qval=qval-256 if qval>=128 else qval
                answer+=qval*sf*E2_FP32[code]*d
    return answer


def dp4a_lane_output(q8_rows: list[list[bytes]], wr: bytes,
                     t: int, row: int, K: int) -> float:
    """Simulate one warp's 32 lane GGUF/Q8 addressing and 8-token reuse."""
    assert K%256==0
    blocks_per_row=K//64
    partial=[0.]*32
    for bbase in range(0,blocks_per_row,4):
        for lane in range(32):
            group=lane>>3
            sub=(lane&7)>>1
            chunk=lane&1
            block_index=bbase+group
            woffset=(row*blocks_per_row+block_index)*36
            w=wr[woffset:woffset+36]
            raw4=as_uint32_le(w[4+sub*8+chunk*4:
                                8+sub*8+chunk*4])
            lo=pack_fp4x2_twice_to_i8x4(raw4,False)
            hi=pack_fp4x2_twice_to_i8x4(raw4,True)
            q=q8_rows[t][block_index*2+sub//2]
            qoffset=4+(sub&1)*16+chunk*4
            qa_lo=as_uint32_le(q[qoffset:qoffset+4])
            qa_hi=as_uint32_le(q[qoffset+8:qoffset+12])
            dot=dp4a_signed(lo,qa_lo,0)
            dot=dp4a_signed(hi,qa_hi,dot)
            partial[lane]+=dot*scale_ue4m3(w[sub])*0.5*f16_d(q)
    # CUDA warp shfl_down tree (only lane0's final value is used).
    for step in (16,8,4,2,1):
        partial=[(partial[i]+partial[i+step])
                 if i+step<32 else partial[i]
                 for i in range(32)]
    return partial[0]


def main() -> None:
    packed_trials=0
    for packed in (0, 0xFFFFFFFF, 0x01234567, 0x89ABCDEF):
        for high in (False,True):
            out=pack_fp4x2_twice_to_i8x4(packed,high)
            actual=[(out>>(i*8))&255 for i in range(4)]
            actual=[v-256 if v>=128 else v for v in actual]
            expected=[E2_X2[(packed>>(i*8+(4 if high else 0)))&15]
                      for i in range(4)]
            assert actual==expected,(packed,high,actual,expected)
            packed_trials+=1

    # Exhaust all signed weight codes against representative Q8 int8s.
    for nib in range(16):
        for qi in (-127,-101,-1,0,1,64,127):
            packed=(nib | nib<<4)*0x01010101
            qbytes=bytes([qi&255]*4)
            a=pack_fp4x2_twice_to_i8x4(packed,False)
            b=as_uint32_le(qbytes)
            assert dp4a_signed(a,b,0)==4*E2_X2[nib]*qi
            packed_trials+=1
    print(f"PASS signed DP4A nibble to int8 packing: {packed_trials} cases")

    shapes=0
    checked=0
    for K in (256,512,1024):
        for T in (1,7,8,9,16,64):
            N=5 if T<=16 else 2
            blocks_per_row=K//64
            # Force corner scales: 0 and 0x7F -> 0, all other values raw.
            wr=bytearray(RNG.randrange(256) for _ in range(N*blocks_per_row*36))
            for b in range(N*blocks_per_row):
                if b%5==0:
                    wr[b*36]=127
                if b%7==0:
                    wr[b*36+1]=0
            q8_rows=[]
            for t in range(T):
                floats=[RNG.uniform(-5,5) for _ in range(K)]
                q8_rows.append([quant_ref(floats[b*32:b*32+32])
                                for b in range(K//32)])
            result=[None]*(T*N)
            for first_t in range(0,T,8):
                for row in range(N):
                    for off in range(8):
                        t=first_t+off
                        if t>=T:
                            continue
                        result[t*N+row]=dp4a_lane_output(q8_rows,wr,t,row,K)
            assert all(x is not None for x in result)
            for t in range(T):
                for row in range(N):
                    expected=reference_output(q8_rows,wr,t,row,K)
                    got=result[t*N+row]
                    if not math.isclose(got,expected,rel_tol=1e-12,abs_tol=1e-9):
                        raise AssertionError((K,T,N,t,row,got,expected))
                    checked+=1
            shapes+=1
    print(f"PASS 8-token warp reuse, tail guards, output N stride: "
          f"{checked} numeric outputs, {shapes} shapes")
    # Repeat deterministic reference 10 times, no state mutation.
    qrows=[[quant_ref([RNG.uniform(-1,1) for _ in range(32)])
            for _ in range(8)]]
    raw=bytes(RNG.randrange(256) for _ in range(4*36))
    baseline=dp4a_lane_output(qrows,raw,0,0,256)
    for _ in range(10):
        assert dp4a_lane_output(qrows,raw,0,0,256)==baseline
    print("PASS ten-repeat CPU deterministic DP4A map")
    print("CPU_ONLY_DP4A_CONTRACT_PASS: no GPU execution/performance evidence")


if __name__=="__main__":
    main()
