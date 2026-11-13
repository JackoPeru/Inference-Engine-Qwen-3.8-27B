"""Host-pinned allreduce microbench (no P2P): D2H + host add + H2D on GPU1."""
import torch, time
torch.cuda.set_device(1)
def bench(n, iters=20):
    a=torch.randn(n,device='cuda:1')
    b=torch.randn(n,device='cuda:1')
    buf=torch.empty(n,dtype=torch.float32,pin_memory=True)
    torch.cuda.synchronize(1)
    t0=time.perf_counter()
    for _ in range(iters):
        buf.copy_(a,non_blocking=True); torch.cuda.synchronize(1)
        # host-side add emulating peer contribution
        buf.add_(1.0)
        b.copy_(buf,non_blocking=True); torch.cuda.synchronize(1)
    t1=time.perf_counter()
    dt=(t1-t0)/iters
    gb=(n*4*2/1e9)/dt
    print(f"n={n} ({n*4/1024:.0f}KB) roundtrip={dt*1e6:.1f}us eff_bw={gb:.1f}GB/s")
bench(5120)       # 1 token hidden
bench(5120*32)    # 32 tokens
bench(5120*512)   # 512 tokens
bench(5120*8192)  # 8k chunk
