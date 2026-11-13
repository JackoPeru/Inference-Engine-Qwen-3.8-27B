import numpy as np, os
def load(p, n): return np.fromfile(p, dtype=np.float32)
pairs = [
 ("qkv","/tmp/e1_cv_qkv.bin","/tmp/e2_qkv.bin"),
 ("cv","/tmp/e1_cv.bin","/tmp/e2_cv.bin"),
 ("beta","/tmp/e1_beta.bin","/tmp/e2_beta.bin"),
 ("gl","/tmp/e1_gl.bin","/tmp/e2_gl.bin"),
 ("o","/tmp/e1_o.bin","/tmp/e2_o.bin"),
 ("core","/tmp/e1_core.bin","/tmp/e2_core.bin"),
 ("y","/tmp/e1_y.bin","/tmp/e2_y.bin"),
]
for name,a,b in pairs:
    try:
        A=load(a,None); B=load(b,None)
        assert A.shape==B.shape, f"{name} shape {A.shape} vs {B.shape}"
        d=np.abs(A-B)
        print(f"{name:5s} n={A.size} maxerr={d.max():.3e} meanerr={d.mean():.3e} Amax={np.abs(A).max():.3f} Bmax={np.abs(B).max():.3f}")
        idx=np.argmax(d)
        if d.max()>1e-4:
            print(f"  first big: idx={idx} A={A.flat[idx]:.6f} B={B.flat[idx]:.6f}")
            # show first 5 of each at start
            print(f"  A[:5]={A.flat[:5]} B[:5]={B.flat[:5]}")
    except Exception as e:
        print(name, "ERR", e)
