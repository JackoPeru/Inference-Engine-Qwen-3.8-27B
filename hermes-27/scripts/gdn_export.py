"""Export tiny GDN block weights+io to flat binary for C++ test."""
import torch, struct
d = torch.load("/tmp/gdn_tiny.pt", weights_only=False)
sd = d["sd"]; x = d["x"].float(); y = d["y_pre"].float()
order = ["in_proj_qkv.weight","in_proj_z.weight","in_proj_b.weight","in_proj_a.weight",
         "conv1d.weight","A_log","dt_bias","norm.weight","out_proj.weight"]
with open("/tmp/gdn_tiny.bin","wb") as f:
    f.write(struct.pack("8i", 1,16,128,2,16,4,16,4))  # B,T,Hid,Hk,Khd,Hv,Vhd,convK
    for k in order:
        t = sd[k].float().contiguous().numpy()
        f.write(struct.pack("i", t.ndim))
        f.write(struct.pack(t.ndim*"i", *t.shape))
        f.write(t.tobytes())
    for name,t in [("x",x.numpy()),("y",y.numpy())]:
        a = t.astype("<f4")
        f.write(struct.pack("i", a.ndim))
        f.write(struct.pack(a.ndim*"i", *a.shape))
        f.write(a.tobytes())
print("exported /tmp/gdn_tiny.bin")
