"""Tiny full-attn block: transformers Qwen3_5DecoderLayer(full) vs numpy replica."""
import torch, math, numpy as np
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config, Qwen3_5DecoderLayer
torch.manual_seed(7)
cfg = Qwen3_5Config(hidden_size=128, intermediate_size=256, num_hidden_layers=1,
    layer_types=["full_attention"], num_attention_heads=4, num_key_value_heads=2,
    head_dim=32, rms_norm_eps=1e-6, hidden_act="silu", output_gate_type="swish",
    attention_dropout=0.0, attention_bias=False,
    attn_output_gate=True, rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
    "partial_rotary_factor": 0.25}, max_position_embeddings=1024)
lyr = Qwen3_5DecoderLayer(cfg, 0).float().eval()
sd = {k: v.clone().float() for k, v in lyr.state_dict().items()}
x = torch.randn(1, 8, 128) * 0.5
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
rot = Qwen3_5TextRotaryEmbedding(cfg)
pids = torch.arange(8).unsqueeze(0).unsqueeze(0).expand(3, 1, 8).clone()
with torch.no_grad():
    cos, sin = rot(x, pids)
    cmask = torch.full((8, 8), float("-inf"))
    cmask = torch.triu(cmask, diagonal=1).unsqueeze(0).unsqueeze(0)
    y = lyr(x, position_embeddings=(cos, sin), attention_mask=cmask)
    y_ref = y[0] if isinstance(y, tuple) else y
print("ref mean/std:", y_ref.mean().item(), y_ref.std().item())
cn = cos[0].numpy()
sn = sin[0].numpy()
print("cos shape:", cn.shape, "sample:", cn.flat[:4])
Hid, NQ, NKV, HD, T, RD = 128, 4, 2, 32, 8, cn.shape[-1]
X = x[0].numpy()
def rms(a, w):
    return a / np.sqrt((a.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-6) * (1 + w)
h = rms(X, sd["input_layernorm.weight"].numpy())
qq = h @ sd["self_attn.q_proj.weight"].numpy().T
hq = NQ * HD
G, Q = qq[:, hq:], qq[:, :hq]
Q = Q.reshape(T, NQ, HD)
K = (h @ sd["self_attn.k_proj.weight"].numpy().T).reshape(T, NKV, HD)
V = (h @ sd["self_attn.v_proj.weight"].numpy().T).reshape(T, NKV, HD)
qn = sd["self_attn.q_norm.weight"].numpy()
kn = sd["self_attn.k_norm.weight"].numpy()
Q = Q / np.sqrt((Q.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-6) * (1 + qn)
K = K / np.sqrt((K.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-6) * (1 + kn)
c = cn
s = sn
H2 = RD // 2
Qh = Q[:, :, :RD].copy()
Q[:, :, :RD] = np.concatenate([Qh[:, :, :H2] * c[:, None, :H2] - Qh[:, :, H2:] * s[:, None, :H2],
                               Qh[:, :, :H2] * s[:, None, :H2] + Qh[:, :, H2:] * c[:, None, :H2]], -1)
Kh = K[:, :, :RD].copy()
K[:, :, :RD] = np.concatenate([Kh[:, :, :H2] * c[:, None, :H2] - Kh[:, :, H2:] * s[:, None, :H2],
                               Kh[:, :, :H2] * s[:, None, :H2] + Kh[:, :, H2:] * c[:, None, :H2]], -1)
K6 = np.repeat(K, NQ // NKV, axis=1)
V6 = np.repeat(V, NQ // NKV, axis=1)
out = np.zeros((T, NQ, HD))
for t in range(T):
    scv = np.einsum("hd,shd->hs", Q[t], K6[:t + 1]) * (HD ** -0.5)
    scv = scv - scv.max(-1, keepdims=True)
    w = np.exp(scv)
    w = w / w.sum(-1, keepdims=True)
    out[t] = np.einsum("hs,shd->hd", w, V6[:t + 1])
att = out.reshape(T, -1) * (1 / (1 + np.exp(-G)))
Y = att @ sd["self_attn.o_proj.weight"].numpy().T
H1 = X + Y
h2 = rms(H1, sd["post_attention_layernorm.weight"].numpy())
g = (1 / (1 + np.exp(-(h2 @ sd["mlp.gate_proj.weight"].numpy().T)))) * (h2 @ sd["mlp.up_proj.weight"].numpy().T)
Y2 = H1 + g @ sd["mlp.down_proj.weight"].numpy().T
err = np.abs(Y2 - y_ref[0].numpy()).max()
print("attn block max err vs transformers:", err)
print("ATTN VERIFY", "PASS" if err < 1e-4 else "FAIL")
