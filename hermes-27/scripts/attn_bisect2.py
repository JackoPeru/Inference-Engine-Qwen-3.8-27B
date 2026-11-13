"""Bisect: torch Q/K/V/G vs numpy Q/K/V/G (same weights/inputs)."""
import torch, numpy as np
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config, Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding, apply_rotary_pos_emb
torch.manual_seed(7)
cfg = Qwen3_5Config(hidden_size=128, intermediate_size=256, num_hidden_layers=1,
    layer_types=["full_attention"], num_attention_heads=4, num_key_value_heads=2,
    head_dim=32, rms_norm_eps=1e-6, hidden_act="silu", output_gate_type="swish",
    attn_output_gate=True, rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
    "partial_rotary_factor": 0.25}, max_position_embeddings=1024,
    attention_dropout=0.0, attention_bias=False)
lyr = Qwen3_5DecoderLayer(cfg, 0).float().eval()
sd = {k: v.clone().float() for k, v in lyr.state_dict().items()}
x = torch.randn(1, 8, 128) * 0.5
rot = Qwen3_5TextRotaryEmbedding(cfg)
pids = torch.arange(8).unsqueeze(0).unsqueeze(0).expand(3, 1, 8).clone()
with torch.no_grad():
    cos, sin = rot(x, pids)
    h = lyr.input_layernorm(x)
    qq = lyr.self_attn.q_proj(h)
    Q_t, G_t = qq.split(128, -1)
    Q_t = lyr.self_attn.q_norm(Q_t.reshape(1, 8, 4, 32)).transpose(1, 2)
    K_t = lyr.self_attn.k_norm((h @ sd["self_attn.k_proj.weight"].T).reshape(1, 8, 2, 32)).transpose(1, 2)
    V_t = (h @ sd["self_attn.v_proj.weight"].T).reshape(1, 8, 2, 32).transpose(1, 2)
    Q_t, K_t = apply_rotary_pos_emb(Q_t, K_t, cos, sin)
    Qt, Kt, Vt, Gt = Q_t[0].numpy(), K_t[0].numpy(), V_t[0].numpy(), G_t[0].numpy()
Hid, NQ, NKV, HD, T = 128, 4, 2, 32, 8
X = x[0].numpy()
def rms(a, w):
    return a / np.sqrt((a.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-6) * (1 + w)
h8 = rms(X, sd["input_layernorm.weight"].numpy())
qq8 = h8 @ sd["self_attn.q_proj.weight"].numpy().T
G8, Q8 = qq8[:, 128:], qq8[:, :128]
Q8 = Q8.reshape(T, NQ, HD)
K8 = (h8 @ sd["self_attn.k_proj.weight"].numpy().T).reshape(T, NKV, HD)
V8 = (h8 @ sd["self_attn.v_proj.weight"].numpy().T).reshape(T, NKV, HD)
qn = sd["self_attn.q_norm.weight"].numpy()
kn = sd["self_attn.k_norm.weight"].numpy()
Q8 = Q8 / np.sqrt((Q8.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-6) * (1 + qn)
K8 = K8 / np.sqrt((K8.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-6) * (1 + kn)
print("cos full shape:", tuple(cos.shape))
print("cos full shape:", tuple(cos.shape))
# mimic layer exactly: cos.unsqueeze(1) broadcast over [B,H,T,D]
Qb = Q8.transpose(1, 0, 2)[None, :, :, :]  # [1,H,T,D]
Kb = K8.transpose(1, 0, 2)[None, :, :, :]
RD = cos.shape[-1]
cs = cos.numpy()
ss = sin.numpy()
# emulate unsqueeze(1): insert axis 1
cs1 = np.expand_dims(cs, 1)
ss1 = np.expand_dims(ss, 1)
H2 = RD // 2
cH = cs1[..., :H2]
sH = ss1[..., :H2]
QH = Qb[..., :RD].copy()
Qb[..., :RD] = np.concatenate([QH[..., :H2] * cH - QH[..., H2:] * sH,
                               QH[..., :H2] * sH + QH[..., H2:] * cH], -1)
KH = Kb[..., :RD].copy()
Kb[..., :RD] = np.concatenate([KH[..., :H2] * cH - KH[..., H2:] * sH,
                               KH[..., :H2] * sH + KH[..., H2:] * cH], -1)
Q8 = Qb[0].transpose(1, 0, 2)
K8 = Kb[0].transpose(1, 0, 2)
print("Qb shape:", Qb.shape)
# torch Q_t is [H,T,D]; numpy Q8 is [T,H,D] -> transpose for compare
print("Q err:", np.abs(Qt.transpose(1, 0, 2) - Q8).max())
print("K err:", np.abs(Kt.transpose(1, 0, 2) - K8).max())
print("V err:", np.abs(Vt.transpose(1, 0, 2) - V8).max())
print("G err:", np.abs(Gt - G8).max())
