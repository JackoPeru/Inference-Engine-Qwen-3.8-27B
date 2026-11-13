"""Final attn bisect: torch-manual vs numpy vs layer, stage by stage."""
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
    cmask = torch.full((8, 8), float("-inf"))
    cmask = torch.triu(cmask, diagonal=1).unsqueeze(0).unsqueeze(0)
    y = lyr(x, position_embeddings=(cos, sin), attention_mask=cmask)
    y_ref = (y[0] if isinstance(y, tuple) else y).numpy()
    # torch manual
    def rmsT(a, w):
        return a * torch.rsqrt((a.float() ** 2).mean(-1, keepdim=True) + 1e-6) * (1 + w)
    h = rmsT(x, sd["input_layernorm.weight"])
    qq = h @ sd["self_attn.q_proj.weight"].T
    Qt4 = qq.reshape(1, 8, 4, 64)
    Qt, Gt = Qt4[..., :32], Qt4[..., 32:]
    Qt = Qt.reshape(1, 8, 4, 32)
    Kt = (h @ sd["self_attn.k_proj.weight"].T).reshape(1, 8, 2, 32)
    Vt = (h @ sd["self_attn.v_proj.weight"].T).reshape(1, 8, 2, 32)
    Qt = rmsT(Qt, sd["self_attn.q_norm.weight"])
    Kt = rmsT(Kt, sd["self_attn.k_norm.weight"])
    Q4 = Qt.transpose(1, 2).clone()
    K4 = Kt.transpose(1, 2).clone()
    Q4, K4 = apply_rotary_pos_emb(Q4, K4, cos, sin)
    K6 = K4.repeat_interleave(2, dim=1)
    V6 = Vt.transpose(1, 2).repeat_interleave(2, dim=1)
    sc = Q4 * (32 ** -0.5) @ K6.transpose(-1, -2)
    w = torch.softmax(sc + cmask, -1)
    att = (w @ V6).transpose(1, 2).reshape(1, 8, -1) * torch.sigmoid(Gt.reshape(1, 8, -1))
    Yt = att @ sd["self_attn.o_proj.weight"].T
    H1 = x + Yt
    h2 = rmsT(H1, sd["post_attention_layernorm.weight"])
    mg = h2 @ sd["mlp.gate_proj.weight"].T
    g = torch.nn.functional.silu(mg) * (h2 @ sd["mlp.up_proj.weight"].T)
    Y2 = H1 + g @ sd["mlp.down_proj.weight"].T
    Y2n = Y2.numpy()
print("torch-manual vs layer:", np.abs(Y2n - y_ref).max())
# numpy replica
Hid, NQ, NKV, HD, T = 128, 4, 2, 32, 8
X = x[0].numpy()
def rms(a, w):
    return a / np.sqrt((a.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-6) * (1 + w)
h8 = rms(X, sd["input_layernorm.weight"].numpy())
qq8 = h8 @ sd["self_attn.q_proj.weight"].numpy().T
Q48 = qq8.reshape(T, NQ, HD * 2)
G8, Q8 = Q48[:, :, HD:], Q48[:, :, :HD]
Q8 = Q8.reshape(T, NQ, HD)
K8 = (h8 @ sd["self_attn.k_proj.weight"].numpy().T).reshape(T, NKV, HD)
V8 = (h8 @ sd["self_attn.v_proj.weight"].numpy().T).reshape(T, NKV, HD)
qn = sd["self_attn.q_norm.weight"].numpy()
kn = sd["self_attn.k_norm.weight"].numpy()
Q8 = Q8 / np.sqrt((Q8.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-6) * (1 + qn)
K8 = K8 / np.sqrt((K8.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-6) * (1 + kn)
cn = cos.numpy()
sn = sin.numpy()
print("cos shape:", cn.shape)
RD = cn.shape[-1]
H2 = RD // 2
Qb = Q8.transpose(1, 0, 2)[None, :, :, :]
Kb = K8.transpose(1, 0, 2)[None, :, :, :]
cH = np.expand_dims(cn, 1)[..., :H2]
sH = np.expand_dims(sn, 1)[..., :H2]
QH = Qb[..., :RD].copy()
Qb[..., :RD] = np.concatenate([QH[..., :H2] * cH - QH[..., H2:] * sH,
                               QH[..., :H2] * sH + QH[..., H2:] * cH], -1)
KH = Kb[..., :RD].copy()
Kb[..., :RD] = np.concatenate([KH[..., :H2] * cH - KH[..., H2:] * sH,
                               KH[..., :H2] * sH + KH[..., H2:] * cH], -1)
Q8 = Qb[0].transpose(1, 0, 2)
K8 = Kb[0].transpose(1, 0, 2)
K6 = np.repeat(K8, NQ // NKV, axis=1)
V6 = np.repeat(V8, NQ // NKV, axis=1)
out = np.zeros((T, NQ, HD))
for t in range(T):
    scv = np.einsum("hd,shd->hs", Q8[t], K6[:t + 1]) * (HD ** -0.5)
    scv = scv - scv.max(-1, keepdims=True)
    w = np.exp(scv)
    w = w / w.sum(-1, keepdims=True)
    out[t] = np.einsum("hs,shd->hd", w, V6[:t + 1])
att = out.reshape(T, -1) * (1 / (1 + np.exp(-G8.reshape(T, -1))))
Y = att @ sd["self_attn.o_proj.weight"].numpy().T
H1n = X + Y
h2n = rms(H1n, sd["post_attention_layernorm.weight"].numpy())
mgn = h2n @ sd["mlp.gate_proj.weight"].numpy().T
gn = (mgn / (1 + np.exp(-mgn))) * (h2n @ sd["mlp.up_proj.weight"].numpy().T)
Y2n = H1n + gn @ sd["mlp.down_proj.weight"].numpy().T
print("numpy vs torch-manual:", np.abs(Y2n - Y2.numpy()).max())
print("numpy vs layer:", np.abs(Y2n - y_ref).max())
