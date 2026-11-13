"""Capture layer's real pre/post-rope Q/K via monkeypatch, compare vs replica."""
import torch, numpy as np
import transformers.models.qwen3_5.modeling_qwen3_5 as M
orig_rope = M.apply_rotary_pos_emb
captured = {}
def spy_rope(q, k, cos, sin, unsqueeze_dim=1):
    captured["q_in"] = q.detach().clone()
    captured["k_in"] = k.detach().clone()
    captured["cos_shape"] = tuple(cos.shape)
    out = orig_rope(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
    captured["q_out"] = out[0].detach().clone()
    captured["k_out"] = out[1].detach().clone()
    return out
M.apply_rotary_pos_emb = spy_rope
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config, Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding
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
print("cos_shape_in_rope:", captured["cos_shape"])
print("q_in shape:", tuple(captured["q_in"].shape))
# my replica
X = x[0].numpy()
def rms(a, w):
    return a / np.sqrt((a.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-6) * (1 + w)
h = rms(X, sd["input_layernorm.weight"].numpy())
qq = h @ sd["self_attn.q_proj.weight"].numpy().T
Q = qq[:, :128].reshape(8, 4, 32)
K = (h @ sd["self_attn.k_proj.weight"].numpy().T).reshape(8, 2, 32)
Q = Q / np.sqrt((Q.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-6) * (1 + sd["self_attn.q_norm.weight"].numpy())
K = K / np.sqrt((K.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-6) * (1 + sd["self_attn.k_norm.weight"].numpy())
# layer q_in is [B,H,T,D] -> transpose to [T,H,D]
Ql = captured["q_in"][0].permute(1, 0, 2).numpy()
Kl = captured["k_in"][0].permute(1, 0, 2).numpy()
print("pre-rope Q err:", np.abs(Q - Ql).max())
print("pre-rope K err:", np.abs(K - Kl).max())
d = np.abs(Q - Ql)
idx = np.unravel_index(np.argmax(d), d.shape)
print("worst (t,h,d):", idx, "mine:", Q[idx], "layer:", Ql[idx])
print("Q[1,0,:4] mine:", Q[1, 0, :4])
print("Q[1,0,:4] layer:", Ql[1, 0, :4])
Qlo = captured["q_out"][0].permute(1, 0, 2).numpy()
print("Q[0,0,:6] mine:", Q[0, 0, :6])
print("Q[0,0,:6] layer-in:", Ql[0, 0, :6])
print("Q[0,0,:6] layer-out:", Qlo[0, 0, :6])
