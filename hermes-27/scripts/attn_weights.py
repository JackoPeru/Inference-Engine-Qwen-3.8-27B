"""Compare layer attn_weights vs manual replica."""
import torch, numpy as np
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config, Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding, apply_rotary_pos_emb, repeat_kv
torch.manual_seed(7)
cfg = Qwen3_5Config(hidden_size=128, intermediate_size=256, num_hidden_layers=1,
    layer_types=["full_attention"], num_attention_heads=4, num_key_value_heads=2,
    head_dim=32, rms_norm_eps=1e-6, hidden_act="silu", output_gate_type="swish",
    attn_output_gate=True, rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
    "partial_rotary_factor": 0.25}, max_position_embeddings=1024,
    attention_dropout=0.0, attention_bias=False)
lyr = Qwen3_5DecoderLayer(cfg, 0).float().eval()
sd = lyr.state_dict()
x = torch.randn(1, 8, 128) * 0.5
rot = Qwen3_5TextRotaryEmbedding(cfg)
pids = torch.arange(8).unsqueeze(0).unsqueeze(0).expand(3, 1, 8).clone()
with torch.no_grad():
    cos, sin = rot(x, pids)
    cmask = torch.full((8, 8), float("-inf"))
    cmask = torch.triu(cmask, diagonal=1).unsqueeze(0).unsqueeze(0)
    y = lyr(x, position_embeddings=(cos, sin), attention_mask=cmask, output_attentions=True)
    y_ref, w_ref = y[0], y[1]
    print("layer out:", y_ref.shape, "weights:", None if w_ref is None else tuple(w_ref.shape))
    # manual replica of the attention core
    def rmsT(a, w):
        return a * torch.rsqrt((a.float() ** 2).mean(-1, keepdim=True) + 1e-6) * (1 + w)
    h = rmsT(x, sd["input_layernorm.weight"])
    qq = h @ sd["self_attn.q_proj.weight"].T
    Qt, Gt = qq.split(128, -1)
    Qt = Qt.reshape(1, 8, 4, 32)
    Kt = (h @ sd["self_attn.k_proj.weight"].T).reshape(1, 8, 2, 32)
    Vt = (h @ sd["self_attn.v_proj.weight"].T).reshape(1, 8, 2, 32)
    Qt = rmsT(Qt, sd["self_attn.q_norm.weight"])
    Kt = rmsT(Kt, sd["self_attn.k_norm.weight"])
    Q4 = Qt.transpose(1, 2).clone()
    K4 = Kt.transpose(1, 2).clone()
    Q4, K4 = apply_rotary_pos_emb(Q4, K4, cos, sin)
    K6 = repeat_kv(K4, 2)
    V6 = repeat_kv(Vt.transpose(1, 2), 2)
    sc = Q4 * (32 ** -0.5) @ K6.transpose(-1, -2)
    w = torch.softmax(sc + cmask, -1)
    print("manual weights vs layer:", (w - w_ref).abs().max().item() if w_ref is not None else "no ref weights")
