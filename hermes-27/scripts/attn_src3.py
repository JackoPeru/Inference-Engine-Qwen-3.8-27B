import transformers.modeling_utils as MU
orig_get = MU.ALL_ATTENTION_FUNCTIONS.get_interface
def spy_get(kind, fallback=None):
    fn = orig_get(kind, fallback)
    print("FN:", fn, "MOD:", getattr(fn, "__module__", "?"))
    return fn
MU.ALL_ATTENTION_FUNCTIONS.get_interface = spy_get
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Config, Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding
import torch
cfg = Qwen3_5Config(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
    layer_types=["full_attention"], num_attention_heads=2, num_key_value_heads=1,
    head_dim=16, rms_norm_eps=1e-6, hidden_act="silu", output_gate_type="swish",
    attn_output_gate=True, rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
    "partial_rotary_factor": 0.25}, max_position_embeddings=64,
    attention_dropout=0.0, attention_bias=False)
lyr = Qwen3_5DecoderLayer(cfg, 0).float().eval()
x = torch.randn(1, 4, 32) * 0.5
rot = Qwen3_5TextRotaryEmbedding(cfg)
pids = torch.arange(4).unsqueeze(0).unsqueeze(0).expand(3, 1, 4).clone()
with torch.no_grad():
    cos, sin = rot(x, pids)
    y = lyr(x, position_embeddings=(cos, sin))
print("ok")
