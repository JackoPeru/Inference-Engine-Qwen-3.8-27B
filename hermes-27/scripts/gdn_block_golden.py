"""Golden: tiny Qwen3_5 linear block forward (transformers exact code, CPU fp32).
Saves outputs for C++ comparison. Prefill T=16 + decode 1 token with cache."""
import torch, json
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet, Qwen3_5Config
torch.manual_seed(42)
cfg = Qwen3_5Config(
    hidden_size=128, intermediate_size=256,
    num_hidden_layers=2, layer_types=["linear_attention","full_attention"],
    num_attention_heads=4, num_key_value_heads=2,
    head_dim=32, linear_key_head_dim=16, linear_num_key_heads=2,
    linear_value_head_dim=16, linear_num_value_heads=4,
    linear_conv_kernel_dim=4, hidden_act="silu",
    rms_norm_eps=1e-6, output_gate_type="swish", attn_output_gate=True,
)
print("key_dim:", 16*2, "value_dim:", 16*4, "conv_dim:", 16*2*2+16*4)
layer = Qwen3_5GatedDeltaNet(cfg, layer_idx=0).float()
layer.eval()
from transformers.cache_utils import DynamicCache
cache = DynamicCache()
x = torch.randn(1, 16, 128) * 0.5
with torch.no_grad():
    y_pre = layer(x)
print("prefill out:", tuple(y_pre.shape), "mean/std:", y_pre.mean().item(), y_pre.std().item())
# decode path: replicate projections manually, then recurrent with carried state
sd = layer.state_dict()
with torch.no_grad():
    qkv = x @ sd['in_proj_qkv.weight'].T
    # causal depthwise conv k=4 + silu (match causal_conv1d_fn, no padding state)
    w = layer.conv1d.weight.squeeze(1)  # [C, K]
    C, K = w.shape
    qkv_t = qkv.transpose(1, 2)  # [B, C, T]
    pad = torch.zeros(1, C, K - 1)
    xp = torch.cat([pad, qkv_t], dim=2)
    conv = torch.stack([torch.nn.functional.conv1d(xp[:, c:c+1, :], w[c:c+1, :].unsqueeze(0)) for c in range(C)], dim=1).squeeze(2).transpose(1, 2)
    conv = torch.nn.functional.silu(conv)
    q, k, v = torch.split(conv, [32, 32, 64], dim=-1)
    q = q.reshape(1, 16, 2, 16).repeat_interleave(2, dim=2)
    k = k.reshape(1, 16, 2, 16).repeat_interleave(2, dim=2)
    v = v.reshape(1, 16, 4, 16)
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule as chunk_fn, torch_recurrent_gated_delta_rule as rec_fn
    b = (x @ sd['in_proj_b.weight'].T).sigmoid()
    gg = (-sd['A_log'].float().exp() * torch.nn.functional.softplus((x @ sd['in_proj_a.weight'].T).float() + sd['dt_bias'])).to(torch.float32)
    o_chunk, S_chunk = chunk_fn(q.float(), k.float(), v.float(), gg, b.float(), output_final_state=True, use_qk_l2norm_in_kernel=True)
    o_rec, S_rec = rec_fn(q.float(), k.float(), v.float(), gg, b.float(), output_final_state=True, use_qk_l2norm_in_kernel=True)
    print("chunk vs recurrent out err:", (o_chunk - o_rec).abs().max().item())
    print("chunk vs recurrent state err:", (S_chunk - S_rec).abs().max().item())
    z = (x @ sd['in_proj_z.weight'].T).reshape(1, 16, 4, 16)
    # RMSNormGated + out_proj to cross-check full block
    core = o_chunk.reshape(-1, 16).float()
    var = core.pow(2).mean(-1, keepdim=True)
    nrm = core * torch.rsqrt(var + 1e-6)
    gated = nrm * torch.nn.functional.silu(z.reshape(-1, 16).float())
    y_full = (gated * sd['norm.weight']).reshape(16, 64) @ sd['out_proj.weight'].T
    print("full-block out err vs layer:", (y_full.reshape(1,16,128) - y_pre).abs().max().item())
torch.save({"y_pre": y_pre, "x": x,
            "sd": {k: v.clone() for k, v in layer.state_dict().items()}},
           "/tmp/gdn_tiny.pt")
print("saved /tmp/gdn_tiny.pt")
