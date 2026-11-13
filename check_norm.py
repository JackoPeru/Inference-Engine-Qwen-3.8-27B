import re
p = '/opt/hermes/venv/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py'
src = open(p).read()
i = src.find('class Qwen3_5RMSNorm(')
print(src[i:i+900])
print('======LLAMA======')
