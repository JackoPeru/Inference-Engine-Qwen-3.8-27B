import re
src = open('/tmp/llama_norm.cpp').read()
i = src.find('llm_graph_context::build_norm')
print(src[i:i+3000])
