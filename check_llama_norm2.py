import re
src = open('/tmp/llama-graph.cpp').read()
i = src.find('build_norm(')
print('FIRST build_norm at', i)
# find the definition: 'ggml_tensor * llm_graph_context::build_norm'
j = src.find('::build_norm')
print(src[j-100:j+3500])
