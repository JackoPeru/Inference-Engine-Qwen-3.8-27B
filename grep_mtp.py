import re
src = open('/tmp/qwen35.cpp').read()
for i, line in enumerate(src.split('\n')):
    if re.search(r'mtp|nextn|eh_proj|enorm|hnorm|shared_head', line, re.I):
        print(f"{i+1}: {line}")
