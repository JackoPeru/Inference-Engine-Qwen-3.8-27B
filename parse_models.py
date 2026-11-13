import json
d = json.load(open('/tmp/models.json'))
for x in d:
    n = x.get('name', '')
    if 'qwen' in n.lower() or 'mtp' in n.lower() or 'next' in n.lower():
        print(n, x.get('size', ''))
