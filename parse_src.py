import json
d = json.load(open('/tmp/src.json'))
for x in d:
    n = x.get('name', '')
    if 'graph' in n.lower() or 'context' in n.lower() or n.endswith('.cpp'):
        print(n, x.get('size', ''))
