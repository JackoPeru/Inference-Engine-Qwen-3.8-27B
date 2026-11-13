#!/usr/bin/env python3
import json, urllib.request
BASE = "http://127.0.0.1:8000"
MODEL = "Qwen3.8-27B-Uncensored-EXL3-3.07bpw"
body = {"model": MODEL, "messages": [{"role": "user", "content": "Rispondi con OK."}],
        "max_tokens": 16, "temperature": 0.0, "stream": False}
req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as r:
    raw = r.read().decode("utf-8", "replace")
print("STATUS", r.status)
print(raw[:3000])
obj = json.loads(raw)
print("KEYS", list(obj.keys()))
ch = obj.get("choices")
print("CHOICES_TYPE", type(ch), (len(ch) if ch else None))
if ch:
    print("C0_TYPE", type(ch[0]), ch[0].keys() if isinstance(ch[0], dict) else ch[0])
