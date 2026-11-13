import json, urllib.request, time
BASE="http://127.0.0.1:8000"
MODEL="Qwen3.8-27B-Uncensored-EXL3-3.07bpw"
PROMPT="Spiega in 2 frasi la gravita."
body={"model":MODEL,"messages":[{"role":"user","content":PROMPT}],"max_tokens":5,"temperature":0.0,"stream":False}
req=urllib.request.Request(BASE+"/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
t0=time.perf_counter()
with urllib.request.urlopen(req,timeout=120) as r:
    obj=json.load(r)
lat=time.perf_counter()-t0
out={"prompt":PROMPT,"latency":lat,"response":obj}
with open("/home/matteo/hermes-27/benchmarks/golden_m2.json","w") as f:
    json.dump(out,f,indent=1,ensure_ascii=False)
ch=obj["choices"][0]["message"]
print("text:",repr(ch.get("content")))
print("usage:",obj.get("usage"))
print("timings:",obj.get("timings"))
print("saved golden_m2.json")
