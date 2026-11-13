#!/usr/bin/env python3
"""M0c: conc1/2/8 brevi + TTFT p50/p95, non distruttivo, max_tokens piccoli."""
import json, time, urllib.request, concurrent.futures, os, statistics
BASE="http://127.0.0.1:8000"
MODEL="Qwen3.8-27B-Uncensored-EXL3-3.07bpw"
def chat(prompt, max_tokens=32, temperature=0.0, timeout=300):
    body={"model":MODEL,"messages":[{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":temperature,"stream":False}
    req=urllib.request.Request(BASE+"/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    t0=time.perf_counter()
    with urllib.request.urlopen(req,timeout=timeout) as r:
        obj=json.load(r)
    lat=time.perf_counter()-t0
    u=obj.get("usage") or {}
    t=obj.get("timings") or {}
    return {"lat":lat,"pt":u.get("prompt_tokens",0),"ct":u.get("completion_tokens",0),
      "ctps":u.get("completion_tokens_per_sec",t.get("predicted_per_second")),
      "ptps":u.get("prompt_tokens_per_sec",t.get("prompt_per_second")),
      "acc":(u.get("completion_tokens_details") or {}).get("accepted_prediction_tokens",t.get("draft_n_accepted"))}
def worker(_):
    try:
        r=chat("Rispondi in una frase breve: cos'e la fotosintesi?",max_tokens=32)
        return {"ok":True,**r}
    except Exception as e:
        return {"ok":False,"err":str(e)[:200]}
out={"runs":[]}
# single sequential x5 for TTFT proxy (first-token ~ prompt processing; use lat/ct for TPOT approx)
lats=[]
for i in range(5):
    try:
        r=chat("Spiega in 2 frasi la gravita.",max_tokens=64)
        lats.append(r["lat"])
        out["runs"].append({"mode":"single","iter":i,**r})
        print(f"single {i}: lat={r['lat']:.2f}s pt={r['pt']} ct={r['ct']} ctps={r['ctps']} acc={r['acc']}")
    except Exception as e:
        print(f"single {i} ERR {str(e)[:200]}")
if lats:
    lats.sort()
    out["single_lat_p50"]=statistics.median(lats)
    out["single_lat_p95"]=lats[int(0.95*(len(lats)-1))]
for conc in [8]:
    t0=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
        res=list(ex.map(worker,range(conc)))
    wall=time.perf_counter()-t0
    toks=sum(x.get("ct",0) for x in res if x.get("ok"))
    errs=sum(1 for x in res if not x.get("ok"))
    ok_lats=sorted([x["lat"] for x in res if x.get("ok")])
    p50=statistics.median(ok_lats) if ok_lats else None
    p95=ok_lats[int(0.95*(len(ok_lats)-1))] if ok_lats else None
    out["runs"].append({"mode":f"conc{conc}","wall":wall,"toks":toks,"errs":errs,"agg_tps":(toks/wall if wall>0 else 0),"p50":p50,"p95":p95,"detail":res})
    print(f"conc{conc}: wall={wall:.2f}s toks={toks} errs={errs} agg={(toks/wall if wall>0 else 0):.1f} p50={p50} p95={p95}")
dest="/home/matteo/hermes-27/benchmarks/baseline_m0c_raw.json"
with open(dest,"w") as f:
    json.dump(out,f,indent=1,ensure_ascii=False)
print("saved",dest)
