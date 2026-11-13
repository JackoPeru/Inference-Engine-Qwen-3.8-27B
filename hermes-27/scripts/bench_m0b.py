#!/usr/bin/env python3
"""M0b v2: gestisce reasoning_content null-content, cattura MTP acceptance."""
import json, time, urllib.request, concurrent.futures, os

BASE = "http://127.0.0.1:8000"
MODEL = "Qwen3.8-27B-Uncensored-EXL3-3.07bpw"

def chat(prompt, max_tokens=256, temperature=0.0, timeout=600):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "stream": False}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        obj = json.load(r)
    lat = time.perf_counter() - t0
    ch = (obj.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    u = obj.get("usage") or {}
    t = obj.get("timings") or {}
    return {"lat": lat, "pt": u.get("prompt_tokens", 0), "ct": u.get("completion_tokens", 0),
            "ptps": u.get("prompt_tokens_per_sec", t.get("prompt_per_second")),
            "ctps": u.get("completion_tokens_per_sec", t.get("predicted_per_second")),
            "acc": (u.get("completion_tokens_details") or {}).get("accepted_prediction_tokens",
                   t.get("draft_n_accepted")),
            "rej": (u.get("completion_tokens_details") or {}).get("rejected_prediction_tokens"),
            "draft_n": t.get("draft_n"),
            "content_len": len(content), "reason_len": len(reasoning)}

def synth(prefix, approx_tokens):
    unit = "La fotosintesi clorofilliana converte luce, acqua e CO2 in glucosio e ossigeno. "
    n = max(1, int(approx_tokens * 4 / len(unit)))
    return prefix + " " + (unit * n)

out = {"runs": []}
for ntok in [1000, 8000]:
    p = synth("Riassumi il seguente testo in 5 punti:", ntok)
    try:
        r = chat(p, max_tokens=256)
        out["runs"].append({"mode": "long", "target": ntok, "ok": True, **{k: r[k] for k in ("pt","ct","lat","ptps","ctps","acc","rej","draft_n","content_len","reason_len")}})
        print(f"long {ntok}: pt={r['pt']} ct={r['ct']} lat={r['lat']:.1f}s ptps={r['ptps']} ctps={r['ctps']} acc={r['acc']} rej={r['rej']}")
    except Exception as e:
        out["runs"].append({"mode": "long", "target": ntok, "ok": False, "err": str(e)[:400]})
        print(f"long {ntok} ERROR {str(e)[:300]}")

def worker(_):
    try:
        r = chat("Rispondi in una frase: cos'e la luna?", max_tokens=64)
        return {"ok": True, "lat": r["lat"], "ct": r["ct"], "ctps": r["ctps"], "acc": r["acc"]}
    except Exception as e:
        return {"ok": False, "err": str(e)[:200]}

for conc in [4]:
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(worker, range(conc)))
    wall = time.perf_counter() - t0
    toks = sum(x.get("ct", 0) for x in res if x.get("ok"))
    errs = sum(1 for x in res if not x.get("ok"))
    out["runs"].append({"mode": f"conc{conc}", "wall": wall, "toks": toks, "errs": errs,
                        "agg_tps": (toks / wall if wall > 0 else 0), "detail": res})
    print(f"conc{conc}: wall={wall:.2f}s toks={toks} errs={errs} agg={(toks/wall if wall>0 else 0):.1f}")

dest = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks", "baseline_m0b_raw.json"))
with open(dest, "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print("saved", dest)
