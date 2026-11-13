#!/usr/bin/env python3
"""M0 baseline: single-request + light concurrency against TabbyAPI 127.0.0.1:8000.
Read-only, non-destructive. Small prompts, short outputs to avoid OOM on full GPUs.
Saves JSON to ../benchmarks/baseline_m0_raw.json
"""
import json, time, urllib.request, concurrent.futures, sys, os

BASE = "http://127.0.0.1:8000"
MODEL = "Qwen3.8-27B-Uncensored-EXL3-3.07bpw"

def chat_once(prompt, max_tokens=64, temperature=0.0, streaming=False, timeout=300):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": streaming,
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    text = ""
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if not streaming:
            obj = json.load(r)
            t1 = time.perf_counter()
            choice = obj["choices"][0]
            text = choice["message"].get("content", "")
            usage = obj.get("usage")
            return {"latency_s": t1 - t0, "text": text, "usage": usage}
        else:
            # SSE parse
            first = True
            buf = b""
            while True:
                chunk = r.read(1)
                if not chunk:
                    break
                buf += chunk
                if buf.endswith(b"\n\n") or buf.endswith(b"\n"):
                    line = buf.decode("utf-8", "replace").strip()
                    buf = b""
                    if not line:
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            obj = json.loads(payload)
                        except Exception:
                            continue
                        if first:
                            ttft = time.perf_counter() - t0
                            first = False
                        try:
                            delta = obj["choices"][0].get("delta", {})
                            text += delta.get("content", "") or ""
                            u = obj.get("usage")
                            if u:
                                usage = u
                        except Exception:
                            pass
            t1 = time.perf_counter()
            return {"latency_s": t1 - t0, "ttft_s": ttft, "text": text, "usage": usage}

def main():
    out = {"env": {}, "runs": []}
    # health + models
    try:
        out["env"]["health"] = json.load(urllib.request.urlopen(BASE + "/health", timeout=15))
    except Exception as e:
        out["env"]["health_error"] = str(e)[:300]
    prompts = {
        "prose_it": "Spiega in 3 frasi cos'e la fotosintesi clorofilliana.",
        "code_py": "Scrivi una funzione Python che calcola il fattoriale in modo iterativo.",
        "reasoning": "Quanto fa 84 * 3 / 2? Mostra i passaggi.",
    }
    # warmup
    try:
        chat_once("Ciao, rispondi con OK.", max_tokens=8, streaming=False)
    except Exception as e:
        print("warmup failed:", str(e)[:500], file=sys.stderr)
    # measured: non-stream totals
    for name, p in prompts.items():
        for i in range(2):
            try:
                r = chat_once(p, max_tokens=64, streaming=False)
                u = r.get("usage") or {}
                ct = u.get("completion_tokens", 0)
                pt = u.get("prompt_tokens", 0)
                lat = r["latency_s"]
                tps = (ct / lat) if lat > 0 and ct else 0
                out["runs"].append({"mode": "nonstream", "prompt": name, "iter": i,
                                    "latency_s": lat, "prompt_tokens": pt,
                                    "completion_tokens": ct, "tok_per_s": tps})
                print(f"{name} iter{i}: lat={lat:.2f}s pt={pt} ct={ct} tps={tps:.1f}")
            except Exception as e:
                out["runs"].append({"mode": "nonstream", "prompt": name, "iter": i, "error": str(e)[:300]})
                print(f"{name} iter{i} ERROR {str(e)[:200]}")
    # streaming TTFT (one prompt)
    try:
        r = chat_once(prompts["prose_it"], max_tokens=64, streaming=True)
        out["runs"].append({"mode": "stream", "prompt": "prose_it",
                            "latency_s": r["latency_s"], "ttft_s": r.get("ttft_s"),
                            "text_len": len(r["text"]), "usage": r.get("usage")})
        print(f"stream: lat={r['latency_s']:.2f}s ttft={r.get('ttft_s')} text_len={len(r['text'])}")
    except Exception as e:
        out["runs"].append({"mode": "stream", "error": str(e)[:300]})
        print("stream ERROR", str(e)[:200])
    # light concurrency x2 (short outputs)
    def worker(k):
        try:
            r = chat_once("Rispondi con una sola frase breve sulla luna.", max_tokens=32, streaming=False)
            u = r.get("usage") or {}
            return {"ok": True, "lat": r["latency_s"], "ct": u.get("completion_tokens", 0)}
        except Exception as e:
            return {"ok": False, "err": str(e)[:200]}
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(worker, k) for k in range(2)]
        res = [f.result() for f in futs]
    wall = time.perf_counter() - t0
    toks = sum(r.get("ct", 0) for r in res if r.get("ok"))
    out["runs"].append({"mode": "concurrency2", "wall_s": wall, "results": res,
                        "agg_tps": (toks / wall) if wall > 0 else 0})
    print(f"concurrency2: wall={wall:.2f}s toks={toks} agg_tps={(toks/wall if wall>0 else 0):.1f}")
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "benchmarks"), exist_ok=True)
    # save next to script dir on server (/home/matteo/hermes-27/scripts -> ../benchmarks)
    dest = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks", "baseline_m0_raw.json"))
    # also try local relative
    try:
        with open(dest, "w") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        print("saved", dest)
    except Exception as e:
        print("save failed", e)

if __name__ == "__main__":
    main()
