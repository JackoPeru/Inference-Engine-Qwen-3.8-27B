"""M3 scheduler sim: continuous batching + chunked prefill, TP2 vs PP2 comm model.
Calibrated on measured EXL3 baseline (conc4 agg 121 tok/s, prefill ~500 tok/s,
decode single ~38 tok/s with MTP). Compares policies, NOT absolute predictions.
Run: python3 sched_sim.py  (saves sched_result.json)
"""
import json, math, random
# measured calibration (EXL3 baseline: single 64tok/1.03s~62tok/s ctps~71;
# conc4 agg 121 tok/s wall 1.85s; conc8x32 agg 101.7): single ~65, cap ~130
PREFILL_TPS = 550.0     # ~400-660 measured
DECODE_TPS_1 = 65.0
DECODE_AGG_CAP = 130.0
TP2_AR_LAT = 145e-6 * 128  # 2 allreduce/layer x64 layers host-staged (~18.6ms/tok!)
PP2_P2P_LAT = 0.6e-3    # PP2 point-to-point activation xfer per microbatch (measured 10MB~1.1ms scaled)
CHUNK = 512             # prefill chunk tokens per step
MAXTOK = 4096           # max active tokens per step
def step_time(n_pre, n_dec, mode):
    t = n_pre / PREFILL_TPS + n_dec / min(DECODE_TPS_1 * max(n_dec, 1), DECODE_AGG_CAP)
    if mode == "tp2":
        t += TP2_AR_LAT  # per decode step (prefill amortized, ignored here)
    else:
        t += PP2_P2P_LAT
    return t
def run(conc, out_len, mode, seed=0):
    rnd = random.Random(seed)
    # all requests arrive at t=0 with prompt 64
    rem_pre = [64]*conc
    rem_dec = [out_len]*conc
    t = 0.0
    first_tok = [None]*conc
    done = [0.0]*conc
    active = list(range(conc))
    while active:
        # chunked prefill: fill up to CHUNK/MAXTOK with waiting prefills first
        budget = min(CHUNK, MAXTOK)
        n_pre = 0
        for i in active:
            take = min(rem_pre[i], budget)
            rem_pre[i] -= take; n_pre += take; budget -= take
        n_dec = sum(1 for i in active if rem_pre[i] == 0)
        dt = step_time(n_pre, n_dec, mode)
        t += dt
        for i in list(active):
            if rem_pre[i] == 0:
                if first_tok[i] is None: first_tok[i] = t
                rem_dec[i] -= 1
                if rem_dec[i] <= 0:
                    done[i] = t
                    active.remove(i)
    tot = conc*out_len
    return {"agg_tps": tot/t, "ttft_p50": sorted(first_tok)[conc//2],
            "tpot_p95": sorted(d)[-max(1, conc//8)] if (d:=[done[i]-first_tok[i] for i in range(conc)]) else 0,
            "wall": t}
out = {}
for mode in ["tp2", "pp2"]:
    for conc in [1, 2, 4, 8, 16]:
        r = run(conc, 64, mode)
        out[f"{mode}_c{conc}"] = {k: round(v, 3) for k, v in r.items()}
        print(f"{mode} conc={conc}: agg={r['agg_tps']:.1f} tok/s ttft={r['ttft_p50']:.2f}s wall={r['wall']:.2f}s")
with open("/tmp/sched_result.json", "w") as f:
    json.dump({"calibration": {"prefill_tps": PREFILL_TPS, "decode_tps_1": DECODE_TPS_1,
                               "tp2_ar_lat_s": TP2_AR_LAT, "pp2_p2p_lat_s": PP2_P2P_LAT,
                               "note": "relative policy comparison only"},
               "runs": out}, f, indent=1)
print("saved /tmp/sched_result.json")
