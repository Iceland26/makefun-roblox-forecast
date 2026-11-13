#!/usr/bin/env python3
"""roblox-up-and-coming-forecast-01 entry: occupancy+drift forecaster, stdlib only."""
import os
import json
import math
from datetime import datetime, timezone

INPUT = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

TAU_RECENT = 28.0
TAU_OLD = 80.0
DRIFT = 0.62
TIME_HORIZON = 40.0
KERNEL_SCALE = 0.9


def ts_of(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def main():
    task = json.load(open(os.path.join(INPUT, "task.json")))
    places = int(task.get("places", 10))

    reads = [json.loads(l) for l in open(os.path.join(INPUT, "history.ndjson")) if l.strip()]

    series = {}
    for r in reads:
        t = ts_of(r["at"])
        for g in r.get("up_and_coming", []):
            if isinstance(g, dict):
                uid = g["universe_id"]
                series.setdefault(uid, []).append((t, g.get("rank"), g.get("playing")))
    for u in series:
        series[u].sort()

    occ = {}
    for uid, s in series.items():
        t0 = s[-1][0]
        w_recent = 0.0
        r_recent = 0.0
        w_old = 0.0
        r_old = 0.0
        for (t, rank, play) in s:
            age_h = (t0 - t) / 3600.0
            w_recent += math.exp(-age_h / TAU_RECENT)
            r_recent += math.exp(-age_h / TAU_RECENT) * rank
            w_old += math.exp(-age_h / TAU_OLD)
            r_old += math.exp(-age_h / TAU_OLD) * rank
        if w_recent <= 0:
            continue
        avg_recent = r_recent / w_recent
        avg_old = r_old / w_old if w_old > 0 else avg_recent
        span_h = (t0 - s[0][0]) / 3600.0
        if span_h > 0:
            drift = (avg_recent - avg_old) * (TIME_HORIZON / span_h) * DRIFT
        else:
            drift = 0.0
        occ[uid] = avg_recent + drift

    ranked = sorted(occ, key=lambda u: occ[u])
    pred = ranked[:places]

    from collections import defaultdict
    pred_rank = {uid: i + 1 for i, uid in enumerate(ranked)}
    place_rows = defaultdict(list)
    for k in range(1, places + 1):
        for uid in pred_rank:
            p = math.exp(-abs(pred_rank[uid] - k) / KERNEL_SCALE)
            if p >= 0.004:
                place_rows[k].append((uid, p))
        place_rows[k].sort(key=lambda x: -x[1])
        place_rows[k] = place_rows[k][:100]
        total = sum(p for _, p in place_rows[k])
        if total > 1.0:
            place_rows[k] = [(u, p / total) for u, p in place_rows[k]]

    probabilities = []
    for k in range(1, places + 1):
        rows = [(uid, round(p, 6)) for uid, p in place_rows[k]]
        s = sum(p for _, p in rows)
        if s > 1.0:
            rows.sort(key=lambda x: -x[1])
            rows[0] = (rows[0][0], round(rows[0][1] - (s - 1.0), 6))
        for uid, p in rows:
            probabilities.append({"universe_id": uid, "place": k, "p": p})

    top10 = list(dict.fromkeys(pred))[:places]

    out = {"probabilities": probabilities, "top10": top10}
    os.makedirs(OUTPUT, exist_ok=True)
    with open(os.path.join(OUTPUT, task.get("output", "forecast.json")), "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()

# sweep-final pick
