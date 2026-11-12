#!/usr/bin/env python3
"""roblox-up-and-coming-forecast-01 entry: momentum forecaster, stdlib only."""
import os
import json
import math
from datetime import datetime, timezone

INPUT = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")


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

    last_chart = reads[-1].get("up_and_coming", [])
    rank_now = {}
    for g in last_chart:
        if isinstance(g, dict):
            rank_now[g["universe_id"]] = g["rank"]
    max_rank = len(rank_now) or 1

    # ---- momentum: predict future playing via log-linear regression ----
    horizon_h = 40.0  # look-ahead hours, ~mid-window

    def predict_playing(s):
        base = s[-1]
        pts = []
        for i in range(max(0, len(s) - 8), len(s)):
            t = (s[i][0] - base[0]) / 3600.0  # hours before last read (<= 0)
            p = s[i][2]
            if p and p > 1:
                pts.append((t, math.log(p)))
        if len(pts) < 3:
            return base[2] or 0
        n = len(pts)
        sx = sum(t for t, _ in pts)
        sy = sum(y for _, y in pts)
        sxx = sum(t * t for t, _ in pts)
        sxy = sum(t * y for t, y in pts)
        den = n * sxx - sx * sx
        if den == 0:
            return base[2] or 0
        slope = (n * sxy - sx * sy) / den  # log-playing per hour
        intercept = (sy - slope * sx) / n
        growth = slope * horizon_h
        growth = max(-3.0, min(growth, 5.0))  # clamp noise
        return math.exp(intercept + growth)

    def median(vals):
        v = sorted(vals)
        n = len(v)
        return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0

    scored = {}
    for uid, s in series.items():
        rk = rank_now.get(uid, max_rank)
        plays = [p for _, _, p in s if p and p > 0]
        accel = 1.0
        if len(plays) >= 4:
            last2 = median(plays[-2:])
            prev2 = median(plays[-4:-2])
            accel = (last2 / prev2) if prev2 > 0 else 1.0
        accel = max(accel, 0.3)
        climb = max(s[0][1] - rk, 0)          # total rank climb (robust, rank not noisy)
        pp = predict_playing(s)               # regression on the (noisy) playing
        rank_norm = (max_rank - rk) / max_rank
        score = (2.0 * rank_norm
                 + 0.6 * math.log1p(climb)
                 + 2.2 * math.log1p(accel)
                 + 0.05 * math.log1p(max(pp, 0.0)))
        scored[uid] = score

    ranked = sorted(scored, key=lambda u: -scored[u])
    pred = ranked[:places]
    cur = [g["universe_id"] for g in sorted(last_chart, key=lambda x: x.get("rank", 999))][:places]

    # ---- probabilities: momentum prediction + current rank + neighbours ----
    probabilities = []
    for k in range(places):
        place = k + 1
        if k < len(pred):
            probabilities.append({"universe_id": pred[k], "place": place, "p": 0.40})
        if k < len(cur):
            probabilities.append({"universe_id": cur[k], "place": place, "p": 0.30})
        for nb in (k - 1, k + 1):
            if 0 <= nb < len(pred):
                probabilities.append({"universe_id": pred[nb], "place": place, "p": 0.10})

    top10 = list(dict.fromkeys(pred))[:places]

    out = {"probabilities": probabilities, "top10": top10}
    os.makedirs(OUTPUT, exist_ok=True)
    with open(os.path.join(OUTPUT, task.get("output", "forecast.json")), "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()