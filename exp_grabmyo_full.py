"""
exp_grabmyo_full.py — GRABMyo cross-day evaluation at the largest available n.

The paper reports GRABMyo on the first 20 participants (40 session pairs). This
re-runs the identical protocol on however many participants the cache holds, and
first REPRODUCES the published 20-participant numbers by restricting to
participants 1-20, so any change at full n is attributable to sample size and not
to a changed pipeline.

Protocol (unchanged): train on session 1, test on sessions 2 and 3; trial 1 of the
target session is the unlabelled calibration set, trials 2-5 are the test set.

  python exp_grabmyo_full.py                 # auto-picks the larger cache
  python exp_grabmyo_full.py --cache data/grabmyo_fw.npz
"""
import argparse
import json
import os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))

src = open(os.path.join(ROOT, "emg_pipeline.py"), encoding="utf-8").read()
head = src.split("# ============================================================== VALIDATION")[0]
ns = {"__file__": os.path.join(ROOT, "emg_pipeline.py")}
exec(compile(head, "h", "exec"), ns)
evaluate = ns["evaluate"]
RNG = np.random.default_rng(0)

PUBLISHED = {"F": dict(raw=0.5154, unitnorm=0.5426, sia=0.6357, pairs=40),
             "W": dict(raw=0.4364, unitnorm=0.4070, sia=0.5246, pairs=40)}


def run(G, pids, site):
    """Returns per-session-pair accuracy arrays for raw / unitnorm / sia."""
    r_, u_, s_ = [], [], []
    for p in pids:
        k1 = f"s1_p{p}_{site}"
        if k1 not in G.files:
            continue
        X1, y1 = G[k1], G[f"s1_p{p}_y"]
        for ses in (2, 3):
            k = f"s{ses}_p{p}_{site}"
            if k not in G.files:
                continue
            X2, y2, t2 = G[k], G[f"s{ses}_p{p}_y"], G[f"s{ses}_p{p}_t"]
            c = t2 == t2.min()
            if c.sum() == 0 or (~c).sum() == 0:
                continue
            for arr, m in ((r_, "raw"), (u_, "unitnorm"), (s_, "sia")):
                arr.append(evaluate(X1, y1, X2[c], X2[~c], y2[~c], m))
    return np.array(r_), np.array(u_), np.array(s_)


def compare(a, b, nboot=20000):
    d = (a - b) * 100.0
    try:
        p = stats.wilcoxon(a, b).pvalue
    except ValueError:
        p = 1.0
    idx = RNG.integers(0, d.size, size=(nboot, d.size))
    bm = d[idx].mean(axis=1)
    pos = (d > 0).sum(); neg = (d < 0).sum()
    rb = (pos - neg) / max(pos + neg, 1)
    return dict(delta_pts=round(float(d.mean()), 1), p=float(p),
                rank_biserial=round(float(rb), 3),
                ci95_pts=[round(float(np.percentile(bm, 2.5)), 1),
                          round(float(np.percentile(bm, 97.5)), 1)])


ap = argparse.ArgumentParser()
ap.add_argument("--cache", default=None)
args = ap.parse_args()

cache = args.cache
if cache is None:
    big = os.path.join(ROOT, "data", "grabmyo_fw_all.npz")
    cache = big if os.path.exists(big) else os.path.join(ROOT, "data", "grabmyo_fw.npz")
G = np.load(cache, allow_pickle=True)
all_pids = sorted({int(k.split("_")[1][1:]) for k in G.files if k.endswith("_y")})
print(f"cache: {os.path.basename(cache)}   participants: {len(all_pids)}\n")

print("=" * 74)
print("STAGE 1 — reproduce the published n=20 numbers (participants 1-20)")
print("=" * 74)
ok = True
for site in ("F", "W"):
    r, u, s = run(G, [p for p in all_pids if p <= 20], site)
    ref = PUBLISHED[site]
    nm = "forearm" if site == "F" else "wrist"
    print(f"  [{nm}] pairs {len(r)} (pub {ref['pairs']})  raw {r.mean():.4f} "
          f"(pub {ref['raw']:.4f})  unitnorm {u.mean():.4f} (pub {ref['unitnorm']:.4f})  "
          f"SIA {s.mean():.4f} (pub {ref['sia']:.4f})")
    for k, v in (("raw", r), ("unitnorm", u), ("sia", s)):
        if abs(v.mean() - ref[k]) > 0.002:
            ok = False
            print(f"    ** MISMATCH on {k} **")
print(f"\n  reproduction: {'OK' if ok else 'FAILED — not extending'}")
if not ok:
    raise SystemExit(1)

print()
print("=" * 74)
print(f"STAGE 2 — full cohort (n = {len(all_pids)} participants)")
print("=" * 74)
out = {"cache": os.path.basename(cache), "n_participants": len(all_pids),
       "participants": all_pids, "sites": {}}
for site in ("F", "W"):
    r, u, s = run(G, all_pids, site)
    nm = "forearm band (2x8)" if site == "F" else "wrist band (2x6)"
    su, ss = compare(u, r), compare(s, r)
    print(f"\n  [{nm}]  session pairs = {len(r)}")
    print(f"    raw        {r.mean():.4f}")
    print(f"    unit-norm  {u.mean():.4f}  ({su['delta_pts']:+.1f} pts, p={su['p']:.4g}, "
          f"CI[{su['ci95_pts'][0]:+.1f},{su['ci95_pts'][1]:+.1f}])")
    print(f"    SIA        {s.mean():.4f}  ({ss['delta_pts']:+.1f} pts, p={ss['p']:.4g}, "
          f"r={ss['rank_biserial']:+.2f}, CI[{ss['ci95_pts'][0]:+.1f},"
          f"{ss['ci95_pts'][1]:+.1f}])")
    out["sites"][site] = dict(name=nm, pairs=int(len(r)),
                              raw=round(float(r.mean()), 4),
                              unitnorm=round(float(u.mean()), 4),
                              sia=round(float(s.mean()), 4),
                              unitnorm_vs_raw=su, sia_vs_raw=ss)

p = os.path.join(ROOT, "results", "grabmyo_full.json")
json.dump(out, open(p, "w"), indent=2)
print(f"\nwritten -> {p}")
