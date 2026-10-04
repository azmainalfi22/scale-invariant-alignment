"""
exp_hierarchical.py — pool the geometry contrast into one mixed-effects model.

The paper tests concentrated-vs-distributed montages with separate paired tests at
each electrode budget (K = 4, 8, 16, 32). That is four tests on twenty people, and
the differential "hotspot trap" claim -- that concentration costs MORE across days
than within -- was never testable at all, because per-participant within-day values
for these montages had not been retained.

The rebuilt pipeline can regenerate them, so this fits the model the critical
review asked for:

    accuracy ~ geometry * session + budget + (1 | participant)

over 20 participants x 4 budgets x 2 geometries x 2 sessions = 320 cells. The
geometry:session interaction IS the hotspot-trap claim, tested once with the power
of the whole design instead of four separate underpowered tests.

Cross-day rows use SIA-corrected features (matching Sec. IV-C, where the geometry
contrast is reported after correction); a raw-feature variant is fitted too.

Run:  python exp_hierarchical.py
"""
import json
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(ROOT, "emg_pipeline.py"), encoding="utf-8").read()
head = src.split("# ============================================================== VALIDATION")[0]
ns = {"__file__": os.path.join(ROOT, "emg_pipeline.py")}
exec(compile(head, "h", "exec"), ns)
get, split = ns["get"], ns["split"]
distributed, concentrated = ns["distributed"], ns["concentrated"]
fit_predict, evaluate = ns["fit_predict"], ns["evaluate"]
SUBJECTS = ns["SUBJECTS"]
BUDGETS = (4, 8, 16, 32)

STORED = {  # from summary_rank_law.json / summary_ht.json, for validation
    (16, "conc", "across", "sia"): 0.4236, (16, "dist", "across", "sia"): 0.5603,
    (16, "conc", "across", "raw"): 0.3728, (16, "dist", "across", "raw"): 0.4854,
    (16, "conc", "within"): 0.8573, (16, "dist", "within"): 0.9268,
}


def amp_map(subj):
    """Day-1 activation map used to place the montages (label-free)."""
    X1, _, _ = get(subj, 1)
    return X1.mean(0)


def within_day(subj, chs):
    """Leave-one-repetition-out accuracy inside session 1."""
    X, y, r = get(subj, 1)
    X = X[:, chs]
    accs = []
    for held in np.unique(r):
        m = r == held
        accs.append((fit_predict(X[~m], y[~m], X[m]) == y[m]).mean())
    return float(np.mean(accs))


rows = []
for s in SUBJECTS:
    amp = amp_map(s)
    for K in BUDGETS:
        for geom, chs in (("conc", concentrated(K, amp)), ("dist", distributed(K, amp))):
            w = within_day(s, chs)
            Xtr, ytr, Xc, Xte, yte = split(s, chs)
            a_raw = evaluate(Xtr, ytr, Xc, Xte, yte, "raw")
            a_sia = evaluate(Xtr, ytr, Xc, Xte, yte, "sia")
            rows.append(dict(pid=s, K=K, geometry=geom, session="within",
                             acc=w, acc_raw=w))
            rows.append(dict(pid=s, K=K, geometry=geom, session="across",
                             acc=a_sia, acc_raw=a_raw))
df = pd.DataFrame(rows)

print("=" * 78)
print("VALIDATION — regenerated cells vs stored values")
print("=" * 78)
ok = True
for key, ref in STORED.items():
    if len(key) == 4:
        K, g, sess, feat = key
        col = "acc" if feat == "sia" else "acc_raw"
    else:
        K, g, sess = key
        col = "acc"
    got = df[(df.K == K) & (df.geometry == g) & (df.session == sess)][col].mean()
    d = got - ref
    flag = "OK" if abs(d) < 0.01 else "** MISMATCH **"
    if abs(d) >= 0.01:
        ok = False
    print(f"  K={K:<3d} {g:<5s} {sess:<7s} {col:<8s} {got:.4f}  stored {ref:.4f}  "
          f"diff {d:+.4f}  {flag}")
print(f"\n  regeneration: {'OK' if ok else 'CHECK'}")

out = {"n_participants": len(SUBJECTS), "budgets": list(BUDGETS),
       "n_cells": int(len(df)), "regeneration_ok": bool(ok), "models": {}}

for feat, col in (("SIA-corrected", "acc"), ("raw", "acc_raw")):
    print()
    print("=" * 78)
    print(f"MIXED MODEL ({feat} cross-day rows) — acc ~ geometry*session + C(K) + (1|pid)")
    print("=" * 78)
    d = df.copy()
    d["y"] = d[col] * 100
    m = smf.mixedlm("y ~ C(geometry, Treatment('conc')) * C(session, Treatment('within')) "
                    "+ C(K)", d, groups=d["pid"]).fit(reml=True)
    names = {n: n for n in m.params.index}
    geo = [n for n in names if "geometry" in n and "session" not in n][0]
    ses = [n for n in names if "session" in n and "geometry" not in n][0]
    inter = [n for n in names if "geometry" in n and "session" in n][0]
    res = {}
    for lbl, n_ in (("geometry (dist-conc, within)", geo),
                    ("session (across-within, conc)", ses),
                    ("geometry x session INTERACTION", inter)):
        est, se, p = m.params[n_], m.bse[n_], m.pvalues[n_]
        lo, hi = m.conf_int().loc[n_]
        star = "  <-- the hotspot-trap claim" if "INTERACTION" in lbl else ""
        print(f"  {lbl:<32s} {est:+7.2f} pts  SE {se:4.2f}  "
              f"95% CI [{lo:+6.2f},{hi:+6.2f}]  p={p:.4g}{star}")
        res[lbl] = dict(estimate_pts=round(float(est), 2), se=round(float(se), 2),
                        ci95=[round(float(lo), 2), round(float(hi), 2)], p=float(p))
    print(f"  n = {int(m.nobs)} cells, {len(SUBJECTS)} participants, "
          f"random-intercept SD {float(np.sqrt(m.cov_re.iloc[0, 0])):.2f} pts")
    res["_n_cells"] = int(m.nobs)
    out["models"][feat] = res

p = os.path.join(ROOT, "results", "hierarchical.json")
df.to_csv(os.path.join(ROOT, "results", "geometry_cells.csv"), index=False)
json.dump(out, open(p, "w"), indent=2)
print(f"\nwritten -> {p}  (+ geometry_cells.csv)")
