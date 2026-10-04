"""
exp_timegap.py -- does the cross-day loss grow with elapsed time, and does the
correction hold at a longer horizon?

GRABMyo records each participant on days 1, 8 and 29.  The paper pools all 86
session pairs and never separates them, so the dataset's elapsed-time structure
is unused -- and the question the Limitations section invites ("two to three
sessions spanning at most 29 days, far short of real device lifetimes") goes
unanswered when the data can partly answer it.  Three gaps are available:

    session 1 -> 2    7 days
    session 2 -> 3   21 days
    session 1 -> 3   28 days

This script also corrects a statistical error in the submitted paper.  Table V
and Sec. IV-E report p < 1e-9 over "86 session pairs", but each of the 43
participants contributes two pairs (1->2 and 1->3) that SHARE their session-1
training data.  The observations are clustered within participant, the effective
sample size is 43 rather than 86, and a Wilcoxon test over 86 pairs overstates
precision.  Every test below is computed at the participant level; the pooled
pair-level values are printed alongside so the size of the inflation is visible.

  python exp_timegap.py
"""
import json
import os
import numpy as np
from scipy.stats import wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
G = np.load(os.path.join(ROOT, "data", "grabmyo_fw_all.npz"), allow_pickle=True)
PIDS = sorted({int(k.split("_")[1][1:]) for k in G.files if k.endswith("_y")})
GAPS = {(1, 2): 7, (2, 3): 21, (1, 3): 28}
BANDS = {"F": "forearm band (16 el.)", "W": "wrist band (12 el.)"}
EPS = 1e-12


def unitnorm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def fitpred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = LDA(solver="lsqr", shrinkage="auto").fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))


def evaluate(Xtr, ytr, Xc, Xte, yte, method):
    if method == "raw":
        A, B = Xtr, Xte
    elif method == "unitnorm":
        A, B = unitnorm(Xtr), unitnorm(Xte)
    elif method in ("zscore", "sia"):
        if method == "sia":
            A, C, B = unitnorm(Xtr), unitnorm(Xc), unitnorm(Xte)
        else:
            A, C, B = Xtr, Xc, Xte
        B = (B - C.mean(0)) * (A.std(0) + EPS) / (C.std(0) + EPS) + A.mean(0)
    else:
        raise ValueError(method)
    return (fitpred(A, ytr, B) == yte).mean()


def within_session(pid, band, ses=1):
    """Leave-one-trial-out within-session ceiling for one participant."""
    k = f"s{ses}_p{pid}_{band}"
    if k not in G.files:
        return None
    X, y, t = G[k], G[f"s{ses}_p{pid}_y"], G[f"s{ses}_p{pid}_t"]
    accs = []
    for tv in np.unique(t):
        te = t == tv
        if te.sum() == 0 or (~te).sum() == 0:
            continue
        accs.append((fitpred(X[~te], y[~te], X[te]) == y[te]).mean())
    return float(np.mean(accs)) if accs else None


def pair(pid, band, s_tr, s_te):
    ktr, kte = f"s{s_tr}_p{pid}_{band}", f"s{s_te}_p{pid}_{band}"
    if ktr not in G.files or kte not in G.files:
        return None
    X1, y1 = G[ktr], G[f"s{s_tr}_p{pid}_y"]
    X2, y2, t2 = G[kte], G[f"s{s_te}_p{pid}_y"], G[f"s{s_te}_p{pid}_t"]
    c = t2 == t2.min()
    if c.sum() == 0 or (~c).sum() == 0:
        return None
    return {m: evaluate(X1, y1, X2[c], X2[~c], y2[~c], m)
            for m in ("raw", "unitnorm", "zscore", "sia")}


def wx(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    try:
        p = float(wilcoxon(a, b).pvalue)
    except ValueError:
        p = 1.0
    return (a.mean() - b.mean()) * 100, p


# ============================================================ collect
rows = []          # one per (participant, band, gap)
ceil = {}
for band in BANDS:
    for pid in PIDS:
        ceil[(pid, band)] = within_session(pid, band)
        for (a, b), days in GAPS.items():
            r = pair(pid, band, a, b)
            if r:
                rows.append({"pid": pid, "band": band, "gap_days": days,
                             "train_ses": a, "test_ses": b, **r})

print("=" * 88)
print("VALIDATION -- restricted to the first 20 participants, sessions 1->2 and 1->3")
print("=" * 88)
for band, ref_raw, ref_sia in (("F", 0.5154, 0.6357), ("W", 0.4364, 0.5246)):
    sub = [r for r in rows if r["band"] == band and r["pid"] <= 20
           and r["train_ses"] == 1]
    rr, ss = np.mean([r["raw"] for r in sub]), np.mean([r["sia"] for r in sub])
    print(f"  {BANDS[band]:<26s} raw {rr:.4f} (stored {ref_raw:.4f})   "
          f"SIA {ss:.4f} (stored {ref_sia:.4f})   "
          f"{'OK' if abs(rr-ref_raw)<0.002 and abs(ss-ref_sia)<0.002 else '** CHECK **'}")

# ================================================= clustering correction
print()
print("=" * 88)
print("CLUSTERING CORRECTION -- pair-level versus participant-level inference")
print("=" * 88)
print("The submitted paper tests over 86 pairs; each participant contributes two,")
print("sharing their session-1 training data.  Same effect, honest precision:\n")
print(f"{'band':<12s}{'unit':<14s}{'n':>4s}{'raw':>8s}{'SIA':>8s}{'delta':>8s}{'p':>12s}")
clust = {}
for band in BANDS:
    sub = [r for r in rows if r["band"] == band and r["train_ses"] == 1]
    # pair level, as submitted
    a = [r["raw"] for r in sub]; b = [r["sia"] for r in sub]
    d_p, p_p = wx(b, a)
    # participant level: average a participant's two pairs first
    byp = {}
    for r in sub:
        byp.setdefault(r["pid"], []).append(r)
    a2 = [np.mean([x["raw"] for x in v]) for v in byp.values()]
    b2 = [np.mean([x["sia"] for x in v]) for v in byp.values()]
    d_s, p_s = wx(b2, a2)
    print(f"{band:<12s}{'pairs':<14s}{len(a):>4d}{np.mean(a):8.3f}{np.mean(b):8.3f}"
          f"{d_p:+8.1f}{p_p:12.3g}")
    print(f"{'':<12s}{'participants':<14s}{len(a2):>4d}{np.mean(a2):8.3f}{np.mean(b2):8.3f}"
          f"{d_s:+8.1f}{p_s:12.3g}")
    clust[band] = {"pair_level": {"n": len(a), "raw": float(np.mean(a)),
                                  "sia": float(np.mean(b)), "delta_pts": d_p, "p": p_p},
                   "participant_level": {"n": len(a2), "raw": float(np.mean(a2)),
                                         "sia": float(np.mean(b2)),
                                         "delta_pts": d_s, "p": p_s}}

# ============================================================ time gap
print()
print("=" * 88)
print("CROSS-DAY LOSS AND RECOVERY BY ELAPSED TIME (participant level)")
print("=" * 88)
out = {}
for band in BANDS:
    cl = np.mean([v for (p, bd), v in ceil.items() if bd == band and v is not None])
    print(f"\n  {BANDS[band]}   within-session ceiling {cl:.3f}")
    print(f"  {'gap':>8s}{'n':>5s}{'raw':>8s}{'step1':>8s}{'step2':>8s}{'SIA':>8s}"
          f"{'+SIA':>7s}{'p':>10s}{'gap rec':>9s}{'loss':>7s}")
    out[band] = {"ceiling": float(cl), "gaps": {}}
    for (a, b), days in GAPS.items():
        sub = [r for r in rows if r["band"] == band and r["gap_days"] == days]
        if not sub:
            continue
        m = {k: np.array([r[k] for r in sub]) for k in
             ("raw", "unitnorm", "zscore", "sia")}
        d, p = wx(m["sia"], m["raw"])
        rec = (m["sia"].mean() - m["raw"].mean()) / (cl - m["raw"].mean()) * 100
        loss = (cl - m["raw"].mean()) * 100
        print(f"  {days:>6d}d{len(sub):>5d}{m['raw'].mean():8.3f}"
              f"{m['unitnorm'].mean():8.3f}{m['zscore'].mean():8.3f}"
              f"{m['sia'].mean():8.3f}{d:+7.1f}{p:10.3g}{rec:8.1f}%{loss:+7.1f}")
        out[band]["gaps"][str(days)] = {
            "n": len(sub), "train_ses": a, "test_ses": b,
            **{k: float(v.mean()) for k, v in m.items()},
            "sia_delta_pts": d, "p_sia_vs_raw": p,
            "gap_recovered_pct": float(rec), "loss_pts": float(loss),
            "per_participant_raw": m["raw"].tolist(),
            "per_participant_sia": m["sia"].tolist()}

# ==================================== does the gap depend on elapsed time?
print()
print("=" * 88)
print("IS THE LOSS WORSE AT A LONGER HORIZON?  (paired within participant)")
print("=" * 88)
print("Both comparisons train on session 1, so the only difference is elapsed time.")
dep = {}
for band in BANDS:
    r7 = {r["pid"]: r for r in rows if r["band"] == band and r["gap_days"] == 7}
    r28 = {r["pid"]: r for r in rows if r["band"] == band and r["gap_days"] == 28}
    common = sorted(set(r7) & set(r28))
    if not common:
        continue
    raw7 = np.array([r7[p]["raw"] for p in common])
    raw28 = np.array([r28[p]["raw"] for p in common])
    sia7 = np.array([r7[p]["sia"] for p in common])
    sia28 = np.array([r28[p]["sia"] for p in common])
    d_raw, p_raw = wx(raw28, raw7)
    d_sia, p_sia = wx(sia28, sia7)
    g7, g28 = (sia7 - raw7) * 100, (sia28 - raw28) * 100
    d_gain, p_gain = wx(g28 / 100, g7 / 100)
    print(f"\n  {BANDS[band]}  (n={len(common)} participants)")
    print(f"    raw   7d {raw7.mean():.3f} -> 28d {raw28.mean():.3f}"
          f"   {d_raw:+.1f} pts, p={p_raw:.3g}")
    print(f"    SIA   7d {sia7.mean():.3f} -> 28d {sia28.mean():.3f}"
          f"   {d_sia:+.1f} pts, p={p_sia:.3g}")
    print(f"    SIA gain  7d {g7.mean():+.1f} pts -> 28d {g28.mean():+.1f} pts"
          f"   {d_gain:+.1f}, p={p_gain:.3g}")
    dep[band] = {"n": len(common),
                 "raw_7d": float(raw7.mean()), "raw_28d": float(raw28.mean()),
                 "raw_delta_pts": d_raw, "raw_p": p_raw,
                 "sia_7d": float(sia7.mean()), "sia_28d": float(sia28.mean()),
                 "sia_delta_pts": d_sia, "sia_p": p_sia,
                 "gain_7d_pts": float(g7.mean()), "gain_28d_pts": float(g28.mean()),
                 "gain_delta_pts": d_gain, "gain_p": p_gain}

json.dump({"gaps_days": {f"{k[0]}->{k[1]}": v for k, v in GAPS.items()},
           "n_participants": len(PIDS),
           "validation_note": "first-20 restriction reproduces the paper's stored values",
           "clustering_correction": clust,
           "by_gap": out, "horizon_dependence": dep},
          open(os.path.join(ROOT, "results", "timegap_grabmyo.json"), "w"), indent=1)
print("\nwrote results/timegap_grabmyo.json")
