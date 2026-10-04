"""
exp_calibsize.py -- how much unlabelled calibration data does step 2 need?

Two reasons this script exists.

1. results/hyser/summary_calibsize.json reported 0.150 for a single calibration
   recording.  That value does not reproduce under any protocol: the correct
   figure is near 0.42.  audit_paper.py only ever compared the paper against
   that stored file and never the file against a re-run, so the error survived.

2. results/calibsize_rebuilt.json then replaced it without a generating script,
   which reintroduced exactly the same weakness.  This script is the generator.

Protocol, fully specified:

    train        every day-1 recording, landmark-distributed montage, K=16
    calibration  k distinct gestures drawn from day-2 repetition 0, one
                 recording each, unlabelled
    test         every day-2 recording outside repetition 0
    full_rep     the whole of day-2 repetition 0 as calibration, no sampling
    tests        Wilcoxon signed-rank against the raw baseline, paired over
                 the 20 participants

Draws are seeded from (draw index, participant index) rather than from a running
stream, so a participant's draws do not depend on loop order and any single
condition can be rerun on its own.

A NOTE ON k=1.  With one recording the per-channel standard deviation is exactly
zero, so step 2 cannot estimate a target scale.  It does not follow that the
result is governed by the epsilon floor: dividing every channel by the same
constant is a uniform positive rescaling and the classifier's decision is
invariant to it.  Sweeping EPS from 1e-12 to 1e-4 moves the k=1 accuracy by less
than 0.0001.  What actually happens is that step 2 degenerates to a mean shift
carrying the source scale; --meanonly reports that variant for comparison.

  python exp_calibsize.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(ROOT, "emg_pipeline.py"), encoding="utf-8").read()
head = src.split("# ============================================================== VALIDATION")[0]
ns = {"__file__": os.path.join(ROOT, "emg_pipeline.py")}
exec(compile(head, "emg_pipeline_head", "exec"), ns)
landmark, get, unitnorm = ns["landmark"], ns["get"], ns["unitnorm"]
fit_predict, SUBJECTS = ns["fit_predict"], ns["SUBJECTS"]

L16 = landmark(16)
EPS = 1e-12
DRAWS = 200
SIZES = (1, 2, 4, 8)
MEANONLY = "--meanonly" in sys.argv


def prepared(subj):
    """Day-1 train, day-2 repetition 0 calibration pool, day-2 remainder test."""
    X1, y1, _ = get(subj, 1)
    X2, y2, r2 = get(subj, 2)
    m = r2 == 0
    return (X1[:, L16], y1, X2[m][:, L16], y2[m], X2[~m][:, L16], y2[~m])


def score(A, ytr, C, B, yte, mS, sS):
    """Step 2 applied with calibration moments from C, then classify."""
    if MEANONLY:
        Bz = (B - C.mean(0)) + mS
    else:
        Bz = (B - C.mean(0)) * sS / (C.std(0) + EPS) + mS
    return (fit_predict(A, ytr, Bz) == yte).mean()


per_subject = {k: [] for k in SIZES}
per_subject["full_repetition"] = []
raw_acc, spread = [], {k: [] for k in SIZES}

for si, s in enumerate(SUBJECTS):
    Xtr, ytr, Xcal, ycal, Xte, yte = prepared(s)
    raw_acc.append((fit_predict(Xtr, ytr, Xte) == yte).mean())
    A, B = unitnorm(Xtr), unitnorm(Xte)
    mS, sS = A.mean(0), A.std(0) + EPS
    gestures = np.unique(ycal)
    where = {g: np.where(ycal == g)[0] for g in gestures}

    for k in SIZES:
        accs = []
        for d in range(DRAWS):
            rng = np.random.default_rng([d, si])
            pick = rng.choice(gestures, size=min(k, len(gestures)), replace=False)
            idx = [rng.choice(where[g]) for g in pick]
            accs.append(score(A, ytr, unitnorm(Xcal[idx]), B, yte, mS, sS))
        per_subject[k].append(float(np.mean(accs)))
        spread[k].append(float(np.std(accs)))

    per_subject["full_repetition"].append(
        float(score(A, ytr, unitnorm(Xcal), B, yte, mS, sS)))

raw = float(np.mean(raw_acc))
print("=" * 78)
print(f"VALIDATION   raw K=16 = {raw:.4f}   (published 0.5638)   "
      f"{'OK' if abs(raw - 0.5638) < 0.002 else '** CHECK **'}")
print("=" * 78)
if MEANONLY:
    print("running the MEAN-SHIFT-ONLY variant: step 2 with no rescaling at all")
print(f"{DRAWS} draws per participant per condition, n={len(SUBJECTS)}")
print()
print(f"{'calibration':<22s}{'acc':>8s}{'vs raw':>9s}{'p':>11s}{'improved':>10s}"
      f"{'draw SD':>9s}")


def summarise(vals):
    v = np.array(vals)
    r = np.array(raw_acc)
    try:
        p = float(wilcoxon(v, r).pvalue)
    except ValueError:
        p = 1.0
    return float(v.mean()), round((v.mean() - r.mean()) * 100, 1), p, int((v > r).sum())


out = {"protocol": ("calibration = k distinct gestures from day-2 repetition 0, "
                    "one recording each, mean over 200 seeded draws; test = all "
                    "day-2 recordings outside repetition 0; landmark-distributed "
                    "montage, K=16"),
       "n": len(SUBJECTS), "raw": round(raw, 4), "draws": DRAWS,
       "eps": EPS, "variant": "meanonly" if MEANONLY else "sia", "sizes": {}}

for k in SIZES:
    mean, gain, p, imp = summarise(per_subject[k])
    out["sizes"][str(k)] = {"mean": round(mean, 4), "gain_pts": gain,
                           "p": float(f"{p:.4g}"), "n_improved": imp}
    label = f"{k} gesture" + ("" if k == 1 else "s")
    print(f"{label:<22s}{mean:8.4f}{gain:+9.1f}{p:11.4g}{imp:>7d}/20"
          f"{np.mean(spread[k]):9.4f}")

mean, gain, p, imp = summarise(per_subject["full_repetition"])
out["full_repetition"] = {"mean": round(mean, 4), "gain_pts": gain,
                          "p": float(f"{p:.4g}"), "n_improved": imp}
print(f"{'full repetition':<22s}{mean:8.4f}{gain:+9.1f}{p:11.4g}{imp:>7d}/20"
      f"{0.0:9.4f}")

print()
print("stability of the sampled conditions (split-half over draws):")
for k in SIZES:
    halves = []
    for si, s in enumerate(SUBJECTS):
        Xtr, ytr, Xcal, ycal, Xte, yte = prepared(s)
        A, B = unitnorm(Xtr), unitnorm(Xte)
        mS, sS = A.mean(0), A.std(0) + EPS
        gestures = np.unique(ycal)
        where = {g: np.where(ycal == g)[0] for g in gestures}
        acc = []
        for d in range(DRAWS):
            rng = np.random.default_rng([d, si])
            pick = rng.choice(gestures, size=min(k, len(gestures)), replace=False)
            idx = [rng.choice(where[g]) for g in pick]
            acc.append(score(A, ytr, unitnorm(Xcal[idx]), B, yte, mS, sS))
        halves.append((np.mean(acc[:DRAWS // 2]), np.mean(acc[DRAWS // 2:])))
    h = np.array(halves)
    print(f"  k={k}: first 100 draws {h[:, 0].mean():.4f}   "
          f"last 100 {h[:, 1].mean():.4f}   difference {abs(h[:, 0].mean() - h[:, 1].mean()):.4f}")

dst = os.path.join(ROOT, "results",
                   "calibsize_meanonly.json" if MEANONLY else "calibsize_rebuilt.json")
with open(dst, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print()
print(f"wrote {os.path.relpath(dst, ROOT)}")
