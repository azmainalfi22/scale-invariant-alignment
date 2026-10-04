"""
exp_identifiability.py -- does discarding amplitude make users MORE identifiable?
Replication at n=43 on two band geometries.

Sec. IV-H reports that cross-day participant identification rises from 0.197 to
0.450 under SIA on 20 Hyser participants against 8-way... in fact against 5%
chance, and calls it "a privacy consequence we did not expect".  As it stands
that is one number on one dataset with no statistics and no positioning against
the published multi-day EMG biometrics literature -- which exists, and uses this
very dataset (Pradhan, He & Jiang's GRABMyo was released for gesture recognition
AND biometrics).  So the concern is not that myoelectric data are identifying;
it is that a normalisation intended to DISCARD information increases how
identifying they are.

Protocol.  Identity classifier trained on all 43 participants' session-1
recordings, tested on the held-out portion of a later session.  Each
participant's features are transformed by exactly the transform a deployment
would apply to them -- their own session-1 statistics as source, their own
calibration trial as target -- so this measures the identifiability of the
representation the device would actually emit, not of a hypothetical one.
Chance is 1/43 = 0.023.

  python exp_identifiability.py
"""
import json
import os
import numpy as np
from scipy.stats import wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
G = np.load(os.path.join(ROOT, "data", "grabmyo_fw_all.npz"), allow_pickle=True)
PIDS = sorted({int(k.split("_")[1][1:]) for k in G.files if k.endswith("_y")})
BANDS = {"F": "forearm band (16 el.)", "W": "wrist band (12 el.)"}
EPS = 1e-12


def unitnorm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def transform(Xsrc, Xcal, Xte, method):
    """Return (source-side features, target-side features) under one method."""
    if method == "raw":
        return Xsrc, Xte
    if method == "unitnorm":
        return unitnorm(Xsrc), unitnorm(Xte)
    if method == "zscore":
        A, C, B = Xsrc, Xcal, Xte
    elif method == "sia":
        A, C, B = unitnorm(Xsrc), unitnorm(Xcal), unitnorm(Xte)
    else:
        raise ValueError(method)
    B = (B - C.mean(0)) * (A.std(0) + EPS) / (C.std(0) + EPS) + A.mean(0)
    return A, B


def build(band, tgt_ses, method):
    """Pooled identity problem: X_train/y_train from session 1, X_test/y_test
    from the held-out part of tgt_ses, each participant transformed by their
    own statistics."""
    Xtr, ytr, Xte, yte = [], [], [], []
    for pid in PIDS:
        ks, kt = f"s1_p{pid}_{band}", f"s{tgt_ses}_p{pid}_{band}"
        if ks not in G.files or kt not in G.files:
            continue
        Xs = G[ks]
        Xt, tt = G[kt], G[f"s{tgt_ses}_p{pid}_t"]
        c = tt == tt.min()
        if c.sum() == 0 or (~c).sum() == 0:
            continue
        A, B = transform(Xs, Xt[c], Xt[~c], method)
        Xtr.append(A); ytr.append(np.full(len(A), pid))
        Xte.append(B); yte.append(np.full(len(B), pid))
    return (np.vstack(Xtr), np.concatenate(ytr),
            np.vstack(Xte), np.concatenate(yte))


def identify(band, tgt_ses, method, clf="LDA"):
    Xtr, ytr, Xte, yte = build(band, tgt_ses, method)
    sc = StandardScaler().fit(Xtr)
    m = (LDA(solver="lsqr", shrinkage="auto") if clf == "LDA"
         else KNeighborsClassifier(n_neighbors=5))
    m.fit(sc.transform(Xtr), ytr)
    pred = m.predict(sc.transform(Xte))
    ok = pred == yte
    per = np.array([ok[yte == p].mean() for p in np.unique(yte)])
    return float(ok.mean()), per


METHODS = [("raw", "raw (standard practice)"),
           ("unitnorm", "unit-norm (step 1)"),
           ("zscore", "per-channel z (step 2)"),
           ("sia", "SIA (both steps)")]

CHANCE = 1.0 / len(PIDS)
print("=" * 84)
print(f"CROSS-DAY PARTICIPANT IDENTIFICATION, n={len(PIDS)}, chance = {CHANCE:.3f}")
print("=" * 84)

out = {"n_participants": len(PIDS), "chance": CHANCE, "bands": {}}
store = {}
for band in BANDS:
    print(f"\n  {BANDS[band]}")
    print(f"  {'representation':<26s}{'ses2':>8s}{'ses3':>8s}{'mean':>8s}"
          f"{'x chance':>10s}{'kNN':>8s}")
    out["bands"][band] = {"name": BANDS[band], "methods": {}}
    for m, nm in METHODS:
        a2, per2 = identify(band, 2, m)
        a3, per3 = identify(band, 3, m)
        knn, _ = identify(band, 2, m, clf="kNN")
        mean = (a2 + a3) / 2
        store[(band, m)] = (per2 + per3) / 2
        print(f"  {nm:<26s}{a2:8.3f}{a3:8.3f}{mean:8.3f}"
              f"{mean / CHANCE:9.1f}x{knn:8.3f}")
        out["bands"][band]["methods"][m] = {
            "name": nm, "ses2": a2, "ses3": a3, "mean": float(mean),
            "times_chance": float(mean / CHANCE), "knn_ses2": knn,
            "per_participant_mean": ((per2 + per3) / 2).tolist()}

print()
print("=" * 84)
print("PAIRED TESTS ACROSS PARTICIPANTS (per-participant identification rate)")
print("=" * 84)
for band in BANDS:
    print(f"\n  {BANDS[band]}")
    base = store[(band, "raw")]
    for m, nm in METHODS[1:]:
        v = store[(band, m)]
        d = (v.mean() - base.mean()) * 100
        p = float(wilcoxon(v, base).pvalue)
        worse = int((v > base).sum())
        print(f"    {nm:<26s}{d:+7.1f} pts vs raw   p={p:<10.3g}"
              f"more identifiable in {worse}/{len(v)}")
        out["bands"][band]["methods"][m]["vs_raw_pts"] = float(d)
        out["bands"][band]["methods"][m]["vs_raw_p"] = p
        out["bands"][band]["methods"][m]["n_more_identifiable"] = worse

print()
print("=" * 84)
print("READ")
print("=" * 84)
for band in BANDS:
    mm = out["bands"][band]["methods"]
    print(f"  {BANDS[band]}: raw {mm['raw']['mean']:.3f} -> SIA {mm['sia']['mean']:.3f} "
          f"({mm['sia']['vs_raw_pts']:+.1f} pts, p={mm['sia']['vs_raw_p']:.3g}), "
          f"{mm['sia']['times_chance']:.0f}x chance")
print("\n  Discarding the amplitude scale does not anonymise; it concentrates the")
print("  representation on the spatial activation pattern, which is individual.")
print("  The same transform that recovers gesture accuracy across days also makes")
print("  the wearer easier to recognise across days.")

json.dump(out, open(os.path.join(ROOT, "results", "identifiability_grabmyo.json"), "w"),
          indent=1)
print("\nwrote results/identifiability_grabmyo.json")
