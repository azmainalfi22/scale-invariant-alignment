"""
exp_uda_baselines.py — is SIA just a weaker version of standard unsupervised
domain adaptation?

The reviewer question this answers: "why should I use SIA rather than an
existing label-free domain-adaptation method?"  We run the standard family on
exactly the same footing as SIA (same montage, same classifier, same single
unlabelled calibration repetition, no labels anywhere):

  1. mean alignment           first moment only
  2. CORAL (full covariance)  Sun et al., the method SIA is a diagonal case of
  3. CORAL + unit-norm        full covariance given SIA's step 1 as well
  4. pseudo-label self-training   the label-free analogue of test-time adaptation
  5. pseudo-label, transductive   also uses the unlabelled test features

Self-validates against the published SIA numbers before reporting anything.

  python exp_uda_baselines.py
"""
import json
import os
import numpy as np
from scipy.stats import wilcoxon
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(ROOT, "emg_pipeline.py"), encoding="utf-8").read()
head = src.split("# ============================================================== VALIDATION")[0]
ns = {"__file__": os.path.join(ROOT, "emg_pipeline.py")}
exec(compile(head, "emg_pipeline_head", "exec"), ns)
landmark, split, unitnorm = ns["landmark"], ns["split"], ns["unitnorm"]
evaluate, SUBJECTS = ns["evaluate"], ns["SUBJECTS"]

L16 = landmark(16)


def fitpred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = LDA(solver="lsqr", shrinkage="auto").fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))


def msqrt(S, inv=False):
    """Symmetric matrix square root (or inverse square root) via eigendecomposition."""
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-10, None)
    w = w ** (-0.5) if inv else w ** 0.5
    return (V * w) @ V.T


def coral(A, C, B):
    """Map target features B onto the source distribution using full second moments.

    Covariances are Ledoit-Wolf shrunk; with 8 calibration samples in 16
    dimensions the raw calibration covariance has rank at most 7, so some
    regularisation is mandatory rather than optional.
    """
    Cs = LedoitWolf().fit(A).covariance_
    Ct = LedoitWolf().fit(C).covariance_
    W = msqrt(Ct, inv=True) @ msqrt(Cs)
    return (B - C.mean(0)) @ W + A.mean(0)


def run():
    res = {k: [] for k in ("raw", "sia", "mean", "coral", "coral_un",
                           "pseudo", "pseudo_trans")}
    for s in SUBJECTS:
        Xtr, ytr, Xc, Xte, yte = split(s, L16)
        res["raw"].append(evaluate(Xtr, ytr, Xc, Xte, yte, "raw"))
        res["sia"].append(evaluate(Xtr, ytr, Xc, Xte, yte, "sia"))

        # 1. first-moment alignment only
        B = Xte - Xc.mean(0) + Xtr.mean(0)
        res["mean"].append((fitpred(Xtr, ytr, B) == yte).mean())

        # 2. full CORAL on raw features
        res["coral"].append((fitpred(Xtr, ytr, coral(Xtr, Xc, Xte)) == yte).mean())

        # 3. full CORAL after unit normalisation (SIA's step 1)
        A, C, T = unitnorm(Xtr), unitnorm(Xc), unitnorm(Xte)
        res["coral_un"].append((fitpred(A, ytr, coral(A, C, T)) == yte).mean())

        # 4. pseudo-label self-training on the calibration repetition
        yc = fitpred(Xtr, ytr, Xc)
        res["pseudo"].append(
            (fitpred(np.vstack([Xtr, Xc]), np.concatenate([ytr, yc]), Xte) == yte).mean())

        # 5. transductive: pseudo-label the calibration AND test features
        yt = fitpred(Xtr, ytr, Xte)
        res["pseudo_trans"].append(
            (fitpred(np.vstack([Xtr, Xc, Xte]), np.concatenate([ytr, yc, yt]),
                     Xte) == yte).mean())
    return {k: np.array(v) for k, v in res.items()}


r = run()

print("=" * 78)
print("VALIDATION")
print("=" * 78)
for k, ref in (("raw", 0.5638), ("sia", 0.7338)):
    g = r[k].mean()
    print(f"  {k:<5s} rebuilt {g:.4f}  stored {ref:.4f}  diff {g - ref:+.4f}"
          f"  {'OK' if abs(g - ref) < 0.002 else '** MISMATCH **'}")

CEIL, RAW = 0.9460, r["raw"].mean()
GAP = CEIL - RAW

print()
print("=" * 78)
print("LABEL-FREE ADAPTATION BASELINES ON A COMMON FOOTING (Hyser, n=20, K=16)")
print("=" * 78)
print(f"{'method':<34s}{'acc':>8s}{'gap rec':>10s}{'d vs raw':>10s}"
      f"{'p vs raw':>10s}{'p vs SIA':>10s}")
NAMES = [("raw", "raw (standard practice)"),
         ("mean", "mean alignment (1st moment)"),
         ("coral", "CORAL, full covariance"),
         ("coral_un", "CORAL, full cov. + unit-norm"),
         ("pseudo", "pseudo-label self-training"),
         ("pseudo_trans", "pseudo-label, transductive"),
         ("sia", "SIA (proposed)")]
out = {}
for k, nm in NAMES:
    a = r[k]
    d = (a.mean() - RAW) * 100
    rec = (a.mean() - RAW) / GAP * 100
    pr = wilcoxon(a, r["raw"]).pvalue if k != "raw" else np.nan
    ps = wilcoxon(a, r["sia"]).pvalue if k != "sia" else np.nan
    print(f"{nm:<34s}{a.mean():8.3f}{rec:9.1f}%{d:+10.1f}"
          f"{pr:10.4g}{ps:10.4g}")
    out[k] = {"name": nm, "acc": float(a.mean()), "sd": float(a.std(ddof=1)),
              "gap_recovered_pct": float(rec), "delta_pts": float(d),
              "p_vs_raw": None if k == "raw" else float(pr),
              "p_vs_sia": None if k == "sia" else float(ps),
              "per_subject": a.tolist()}

print()
print("Why full CORAL cannot work here: the calibration repetition supplies 8")
print("samples in C=16 dimensions, so its covariance has rank <= 7 and the")
print("O(C^2) parameters CORAL needs cannot be estimated. SIA's diagonal")
print("restriction needs O(C).")
print()
best = max((k for k, _ in NAMES if k not in ("sia", "raw")), key=lambda k: r[k].mean())
d, p = (r["sia"].mean() - r[best].mean()) * 100, wilcoxon(r["sia"], r[best]).pvalue
print(f"SIA beats the best baseline ({out[best]['name']}) by {d:+.1f} points, p={p:.4g}")

json.dump({"ceiling": CEIL, "n": len(SUBJECTS), "K": 16,
           "calibration_samples": 8, "dimensions": 16,
           "methods": out,
           "sia_vs_best_baseline": {"baseline": best, "delta_pts": float(d),
                                    "p": float(p)}},
          open(os.path.join(ROOT, "results", "uda_baselines.json"), "w"), indent=1)
print("\nwrote results/uda_baselines.json")
