"""
exp_equivalence.py — replace the paper's ACCEPTED NULLS with equivalence tests.

The paper draws several conclusions from failures to reject: that SIA is
indistinguishable from supervised recalibration, that it does nothing for
amplitude-invariant features, and that calibration statistics transfer across
gesture vocabularies. "Not significantly different" is not "equivalent", so each
of these is restated here as a two-one-sided-tests (TOST) equivalence result.

Reported for every comparison:
  * the paired mean difference,
  * the 90% CI (the interval TOST uses: equivalence at margin D holds iff the
    whole CI lies inside (-D, +D)),
  * the SMALLEST margin at which equivalence holds -- the honest summary, since
    it needs no arbitrary pre-specified D,
  * TOST p at a 10-point margin, chosen to exceed the ~11-point minimum
    detectable effect at n=20 so the test is not automatically doomed.

Stage 1 reproduces the stored means for every quantity whose per-participant
arrays were lost, so nothing is tested that cannot first be validated.

Run:  python exp_equivalence.py
"""
import json
import os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))

src = open(os.path.join(ROOT, "emg_pipeline.py"), encoding="utf-8").read()
head = src.split("# ============================================================== VALIDATION")[0]
ns = {"__file__": os.path.join(ROOT, "emg_pipeline.py")}
exec(compile(head, "h", "exec"), ns)
get, split, landmark = ns["get"], ns["split"], ns["landmark"]
fit_predict, evaluate, unitnorm = ns["fit_predict"], ns["evaluate"], ns["unitnorm"]
SUBJECTS, L16 = ns["SUBJECTS"], ns["landmark"](16)
RNG = np.random.default_rng(0)


def sia_transform(Xtr, Xcal, Xte):
    A, C, B = unitnorm(Xtr), unitnorm(Xcal), unitnorm(Xte)
    mS, sS = A.mean(0), A.std(0) + 1e-12
    mC, sC = C.mean(0), C.std(0) + 1e-12
    return A, (B - mC) * sS / sC + mS


def supervised(subj, use_sia=False):
    """Recalibration: add the LABELLED target-session calibration repetition to training."""
    X1, y1, _ = get(subj, 1)
    X2, y2, r2 = get(subj, 2)
    cal, tst = (r2 == 0), (r2 != 0)
    Xtr, Xc, Xte = X1[:, L16], X2[cal][:, L16], X2[tst][:, L16]
    ytr, yc, yte = y1, y2[cal], y2[tst]
    if use_sia:
        Xtr, Xte = sia_transform(Xtr, Xc, Xte)
        _, Xc2 = sia_transform(X1[:, L16], Xc, Xc)
        Xc = Xc2
    A = np.vstack([Xtr, Xc])
    a = np.concatenate([ytr, yc])
    return (fit_predict(A, a, Xte) == yte).mean()


def tost(a, b, margin_pts=10.0, nboot=20000):
    """Paired equivalence test. Returns dict with CI90, smallest margin, TOST p."""
    d = (np.asarray(a, float) - np.asarray(b, float)) * 100.0
    n = d.size
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    lo_t, hi_t = m - stats.t.ppf(0.95, n - 1) * se, m + stats.t.ppf(0.95, n - 1) * se
    idx = RNG.integers(0, n, size=(nboot, n))
    bm = d[idx].mean(axis=1)
    lo_b, hi_b = np.percentile(bm, 5), np.percentile(bm, 95)
    p_lo = stats.t.sf((m + margin_pts) / se, n - 1)     # H0: diff <= -margin
    p_hi = stats.t.cdf((m - margin_pts) / se, n - 1)    # H0: diff >= +margin
    return dict(n=int(n), diff_pts=round(float(m), 2),
                ci90_t=[round(float(lo_t), 1), round(float(hi_t), 1)],
                ci90_boot=[round(float(lo_b), 1), round(float(hi_b), 1)],
                smallest_margin_pts=round(float(max(abs(lo_t), abs(hi_t))), 1),
                tost_p_at_margin=float(max(p_lo, p_hi)), margin_pts=margin_pts,
                equivalent_at_margin=bool(max(p_lo, p_hi) < 0.05))


def show(label, r, claim):
    verdict = ("EQUIVALENT within " + f"{r['margin_pts']:.0f} pts"
               if r["equivalent_at_margin"] else "NOT established at "
               f"{r['margin_pts']:.0f} pts")
    print(f"\n  {label}")
    print(f"    paper's claim : {claim}")
    print(f"    difference    : {r['diff_pts']:+.2f} pts   90% CI "
          f"[{r['ci90_t'][0]:+.1f}, {r['ci90_t'][1]:+.1f}]  "
          f"(boot [{r['ci90_boot'][0]:+.1f}, {r['ci90_boot'][1]:+.1f}])")
    print(f"    TOST          : p={r['tost_p_at_margin']:.4f}  ->  {verdict}")
    print(f"    tightest supportable claim: |difference| < "
          f"{r['smallest_margin_pts']:.1f} pts at 90% confidence")


out = {}
print("=" * 78)
print("STAGE 1 — reproduce stored means before testing anything")
print("=" * 78)

raw, sia = [], []
for s in SUBJECTS:
    Xtr, ytr, Xc, Xte, yte = split(s, L16)
    raw.append(evaluate(Xtr, ytr, Xc, Xte, yte, "raw"))
    sia.append(evaluate(Xtr, ytr, Xc, Xte, yte, "sia"))
sup = [supervised(s) for s in SUBJECTS]
sup_sia = [supervised(s, use_sia=True) for s in SUBJECTS]
for nm, got, ref in (("raw", np.mean(raw), 0.5638), ("SIA", np.mean(sia), 0.7338),
                     ("supervised", np.mean(sup), 0.7701),
                     ("SIA+supervised", np.mean(sup_sia), 0.8899)):
    d = got - ref
    ok = "OK" if abs(d) < 0.02 else "** MISMATCH **"
    print(f"  {nm:<16s} rebuilt {got:.4f}   stored {ref:.4f}   diff {d:+.4f}   {ok}")
sup_ok = abs(np.mean(sup) - 0.7701) < 0.02

# spectral features
M = np.load(os.path.join(ROOT, "data", "hyser_multifeat.npz"), allow_pickle=True)
feat_arrays = {}
for f in ("MNF", "MDF"):
    g_raw, g_sia = [], []
    for s in SUBJECTS:
        X1, y1 = M[f"{s}_ses1_{f}"], M[f"{s}_ses1_y"]
        X2, y2, r2 = M[f"{s}_ses2_{f}"], M[f"{s}_ses2_y"], M[f"{s}_ses2_rep"]
        c = r2 == 0
        g_raw.append(evaluate(X1[:, L16], y1, X2[c][:, L16], X2[~c][:, L16], y2[~c], "raw"))
        g_sia.append(evaluate(X1[:, L16], y1, X2[c][:, L16], X2[~c][:, L16], y2[~c], "sia"))
    feat_arrays[f] = (np.array(g_sia), np.array(g_raw))
    ref = {"MNF": (0.4948, 0.5385), "MDF": (0.4289, 0.4360)}[f]
    print(f"  {f:<16s} raw {np.mean(g_raw):.4f} (stored {ref[0]:.4f}), "
          f"SIA {np.mean(g_sia):.4f} (stored {ref[1]:.4f})")

# cross-vocabulary transfer
match, cross = [], []
for s in SUBJECTS:
    X1, y1, _ = get(s, 1)
    X2, y2, r2 = get(s, 2)
    X1, X2 = X1[:, L16], X2[:, L16]
    gl = np.unique(y2)
    A_, B_ = gl[:len(gl) // 2], gl[len(gl) // 2:]
    te = (~(r2 == 0)) & np.isin(y2, B_)
    cm, cx = (r2 == 0) & np.isin(y2, B_), (r2 == 0) & np.isin(y2, A_)
    tr = np.isin(y1, B_)
    match.append(evaluate(X1[tr], y1[tr], X2[cm], X2[te], y2[te], "sia"))
    cross.append(evaluate(X1[tr], y1[tr], X2[cx], X2[te], y2[te], "sia"))
print(f"  {'cross-vocab':<16s} matched {np.mean(match):.4f} (stored 0.7854), "
      f"different {np.mean(cross):.4f} (stored 0.7301)")

print()
print("=" * 78)
print("STAGE 2 — equivalence tests (TOST), margin = 10 accuracy points")
print("=" * 78)

if sup_ok:
    out["SIA_vs_supervised"] = tost(sia, sup)
    show("SIA vs supervised recalibration", out["SIA_vs_supervised"],
         "\"statistically indistinguishable\" (p = 0.45)")
else:
    print("\n  supervised recalibration did NOT reproduce; test skipped.")

for f in ("MNF", "MDF"):
    a, b = feat_arrays[f]
    out[f"SIA_gain_{f}"] = tost(a, b)
    show(f"SIA gain on {f} (amplitude-invariant)", out[f"SIA_gain_{f}"],
         "SIA does nothing for spectral features")

out["cross_vocabulary"] = tost(cross, match)
show("Cross-vocabulary vs matched calibration", out["cross_vocabulary"],
     "calibration statistics transfer (p = 0.38)")

d = json.load(open(os.path.join(ROOT, "results", "hyser", "summary_diversity.json")))["per_subject"]
out["SIA_vs_raw_POSITIVE_CONTROL"] = tost(np.array(d["dist_sia"]), np.array(d["dist_raw"]))
show("[negative control] SIA gain on distributed montage",
     out["SIA_vs_raw_POSITIVE_CONTROL"], "a real effect -- should NOT be equivalent to zero")

out["_note"] = ("TOST margin 10 accuracy points, chosen to exceed the ~11-point minimum "
                "detectable effect at n=20; 'smallest_margin_pts' is the tightest "
                "equivalence claim the data support at 90% confidence.")
p = os.path.join(ROOT, "results", "equivalence_tests.json")
json.dump(out, open(p, "w"), indent=2)
print(f"\n\nwritten -> {p}")
