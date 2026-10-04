"""
exp_posterior.py — posterior intervals for the results that sit on the 0.05 boundary.

Several conclusions in this work land at p = 0.044-0.074, where a reject/accept
verdict carries almost no information: p = 0.049 and p = 0.051 mean nearly the
same thing but get reported as opposite decisions. This replaces that dichotomy
with the quantity a reader actually wants -- the posterior probability that the
effect has the claimed sign, plus a credible interval.

Model: paired differences with a flat prior on the mean and Jeffreys prior on
log sigma, giving a Student-t posterior for the mean, t_{n-1}(xbar, s/sqrt(n)).
This is deliberately the most standard, least assumption-laden choice; with these
sample sizes the credible interval coincides numerically with the t confidence
interval, so nothing hinges on the prior. For the mixed-model interaction the
posterior is taken as normal(estimate, SE) from the fitted model.

Run:  python exp_posterior.py
"""
import json
import os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, "results")


def posterior(a, b=None, label="", claim_positive=True):
    """Student-t posterior for the paired mean difference, in accuracy points."""
    d = np.asarray(a, float) * 100 if b is None else (np.asarray(a, float) -
                                                      np.asarray(b, float)) * 100
    n = d.size
    m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    lo, hi = stats.t.ppf([0.025, 0.975], n - 1, loc=m, scale=se)
    p_pos = float(stats.t.sf(0, n - 1, loc=m, scale=se))
    p_dir = p_pos if claim_positive else 1 - p_pos
    try:
        pw = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        pw = 1.0
    return dict(label=label, n=int(n), mean_pts=round(float(m), 2),
                ci95=[round(float(lo), 2), round(float(hi), 2)],
                p_wilcoxon=round(pw, 4),
                posterior_prob_claimed_direction=round(p_dir, 4),
                prob_effect_exceeds_5pts=round(
                    float(stats.t.sf(5, n - 1, loc=abs(m), scale=se)), 3))


def show(r):
    print(f"  {r['label']:<42s} {r['mean_pts']:+6.2f} pts  "
          f"CI[{r['ci95'][0]:+6.2f},{r['ci95'][1]:+6.2f}]  "
          f"p={r['p_wilcoxon']:<7.4f} P(direction)={r['posterior_prob_claimed_direction']:.3f}")


out = {"_note": "Student-t posterior on paired differences (flat prior on mean, "
                "Jeffreys on log sigma). P(direction) = posterior probability the "
                "effect has the claimed sign.", "results": []}

print("=" * 96)
print("POSTERIOR INTERVALS FOR THE MARGINAL RESULTS")
print("=" * 96)

# --- 1. corruption: shift2 + SIA (p = 0.049) --------------------------------
c = json.load(open(os.path.join(R, "corruption_stats.json")))["per_subject"]
out["results"].append(posterior(c["shift2_sia"], c["shift2"],
                                "SIA repairs 2-electrode displacement"))

# --- 2. split-half oracle, raw and SIA (p = 0.059 / 0.055) ------------------
s = json.load(open(os.path.join(R, "sia_plus_shift.json")))["per_subject"]
out["results"].append(posterior(s["half_raw"], s["base_raw"],
                                "debiased shift oracle, raw features"))
out["results"].append(posterior(s["half_sia"], s["base_sia"],
                                "debiased shift oracle, on top of SIA"))

# --- 3. geometry contrast at K=32 after SIA (p = 0.051) --------------------
import csv
cells = list(csv.DictReader(open(os.path.join(R, "geometry_cells.csv"))))
for K in (4, 8, 16, 32):
    d_, c_ = {}, {}
    for row in cells:
        if int(row["K"]) == K and row["session"] == "across":
            (d_ if row["geometry"] == "dist" else c_)[row["pid"]] = float(row["acc"])
    pids = sorted(set(d_) & set(c_))
    out["results"].append(posterior([d_[p] for p in pids], [c_[p] for p in pids],
                                    f"distributed - concentrated at K={K} (SIA)"))

# --- 4. Laplacian SIA gain (p = 0.050) -------------------------------------
src = open(os.path.join(ROOT, "emg_pipeline.py"), encoding="utf-8").read()
ns = {"__file__": os.path.join(ROOT, "emg_pipeline.py")}
exec(compile(src.split("# ====================================================="
                       "========= VALIDATION")[0], "h", "exec"), ns)
evaluate, landmark, SUBJECTS = ns["evaluate"], ns["landmark"], ns["SUBJECTS"]
L16 = landmark(16)
M = np.load(os.path.join(ROOT, "data", "hyser_multifeat.npz"), allow_pickle=True)
lap_raw, lap_sia = [], []
for sub in SUBJECTS:
    X1, y1 = M[f"{sub}_ses1_LAP"], M[f"{sub}_ses1_y"]
    X2, y2, r2 = M[f"{sub}_ses2_LAP"], M[f"{sub}_ses2_y"], M[f"{sub}_ses2_rep"]
    cm = r2 == 0
    lap_raw.append(evaluate(X1[:, L16], y1, X2[cm][:, L16], X2[~cm][:, L16], y2[~cm], "raw"))
    lap_sia.append(evaluate(X1[:, L16], y1, X2[cm][:, L16], X2[~cm][:, L16], y2[~cm], "sia"))
out["results"].append(posterior(lap_sia, lap_raw, "SIA gain on Laplacian montage"))

for r in out["results"]:
    show(r)

# --- 5. mixed-model interaction (normal posterior from estimate/SE) --------
h = json.load(open(os.path.join(R, "hierarchical.json")))["models"]
print()
print("=" * 96)
print("MIXED-MODEL INTERACTION (the hotspot-trap differential)")
print("=" * 96)
out["interaction"] = {}
for feat in h:
    k = "geometry x session INTERACTION"
    est, se, p = h[feat][k]["estimate_pts"], h[feat][k]["se"], h[feat][k]["p"]
    pp = float(stats.norm.sf(0, loc=est, scale=se))
    p5 = float(stats.norm.sf(5, loc=est, scale=se))
    print(f"  {feat:<16s} {est:+.2f} pts  SE {se:.2f}  p={p:.4f}   "
          f"P(trap > 0) = {pp:.3f}   P(trap > 5 pts) = {p5:.3f}")
    out["interaction"][feat] = dict(estimate_pts=est, se=se, p=p,
                                    posterior_prob_positive=round(pp, 3),
                                    posterior_prob_exceeds_5pts=round(p5, 3))

p_ = os.path.join(R, "posterior_intervals.json")
json.dump(out, open(p_, "w"), indent=2)
print(f"\nwritten -> {p_}")
