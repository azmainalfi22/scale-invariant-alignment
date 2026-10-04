"""
verify_stats.py — recompute every headline statistic in the paper directly from the
per-subject arrays stored in results/*.json.

Purpose: the original analysis scripts were lost (see CRITICAL_ANALYSIS_REPORT.md 2.9d).
Many result files still contain full per-subject arrays, so all paired comparisons,
effect sizes and bootstrap CIs can be regenerated without them. Every number printed
here is derived from a file in results/ and nothing else.

Run:  python verify_stats.py
"""
import json
import os
import numpy as np
from scipy.stats import wilcoxon

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
RNG = np.random.default_rng(0)          # fixed seed => reproducible CIs


def load(rel):
    with open(os.path.join(R, rel), "r", encoding="utf-8") as f:
        return json.load(f)


def rank_biserial(a, b):
    """Paired rank-biserial correlation, matching the convention r = (favourable
    signed-rank mass - unfavourable) / total, computed on non-zero differences."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    if d.size == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    pos = ranks[d > 0].sum()
    neg = ranks[d < 0].sum()
    return (pos - neg) / (pos + neg)


def boot_ci(a, b, n=20000):
    """Bootstrap 95% CI of the paired mean difference (a - b), in accuracy points."""
    d = np.asarray(a, float) - np.asarray(b, float)
    idx = RNG.integers(0, d.size, size=(n, d.size))
    means = d[idx].mean(axis=1) * 100.0
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def compare(label, a, b, note=""):
    """Paired comparison a vs b. Returns a dict and prints one line."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    assert a.shape == b.shape, f"{label}: shape mismatch {a.shape} vs {b.shape}"
    diff = (a.mean() - b.mean()) * 100.0
    try:
        p = wilcoxon(a, b).pvalue
    except ValueError:            # all differences zero
        p = 1.0
    lo, hi = boot_ci(a, b)
    r = rank_biserial(a, b)
    print(f"  {label:<46s} n={a.size:2d}  {a.mean():.4f} vs {b.mean():.4f}  "
          f"D={diff:+6.1f} pts  p={p:.4g}  r={r:+.2f}  CI[{lo:+.1f},{hi:+.1f}] {note}")
    return dict(n=int(a.size), mean_a=round(float(a.mean()), 4),
                mean_b=round(float(b.mean()), 4), delta_pts=round(float(diff), 1),
                p=float(p), rank_biserial=round(float(r), 3),
                ci95_pts=[round(float(lo), 1), round(float(hi), 1)])


out = {}

# ---------------------------------------------------------------- inter-day core
print("\n[1] Inter-day representations (Hyser, K=16, n=20) -- summary_adapt.json")
ad = load("hyser/summary_adapt.json")["per_subject"]
ceiling = load("hyser/summary_adapt.json")["within_day_ceiling"]
raw = ad["raw"]
out["interday"] = {}
for k in ["unitnorm", "log_unitnorm", "chan_gain", "chan_gain_unitnorm",
          "coral_diag", "coral_on_unitnorm", "rank"]:
    if k in ad:
        gap = (np.mean(ad[k]) - np.mean(raw)) / (ceiling - np.mean(raw)) * 100
        out["interday"][k] = compare(f"{k} vs raw", ad[k], raw,
                                     note=f"gap_rec={gap:.1f}%")

# ---------------------------------------------------------- classical normalisers
print("\n[2] Classical normalisation (summary_scaledrift.json per-subject)")
sd = load("hyser/summary_scaledrift.json")["per_subject"]
out["classical"] = {}
for k in ["log", "unitnorm", "proportion", "calib_scale"]:
    if k in sd:
        out["classical"][k] = compare(f"{k} vs raw", sd[k], sd["raw"])

# ------------------------------------------------------------ spatial corrections
print("\n[3] Spatial corrections (summary_scr.json / summary_carl.json, n=20)")
scr = load("hyser/summary_scr.json")
carl = load("hyser/summary_carl.json")
s, c = scr["per_subject"], carl["per_subject"]
out["spatial"] = {
    "shift_vs_raw":        compare("shift compensation vs raw",      s["E"], s["A"]),
    "shift_oracle_vs_raw": compare("shift ORACLE vs raw",            s["F"], s["A"]),
    "shift_oracle_vs_est": compare("shift ORACLE vs its estimate",   s["F"], s["E"]),
    "rank_vs_raw":         compare("rank re-localisation vs raw",    c["C"], c["A"]),
    "rank_oracle_vs_raw":  compare("rank ORACLE vs raw",             c["D"], c["A"]),
    "stale_vs_raw":        compare("stale individualised vs raw",    c["B"], c["A"]),
}
print(f"    mean estimated array movement = {scr['mean_shift_electrodes']} electrodes")

# ------------------------------------------------------------ geometry: conc/dist
print("\n[4] Concentrated vs distributed, K=16 inter-day (summary_diversity.json)")
dv = load("hyser/summary_diversity.json")["per_subject"]
out["geometry"] = {
    "dist_vs_conc_raw": compare("distributed vs concentrated (raw)",
                                dv["dist_raw"], dv["conc_raw"]),
    "dist_vs_conc_sia": compare("distributed vs concentrated (+SIA)",
                                dv["dist_sia"], dv["conc_sia"]),
    "sia_on_conc":      compare("SIA gain, concentrated",  dv["conc_sia"], dv["conc_raw"]),
    "sia_on_dist":      compare("SIA gain, distributed",   dv["dist_sia"], dv["dist_raw"]),
}

# -------------------------------------------------------------- within-session
print("\n[5] Within-session selection at K=8 (exp3_ablation.json, n=20)")
ab = load("hyser/exp3_ablation.json")
out["within_k8"] = {
    "amp_vs_fixed":      compare("amplitude-selected vs fixed",  ab["amp"], ab["fixed"]),
    "var_vs_fixed":      compare("variance-selected vs fixed",   ab["var"], ab["fixed"]),
    "perarray_vs_fixed": compare("coverage-constrained vs fixed", ab["perarray"], ab["fixed"]),
}

# ------------------------------------------------------------------ sparse patch
print("\n[6] Sparse-patch pipeline across days (summary_pipeline.json, n=20)")
pl = load("hyser/summary_pipeline.json")["per_subject"]
out["sparse_pipeline"] = {
    "indiv_vs_fixed": compare("individualised patch vs fixed patch", pl["P1"], pl["P0"]),
    "indiv_sia_vs_fixed": compare("individualised patch +SIA vs fixed", pl["P2"], pl["P0"]),
}

# --------------------------------------------------------------------- responders
print("\n[7] Responder analysis (summary_privacy.json)")
pv = load("hyser/summary_privacy.json")["responder"]
delta = np.asarray(pv["per_subject_delta"], float)
idx = RNG.integers(0, delta.size, size=(20000, delta.size))
bmeans = delta[idx].mean(axis=1) * 100
print(f"  improved={int((delta>0).sum())}/{delta.size}  worsened={int((delta<0).sum())}  "
      f"mean={delta.mean()*100:+.1f}  median={np.median(delta)*100:+.1f}  "
      f"CI[{np.percentile(bmeans,2.5):+.1f},{np.percentile(bmeans,97.5):+.1f}]")
out["responders"] = dict(n=int(delta.size), improved=int((delta > 0).sum()),
                         worsened=int((delta < 0).sum()),
                         mean_pts=round(float(delta.mean()*100), 1),
                         median_pts=round(float(np.median(delta)*100), 1),
                         ci95_pts=[round(float(np.percentile(bmeans, 2.5)), 1),
                                   round(float(np.percentile(bmeans, 97.5)), 1)])

# ------------------------------------------------------------------- GRABMyo n=8
print("\n[8] GRABMyo forearm, n=8 participants / 16 pairs (grabmyo_sia.json)")
gm = load("grabmyo_sia.json")["per_pair"]
out["grabmyo_n8"] = {
    "unitnorm_vs_raw": compare("unit-norm alone vs raw", gm["unitnorm"], gm["raw"]),
    "sia_vs_raw":      compare("SIA vs raw",             gm["sia"], gm["raw"]),
    "sia_vs_unitnorm": compare("SIA vs unit-norm alone", gm["sia"], gm["unitnorm"]),
}

# ------------------------------------------------------------------ rank vs gain
print("\n[9] Effective rank vs SIA gain (summary_rank_law.json)")
rl = load("hyser/summary_rank_law.json")["H7"]
print(f"  Spearman rho={rl['spearman_rho']:.3f} p={rl['p']:.3f}  "
      f"mean effective rank={rl['eff_rank_mean']:.2f} of 16")

with open(os.path.join(R, "verified_stats.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("\nWrote results/verified_stats.json")
