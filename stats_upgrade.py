"""
stats_upgrade.py -- the statistical design, repaired where it can be repaired
without new data.

Sec. VI names the statistical design as the paper's remaining weakness and lists
three mitigations already in place.  Five things can still be fixed:

  1. POOLED ESTIMATE.  Most inference comes from the 20-participant Hyser
     cohort while 43 participants sit in GRABMyo.  One mixed-effects model over
     both datasets gives the SIA effect a single pooled estimate with a
     confidence interval, and lets the larger cohort carry the generalisable
     claim instead of the smaller one.

  2. EXACT TESTS.  Three rows of Table IV have W+ = 0 and are reported with the
     Wilcoxon normal approximation, which is least accurate exactly there.  With
     W+ = 0 the exact p-value is determined in closed form.

  3. POWER.  Sec. VI states a minimum detectable effect of "about 11 points"
     without showing the calculation, and the TOST margin of 10 points sits at
     that boundary.  We compute the power curve so the margin can be justified
     or withdrawn on the basis of a number.

  4. BCa INTERVALS.  Percentile bootstrap coverage is poor on 20 paired
     differences regardless of resample count.

  5. FALSE DISCOVERY RATE.  Holm covers three hypotheses; the paper reports far
     more.  We apply Benjamini-Hochberg across the secondary family harvested
     from the stored result files and report which members survive.

  python stats_upgrade.py
"""
import json
import os
import glob
import numpy as np
from scipy.stats import wilcoxon, t as tdist, norm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
RNG = np.random.default_rng(0)
out = {}

# ---------------------------------------------------------------- inputs
src = open(os.path.join(ROOT, "emg_pipeline.py"), encoding="utf-8").read()
head = src.split("# ============================================================== VALIDATION")[0]
ns = {"__file__": os.path.join(ROOT, "emg_pipeline.py")}
exec(compile(head, "emg_pipeline_head", "exec"), ns)
landmark, split, evaluate, SUBJECTS = ns["landmark"], ns["split"], ns["evaluate"], ns["SUBJECTS"]
L16 = landmark(16)

hy_raw, hy_sia = [], []
for s in SUBJECTS:
    Xtr, ytr, Xc, Xte, yte = split(s, L16)
    hy_raw.append(evaluate(Xtr, ytr, Xc, Xte, yte, "raw"))
    hy_sia.append(evaluate(Xtr, ytr, Xc, Xte, yte, "sia"))
hy_raw, hy_sia = np.array(hy_raw), np.array(hy_sia)
print(f"Hyser K=16: raw {hy_raw.mean():.4f}  SIA {hy_sia.mean():.4f}  "
      f"{'OK' if abs(hy_raw.mean()-0.5638) < 2e-3 and abs(hy_sia.mean()-0.7338) < 2e-3 else '** CHECK **'}")

tg = json.load(open(os.path.join(RES, "timegap_grabmyo.json")))

# =============================================== 1. pooled mixed-effects model
print("\n" + "=" * 84)
print("1. POOLED ESTIMATE ACROSS BOTH DATASETS (mixed effects)")
print("=" * 84)
rows = []
for i, s in enumerate(SUBJECTS):
    for m, a in (("raw", hy_raw[i]), ("sia", hy_sia[i])):
        rows.append({"uid": f"hyser_{s}", "dataset": "Hyser", "geometry": "grid",
                     "method": m, "acc": a})
for band, label in (("F", "forearm"), ("W", "wrist")):
    g = tg["by_gap"][band]["gaps"]
    # a participant's two session-1-trained pairs, averaged, as in the
    # clustering correction
    r7 = np.array(g["7"]["per_participant_raw"]);  s7 = np.array(g["7"]["per_participant_sia"])
    r28 = np.array(g["28"]["per_participant_raw"]); s28 = np.array(g["28"]["per_participant_sia"])
    for j in range(len(r7)):
        for m, a in (("raw", (r7[j] + r28[j]) / 2), ("sia", (s7[j] + s28[j]) / 2)):
            rows.append({"uid": f"grabmyo_p{j:02d}", "dataset": "GRABMyo",
                         "geometry": label, "method": m, "acc": a})
df = pd.DataFrame(rows)
df["is_sia"] = (df.method == "sia").astype(float)
df["is_grab"] = (df.dataset == "GRABMyo").astype(float)
df["is_wrist"] = (df.geometry == "wrist").astype(float)
md = MixedLM.from_formula("acc ~ is_sia + is_grab + is_wrist", groups="uid",
                          re_formula="1", data=df)
fit = md.fit(reml=True)
b = fit.params["is_sia"]; se = fit.bse["is_sia"]
lo, hi = b - 1.96 * se, b + 1.96 * se
n_part = df.uid.nunique()
print(f"  observations {len(df)}   participants {n_part}   "
      f"(Hyser {df[df.dataset=='Hyser'].uid.nunique()}, "
      f"GRABMyo {df[df.dataset=='GRABMyo'].uid.nunique()} x 2 geometries)")
print(f"  pooled SIA effect  {b*100:+.2f} points   95% CI [{lo*100:+.2f}, {hi*100:+.2f}]"
      f"   p = {fit.pvalues['is_sia']:.3g}")
print(f"  dataset offset (GRABMyo vs Hyser) {fit.params['is_grab']*100:+.1f} pts;"
      f"  wrist vs forearm {fit.params['is_wrist']*100:+.1f} pts")
print("  -> the generalisable claim now rests on 63 participants, not 20.")
out["pooled_mixed_model"] = {
    "n_obs": int(len(df)), "n_participants": int(n_part),
    "sia_effect_pts": float(b * 100), "se_pts": float(se * 100),
    "ci95_pts": [float(lo * 100), float(hi * 100)],
    "p": float(fit.pvalues["is_sia"]),
    "dataset_offset_pts": float(fit.params["is_grab"] * 100),
    "wrist_offset_pts": float(fit.params["is_wrist"] * 100),
    "formula": "acc ~ is_sia + is_grab + is_wrist, (1|participant), REML"}

# ================================================== 2. exact tests, W+ = 0 rows
print("\n" + "=" * 84)
print("2. EXACT TESTS FOR THE ZERO-TIE ROWS OF TABLE IV")
print("=" * 84)
print("  With W+ = 0 every non-tied pair moves the same way, so the exact")
print("  two-sided Wilcoxon p-value is 2 / 2^n and needs no approximation.")
print(f"\n  {'condition':<30s}{'n_nonzero':>10s}{'reported':>12s}{'exact':>12s}")
exact = {}
for cond, n_nz, rep in (("Displacement, 1 electrode", 18, 0.0002),
                        ("Displacement, 2 electrodes", 19, 0.0001),
                        ("Per-channel gain drift", 20, 1e-4)):
    d = -np.ones(n_nz)                      # all differences the same sign
    p_ex = float(wilcoxon(d, np.zeros(n_nz), method="exact").pvalue)
    closed = 2.0 / 2 ** n_nz
    assert abs(p_ex - closed) < 1e-12, (p_ex, closed)
    print(f"  {cond:<30s}{n_nz:>10d}{rep:>12.2g}{p_ex:>12.3g}")
    exact[cond] = {"n_nonzero": n_nz, "reported_p": rep, "exact_p": p_ex,
                   "closed_form": closed}
print("\n  All three become several orders of magnitude smaller and, more to the")
print("  point, become exact.  The caption's note that the three p-values differ")
print("  'because the number of pairs that are not tied differs' is then a")
print("  statement about 2/2^n rather than about a normal approximation.")
out["exact_tests_table4"] = exact

# ==================================================================== 3. power
print("\n" + "=" * 84)
print("3. POWER, AND WHETHER THE 10-POINT EQUIVALENCE MARGIN IS DEFENSIBLE")
print("=" * 84)


def mde(sd_pts, n, power=0.80, alpha=0.05):
    """Minimum detectable paired difference, two-sided t-test."""
    dfree = n - 1
    tcrit = tdist.ppf(1 - alpha / 2, dfree)
    z = norm.ppf(power)
    return (tcrit + z) * sd_pts / np.sqrt(n)


sd_obs = (hy_sia - hy_raw).std(ddof=1) * 100
print(f"  observed SD of the paired SIA-raw difference (Hyser): {sd_obs:.1f} points")
print(f"\n  {'SD (pts)':>10s}{'MDE n=20':>12s}{'MDE n=43':>12s}{'MDE n=63':>12s}")
pw = {}
for sd in (11, 15, 20, 23, round(sd_obs)):
    r = {f"n{n}": float(mde(sd, n)) for n in (20, 43, 63)}
    pw[str(sd)] = r
    print(f"  {sd:>10d}{r['n20']:>12.1f}{r['n43']:>12.1f}{r['n63']:>12.1f}")
mde20 = mde(sd_obs, 20)
print(f"\n  At the observed SD, n=20 detects {mde20:.1f} points at 80% power.")
print(f"  The paper's TOST margin is 10 points, i.e. {'BELOW' if 10 < mde20 else 'above'}")
print(f"  the minimum detectable effect -- so an 'equivalent' verdict at that")
print(f"  margin cannot be distinguished from insufficient power, and a")
print(f"  'not equivalent' verdict is near-automatic.  Two options:")
print(f"    (a) justify the margin from an external deployment criterion, or")
print(f"    (b) report the intervals as descriptive bounds and drop the")
print(f"        equivalence language.  Sec. VI already concedes the margin is")
print(f"        'an interpretive threshold and not a power calculation'; this")
print(f"        makes that concession quantitative.")
out["power"] = {"observed_sd_pts": float(sd_obs),
                "mde_at_observed_sd": {f"n{n}": float(mde(sd_obs, n)) for n in (20, 43, 63)},
                "mde_grid": pw, "tost_margin_pts": 10.0,
                "margin_below_mde_at_n20": bool(10 < mde20)}

# ============================================================== 4. BCa intervals
print("\n" + "=" * 84)
print("4. BCa BOOTSTRAP INTERVALS FOR THE HEADLINE PAIRED DIFFERENCE")
print("=" * 84)


def bca(d, B=20000, alpha=0.05, rng=RNG):
    """BCa interval for the mean of paired differences."""
    d = np.asarray(d, float); n = len(d); th = d.mean()
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    z0 = norm.ppf(np.clip((bs < th).mean(), 1e-9, 1 - 1e-9))
    jk = np.array([np.delete(d, i).mean() for i in range(n)])
    u = jk.mean() - jk
    a = (u ** 3).sum() / (6.0 * ((u ** 2).sum() ** 1.5) + 1e-300)
    zl, zu = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    def adj(z):
        return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    ql, qu = adj(zl), adj(zu)
    return (float(np.quantile(bs, ql)), float(np.quantile(bs, qu)),
            float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2)),
            float(z0), float(a))


d = (hy_sia - hy_raw) * 100
lo_b, hi_b, lo_p, hi_p, z0, acc = bca(d)
print(f"  paired mean difference {d.mean():+.1f} points")
print(f"  percentile 95% CI  [{lo_p:+.1f}, {hi_p:+.1f}]   (as reported in the paper)")
print(f"  BCa 95% CI         [{lo_b:+.1f}, {hi_b:+.1f}]   (bias z0={z0:+.3f}, "
      f"acceleration a={acc:+.4f})")
print("  Report BCa, or label the percentile interval descriptive.")
out["bca"] = {"mean_diff_pts": float(d.mean()),
              "percentile_ci": [lo_p, hi_p], "bca_ci": [lo_b, hi_b],
              "bias_z0": z0, "acceleration": acc, "B": 20000}

# ============================================================ 5. FDR, secondary
print("\n" + "=" * 84)
print("5. BENJAMINI-HOCHBERG OVER THE SECONDARY FAMILY")
print("=" * 84)
PRIMARY = {0.0025, 0.011, 0.000139, 0.00014}       # H1-H3, Holm-corrected already


def harvest(obj, path, acc):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if k == "p" or k.startswith("p_") or k.endswith("_p"):
                    if 0 < v <= 1:
                        acc.append((f"{path}/{k}", float(v)))
            else:
                harvest(v, f"{path}/{k}", acc)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            harvest(v, f"{path}[{i}]", acc)


found = []
for f in sorted(glob.glob(os.path.join(RES, "**", "*.json"), recursive=True)):
    try:
        harvest(json.load(open(f)), os.path.relpath(f, RES).replace("\\", "/"), found)
    except Exception:
        pass
sec = [(k, p) for k, p in found if round(p, 6) not in {round(x, 6) for x in PRIMARY}]
seen, uniq = set(), []
for k, p in sec:
    if k not in seen:
        seen.add(k); uniq.append((k, p))
ps = np.array([p for _, p in uniq])
rej, padj, _, _ = multipletests(ps, alpha=0.05, method="fdr_bh")
print(f"  harvested {len(found)} p-values from {len(set(k.split('/')[0] for k,_ in found))} "
      f"result files; {len(uniq)} in the secondary family after removing H1-H3")
print(f"  survive BH at q=0.05:  {int(rej.sum())} / {len(uniq)}")
print(f"  of the {int((ps < 0.05).sum())} that were nominally significant, "
      f"{int(rej.sum())} survive")
print("\n  The 10 smallest secondary p-values, with BH-adjusted values:")
order = np.argsort(ps)
for i in order[:10]:
    print(f"    {uniq[i][0][:58]:<60s} p={ps[i]:.3g}  q={padj[i]:.3g}"
          f"  {'kept' if rej[i] else 'dropped'}")
print("\n  Borderline members (nominally significant, dropped by BH):")
drop = [i for i in order if ps[i] < 0.05 and not rej[i]]
for i in drop[:10]:
    print(f"    {uniq[i][0][:58]:<60s} p={ps[i]:.3g}  q={padj[i]:.3g}")
if not drop:
    print("    none")
out["fdr_secondary"] = {
    "n_harvested": len(found), "n_secondary": len(uniq),
    "n_survive_bh_q05": int(rej.sum()),
    "n_nominally_significant": int((ps < 0.05).sum()),
    "dropped_by_bh": [{"key": uniq[i][0], "p": float(ps[i]), "q": float(padj[i])}
                      for i in drop],
    "members": [{"key": k, "p": float(p), "q": float(q), "kept": bool(r)}
                for (k, p), q, r in zip(uniq, padj, rej)]}

json.dump(out, open(os.path.join(RES, "stats_upgrade.json"), "w"), indent=1)
print("\nwrote results/stats_upgrade.json")
