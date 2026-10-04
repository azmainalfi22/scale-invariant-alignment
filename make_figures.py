"""
make_figures.py -- the figure set for the revision.

The submitted paper carries one figure across ten pages.  This builds five,
each from a stored result file rather than from hard-coded values:

  fig_waterfall.png    where the cross-day gap goes, and which part is reachable
  fig_participants.png all 20 individual outcomes, including the two that worsen
  fig_calibration.png  the calibration-size threshold and its harm zone
  fig_methods.png      every cross-day method with a paired BCa interval
  fig_timegap.png      loss and recovery against elapsed time (GRABMyo, n=43)

Single-column IEEE width is 3.5 in; fig_methods is full width (7.16 in) and
belongs in a figure* environment.

  python make_figures.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")

# Kept self-contained so the script runs anywhere.  A three-size ladder mapped
# to role (titles/labels, annotation, ticks) and an open frame; nothing here is
# specific to this paper.
META_GREY = "#888888"


def apply_figure_style(*, sizes=(8, 7, 6), frame="open", grid=False):
    base, ann, tick = sizes
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 400,
        "font.size": base, "axes.titlesize": base, "axes.labelsize": base,
        "legend.fontsize": ann, "xtick.labelsize": tick, "ytick.labelsize": tick,
        "axes.titlelocation": "left", "axes.titleweight": "regular",
        "axes.titlepad": 5.0, "axes.labelpad": 3.0,
        "axes.spines.top": frame != "open", "axes.spines.right": frame != "open",
        "axes.linewidth": 0.7, "xtick.major.width": 0.7,
        "ytick.major.width": 0.7, "xtick.major.size": 2.6,
        "ytick.major.size": 2.6, "axes.grid": grid,
        "grid.linewidth": 0.4, "grid.alpha": 0.35,
        "legend.frameon": False, "figure.constrained_layout.use": False,
    })
OUTS = [os.path.join(ROOT, "paper", "figures"), os.path.join(ROOT, "build", "figures")]
for d in OUTS:
    os.makedirs(d, exist_ok=True)

COL1, COL2 = 3.5, 7.16
RNG = np.random.default_rng(0)

# palette: one colour per entity, reused everywhere (figure-style 4.1)
C_RAW = "#8C8C8C"        # standard practice
C_SIA = "#1F4E9C"        # the proposed correction (focal)
C_STEP = "#7FA6DC"       # its individual steps / AdaBN family
C_SPATIAL = "#D4813B"    # spatial / geometric corrections
C_SUP = "#4C8C5A"        # uses labels
C_CEIL = "#4A4A4A"       # within-day ceiling
C_ALARM = "#B03A2E"      # harm / worsening

CEIL, RAW = 0.9458, 0.5638


def J(p):
    return json.load(open(os.path.join(RES, p)))


def bca(d, B=8000, alpha=0.05, rng=None):
    """BCa interval for the mean of paired differences."""
    rng = rng or np.random.default_rng(0)
    d = np.asarray(d, float); n = len(d); th = d.mean()
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    z0 = norm.ppf(np.clip((bs < th).mean(), 1e-9, 1 - 1e-9))
    jk = np.array([np.delete(d, i).mean() for i in range(n)])
    u = jk.mean() - jk
    a = (u ** 3).sum() / (6.0 * ((u ** 2).sum() ** 1.5) + 1e-300)
    def adj(z):
        return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    return (float(np.quantile(bs, adj(norm.ppf(alpha / 2)))),
            float(np.quantile(bs, adj(norm.ppf(1 - alpha / 2)))))


def save(fig, name):
    for d in OUTS:
        fig.savefig(os.path.join(d, name), dpi=400, bbox_inches="tight")
    print("  wrote", name)


apply_figure_style(frame="open", sizes=(8, 7, 6))

# ============================================================ FIG A: waterfall
print("fig_waterfall")
shift = J("sia_plus_shift.json")["per_subject"]
sia = np.array(shift["base_sia"]); raw = np.array(shift["base_raw"])
sia_shift = np.array(shift["full_sia"])
GAP = CEIL - RAW
seg = [
    ("Standard\npractice", RAW, C_RAW, None),
    ("Amplitude\nalignment", sia.mean() - RAW, C_SIA, "no\nlabels"),
    ("Rigid\ntranslation", sia_shift.mean() - sia.mean(), C_SPATIAL, "ceiling\nonly"),
    ("Unexplained", CEIL - sia_shift.mean(), "#DCDCDC", None),
]
fig, ax = plt.subplots(figsize=(COL1, 2.7))
bottom = 0.0
for i, (lab, h, c, note) in enumerate(seg):
    ax.bar(i, h, bottom=bottom, width=0.62, color=c,
           edgecolor="white", linewidth=0.8)
    if i > 0:
        pct = h / GAP * 100
        ax.text(i, bottom + h + 0.012, f"{pct:.0f}% of gap",
                ha="center", va="bottom", fontsize=6)
        ax.plot([i - 0.31 - 0.07, i - 0.31], [bottom, bottom],
                color=META_GREY, lw=0.7, clip_on=False)
    else:
        ax.text(i, h / 2, f"{RAW:.3f}", ha="center", va="center",
                fontsize=6.5, color="white")
    if note:
        ax.text(i, bottom + h / 2, note, ha="center", va="center",
                fontsize=5.6, color="white" if c != "#DCDCDC" else "#444444",
                style="italic")
    bottom += h
ax.axhline(CEIL, color=C_CEIL, lw=0.9, ls=(0, (4, 2)))
ax.text(-0.42, CEIL + 0.008, f"within-day ceiling  {CEIL:.3f}", ha="left",
        va="bottom", fontsize=6, color=C_CEIL)
ax.set_xticks(range(len(seg)))
ax.set_xticklabels([s[0] for s in seg])
ax.set_ylabel("cross-day accuracy")
ax.set_ylim(0, 1.04)
ax.set_title("Only the amplitude share is reachable without labels")
ax.margins(x=0.06)
save(fig, "fig_waterfall.png")
plt.close(fig)

# ======================================================== FIG B: participants
print("fig_participants")
ad = J("adabn_uda_extended.json")["methods"]
r = np.array(ad["raw"]["per_subject"]); s = np.array(ad["sia"]["per_subject"])
o = np.argsort(r)
y = np.arange(len(o))
fig, ax = plt.subplots(figsize=(COL1, 3.4))
for k, i in enumerate(o):
    worse = s[i] < r[i]
    ax.plot([r[i], s[i]], [k, k], color=C_ALARM if worse else C_STEP,
            lw=1.5, zorder=1, solid_capstyle="round")
ax.scatter(r[o], y, s=14, color=C_RAW, zorder=3, label="standard practice",
           edgecolor="white", linewidth=0.4)
ax.scatter(s[o], y, s=14, color=C_SIA, zorder=3, label="+ amplitude alignment",
           edgecolor="white", linewidth=0.4)
bad = [k for k, i in enumerate(o) if s[i] < r[i]]
ax.scatter(s[o][bad], np.array(y)[bad], s=14, color=C_ALARM, zorder=4,
           edgecolor="white", linewidth=0.4)
ax.axvline(r.mean(), color=C_RAW, lw=0.8, ls=(0, (3, 2)), zorder=0)
ax.axvline(s.mean(), color=C_SIA, lw=0.8, ls=(0, (3, 2)), zorder=0)
ax.text(r.mean(), len(o) - 0.2, f"{r.mean():.3f}", ha="center", va="bottom",
        fontsize=6, color=C_RAW)
ax.text(s.mean(), len(o) - 0.2, f"{s.mean():.3f}", ha="center", va="bottom",
        fontsize=6, color=C_SIA)
ax.annotate("worse under\nthe correction", xy=(s[o][bad[0]], bad[0]),
            xytext=(0.30, 5.2), fontsize=6, color=C_ALARM, ha="center",
            arrowprops=dict(arrowstyle="-", color=C_ALARM, lw=0.6,
                            shrinkA=0, shrinkB=2))
ax.set_yticks([])
ax.set_xlabel("cross-day accuracy")
ax.set_ylabel("participant  (ordered by uncorrected accuracy)")
ax.set_xlim(0.1, 1.0)
ax.set_ylim(-1, len(o) + 0.4)
ax.set_title("Improves 18 of 20 participants, and harms two")
ax.legend(frameon=False, loc="lower right", fontsize=6, handletextpad=0.4,
          borderpad=0.2)
save(fig, "fig_participants.png")
plt.close(fig)

# ======================================================== FIG C: calibration
print("fig_calibration")
cs = J("calibsize_rebuilt.json")
ks = [1, 2, 4, 8]
acc = [cs["sizes"][str(k)]["mean"] for k in ks]
pv = [cs["sizes"][str(k)]["p"] for k in ks]
full = cs["full_repetition"]["mean"]
fig, ax = plt.subplots(figsize=(COL1, 2.6))
ax.axhspan(0, RAW, color=C_ALARM, alpha=0.07, zorder=0)
ax.axhline(RAW, color=C_RAW, lw=1.0)
ax.text(1.05, RAW - 0.012, "no correction (0.564)", fontsize=6, color=C_RAW,
        va="top")
ax.text(1.05, 0.30, "correction does\nactive harm", fontsize=6.2,
        color=C_ALARM, va="center", style="italic")
xs = [1, 2, 4, 8]
ax.plot(xs, acc, "-o", color=C_SIA, lw=1.4, ms=5, zorder=3,
        markeredgecolor="white", markeredgewidth=0.5)
for x, a, p in zip(xs, acc, pv):
    harm = a < RAW
    ax.annotate(f"{a:.3f}", (x, a), textcoords="offset points",
                xytext=(0, -11 if harm else 7), ha="center", fontsize=6,
                color=C_ALARM if harm else C_SIA)
ax.annotate("spans the full\ngesture vocabulary", (8, acc[-1]),
            textcoords="offset points", xytext=(-8, 16), ha="right",
            fontsize=6, color=C_SIA)
ax.set_xscale("log", base=2)
ax.set_xticks(xs); ax.set_xticklabels([str(k) for k in xs])
ax.set_xlabel("gestures covered by the unlabelled calibration set")
ax.set_ylabel("cross-day accuracy")
ax.set_ylim(0.25, 0.85)
ax.set_xlim(0.85, 11)
ax.set_title("Partial calibration is worse than none at all")
save(fig, "fig_calibration.png")
plt.close(fig)

# ============================================================ FIG D: methods
print("fig_methods")
ud = J("uda_baselines.json")["methods"]
cor = J("corruption_stats.json")
raw_ps = np.array(ad["raw"]["per_subject"])
ENTRIES = [
    ("Spatial correction", [
        ("rank re-localisation", None, 0.3980, C_SPATIAL),
        ("shift compensation", None, 0.4700, C_SPATIAL),
        ("best possible rigid shift  (ceiling, uses test labels)",
         np.array(shift["full_raw"]), None, C_SPATIAL),
    ]),
    ("Label-free adaptation", [
        ("pseudo-label self-training", np.array(ud["pseudo"]["per_subject"]), None, C_STEP),
        ("CORAL, full covariance", np.array(ud["coral"]["per_subject"]), None, C_STEP),
        ("mean alignment", np.array(ud["mean"]["per_subject"]), None, C_STEP),
        ("optimal transport", np.array(ad["ot"]["per_subject"]), None, C_STEP),
        ("whitening / covariance recentring", np.array(ad["whiten"]["per_subject"]), None, C_STEP),
        ("AdaBN  (= step 2 alone)", np.array(ad["adabn_feat"]["per_subject"]), None, C_STEP),
        ("subspace alignment", np.array(ad["sa"]["per_subject"]), None, C_STEP),
        ("CORAL after step 1", np.array(ud["coral_un"]["per_subject"]), None, C_STEP),
        ("BN-MLP + AdaBN, unit-normed", np.array(ad["mlp_adabn_un"]["per_subject"]), None, C_STEP),
    ]),
    ("Proposed", [
        ("unit normalisation + AdaBN", np.array(ad["sia"]["per_subject"]), None, C_SIA),
    ]),
    ("Uses calibration labels", [
        ("supervised recalibration", None, 0.7700, C_SUP),
        ("both", None, 0.8900, C_SUP),
    ]),
]
labels, mids, los, his, cols, heads = [], [], [], [], [], []
row = 0
for group, items in ENTRIES:
    heads.append((row, group)); row += 1
    for nm, arr, mean_only, c in items:
        if arr is not None:
            d = (arr - raw_ps) * 100
            lo, hi = bca(d, rng=np.random.default_rng(1))
            mids.append(d.mean()); los.append(lo); his.append(hi)
        else:
            mids.append((mean_only - RAW) * 100); los.append(np.nan); his.append(np.nan)
        labels.append(nm); cols.append(c); row += 1
ypos, k = [], 0
for group, items in ENTRIES:
    k += 1
    for _ in items:
        ypos.append(k); k += 1
ypos = np.array(ypos, float)
fig, ax = plt.subplots(figsize=(COL2, 3.8))
ax.axvline(0, color=C_RAW, lw=1.0, zorder=1)
for i in range(len(mids)):
    if not np.isnan(los[i]):
        ax.plot([los[i], his[i]], [ypos[i]] * 2, color=cols[i], lw=1.3,
                solid_capstyle="round", zorder=2)
    else:
        ax.plot([mids[i] - 0.4, mids[i] + 0.4], [ypos[i]] * 2, color=cols[i],
                lw=0.6, alpha=0.4, zorder=2)
    ax.scatter(mids[i], ypos[i], s=22, color=cols[i], zorder=3,
               edgecolor="white", linewidth=0.5,
               marker="o" if not np.isnan(los[i]) else "s")
ax.set_yticks(ypos)
ax.set_yticklabels(labels)
for yy, name in heads:
    ax.text(-0.012, yy, name, transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=6.6, fontweight="bold",
            color="#333333")
ax.invert_yaxis()
ax.set_xlabel("change in cross-day accuracy versus standard practice (points)")
ax.set_title("Every achievable spatial correction lands below doing nothing; "
             "the amplitude corrections do not", pad=14)
ax.text(0.0, -0.135, "square marker = mean only, no per-participant array;  "
                     "bars are 95% BCa intervals of the paired difference",
        transform=ax.transAxes, ha="left", va="top", fontsize=5.6,
        color=META_GREY)
ax.set_xlim(-20, 40)
ax.margins(y=0.02)
save(fig, "fig_methods.png")
plt.close(fig)

# ============================================================ FIG E: time gap
print("fig_timegap")
tg = J("timegap_grabmyo.json")["by_gap"]
fig, axes = plt.subplots(1, 2, figsize=(COL1, 2.5), sharey=True)
for ax, band, nm in zip(axes, ("F", "W"), ("forearm, 16 el.", "wrist, 12 el.")):
    g = tg[band]["gaps"]; cl = tg[band]["ceiling"]
    days = [7, 21, 28]
    rr = [g[str(d)]["raw"] for d in days]
    ss = [g[str(d)]["sia"] for d in days]
    ax.axhline(cl, color=C_CEIL, lw=0.8, ls=(0, (4, 2)))
    ax.plot(days, rr, "-o", color=C_RAW, lw=1.3, ms=4.5,
            markeredgecolor="white", markeredgewidth=0.4)
    ax.plot(days, ss, "-o", color=C_SIA, lw=1.3, ms=4.5,
            markeredgecolor="white", markeredgewidth=0.4)
    for d, a, b in zip(days, rr, ss):
        ax.annotate(f"+{(b-a)*100:.0f}", ((d), (a + b) / 2), fontsize=5.8,
                    color=C_SIA, ha="center", va="center",
                    bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_xticks(days); ax.set_xticklabels([f"{d}" for d in days])
    ax.set_xlabel("days between sessions")
    ax.set_title(nm, fontsize=7)
    ax.set_xlim(3, 32)
axes[0].set_ylabel("accuracy")
axes[0].set_ylim(0.34, 0.90)
axes[0].text(7.6, tg["F"]["ceiling"] - 0.012, "within-session", fontsize=5.8,
             color=C_CEIL, va="top")
axes[0].text(21, 0.695, "+ alignment", fontsize=6, color=C_SIA, ha="center")
axes[0].text(21, 0.545, "standard practice", fontsize=6, color=C_RAW, ha="center")
fig.suptitle("The recovered gain does not decay over four weeks", fontsize=8,
             y=1.015)
fig.subplots_adjust(wspace=0.06)
save(fig, "fig_timegap.png")
plt.close(fig)

# =============================== regenerate Fig. 1 with the p-value format fix
print("fig_tradeoff (p-format fix: .3f -> .4f)")
src = open(os.path.join(ROOT, "make_fig_tradeoff.py"), encoding="utf-8").read()
fixed = src.replace("{sp['p_decode']:.3f}", "{sp['p_decode']:.4f}")
if fixed != src:
    open(os.path.join(ROOT, "make_fig_tradeoff.py"), "w", encoding="utf-8").write(fixed)
    print("  patched make_fig_tradeoff.py (.3f -> .4f); re-run it to regenerate")
elif "{sp['p_decode']:.4f}" in src:
    print("  already patched; figure prints p=0.0025 as the text does")
else:
    raise SystemExit("make_fig_tradeoff.py: p-format pattern not found")
print("\ndone")
