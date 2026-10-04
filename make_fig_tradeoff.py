"""
make_fig_tradeoff.py — regenerate Fig. 1 (the amplitude-stability trade-off).

The original plotting script was lost with the rest. The previous PNG had two
labelling faults, both fixed here:

  * its LEFT panel plotted the sparse nine-electrode patch result but labelled
    the bars "Distributed" / "Concentrated", which Sec. III-D defines as the
    K=16 montage configurations. The same two words therefore meant different
    things in the two panels of one figure.
  * its title read "amplitude-robustness trade-off" while the caption in the
    paper read "amplitude-stability trade-off".

Values are read from the stored result files, not hard-coded.

  python make_fig_tradeoff.py
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, "results")

sp = json.load(open(os.path.join(R, "hyser", "FINAL_all20.json")))["exp5_sparse_positive"]
dv = json.load(open(os.path.join(R, "hyser", "summary_diversity.json")))
m, pv = dv["means"], dv["p"]

DARK_B, LIGHT_B = "#4C72B0", "#A9C0DE"
DARK_O, LIGHT_O = "#DD8452", "#F2C0A0"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.2))

# ---- left: sparse nine-electrode patch, landmark site vs individual best site
d = sp["acc_delta_mean"] * 100   # stored mean of per-participant differences
ax1.bar([0, 1], [sp["acc_fixed_mean"], sp["acc_indiv_mean"]],
        color=[DARK_B, DARK_O], width=0.55, edgecolor="black", linewidth=0.6)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["Landmark site", "Individual best site"])
ax1.set_ylabel("accuracy")
ax1.set_ylim(0, 1)
ax1.set_title(f"WITHIN session, sparse 9-electrode patch\n"
              f"individualising wins (+{d:.1f} pts, $p$={sp['p_decode']:.4f})", fontsize=10)
ax1.text(1, sp["acc_indiv_mean"] + 0.02, "**", ha="center", fontsize=13)

# ---- right: K=16 montage geometry, distributed vs concentrated, raw and +SIA
x = [0, 0.62, 1.6, 2.22]
ax2.bar(x[0], m["dist_raw"], 0.55, color=DARK_B, edgecolor="black", linewidth=0.6, label="raw")
ax2.bar(x[1], m["dist_sia"], 0.55, color=LIGHT_B, edgecolor="black", linewidth=0.6, label="+ SIA")
ax2.bar(x[2], m["conc_raw"], 0.55, color=DARK_O, edgecolor="black", linewidth=0.6)
ax2.bar(x[3], m["conc_sia"], 0.55, color=LIGHT_O, edgecolor="black", linewidth=0.6)
ax2.set_xticks([(x[0] + x[1]) / 2, (x[2] + x[3]) / 2])
ax2.set_xticklabels(["Distributed", "Concentrated"])
ax2.set_ylabel("inter-day accuracy")
ax2.set_ylim(0, 1)
ax2.set_title(f"ACROSS days, matched budget ($K$=16)\n"
              f"concentrating loses ($p$={pv['dist_sia_vs_conc_sia_p']:.3f}); SIA does not close it",
              fontsize=10)
ax2.legend(frameon=False, loc="upper right", fontsize=9)

for ax in (ax1, ax2):
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("The amplitude–stability trade-off of individualised placement",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])

for dest in (os.path.join(ROOT, "paper", "figures", "fig_tradeoff.png"),
             os.path.join(ROOT, "build", "figures", "fig_tradeoff.png")):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    fig.savefig(dest, dpi=200)
    print("written ->", dest)

print(f"\nleft  panel: {sp['acc_fixed_mean']:.3f} -> {sp['acc_indiv_mean']:.3f} "
      f"(+{d:.1f} pts, p={sp['p_decode']})")
print(f"right panel: dist {m['dist_raw']:.4f}/{m['dist_sia']:.4f}  "
      f"conc {m['conc_raw']:.4f}/{m['conc_sia']:.4f}  p={pv['dist_sia_vs_conc_sia_p']:.5f}")
