"""
exp_step_decomposition.py -- what each of the two steps contributes, per geometry.

results/step_decomposition_grabmyo.json was written without a generating script,
so the wrist qualification in Sec. IV-E rested on a file nothing could rebuild.
This reproduces it.

The finding it carries: on the wrist band, per-channel standardisation ALONE
reaches 0.584 against 0.542 for the full two-step method, so step 1 costs 4.2
points there.  Step 1's contribution runs

    Hyser 8x8 grid (256 ch)   +7.8
    GRABMyo forearm (16 el.)  +1.7
    GRABMyo wrist   (12 el.)  -4.2

which tracks the spatial-information argument of Sec. IV-G with a sign change.
It is a qualification of the method, not of the science: step 2 helps on every
geometry, and it is step 2 that the paper identifies as AdaBN.

Protocol is inherited verbatim from exp_timegap.py -- train on session 1, the
earliest trial of the test session as unlabelled calibration, every test-session
trial outside it as test, tests at the participant level because each
participant contributes two session pairs that share their training data.

  python exp_step_decomposition.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(ROOT, "exp_timegap.py"), encoding="utf-8").read()
head = src.split("# ============================================================ collect")[0]
ns = {"__file__": os.path.join(ROOT, "exp_timegap.py")}
exec(compile(head, "exp_timegap_head", "exec"), ns)
PIDS, GAPS, BANDS = ns["PIDS"], ns["GAPS"], ns["BANDS"]
pair, wx = ns["pair"], ns["wx"]

METHODS = ("raw", "unitnorm", "zscore", "sia")
# step 1 = unit normalisation, step 2 = per-channel standardisation (AdaBN).
CONTRASTS = (("step1_vs_raw", "unitnorm", "raw"),
             ("step2_vs_raw", "zscore", "raw"),
             ("sia_vs_raw", "sia", "raw"),
             ("sia_vs_step2", "sia", "zscore"),
             ("step2_vs_step1", "zscore", "unitnorm"))

# ------------------------------------------------------------------ collect
rows = []
for band in BANDS:
    for pid in PIDS:
        for (a, b) in GAPS:
            if a != 1:            # session-1 training only, as in Table V
                continue
            r = pair(pid, band, a, b)
            if r:
                rows.append({"pid": pid, "band": band, **r})

out = {}
print("=" * 84)
print("PER-STEP DECOMPOSITION -- participant level, session-1 training")
print("=" * 84)
for band in BANDS:
    sub = [r for r in rows if r["band"] == band]
    byp = {}
    for r in sub:
        byp.setdefault(r["pid"], []).append(r)
    # average a participant's session pairs before testing: the two pairs share
    # their session-1 training data and are not independent observations.
    per = {m: np.array([np.mean([x[m] for x in v]) for v in byp.values()])
           for m in METHODS}
    rec = {"n": len(byp)}
    rec.update({m: float(per[m].mean()) for m in METHODS})
    print(f"{BANDS[band]}   n={len(byp)}")
    for m in METHODS:
        print(f"    {m:<10s} {per[m].mean():.4f}")
    for name, hi, lo in CONTRASTS:
        d, p = wx(per[hi], per[lo])
        rec[name] = {"delta_pts": round(d, 1), "p": float(f"{p:.4g}")}
        print(f"    {name:<16s} {d:+6.1f} pts   p={p:.4g}")
    print()
    out[band] = rec

# ------------------------------------------------------------- self-check
REF = os.path.join(ROOT, "results", "step_decomposition_grabmyo.json")
if os.path.exists(REF):
    old = json.load(open(REF))
    print("=" * 84)
    print("AGREEMENT WITH THE STORED FILE")
    print("=" * 84)
    worst = 0.0
    for band in BANDS:
        for m in METHODS:
            if m in old.get(band, {}):
                d = abs(out[band][m] - old[band][m])
                worst = max(worst, d)
                print(f"  {band} {m:<10s} rebuilt {out[band][m]:.4f}   "
                      f"stored {old[band][m]:.4f}   diff {d:.2e}")
    print(f"  worst absolute difference: {worst:.2e}"
          f"   {'OK' if worst < 5e-4 else '** CHECK **'}")

with open(REF, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print(f"wrote {os.path.relpath(REF, ROOT)}")
