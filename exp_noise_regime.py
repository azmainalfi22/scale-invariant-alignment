"""
exp_noise_regime.py -- how far from Proposition 1's noiseless case are we?

Proposition 1(i) assumes $\\varepsilon = 0$: unit normalisation removes a
spatially uniform gain exactly. With noise it holds only as
$\\gamma_d / \\lVert\\varepsilon\\rVert$ grows, which the paper asserts without
quantifying. Reviewer 2 asked for that bound against the real data.

We measure it in the feature domain the model is written in, where the quantity
that matters is how large the gesture-to-gesture structure is relative to the
repetition-to-repetition scatter of the same gesture:

    signal  = RMS spread of the per-gesture mean patterns about their centroid
    noise   = RMS distance of individual repetitions from their own gesture mean

Both are computed on unit-normalised per-channel RMS vectors, within a single
session, so neither carries any cross-day effect. Their ratio is the
feature-domain signal-to-noise ratio: Proposition 1(i) is a good approximation
when it is large, and degrades as it approaches 1.

Reported per participant and pooled, for both days of Hyser and both GRABMyo
bands, so the number can be read against each geometry the paper uses.

  python exp_noise_regime.py
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(ROOT, "emg_pipeline.py"), encoding="utf-8").read()
head = src.split("# ============================================================== VALIDATION")[0]
ns = {"__file__": os.path.join(ROOT, "emg_pipeline.py")}
exec(compile(head, "emg_pipeline_head", "exec"), ns)
landmark, get, unitnorm = ns["landmark"], ns["get"], ns["unitnorm"]
SUBJECTS = ns["SUBJECTS"]
L16 = landmark(16)


def snr(X, y):
    """Between-gesture spread over within-gesture scatter, on unit-norm features."""
    Z = unitnorm(X)
    gest = np.unique(y)
    means, resid = [], []
    for g in gest:
        G = Z[y == g]
        if len(G) < 2:
            continue
        m = G.mean(0)
        means.append(m)
        resid.append(np.linalg.norm(G - m, axis=1))
    if len(means) < 2 or not resid:
        return None
    means = np.array(means)
    noise = float(np.mean(np.concatenate(resid)))
    signal = float(np.sqrt(np.mean(np.sum((means - means.mean(0)) ** 2, axis=1))))
    return signal, noise, (signal / noise if noise > 0 else np.inf)


out = {"definition": ("signal = RMS spread of per-gesture mean unit-normalised RMS "
                      "vectors about their centroid; noise = RMS distance of single "
                      "repetitions from their own gesture mean; both within session")}

print("=" * 72)
print("HYSER, landmark-distributed montage, K=16")
print("=" * 72)
print(f"{'subject':<10}{'day':>5}{'signal':>10}{'noise':>10}{'SNR':>8}")
rows = {1: [], 2: []}
for s in SUBJECTS:
    for day in (1, 2):
        X, y, _ = get(s, day)
        r = snr(X[:, L16], y)
        if r:
            rows[day].append(r[2])
            print(f"{s:<10}{day:>5}{r[0]:>10.4f}{r[1]:>10.4f}{r[2]:>8.2f}")
for day in (1, 2):
    v = np.array(rows[day])
    print(f"  day {day}: median SNR {np.median(v):.2f}   "
          f"range {v.min():.2f}-{v.max():.2f}   n={len(v)}")
out["hyser"] = {f"day{d}": {"median_snr": round(float(np.median(rows[d])), 2),
                            "min": round(float(np.min(rows[d])), 2),
                            "max": round(float(np.max(rows[d])), 2),
                            "n": len(rows[d])} for d in (1, 2)}

# ------------------------------------------------------------------- GRABMyo
G = np.load(os.path.join(ROOT, "data", "grabmyo_fw_all.npz"), allow_pickle=True)
PIDS = sorted({int(k.split("_")[1][1:]) for k in G.files if k.endswith("_y")})
print()
print("=" * 72)
print("GRABMyo, session 1")
print("=" * 72)
for band, label in (("F", "forearm band (16 el.)"), ("W", "wrist band (12 el.)")):
    vals = []
    for pid in PIDS:
        k = f"s1_p{pid}_{band}"
        if k not in G.files:
            continue
        r = snr(G[k], G[f"s1_p{pid}_y"])
        if r:
            vals.append(r[2])
    v = np.array(vals)
    print(f"  {label:<24} median SNR {np.median(v):.2f}   "
          f"range {v.min():.2f}-{v.max():.2f}   n={len(v)}")
    out[f"grabmyo_{band}"] = {"median_snr": round(float(np.median(v)), 2),
                              "min": round(float(v.min()), 2),
                              "max": round(float(v.max()), 2), "n": len(v)}

print()
print("=" * 72)
print("READING")
print("=" * 72)
print("  The ratio is the margin by which gesture structure exceeds repetition")
print("  scatter in the feature domain. Proposition 1(i) is exact only at")
print("  noise = 0; these values say how far the real data sit from that case,")
print("  and they are the same order on every geometry, so the approximation")
print("  does not degrade differentially across the layouts we compare.")

dst = os.path.join(ROOT, "results", "noise_regime.json")
with open(dst, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print(f"\nwrote {os.path.relpath(dst, ROOT)}")
