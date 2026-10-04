"""
exp_sia_plus_shift.py — does a rigid geometric correction add anything ON TOP of SIA?

Open question left by paper_revised Sec. IV-F, which states that the residual
cross-day loss "is reached by neither scale nor rigid spatial correction". Each
method ALONE was tested; their COMPOSITION was not. This runs it.

Protocol (matches the accuracy-oracle of emg_pipeline.py EXP 1):
    train on day-1 at the ORIGINAL montage sites;
    read day-2 calibration AND test at the SHIFTED sites;
    search all integer translations (dr,dc) in [-3,3]^2 inside each array.

Two estimators are reported, because they answer different questions:

  (a) FULL-TEST ORACLE — pick the shift maximising accuracy on the whole day-2
      test set. This is the paper's existing protocol, so it is comparable to the
      published 0.6794. It is NOT attainable and it is upward-biased: maximising
      over ~49 candidates on the same data it is scored on inflates the result
      even when no real shift helps. Reported for both raw and SIA, so the two
      share an identical bias and the CONTRAST between them is interpretable.

  (b) SPLIT-HALF ORACLE — pick the shift on one half of the day-2 test
      repetitions, score it on the held-out half, both directions, averaged.
      This removes the selection bias and estimates what a perfect *label-using*
      shift chooser would actually deliver.

Both are bounded below by their own do-nothing baseline, because (0,0) is in the
search space.

Run:  python exp_sia_plus_shift.py
"""
import json
import os
import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---- reuse the validated machinery from the rebuilt pipeline ----------------
src = open(os.path.join(ROOT, "emg_pipeline.py"), encoding="utf-8").read()
head = src.split("# ============================================================== VALIDATION")[0]
ns = {"__file__": os.path.join(ROOT, "emg_pipeline.py")}
exec(compile(head, "emg_pipeline_head", "exec"), ns)

get, split, landmark = ns["get"], ns["split"], ns["landmark"]
shift_channels, unitnorm = ns["shift_channels"], ns["unitnorm"]
fit_predict, evaluate, paired = ns["fit_predict"], ns["evaluate"], ns["paired"]
SUBJECTS = ns["SUBJECTS"]
L16 = landmark(16)
SHIFTS = [(dr, dc) for dr in range(-3, 4) for dc in range(-3, 4)]


def correct_vec(Xtr, ytr, Xcal, Xte, yte, method):
    """Per-sample correctness, so accuracy can be taken on arbitrary subsets."""
    if method == "raw":
        A, B = Xtr, Xte
    elif method == "sia":
        A, C, B = unitnorm(Xtr), unitnorm(Xcal), unitnorm(Xte)
        mS, sS = A.mean(0), A.std(0) + 1e-12
        mC, sC = C.mean(0), C.std(0) + 1e-12
        B = (B - mC) * sS / sC + mS
    else:
        raise ValueError(method)
    return fit_predict(A, ytr, B) == yte


# ---- sanity: reproduce the two published anchors before doing anything new --
base_raw, base_sia = [], []
for s in SUBJECTS:
    Xtr, ytr, Xc, Xte, yte = split(s, L16)
    base_raw.append(evaluate(Xtr, ytr, Xc, Xte, yte, "raw"))
    base_sia.append(evaluate(Xtr, ytr, Xc, Xte, yte, "sia"))
print("=" * 74)
print("ANCHOR CHECK (must match published values before new results are used)")
print("=" * 74)
for nm, got, ref in (("raw", np.mean(base_raw), 0.5638), ("SIA", np.mean(base_sia), 0.7338)):
    d = got - ref
    print(f"  {nm:<4s} rebuilt {got:.4f}  stored {ref:.4f}  diff {d:+.4f}"
          f"  {'OK' if abs(d) < 0.02 else '** MISMATCH **'}")
assert abs(np.mean(base_raw) - 0.5638) < 0.02 and abs(np.mean(base_sia) - 0.7338) < 0.02

# ---- the experiment --------------------------------------------------------
full = {"raw": [], "sia": []}      # full-test oracle
half = {"raw": [], "sia": []}      # split-half oracle
nonzero = {"raw": 0, "sia": 0}

for s in SUBJECTS:
    X1, y1, _ = get(s, 1)
    X2, y2, r2 = get(s, 2)
    cal, tst = (r2 == 0), (r2 != 0)
    yte = y2[tst]
    reps = r2[tst]
    uniq = np.unique(reps)
    hA = np.isin(reps, uniq[: len(uniq) // 2])   # first half of test reps
    hB = ~hA
    Xtr0 = X1[:, L16]

    acc = {"raw": [], "sia": []}
    for dr, dc in SHIFTS:
        chs = shift_channels(L16, dr, dc)
        if chs is None:
            acc["raw"].append(None); acc["sia"].append(None); continue
        Xc, Xte = X2[cal][:, chs], X2[tst][:, chs]
        for m in ("raw", "sia"):
            acc[m].append(correct_vec(Xtr0, y1, Xc, Xte, yte, m))

    for m in ("raw", "sia"):
        v = acc[m]
        ok = [i for i, c in enumerate(v) if c is not None]
        # (a) full-test oracle
        best = max(ok, key=lambda i: v[i].mean())
        full[m].append(v[best].mean())
        if SHIFTS[best] != (0, 0):
            nonzero[m] += 1
        # (b) split-half oracle, both directions
        iA = max(ok, key=lambda i: v[i][hA].mean())
        iB = max(ok, key=lambda i: v[i][hB].mean())
        half[m].append((v[iA][hB].mean() + v[iB][hA].mean()) / 2)

print()
print("=" * 74)
print("RESULT — does a rigid shift add anything on top of SIA?")
print("=" * 74)
out = {"n": len(SUBJECTS), "K": 16, "shifts_searched": len(SHIFTS)}

print("\n(a) FULL-TEST ORACLE  [upward-biased; comparable to the paper's 0.6794]")
for m, b in (("raw", base_raw), ("sia", base_sia)):
    d, p = paired(full[m], b)
    print(f"    {m:<4s} {np.mean(b):.4f} -> {np.mean(full[m]):.4f}   "
          f"({d:+.1f} pts, p={p:.4g}), non-zero shift {nonzero[m]}/{len(SUBJECTS)}")
    out[f"full_oracle_{m}"] = {"base": round(float(np.mean(b)), 4),
                               "oracle": round(float(np.mean(full[m])), 4),
                               "delta_pts": round(float(d), 1), "p": float(p),
                               "nonzero_shift": nonzero[m]}

print("\n(b) SPLIT-HALF ORACLE  [selection bias removed; attainable with labels]")
for m, b in (("raw", base_raw), ("sia", base_sia)):
    d, p = paired(half[m], b)
    print(f"    {m:<4s} {np.mean(b):.4f} -> {np.mean(half[m]):.4f}   ({d:+.1f} pts, p={p:.4g})")
    out[f"halfsplit_oracle_{m}"] = {"base": round(float(np.mean(b)), 4),
                                    "oracle": round(float(np.mean(half[m])), 4),
                                    "delta_pts": round(float(d), 1), "p": float(p)}

print("\n(c) THE COMPARISON THAT ANSWERS THE QUESTION")
dr_, pr_ = paired(full["raw"], base_raw)
ds_, ps_ = paired(full["sia"], base_sia)
print(f"    shift gain on RAW features : {dr_:+.1f} pts (p={pr_:.4g})")
print(f"    shift gain on SIA features : {ds_:+.1f} pts (p={ps_:.4g})")
print(f"    -> the geometric gain shrinks by {dr_ - ds_:+.1f} points once scale is corrected")
d3, p3 = paired(full["sia"], full["raw"])
print(f"    SIA+oracle {np.mean(full['sia']):.4f} vs raw+oracle {np.mean(full['raw']):.4f}"
      f"  ({d3:+.1f} pts, p={p3:.4g})")
out["shift_gain_raw_pts"] = round(float(dr_), 1)
out["shift_gain_sia_pts"] = round(float(ds_), 1)
out["gain_shrinkage_pts"] = round(float(dr_ - ds_), 1)
out["sia_oracle_vs_raw_oracle"] = {"delta_pts": round(float(d3), 1), "p": float(p3)}
out["per_subject"] = {k: [round(float(x), 4) for x in v]
                      for k, v in (("full_raw", full["raw"]), ("full_sia", full["sia"]),
                                   ("half_raw", half["raw"]), ("half_sia", half["sia"]),
                                   ("base_raw", base_raw), ("base_sia", base_sia))}

p = os.path.join(ROOT, "results", "sia_plus_shift.json")
json.dump(out, open(p, "w"), indent=2)
print(f"\nwritten -> {p}")
