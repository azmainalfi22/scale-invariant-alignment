"""
exp_corruption_stats.py — regenerate Table IV (controlled corruption) WITH statistics.

The original run stored only means and SDs, so Table IV of the paper carries no
significance tests and is missing a `shift2 + SIA` row. The rebuilt pipeline makes
both recoverable.

Protocol (as stated in the paper, mirroring the cross-day protocol on within-day data):
    train on repetitions 0-3 (clean, original sites);
    repetition 4 = unlabelled calibration, CORRUPTED;
    repetition 5 = test, CORRUPTED.

Corruptions:
    displacement d : the corrupted repetitions are read at channel positions
                     translated by d electrodes (the array "moved");
    gain drift     : the corrupted repetitions are multiplied by a per-channel
                     log-normal gain vector, exp(N(0, sigma^2)).

Because the original script's displacement axis and log-normal sigma were lost,
STAGE 1 searches for the configuration that reproduces the stored means before
STAGE 2 reports anything. The gain condition is averaged over several seeds, which
the single-seed original could not do.

Run:  python exp_corruption_stats.py
"""
import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---- reuse validated machinery --------------------------------------------
src = open(os.path.join(ROOT, "emg_pipeline.py"), encoding="utf-8").read()
head = src.split("# ============================================================== VALIDATION")[0]
ns = {"__file__": os.path.join(ROOT, "emg_pipeline.py")}
exec(compile(head, "emg_pipeline_head", "exec"), ns)
get, landmark, shift_channels = ns["get"], ns["landmark"], ns["shift_channels"]
fit_predict, evaluate, paired = ns["fit_predict"], ns["evaluate"], ns["paired"]
SUBJECTS, L16 = ns["SUBJECTS"], ns["landmark"](16)

vsrc = open(os.path.join(ROOT, "verify_stats.py"), encoding="utf-8").read()
vns = {"__file__": os.path.join(ROOT, "verify_stats.py")}
exec(compile(vsrc.split("out = {}")[0], "verify_stats_head", "exec"), vns)
compare = vns["compare"]

STORED = {"clean": 0.9286, "shift1": 0.6101, "shift2": 0.4092,
          "shift1_sia": 0.7914, "gain": 0.5223, "gain_sia": 0.9524}


def run(cond, axis=(0, 1), sigma=0.5, seed=0, method="raw"):
    """Per-participant accuracies for one corruption condition."""
    out = []
    rng = np.random.default_rng(seed)
    for s in SUBJECTS:
        X, y, r = get(s, 1)
        tr, ca, te = r <= 3, r == 4, r == 5
        Xtr = X[tr][:, L16]
        if cond == "clean":
            Xc, Xte = X[ca][:, L16], X[te][:, L16]
        elif cond.startswith("shift"):
            d = int(cond[5])
            chs = shift_channels(L16, axis[0] * d, axis[1] * d)
            if chs is None:
                continue
            Xc, Xte = X[ca][:, chs], X[te][:, chs]
        elif cond == "gain":
            g = np.exp(rng.normal(0.0, sigma, size=len(L16)))
            Xc, Xte = X[ca][:, L16] * g, X[te][:, L16] * g
        else:
            raise ValueError(cond)
        if method == "raw":
            out.append((fit_predict(Xtr, y[tr], Xte) == y[te]).mean())
        else:
            out.append(evaluate(Xtr, y[tr], Xc, Xte, y[te], method))
    return np.array(out)


# ================================================= STAGE 1: recover the config
print("=" * 78)
print("STAGE 1 — recover the lost corruption parameters by reproducing stored means")
print("=" * 78)
clean = run("clean")
print(f"  clean  {clean.mean():.4f}   stored {STORED['clean']:.4f}   "
      f"diff {clean.mean()-STORED['clean']:+.4f}")

print("\n  displacement axis search (target: shift1 0.6101, shift2 0.4092)")
best_axis, best_err = None, 1e9
for nm, ax in (("row + (longitudinal)", (1, 0)), ("row - (longitudinal)", (-1, 0)),
               ("col + (circumferential)", (0, 1)), ("col - (circumferential)", (0, -1))):
    s1, s2 = run("shift1", axis=ax).mean(), run("shift2", axis=ax).mean()
    err = abs(s1 - STORED["shift1"]) + abs(s2 - STORED["shift2"])
    print(f"    {nm:<24s} shift1 {s1:.4f}  shift2 {s2:.4f}   |err| {err:.4f}")
    if err < best_err:
        best_axis, best_err = ax, err
print(f"    -> chosen axis {best_axis}")

print("\n  log-normal sigma search (target: gain 0.5223)")
best_sig, best_gerr = None, 1e9
for sig in (0.25, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5):
    v = np.mean([run("gain", sigma=sig, seed=k).mean() for k in range(3)])
    err = abs(v - STORED["gain"])
    print(f"    sigma={sig:<5.2f} gain {v:.4f}   |err| {err:.4f}")
    if err < best_gerr:
        best_sig, best_gerr = sig, err
print(f"    -> chosen sigma {best_sig}")

REPRO = best_err < 0.05 and best_gerr < 0.05
print(f"\n  reproduction of stored means: "
      f"{'CLOSE (table values retained)' if REPRO else 'NOT EXACT (table regenerated)'}")

# ================================================= STAGE 2: the table, with stats
print()
print("=" * 78)
print("STAGE 2 — Table IV with per-participant statistics")
print("=" * 78)
SEEDS = range(5)
res = {
    "clean": clean,
    "clean_sia": run("clean", method="sia"),
    "shift1": run("shift1", axis=best_axis),
    "shift1_sia": run("shift1", axis=best_axis, method="sia"),
    "shift2": run("shift2", axis=best_axis),
    "shift2_sia": run("shift2", axis=best_axis, method="sia"),   # <-- previously missing
    "gain": np.mean([run("gain", sigma=best_sig, seed=k) for k in SEEDS], axis=0),
    "gain_sia": np.mean([run("gain", sigma=best_sig, seed=k, method="sia")
                         for k in SEEDS], axis=0),
}
gain_seed_sd = float(np.std([run("gain", sigma=best_sig, seed=k).mean() for k in SEEDS]))

print()
out = {"protocol": "train reps 0-3, calib rep 4 (corrupted), test rep 5 (corrupted)",
       "n": len(clean), "K": 16, "displacement_axis": list(best_axis),
       "lognormal_sigma": best_sig, "gain_seeds": len(SEEDS),
       "gain_across_seed_sd": round(gain_seed_sd, 4),
       "reproduces_stored_means": bool(REPRO), "stored_for_reference": STORED,
       "means": {k: round(float(v.mean()), 4) for k, v in res.items()},
       "comparisons": {}}

for lab, a, b in (("clean+SIA vs clean", "clean_sia", "clean"),
                  ("shift1 vs clean", "shift1", "clean"),
                  ("shift1+SIA vs shift1", "shift1_sia", "shift1"),
                  ("shift2 vs clean", "shift2", "clean"),
                  ("shift2+SIA vs shift2", "shift2_sia", "shift2"),
                  ("gain vs clean", "gain", "clean"),
                  ("gain+SIA vs gain", "gain_sia", "gain"),
                  ("gain+SIA vs clean", "gain_sia", "clean")):
    out["comparisons"][lab] = compare(lab, res[a], res[b])

out["per_subject"] = {k: [round(float(x), 4) for x in v] for k, v in res.items()}
p = os.path.join(ROOT, "results", "corruption_stats.json")
json.dump(out, open(p, "w"), indent=2)
print(f"\nwritten -> {p}")
