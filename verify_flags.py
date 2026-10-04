"""verify_flags.py — resolve the five AUTHOR-CHECK flags against the stored arrays.

B1  amplitude retention at 1/2/3 electrodes of offset
B2  Table IV rows printed with r = -1.00 and three different p-values
B3  full-calibration condition 0.728 (+16.4) vs SIA 0.7338 (+17.0)
B4  "2 of 19 paired participants" against a cohort of 20
B5  which eight Hyser gestures, and which task type

  python verify_flags.py
"""
import json
import os
import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(ROOT, "emg_pipeline.py"), encoding="utf-8").read()
head = src.split("# ============================================================== VALIDATION")[0]
ns = {"__file__": os.path.join(ROOT, "emg_pipeline.py")}
exec(compile(head, "emg_pipeline_head", "exec"), ns)
get, rc, ch_of, GRID = ns["get"], ns["rc"], ns["ch_of"], ns["GRID"]
SUBJECTS, CACHE = ns["SUBJECTS"], ns["CACHE"]

BAR = "=" * 78


# ------------------------------------------------------------------------ B1
print(BAR)
print("B1  amplitude retention at 1/2/3 electrodes of offset")
print(BAR)
cols, rows = {d: [] for d in range(-4, 5)}, {d: [] for d in range(-4, 5)}
for s in SUBJECTS:
    X = get(s, 1)[0]
    m = X.mean(0)
    pk = int(np.argmax(m))
    b, r0, c0 = rc(pk)
    for d in range(-4, 5):
        c = c0 + d
        cols[d].append(m[ch_of(b, r0, c)] / m[pk] * 100 if 0 <= c < GRID else np.nan)
        r = r0 + d
        rows[d].append(m[ch_of(b, r, c0)] / m[pk] * 100 if 0 <= r < GRID else np.nan)

col_mean = {d: np.nanmean(v) for d, v in cols.items()}
row_mean = {d: np.nanmean(v) for d, v in rows.items()}
print("  across columns, mean %% retained:")
print("   ", {d: round(col_mean[d], 1) for d in range(-4, 5)})
print("  along rows, mean %% retained:")
print("   ", {d: round(row_mean[d], 1) for d in range(-4, 5)})

stored = json.load(open(os.path.join(ROOT, "results", "hyser", "FINAL_all20.json")))
st = stored["exp2_variability_tolerance"]
print(f"\n  stored retention_pct_mean : {st['retention_pct_mean']}")
print(f"  stored retention_at_1_2_3 : {st['retention_at_1_2_3_off']}   (n = {st['n']})")

one_sided = [st["retention_pct_mean"][5], st["retention_pct_mean"][6], st["retention_pct_mean"][7]]
two_sided = [round((st["retention_pct_mean"][4 + k] + st["retention_pct_mean"][4 - k]) / 2, 1)
             for k in (1, 2, 3)]
print(f"\n  stored, POSITIVE offsets only      : {one_sided}")
print(f"  stored, averaged over +/- offsets  : {two_sided}")
print("  paper currently prints             : [54.1, 44.0, 38.1]")

TWIN = [82.9, 55.2, 35.5]
for name, real in (("positive-only", one_sided), ("symmetric", two_sided)):
    rmse = float(np.sqrt(np.mean([(t - r) ** 2 for t, r in zip(TWIN, real)])))
    print(f"  twin RMSE against {name:<14s}: {rmse:.1f} pts   "
          f"(1-electrode gap {TWIN[0] - real[0]:.1f})")
print("  paper reports the twin overestimating one-electrode retention by 28.8 pts")

# ------------------------------------------------------------------------ B2
print()
print(BAR)
print("B2  Table IV rows printed with r = -1.00")
print(BAR)
cs = json.load(open(os.path.join(ROOT, "results", "corruption_stats.json")))


def walk(d, path=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(d, list) and len(d) == 20 and all(isinstance(x, (int, float)) for x in d):
        yield path, d


arrays = dict(walk(cs))
print(f"  per-participant arrays found: {sorted(arrays)}")
clean = None
for k in arrays:
    if "clean" in k and "sia" not in k:
        clean = np.array(arrays[k])
for k in sorted(arrays):
    a = np.array(arrays[k])
    if clean is None or k == "clean" or len(a) != 20:
        continue
    d = a - clean
    if not np.any(d):
        continue
    try:
        p_exact = wilcoxon(a, clean, mode="exact").pvalue
    except Exception:
        p_exact = wilcoxon(a, clean).pvalue
    p_approx = wilcoxon(a, clean, mode="approx").pvalue
    nneg, npos = int((d < 0).sum()), int((d > 0).sum())
    rb = (npos - nneg) / (npos + nneg) if npos + nneg else 0
    print(f"  {k:<34s} mean{a.mean():7.4f}  neg{nneg:3d} pos{npos:3d}  "
          f"p_exact={p_exact:.3g}  p_approx={p_approx:.3g}")

# ------------------------------------------------------------------------ B3
print()
print(BAR)
print("B3  full-calibration 0.728 (+16.4) vs SIA 0.7338 (+17.0)")
print(BAR)
cal = os.path.join(ROOT, "results", "hyser", "summary_calibsize.json")
if os.path.exists(cal):
    d = json.load(open(cal))
    print("  summary_calibsize.json keys:", list(d)[:12])
    print(json.dumps(d, indent=1)[:1200])

# ------------------------------------------------------------------------ B4
print()
print(BAR)
print("B4  \"2 of 19 paired participants\"")
print(BAR)
have2 = [s for s in SUBJECTS if f"{s}_ses2_X" in CACHE.files]
print(f"  participants with a day-2 session in the cache: {len(have2)} of {len(SUBJECTS)}")
same, pairs, ties = 0, 0, []
for s in SUBJECTS:
    m1 = get(s, 1)[0].mean(0)
    m2 = get(s, 2)[0].mean(0)
    pairs += 1
    if int(np.argmax(m1)) == int(np.argmax(m2)):
        same += 1
        ties.append(s)
print(f"  peak channel recurs on day 2 in {same} of {pairs} participants  ({ties})")
d1 = json.load(open(os.path.join(ROOT, "results", "hyser", "summary_hyser.json")))
subs = d1["subjects"]
print(f"  summary_hyser.json records peaks for {len(subs)} sessions: "
      f"{sorted({k.split('_')[0] for k in subs})}")

# ------------------------------------------------------------------------ B5
print()
print(BAR)
print("B5  which Hyser gestures, and which task type")
print(BAR)
raw = os.path.join(ROOT, "data", "hyser")
if os.path.isdir(raw):
    ses = os.path.join(raw, "s01_ses1")
    files = sorted(os.listdir(ses))
    print(f"  {ses} holds {len(files)} files; first 12:")
    print("   ", files[:12])
y, rep = get("s01", 1)[1], get("s01", 1)[2]
print(f"  cache label set: {sorted(set(y.tolist()))}, repetitions {sorted(set(rep.tolist()))}")
lens = {s: get(s, 1)[0].shape for s in SUBJECTS[:3]}
print(f"  feature-matrix shapes: {lens}")
