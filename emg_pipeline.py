"""
emg_pipeline.py — rebuilt cross-day sEMG evaluation pipeline.

The original analysis scripts were lost (see CRITICAL_ANALYSIS_REPORT.md 2.9d).
This reconstructs the evaluation from the surviving feature cache
`data/hyser_features.npz`, and is validated by reproducing the stored
published values before any new experiment is run.

Channel layout (decoded from results/hyser/summary_crossday.json):
    block = ch // 64        # 0=ED, 1=EP, 2=FD, 3=FP
    idx   = ch %  64
    r0, c0 = idx // 8, idx % 8      # 0-indexed grid position inside the 8x8 array
    printed name row/col = 8-r0 / 8-c0

Run:  python emg_pipeline.py
"""
import json
import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = np.load(os.path.join(ROOT, "data", "hyser_features.npz"), allow_pickle=True)
SUBJECTS = sorted({k[:3] for k in CACHE.files if k.endswith("_X")})
NARR, GRID = 4, 8
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------- geometry
def rc(ch):
    """(block, row0, col0) for a channel index."""
    b, i = ch // 64, ch % 64
    return b, i // 8, i % 8


def ch_of(b, r, c):
    return b * 64 + r * 8 + c


CENTRE = [ch_of(b, r, c) for b in range(NARR) for r in (3, 4) for c in (3, 4)]  # K=16 landmark


def landmark(K):
    """K channels closest to the geometric centre of their arrays, K/4 per array."""
    per = K // NARR
    out = []
    for b in range(NARR):
        d = [(abs(r - 3.5) ** 2 + abs(c - 3.5) ** 2, ch_of(b, r, c))
             for r in range(GRID) for c in range(GRID)]
        d.sort()
        out += [ch for _, ch in d[:per]]
    return np.array(sorted(out))


def distributed(K, amp):
    """K/R strongest channels inside each array (coverage-constrained)."""
    per, out = K // NARR, []
    for b in range(NARR):
        chs = np.arange(b * 64, (b + 1) * 64)
        out += list(chs[np.argsort(-amp[chs])[:per]])
    return np.array(sorted(out))


def concentrated(K, amp):
    """K channels nearest the global peak, within that peak's array."""
    pk = int(np.argmax(amp))
    b, r, c = rc(pk)
    chs = [(abs(rr - r) ** 2 + abs(cc - c) ** 2, ch_of(b, rr, cc))
           for rr in range(GRID) for cc in range(GRID)]
    chs.sort()
    return np.array(sorted(ch for _, ch in chs[:K]))


def shift_channels(chs, dr, dc):
    """Translate a channel set by (dr,dc) inside each array; None if it leaves the grid."""
    out = []
    for ch in chs:
        b, r, c = rc(ch)
        r2, c2 = r + dr, c + dc
        if not (0 <= r2 < GRID and 0 <= c2 < GRID):
            return None
        out.append(ch_of(b, r2, c2))
    return np.array(out)


# ------------------------------------------------------------------- data
def get(subj, ses):
    p = f"{subj}_ses{ses}"
    return CACHE[p + "_X"], CACHE[p + "_y"], CACHE[p + "_rep"]


# ---------------------------------------------------------------- methods
def unitnorm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = LDA(solver="lsqr", shrinkage="auto").fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))


def evaluate(Xtr, ytr, Xcal, Xte, yte, method="raw"):
    """Cross-day evaluation under one representation."""
    if method == "raw":
        A, B = Xtr, Xte
    elif method == "unitnorm":
        A, B = unitnorm(Xtr), unitnorm(Xte)
    elif method in ("sia", "zscore"):
        if method == "sia":
            A, C, B = unitnorm(Xtr), unitnorm(Xcal), unitnorm(Xte)
        else:
            A, C, B = Xtr, Xcal, Xte
        mS, sS = A.mean(0), A.std(0) + 1e-12
        mC, sC = C.mean(0), C.std(0) + 1e-12
        B = (B - mC) * sS / sC + mS
    else:
        raise ValueError(method)
    return (fit_predict(A, ytr, B) == yte).mean()


def split(subj, chs, calib_rep=0):
    """Day-1 train / day-2 calibration / day-2 test on a channel subset."""
    X1, y1, _ = get(subj, 1)
    X2, y2, r2 = get(subj, 2)
    m = r2 == calib_rep
    return X1[:, chs], y1, X2[m][:, chs], X2[~m][:, chs], y2[~m]


def paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    try:
        p = wilcoxon(a, b).pvalue
    except ValueError:
        p = 1.0
    return (a.mean() - b.mean()) * 100, p


# ============================================================== VALIDATION
print("=" * 74)
print("VALIDATION — reproduce stored published values (K=16, fixed landmark)")
print("=" * 74)
L16 = landmark(16)
raw, sia, un, zs = [], [], [], []
for s in SUBJECTS:
    Xtr, ytr, Xc, Xte, yte = split(s, L16)
    raw.append(evaluate(Xtr, ytr, Xc, Xte, yte, "raw"))
    un.append(evaluate(Xtr, ytr, Xc, Xte, yte, "unitnorm"))
    zs.append(evaluate(Xtr, ytr, Xc, Xte, yte, "zscore"))
    sia.append(evaluate(Xtr, ytr, Xc, Xte, yte, "sia"))
ref = {"raw": 0.5638, "unitnorm": 0.6426, "zscore": 0.6561, "SIA": 0.7338}
got = {"raw": np.mean(raw), "unitnorm": np.mean(un), "zscore": np.mean(zs), "SIA": np.mean(sia)}
for k in ref:
    d = got[k] - ref[k]
    print(f"  {k:<10s} rebuilt {got[k]:.4f}   stored {ref[k]:.4f}   diff {d:+.4f}"
          f"   {'OK' if abs(d) < 0.02 else '** MISMATCH **'}")
dpts, p = paired(sia, raw)
print(f"  SIA vs raw: {dpts:+.1f} pts, p={p:.4g}   (stored +17.0, p=1.39e-4)")



print("\nValidation complete.\n")

# ====================================================== EXP 1: ACCURACY-ORACLE
# The sharpest test of the geometric claim. For each participant, search ALL
# integer (dr,dc) translations of the montage and keep the one that maximises
# day-2 TEST accuracy. This uses test labels and is therefore NOT a method --
# it is an upper bound on any rigid-translation correction. Because (0,0) is in
# the search space, it is bounded below by the do-nothing baseline by construction.
print("=" * 74)
print("EXP 1 — TRUE ACCURACY-ORACLE over integer montage translations (K=16)")
print("=" * 74)
orc, orc_sh, base = [], [], []
for s in SUBJECTS:
    best, bestsh = -1, None
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            chs = shift_channels(L16, dr, dc)
            if chs is None:
                continue
            Xtr, ytr, Xc, Xte, yte = split(s, chs)
            # train on day-1 at the ORIGINAL sites, read day-2 at the shifted sites
            Xtr0 = get(s, 1)[0][:, L16]
            a = (fit_predict(Xtr0, ytr, Xte) == yte).mean()
            if a > best:
                best, bestsh = a, (dr, dc)
    orc.append(best); orc_sh.append(bestsh)
    Xtr, ytr, Xc, Xte, yte = split(s, L16)
    base.append(evaluate(Xtr, ytr, Xc, Xte, yte, "raw"))
d, p = paired(orc, base)
print(f"  do-nothing baseline        {np.mean(base):.4f}")
print(f"  ACCURACY-ORACLE shift      {np.mean(orc):.4f}   ({d:+.1f} pts, p={p:.4g})")
print(f"  SIA (label-free)           {np.mean(sia):.4f}")
d2, p2 = paired(sia, orc)
print(f"  SIA vs accuracy-oracle     {d2:+.1f} pts, p={p2:.4g}")
nz = sum(1 for x in orc_sh if x != (0, 0))
print(f"  oracle chose a non-zero shift for {nz}/{len(SUBJECTS)} participants")
print(f"  ceiling of rigid translation = {np.mean(orc):.4f} vs within-day 0.9458")

# ============================== EXP 2: clean + SIA control (Table IV gap)
print()
print("=" * 74)
print("EXP 2 — missing 'clean + SIA' control for the controlled-corruption table")
print("=" * 74)
# Within-day protocol mirroring the corruption experiment: train reps 0-3,
# calibrate on rep 4, test on rep 5, no corruption applied.
cl, cl_sia = [], []
for s in SUBJECTS:
    X, y, r = get(s, 1)
    X = X[:, L16]
    tr, ca, te = r <= 3, r == 4, r == 5
    cl.append((fit_predict(X[tr], y[tr], X[te]) == y[te]).mean())
    cl_sia.append(evaluate(X[tr], y[tr], X[ca], X[te], y[te], "sia"))
d, p = paired(cl_sia, cl)
print(f"  clean (no corruption)      {np.mean(cl):.4f}")
print(f"  clean + SIA                {np.mean(cl_sia):.4f}   ({d:+.1f} pts, p={p:.4g})")
print(f"  -> SIA's effect on UNCORRUPTED data is {d:+.1f} pts; the +43.0 gain-repair")
print(f"     figure should be discounted by this amount.")

# ====================== EXP 3: cross-gesture-set calibration transfer
print()
print("=" * 74)
print("EXP 3 — cross-gesture-set calibration transfer (gain vs effort discriminator)")
print("=" * 74)
# Estimate SIA statistics on gestures {1..4}, apply to test gestures {5..8}.
match, cross = [], []
for s in SUBJECTS:
    X1, y1, _ = get(s, 1)
    X2, y2, r2 = get(s, 2)
    X1, X2 = X1[:, L16], X2[:, L16]
    gl = np.unique(y2)
    A_, B_ = gl[:len(gl) // 2], gl[len(gl) // 2:]
    te = (~(r2 == 0)) & np.isin(y2, B_)
    ca_m = (r2 == 0) & np.isin(y2, B_)      # matched vocabulary
    ca_x = (r2 == 0) & np.isin(y2, A_)      # different vocabulary
    tr = np.isin(y1, B_)
    if te.sum() == 0 or ca_m.sum() == 0 or ca_x.sum() == 0:
        continue
    match.append(evaluate(X1[tr], y1[tr], X2[ca_m], X2[te], y2[te], "sia"))
    cross.append(evaluate(X1[tr], y1[tr], X2[ca_x], X2[te], y2[te], "sia"))
d, p = paired(cross, match)
print(f"  calibration on MATCHED gestures   {np.mean(match):.4f}")
print(f"  calibration on DIFFERENT gestures {np.mean(cross):.4f}   ({d:+.1f} pts, p={p:.4g})")
print(f"  n={len(match)} participants")

# ====================== EXP 4: GRABMyo unit-norm alone at full n
print()
print("=" * 74)
print("EXP 4 — GRABMyo forearm band: unit-norm alone at full n (was n=8 pilot)")
print("=" * 74)
G = np.load(os.path.join(ROOT, "data", "grabmyo_fw.npz"), allow_pickle=True)
pids = sorted({int(k.split('_')[1][1:]) for k in G.files if k.endswith('_y')})
for site in ("F", "W"):
    r_, u_, s_ = [], [], []
    for p_ in pids:
        try:
            X1, y1, t1 = G[f"s1_p{p_}_{site}"], G[f"s1_p{p_}_y"], G[f"s1_p{p_}_t"]
        except KeyError:
            continue
        for ses in (2, 3):
            k = f"s{ses}_p{p_}_{site}"
            if k not in G.files:
                continue
            X2, y2, t2 = G[k], G[f"s{ses}_p{p_}_y"], G[f"s{ses}_p{p_}_t"]
            c = t2 == t2.min()
            if c.sum() == 0 or (~c).sum() == 0:
                continue
            r_.append(evaluate(X1, y1, X2[c], X2[~c], y2[~c], "raw"))
            u_.append(evaluate(X1, y1, X2[c], X2[~c], y2[~c], "unitnorm"))
            s_.append(evaluate(X1, y1, X2[c], X2[~c], y2[~c], "sia"))
    du, pu = paired(u_, r_)
    ds, ps = paired(s_, r_)
    nm = "forearm" if site == "F" else "wrist"
    print(f"  [{nm}] pairs={len(r_)}  raw {np.mean(r_):.4f}")
    print(f"      unit-norm alone {np.mean(u_):.4f}  ({du:+.1f} pts, p={pu:.4g})")
    print(f"      SIA             {np.mean(s_):.4f}  ({ds:+.1f} pts, p={ps:.4g})")
