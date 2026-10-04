"""
exp_adabn_uda.py -- the baselines a domain-adaptation reviewer will ask for.

exp_uda_baselines.py already runs mean alignment, full CORAL, CORAL-after-step-1
and pseudo-label self-training.  Three gaps remain, and the first is the one
that matters:

  1. ADAPTIVE BATCH NORMALISATION (AdaBN, Li et al., Pattern Recognition 80,
     pp. 109-117, 2018).  SIA's second step replaces per-channel first and
     second moments with statistics recomputed on unlabelled target data --
     which is what AdaBN does.  The paper describes step 2 as the diagonal case
     of correlation alignment and never names AdaBN, so we (a) demonstrate the
     algebraic identity in the feature domain and (b) run AdaBN in its native
     setting, a network with BatchNorm layers, where the statistics of EVERY
     layer are re-estimated from the calibration repetition.

  2. Methods that relax the diagonal in a direction other than CORAL's:
     whitening (each domain to identity), subspace alignment, and Riemannian
     covariance recentring.

  3. A classifier that learns its own representation, so that the
     "not an artefact of the classifier" claim rests on more than three
     linear-ish models over one fixed 16-dimensional feature vector.

Every label-free method sees exactly the same unlabelled calibration
repetition, the same landmark-distributed montage at K=16, and no labels
anywhere.  Self-validates against the published values first.

  python exp_adabn_uda.py
"""
import json
import os
import numpy as np
from scipy.stats import wilcoxon
from sklearn.covariance import LedoitWolf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(ROOT, "emg_pipeline.py"), encoding="utf-8").read()
head = src.split("# ============================================================== VALIDATION")[0]
ns = {"__file__": os.path.join(ROOT, "emg_pipeline.py")}
exec(compile(head, "emg_pipeline_head", "exec"), ns)
landmark, split, unitnorm = ns["landmark"], ns["split"], ns["unitnorm"]
evaluate, SUBJECTS = ns["evaluate"], ns["SUBJECTS"]

L16 = landmark(16)
EPS = 1e-12


# --------------------------------------------------------------- shared helpers
def fitpred(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = LDA(solver="lsqr", shrinkage="auto").fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))


def msqrt(S, inv=False):
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-10, None)
    w = w ** (-0.5) if inv else w ** 0.5
    return (V * w) @ V.T


# ------------------------------------------------- feature-domain adaptations
def adabn_features(A, C, B):
    """AdaBN in the feature domain: replace source per-channel moments with
    calibration ones.  Identical to SIA's step 2 by construction -- we report
    both so the identity is visible in the table rather than asserted."""
    return (B - C.mean(0)) * (A.std(0) + EPS) / (C.std(0) + EPS) + A.mean(0)


def whiten(A, C, B):
    """Each domain whitened to identity by its own second moments (no
    re-colouring).  The full-matrix counterpart of standardising per channel."""
    Ws = msqrt(LedoitWolf().fit(A).covariance_, inv=True)
    Wt = msqrt(LedoitWolf().fit(C).covariance_, inv=True)
    return (A - A.mean(0)) @ Ws, (B - C.mean(0)) @ Wt


def subspace_align(A, C, B, d=8):
    """Subspace alignment (Fernando et al.): principal subspaces of source and
    target are matched by a linear map, both domains are projected into it."""
    def pcs(X, d):
        Xc = X - X.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        return Vt[:d].T
    d = min(d, A.shape[1], max(1, C.shape[0] - 1))
    Ps, Pt = pcs(A, d), pcs(C, d)
    M = Ps.T @ Pt
    return (A - A.mean(0)) @ Ps @ M, (B - C.mean(0)) @ Pt


def riemann_recentre(A, C, B):
    """Covariance recentring in the spirit of Riemannian transfer (Zanini et
    al., IEEE TBME 2017): each domain is mapped by the inverse square root of
    its own feature covariance, putting both at the identity.

    NOTE, and worth stating in the paper rather than hiding: Riemannian
    recentring is defined on the SPD manifold, where the features ARE
    covariance matrices.  Our features are vectors, so the manifold has no
    purchase and the construction degenerates to `whiten` above -- the two
    rows agree to 0.0 per participant.  We keep both names only to record
    that the Riemannian baseline was considered and reduces to whitening
    here; the table should carry one row, not two.
    """
    Ws = msqrt(LedoitWolf().fit(A).covariance_, inv=True)
    Wt = msqrt(LedoitWolf().fit(C).covariance_, inv=True)
    return (A - A.mean(0)) @ Ws, (B - C.mean(0)) @ Wt


def ot_barycentric(A, ytr, C, B):
    """Optimal-transport adaptation, exact (not entropic).  Target features are
    transported onto the source cloud by the barycentric map of the optimal
    coupling between the calibration set and the source set."""
    from scipy.optimize import linprog
    ns_, nt_ = len(A), len(C)
    M = ((C[:, None, :] - A[None, :, :]) ** 2).sum(-1)          # nt x ns
    # solve the transportation problem with uniform marginals
    c = M.ravel()
    Aeq, beq = [], []
    for i in range(nt_):
        row = np.zeros((nt_, ns_)); row[i, :] = 1
        Aeq.append(row.ravel()); beq.append(1.0 / nt_)
    for j in range(ns_ - 1):
        col = np.zeros((nt_, ns_)); col[:, j] = 1
        Aeq.append(col.ravel()); beq.append(1.0 / ns_)
    r = linprog(c, A_eq=np.array(Aeq), b_eq=np.array(beq), bounds=(0, None),
                method="highs")
    G = r.x.reshape(nt_, ns_)
    G = G / np.clip(G.sum(1, keepdims=True), 1e-12, None)
    Chat = G @ A                                                 # transported calibration
    # apply the same shift/scale the calibration set underwent, to the test set
    shift = Chat.mean(0) - C.mean(0)
    scale = (Chat.std(0) + EPS) / (C.std(0) + EPS)
    return (B - C.mean(0)) * scale + C.mean(0) + shift


# --------------------------------------------------- BatchNorm MLP, for AdaBN
class MLPBN:
    """16 -> BN -> Linear -> BN -> ReLU -> Linear -> softmax, numpy, full batch.

    Small by design: 48 training samples per participant.  BatchNorm sits where
    AdaBN needs it, so `predict(..., stats=...)` can substitute target-domain
    statistics at BOTH normalisation layers -- which is AdaBN as published,
    rather than an input-rescaling stand-in for it.
    """

    def __init__(self, n_in, n_hid, n_out, seed=0, l2=1e-3):
        g = np.random.default_rng(seed)
        self.W1 = g.normal(0, np.sqrt(2.0 / n_in), (n_in, n_hid))
        self.b1 = np.zeros(n_hid)
        self.W2 = g.normal(0, np.sqrt(2.0 / n_hid), (n_hid, n_out))
        self.b2 = np.zeros(n_out)
        self.g0, self.B0 = np.ones(n_in), np.zeros(n_in)
        self.g1, self.B1 = np.ones(n_hid), np.zeros(n_hid)
        self.l2, self.n_out = l2, n_out

    def _bn(self, X, m, s, g, b):
        return (X - m) / (s + 1e-5) * g + b

    def forward(self, X, stats, cache=False):
        m0, s0, m1, s1 = stats
        h0 = self._bn(X, m0, s0, self.g0, self.B0)
        z1 = h0 @ self.W1 + self.b1
        h1 = self._bn(z1, m1, s1, self.g1, self.B1)
        a1 = np.maximum(h1, 0)
        z2 = a1 @ self.W2 + self.b2
        z2 = z2 - z2.max(1, keepdims=True)
        P = np.exp(z2); P /= P.sum(1, keepdims=True)
        if cache:
            return P, (X, h0, z1, h1, a1)
        return P

    def source_stats(self, X):
        m0, s0 = X.mean(0), X.std(0)
        z1 = self._bn(X, m0, s0, self.g0, self.B0) @ self.W1 + self.b1
        return m0, s0, z1.mean(0), z1.std(0)

    def target_stats(self, Xc):
        """AdaBN: re-estimate every BatchNorm layer's statistics on unlabelled
        target data, propagating through the network as they are replaced."""
        m0, s0 = Xc.mean(0), Xc.std(0)
        z1 = self._bn(Xc, m0, s0, self.g0, self.B0) @ self.W1 + self.b1
        return m0, s0, z1.mean(0), z1.std(0)

    def fit(self, X, y, epochs=1500, lr=0.02):
        n = len(X)
        Y = np.zeros((n, self.n_out)); Y[np.arange(n), y] = 1
        params = ["W1", "b1", "W2", "b2", "g0", "B0", "g1", "B1"]
        mom = {p: np.zeros_like(getattr(self, p)) for p in params}
        vel = {p: np.zeros_like(getattr(self, p)) for p in params}
        for t in range(1, epochs + 1):
            m0, s0 = X.mean(0), X.std(0)
            h0 = self._bn(X, m0, s0, self.g0, self.B0)
            z1 = h0 @ self.W1 + self.b1
            m1, s1 = z1.mean(0), z1.std(0)
            h1 = self._bn(z1, m1, s1, self.g1, self.B1)
            a1 = np.maximum(h1, 0)
            z2 = a1 @ self.W2 + self.b2
            z2s = z2 - z2.max(1, keepdims=True)
            P = np.exp(z2s); P /= P.sum(1, keepdims=True)
            dz2 = (P - Y) / n
            g = {}
            g["W2"] = a1.T @ dz2 + self.l2 * self.W2
            g["b2"] = dz2.sum(0)
            da1 = dz2 @ self.W2.T
            dh1 = da1 * (h1 > 0)
            g["g1"] = (dh1 * ((z1 - m1) / (s1 + 1e-5))).sum(0)
            g["B1"] = dh1.sum(0)
            dz1 = dh1 * self.g1 / (s1 + 1e-5)
            g["W1"] = h0.T @ dz1 + self.l2 * self.W1
            g["b1"] = dz1.sum(0)
            dh0 = dz1 @ self.W1.T
            g["g0"] = (dh0 * ((X - m0) / (s0 + 1e-5))).sum(0)
            g["B0"] = dh0.sum(0)
            for p in params:                                     # Adam
                mom[p] = 0.9 * mom[p] + 0.1 * g[p]
                vel[p] = 0.999 * vel[p] + 0.001 * g[p] ** 2
                mh = mom[p] / (1 - 0.9 ** t)
                vh = vel[p] / (1 - 0.999 ** t)
                setattr(self, p, getattr(self, p) - lr * mh / (np.sqrt(vh) + 1e-8))
        return self


def mlp_pair(Xtr, ytr, Xc, Xte, yte, seeds=(0, 1, 2)):
    """Returns (source-only accuracy, AdaBN accuracy), averaged over seeds."""
    cl = np.unique(ytr)
    remap = {c: i for i, c in enumerate(cl)}
    yi = np.array([remap[v] for v in ytr])
    so, ad = [], []
    for sd in seeds:
        net = MLPBN(Xtr.shape[1], 32, len(cl), seed=sd).fit(Xtr, yi)
        ss = net.source_stats(Xtr)
        ts = net.target_stats(Xc)
        so.append((cl[net.forward(Xte, ss).argmax(1)] == yte).mean())
        ad.append((cl[net.forward(Xte, ts).argmax(1)] == yte).mean())
    return float(np.mean(so)), float(np.mean(ad))


# --------------------------------------------------------------------- run
def run():
    keys = ("raw", "sia", "zscore", "adabn_feat", "adabn_feat_un",
            "whiten", "sa", "riemann", "ot",
            "mlp_src", "mlp_adabn", "mlp_src_un", "mlp_adabn_un")
    res = {k: [] for k in keys}
    for s in SUBJECTS:
        Xtr, ytr, Xc, Xte, yte = split(s, L16)
        res["raw"].append(evaluate(Xtr, ytr, Xc, Xte, yte, "raw"))
        res["sia"].append(evaluate(Xtr, ytr, Xc, Xte, yte, "sia"))
        res["zscore"].append(evaluate(Xtr, ytr, Xc, Xte, yte, "zscore"))

        # AdaBN in the feature domain, on raw and on unit-normalised features
        res["adabn_feat"].append(
            (fitpred(Xtr, ytr, adabn_features(Xtr, Xc, Xte)) == yte).mean())
        A, C, T = unitnorm(Xtr), unitnorm(Xc), unitnorm(Xte)
        res["adabn_feat_un"].append(
            (fitpred(A, ytr, adabn_features(A, C, T)) == yte).mean())

        # full-matrix alternatives to the diagonal
        As, Bs = whiten(Xtr, Xc, Xte)
        res["whiten"].append((fitpred(As, ytr, Bs) == yte).mean())
        As, Bs = subspace_align(Xtr, Xc, Xte)
        res["sa"].append((fitpred(As, ytr, Bs) == yte).mean())
        As, Bs = riemann_recentre(Xtr, Xc, Xte)
        res["riemann"].append((fitpred(As, ytr, Bs) == yte).mean())
        res["ot"].append(
            (fitpred(Xtr, ytr, ot_barycentric(Xtr, ytr, Xc, Xte)) == yte).mean())

        # a classifier that learns its own representation, with and without AdaBN
        a, b = mlp_pair(Xtr, ytr, Xc, Xte, yte)
        res["mlp_src"].append(a); res["mlp_adabn"].append(b)
        a, b = mlp_pair(A, ytr, C, T, yte)
        res["mlp_src_un"].append(a); res["mlp_adabn_un"].append(b)
    return {k: np.array(v) for k, v in res.items()}


r = run()

print("=" * 82)
print("VALIDATION -- reproduce published values before reporting anything new")
print("=" * 82)
ok = True
for k, ref in (("raw", 0.5638), ("zscore", 0.6561), ("sia", 0.7338)):
    g = r[k].mean(); d = g - ref
    ok &= abs(d) < 0.002
    print(f"  {k:<12s} rebuilt {g:.4f}  stored {ref:.4f}  diff {d:+.4f}"
          f"  {'OK' if abs(d) < 0.002 else '** MISMATCH **'}")

print()
print("=" * 82)
print("IDENTITY CHECK -- feature-domain AdaBN versus SIA's step 2")
print("=" * 82)
mx = np.abs(r["adabn_feat"] - r["zscore"]).max()
print(f"  per-participant max |AdaBN(features) - per-channel z| = {mx:.2e}")
print(f"  AdaBN(features) {r['adabn_feat'].mean():.4f}   "
      f"per-channel z {r['zscore'].mean():.4f}")
print("  -> step 2 of SIA IS AdaBN applied in the feature domain, not merely")
print("     analogous to it.  The paper should cite Li et al. (2018) and say so.")
mx_un = np.abs(r["adabn_feat_un"] - r["sia"]).max()
print(f"\n  and with step 1 in front: max |AdaBN(unit-normed) - SIA| = {mx_un:.2e}")
print(f"  AdaBN after unit-norm {r['adabn_feat_un'].mean():.4f}   "
      f"SIA {r['sia'].mean():.4f}")
print("  -> SIA = unit normalisation + AdaBN.  The contribution is step 1 and")
print("     the evidence for which axis to correct, not a new estimator.")

CEIL = 0.9460
RAW = r["raw"].mean()
GAP = CEIL - RAW

print()
print("=" * 82)
print("EXTENDED LABEL-FREE ADAPTATION TABLE (Hyser, n=20, K=16)")
print("=" * 82)
print(f"{'method':<38s}{'acc':>7s}{'gap rec':>9s}{'d raw':>8s}"
      f"{'p vs raw':>10s}{'p vs SIA':>10s}")
NAMES = [
    ("raw", "raw (standard practice)"),
    ("whiten", "whitening, each domain to identity"),
    ("sa", "subspace alignment"),
    ("riemann", "Riemannian covariance recentring"),
    ("ot", "optimal transport, exact"),
    ("adabn_feat", "AdaBN, feature domain (= step 2)"),
    ("mlp_src", "BN-MLP, source only"),
    ("mlp_adabn", "BN-MLP + AdaBN (all layers)"),
    ("mlp_src_un", "BN-MLP, source only, unit-normed"),
    ("mlp_adabn_un", "BN-MLP + AdaBN, unit-normed"),
    ("sia", "SIA (proposed)"),
]
out = {}
for k, nm in NAMES:
    a = r[k]
    d = (a.mean() - RAW) * 100
    rec = (a.mean() - RAW) / GAP * 100
    pr = np.nan if k == "raw" else wilcoxon(a, r["raw"]).pvalue
    ps = np.nan if k == "sia" else wilcoxon(a, r["sia"]).pvalue
    print(f"{nm:<38s}{a.mean():7.3f}{rec:8.1f}%{d:+8.1f}{pr:10.4g}{ps:10.4g}")
    out[k] = {"name": nm, "acc": float(a.mean()), "sd": float(a.std(ddof=1)),
              "gap_recovered_pct": float(rec), "delta_pts": float(d),
              "p_vs_raw": None if k == "raw" else float(pr),
              "p_vs_sia": None if k == "sia" else float(ps),
              "per_subject": a.tolist()}

print()
print("=" * 82)
print("WHAT THIS SETTLES")
print("=" * 82)
d_mlp = (r["mlp_adabn"].mean() - r["mlp_src"].mean()) * 100
p_mlp = wilcoxon(r["mlp_adabn"], r["mlp_src"]).pvalue
print(f"  AdaBN helps a network that learns its own features too: "
      f"{d_mlp:+.1f} pts, p={p_mlp:.4g}")
d_un = (r["mlp_adabn_un"].mean() - r["mlp_adabn"].mean()) * 100
p_un = wilcoxon(r["mlp_adabn_un"], r["mlp_adabn"]).pvalue
print(f"  and step 1 still adds on top of AdaBN in that model: "
      f"{d_un:+.1f} pts, p={p_un:.4g}")
best = max((k for k, _ in NAMES if k not in ("sia", "raw")),
           key=lambda k: r[k].mean())
d_b = (r["sia"].mean() - r[best].mean()) * 100
p_b = wilcoxon(r["sia"], r[best]).pvalue
print(f"  strongest non-SIA method: {out[best]['name']} "
       f"({r[best].mean():.3f}); SIA {d_b:+.1f} pts, p={p_b:.4g}")

json.dump({"ceiling": CEIL, "raw": float(RAW), "n": len(SUBJECTS), "K": 16,
           "validation_passed": bool(ok),
           "identity_adabn_vs_step2_max_abs_diff": float(mx),
           "identity_adabn_unitnorm_vs_sia_max_abs_diff": float(mx_un),
           "methods": out,
           "mlp_adabn_effect": {"delta_pts": float(d_mlp), "p": float(p_mlp)},
           "mlp_step1_on_top_of_adabn": {"delta_pts": float(d_un), "p": float(p_un)},
           "sia_vs_best_other": {"method": best, "delta_pts": float(d_b),
                                 "p": float(p_b)}},
          open(os.path.join(ROOT, "results", "adabn_uda_extended.json"), "w"),
          indent=1)
print("\nwrote results/adabn_uda_extended.json")
