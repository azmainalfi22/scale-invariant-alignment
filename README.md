# Code and computation for *What Changes and What Survives Across Days in Surface EMG Decoding*

IEEE Conference on Engineering Informatics 2026.
Maria Kabir (Swinburne University of Technology) and Mohammad Azmain Hossain Alfi
(BRAC University).

This repository is self-contained: 21 scripts, the cached feature matrices every
script reads, the Hyser gesture-label files, and every stored result file the
paper cites. Nothing here needs the raw recordings.

```bash
pip install numpy scipy scikit-learn matplotlib
python emg_pipeline.py        # start here: it validates before it computes
```

## Running a script rewrites its result file

Each script writes its own output into `results/`, so a run leaves the tree
dirty. That is deliberate: `git diff` after a run is the comparison. On a
different numpy/scipy build you may see changes in the last few floating-point
digits of a stored value. A full clean-clone run of
`exp_step_decomposition.py`, `exp_noise_regime.py` and `verify_stats.py` on
numpy 2.2.6 / scipy 1.15.3 moved exactly one digit, in the seventeenth
significant figure of one *p*-value, and no reported number changed. If a run
moves a digit the paper actually prints, stop and find out why.

## The flat layout is load-bearing

Every script does `ROOT = os.path.dirname(os.path.abspath(__file__))` and then
reads `data/` and `results/` beside itself. **Do not move the `.py` files into
subfolders.**

## Start with the validation

`emg_pipeline.py` reproduces the paper's four cross-day anchors *before* running
anything new, and prints the comparison:

| | rebuilt | stored |
|---|---|---|
| raw features, K = 16 | 0.5638 | 0.5638 |
| unit normalisation only (step 1) | 0.6426 | 0.6426 |
| per-channel standardisation only (step 2, AdaBN) | 0.6561 | 0.6561 |
| SIA (both steps) | 0.7338 | 0.7338 |

Every experiment script performs the same check against the values it depends on
before reporting anything. **If those four numbers ever drift, stop and find out
why before trusting any other output.** That property is the only reason output
from this pipeline can be trusted at all — see "Provenance" below.

## Which script produces which reported number

### The method and the headline result

| script | what it produces |
|---|---|
| `emg_pipeline.py` | the evaluation pipeline all others import: montages, the four cross-day methods, the Hyser split, the paired tests. Section IV-E's 0.564 → 0.734. |
| `exp_sia_plus_shift.py` | Section IV-D. The best-possible-shift ceiling (0.679), its additivity with SIA (0.850, 75.0 %), and the held-out half-split that debiases it (+8.7 / +8.2). |
| `exp_corruption_stats.py` | Section IV-F. Controlled corruption of held-out within-day data: injected displacement and injected gain drift, with exact tests for the zero-tie rows. |

### Mechanism and scope

| script | what it produces |
|---|---|
| `exp_uda_baselines.py` | Section IV-E. Mean alignment, full-covariance CORAL, pseudo-label self-training, and CORAL after step 1 (0.713). |
| `exp_adabn_uda.py` | The AdaBN identity. Shows `max abs(AdaBN(unit-normed) − SIA) = 0.00e+00` across all 20 participants, which is what licenses naming step 2 in Section III-E. Also runs whitening, subspace alignment, Riemannian recentring, exact optimal transport, and a BatchNorm MLP. |
| `exp_step_decomposition.py` | Section IV-E, the geometry ordering: what step 1 adds on top of step 2, +7.8 / +1.7 / −4.2 on grid, forearm and wrist. Reproduces `results/step_decomposition_grabmyo.json` to machine zero. |
| `exp_calibsize.py` | Section IV-H, the calibration-size sweep. **Reporting this correctly is why the camera-ready differs from the accepted version** — see "Provenance". Includes `--meanonly`, the mean-shift-only variant. |
| `exp_representation.py` | Feature sets and electrode counts beyond the default; the practitioner representation reaching 0.841 at K = 64. |
| `exp_identifiability.py` | Section IV-H, "normalisation is not anonymisation": cross-day participant identification rising under SIA. |
| `exp_noise_regime.py` | Section III-A. The feature-domain signal-to-noise ratio that bounds how far the data sit from Proposition 1's noiseless case: median 4.3 on the Hyser grid, 3.1 forearm, 2.8 wrist. |

### External validation and statistics

| script | what it produces |
|---|---|
| `cache_grabmyo_fw.py` | builds the GRABMyo forearm/wrist caches from the raw recordings (the only script that needs them) |
| `exp_grabmyo_full.py` | Section IV-G, all 43 participants and both band geometries |
| `exp_timegap.py` | the 7-, 21- and 28-day horizons, **and the clustering correction**: pair-level against participant-level inference, which is why the paper's GRABMyo *p*-values are 2.0 × 10⁻⁶ and 7.1 × 10⁻⁷ rather than < 10⁻⁹ |
| `exp_hierarchical.py` | the pooled mixed-effects model over both datasets |
| `exp_equivalence.py` | the TOST equivalence tests and their margins |
| `exp_posterior.py` | Student-*t* posteriors for results near the 0.05 boundary |
| `stats_upgrade.py` | BCa intervals, exact tests for the zero-tie conditions, the Benjamini–Hochberg pass over the secondary family, and minimum detectable effects |
| `verify_stats.py`, `verify_flags.py` | recompute means, *p*-values, effect sizes and intervals from the stored per-participant arrays; `verify_flags.py` also checks the Section III-B gesture-selection claims against `data/hyser/*/label_*.txt` |

### Figures

`make_fig_tradeoff.py` draws the paper's figure. `make_figures.py` draws a larger
set built for the journal extension; only the trade-off figure appears in the
conference paper.

## Provenance, stated plainly

**The original analysis scripts were lost during the project.** The cached
feature matrices, the raw recordings and the stored result files survived.
`emg_pipeline.py` is a reconstruction from those caches, and it earns trust only
through the validation above: it reproduces the published anchors to four decimal
places before it computes anything new. Every script added since does the same.

Two reported quantities come from re-runs rather than from the original scripts:

- **The calibration-size sweep** (`exp_calibsize.py`). The accepted version of the
  paper reported 0.150 for one calibration gesture. That value does not
  reproduce. It is not a protocol disagreement: this script, run under the
  protocol the paper describes, reproduces the *same sweep's* eight-gesture
  condition exactly at 0.7281, the stored value, so only the small-*k* end was
  ever wrong. The correct figure is 0.417, and the camera-ready reports it.
- **The GRABMyo *p*-values** (`exp_timegap.py`). Each of the 43 participants
  contributes two session pairs that share their session-1 training data. Those
  are not independent observations. Every test is now computed on participant
  means, *n* = 43; point estimates are unchanged to three decimals.

Both corrections were found by re-running the computation rather than by reading
the paper: the audit that compared the manuscript against the stored result files
could not have caught the first, because it never compared a stored file against a
fresh re-run. `exp_calibsize.py` regenerates `results/calibsize_rebuilt.json`, and
`--meanonly` writes `results/calibsize_meanonly.json`.

## Raw recordings

Not included, and not needed: no analysis script reads them. Only
`cache_grabmyo_fw.py` does, and only to rebuild `data/*.npz` from scratch.

- **Hyser** — PhysioNet, `doi:10.13026/ym7v-bh53`
- **GRABMyo** — PhysioNet, `doi:10.13026/rrvt-9s97`

## Licence and citation

No licence file is included yet, so default copyright applies. If you want this
reusable, add one (MIT or BSD-3-Clause is usual for research code).

Please cite the paper if you use this:

> M. Kabir and M. A. H. Alfi, "What Changes and What Survives Across Days in
> Surface EMG Decoding," in *Proc. IEEE Int. Conf. Engineering Informatics
> (ICEI)*, 2026.

## What is in `data/`

Derived per-channel features, not raw recordings: RMS and the time-domain and
spectral feature matrices the scripts read, plus the Hyser gesture-label files.
Both source datasets are publicly released for secondary research use and are
cited above; please cite them too if you use these caches.
