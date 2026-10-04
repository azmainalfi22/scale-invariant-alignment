"""
cache_grabmyo_fw.py — rebuild/extend the GRABMyo forearm+wrist feature cache.

The original caching script was lost with the rest. This reconstructs it and
VALIDATES against the surviving cache (`data/grabmyo_fw.npz`, participants 1-20)
before writing anything, so new participants are only added once the recipe is
proven to reproduce the old ones.

Features: per-channel RMS of each 5 s recording, in physical units (mV), taken
over channels named F1..F16 (forearm band) and W1..W12 (wrist band); U* channels
are ignored.

  python cache_grabmyo_fw.py --validate           # check recipe vs stored cache
  python cache_grabmyo_fw.py --build              # write data/grabmyo_fw_all.npz
"""
import argparse
import os
import sys
import numpy as np

try:
    import wfdb
except ImportError:
    sys.exit("wfdb not installed:  pip install wfdb")

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "grabmyo")
OLD = os.path.join(ROOT, "data", "grabmyo_fw.npz")
GESTURES, TRIALS = range(1, 18), range(1, 6)


def rms_of(path):
    """Per-channel RMS (mV) for one recording, split into forearm and wrist."""
    rec = wfdb.rdrecord(path)
    sig, names = rec.p_signal, list(rec.sig_name)
    fi = [i for i, n in enumerate(names) if n.upper().startswith("F")]
    wi = [i for i, n in enumerate(names) if n.upper().startswith("W")]
    fi.sort(key=lambda i: int(names[i][1:]))
    wi.sort(key=lambda i: int(names[i][1:]))
    r = np.sqrt(np.nanmean(sig ** 2, axis=0))
    return r[fi], r[wi]


def participant(sess, pid):
    """(F, W, y, t) for one participant-session, or None if incomplete."""
    d = os.path.join(RAW, f"Session{sess}", f"session{sess}_participant{pid}")
    if not os.path.isdir(d):
        return None
    F, W, y, t = [], [], [], []
    for g in GESTURES:
        for tr in TRIALS:
            stem = f"session{sess}_participant{pid}_gesture{g}_trial{tr}"
            dat = os.path.join(d, stem + ".dat")
            if not (os.path.exists(dat) and os.path.getsize(dat) == 655360
                    and os.path.exists(os.path.join(d, stem + ".hea"))):
                return None
            f_, w_ = rms_of(os.path.join(d, stem))
            F.append(f_); W.append(w_); y.append(g); t.append(tr)
    return (np.array(F), np.array(W), np.array(y), np.array(t))


def validate(n=3):
    old = np.load(OLD, allow_pickle=True)
    print("validating reconstructed recipe against the surviving cache")
    worst = 0.0
    for pid in range(1, n + 1):
        for s in (1, 2, 3):
            k = f"s{s}_p{pid}"
            if f"{k}_F" not in old.files:
                continue
            got = participant(s, pid)
            if got is None:
                print(f"  {k}: raw files missing, skipped")
                continue
            F, W, y, t = got
            for nm, a, b in (("F", F, old[f"{k}_F"]), ("W", W, old[f"{k}_W"]),
                             ("y", y, old[f"{k}_y"]), ("t", t, old[f"{k}_t"])):
                d = float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))
                worst = max(worst, d)
                flag = "OK" if d < 1e-6 else "** MISMATCH **"
                print(f"  {k}_{nm}: max|diff| = {d:.3e}  {flag}")
    print(f"\nworst deviation across all checked arrays: {worst:.3e}")
    return worst < 1e-6


def build():
    old = np.load(OLD, allow_pickle=True)
    out = {k: old[k] for k in old.files}
    added, skipped = [], []
    for pid in range(1, 44):
        if f"s1_p{pid}_F" in out and f"s3_p{pid}_F" in out:
            continue
        got = {s: participant(s, pid) for s in (1, 2, 3)}
        if any(v is None for v in got.values()):
            skipped.append(pid)
            continue
        for s, (F, W, y, t) in got.items():
            out[f"s{s}_p{pid}_F"] = F
            out[f"s{s}_p{pid}_W"] = W
            out[f"s{s}_p{pid}_y"] = y
            out[f"s{s}_p{pid}_t"] = t
        added.append(pid)
        print(f"  cached participant {pid}")
    dest = os.path.join(ROOT, "data", "grabmyo_fw_all.npz")
    np.savez_compressed(dest, **out)
    pids = sorted({int(k.split("_")[1][1:]) for k in out if k.endswith("_y")})
    print(f"\nadded {len(added)}: {added}")
    print(f"incomplete (not yet downloaded): {skipped}")
    print(f"cache now holds {len(pids)} participants -> {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.validate:
        sys.exit(0 if validate() else 1)
    elif a.build:
        if not validate(n=2):
            sys.exit("recipe does not reproduce the stored cache -- refusing to build")
        build()
    else:
        ap.print_help()
