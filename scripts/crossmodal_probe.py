#!/usr/bin/env python3
"""
crossmodal_probe.py -- does AMD identity transfer between OCT and fundus?

Two questions, both answered from feature caches that already exist. No
GPU, no re-caching, runs in seconds.

  1. MODALITY SEPARABILITY. Can a linear model tell an OCT B-scan from a
     fundus photograph using the same frozen ConvNeXt features? Expected
     to be near-perfect -- the two are different imaging physics -- but it
     puts a number on the gap. For reference, NEH vs Kermany (two OCT
     scanners) was 0.996 against a 0.816 baseline.

  2. CROSS-MODAL TRANSFER. Train a classifier on one modality in a shared
     two-class space and test it on the other. Both directions, because a
     one-way result would itself be informative.

        NORMAL  <- OCT NORMAL          <- CFP normal
        AMD     <- OCT DRUSEN + CNV    <- CFP amd

     OCT's DME and CFP's DIABETIC / CATARACT are dropped: DME has no
     fundus counterpart in these cohorts and cataract is a lens opacity
     that does not appear in a retinal B-scan.

Interpretation:

  * transfer clearly above chance  -> disease signal survives the modality
                                      change; a joint or multimodal model
                                      is worth building
  * transfer at or near chance     -> the frozen representation encodes
                                      modality far more strongly than
                                      pathology, and pooling the two would
                                      mostly teach a model which camera
                                      took the picture

Note this is a transfer test, not a multimodal fusion. The cohorts share
no patients, so there is no paired OCT+fundus data here to fuse.

Usage:
    python crossmodal_probe.py
    python crossmodal_probe.py --cfp-track cfp_hyamd --seeds 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score, confusion_matrix)

try:
    from scripts.config import Config
except ImportError:
    from config import Config


# Shared two-class space. Values are the y_label strings as they appear in
# each track's manifest.
SHARED = {
    "oct":          {"NORMAL": "NORMAL", "DRUSEN": "AMD", "CNV": "AMD"},
    "cfp_amdnet23": {"NORMAL": "NORMAL", "AMD": "AMD"},
    "cfp_odir":     {"NORMAL": "NORMAL", "AMD": "AMD"},
    # HYAMD's CONTROL is diabetic-retinopathy patients WITHOUT AMD, not
    # healthy eyes, so mapping it to NORMAL would compare AMD-vs-DR on one
    # side against AMD-vs-healthy on the other. Included so the asymmetry
    # can be measured, but it is not the primary comparison.
    "cfp_hyamd":    {"CONTROL": "NORMAL", "AMD": "AMD"},
}
CLASSES = ["NORMAL", "AMD"]


def track_config(track):
    if track not in Config.TRACKS:
        sys.exit(f"unknown track {track!r}; valid: {sorted(Config.TRACKS)}")
    return Config.TRACKS[track]


def load_track(track, splits=("train", "val", "test")):
    """Cached features + the manifest rows they correspond to."""
    t = track_config(track)
    cache_dir = os.path.join(Config.BASE_DIR, "features", track)
    tag = f"{Config.MODEL_NAME}_{Config.IMAGE_SIZE}_{t['resize']}"

    man_path = os.path.join(Config.MANIFEST_DIR, t["manifest"])
    if not os.path.isfile(man_path):
        sys.exit(f"manifest not found: {man_path}")
 
    # Read in chunks, keeping only the columns this probe uses. The full
    # manifest read has repeatedly exhausted the Windows commit limit on
    # this machine, and only four or five of the fifteen columns are
    # needed here.
    keep = ["Directory", "split", "y_label", "patient_key"]
    keep += list(t.get("filter") or {})
    man = pd.concat(
        [c for c in pd.read_csv(man_path, chunksize=20000,
                                usecols=lambda x: x in keep)],
        ignore_index=True)

    filt = t.get("filter")
    if filt:
        for col, vals in filt.items():
            man = man[man[col].isin(vals)]
        man = man.reset_index(drop=True)

    X, y_lab, pat = [], [], []
    for sp in splits:
        f = os.path.join(cache_dir, f"{tag}_{sp}.npz")
        if not os.path.isfile(f):
            sys.exit(f"no cache for {track}/{sp}: {f}\n"
                     f"Run: python scripts/cache_features.py --track {track}")
        d = np.load(f)
        sub = man[man["split"] == sp].reset_index(drop=True)
        idx = d["index"]
        X.append(d["features"])
        y_lab.append(sub.loc[idx, "y_label"].to_numpy())
        pat.append(sub.loc[idx, "patient_key"].to_numpy())

    return (np.concatenate(X), np.concatenate(y_lab), np.concatenate(pat))


def to_shared(track, labels):
    """Map a track's labels into the shared space; None where unmapped."""
    m = SHARED.get(track)
    if m is None:
        sys.exit(f"no shared-space mapping defined for track {track!r}")
    return np.array([m.get(l) for l in labels], dtype=object)


def _score(yte, pred, prob):
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro",
                                   zero_division=0)),
        "auc": float(roc_auc_score(yte, prob)) if len(set(yte)) == 2 else None,
    }


def fit_eval(Xtr, ytr, Xte, yte, n_boot=1000, seed=0):
    """
    Point estimate plus a bootstrap interval over the TEST set.

    Note logistic regression on fixed features is convex and deterministic:
    refitting with different random_state values returns the identical
    model, so a "seed sweep" here would report a spurious sd of zero. The
    uncertainty that matters is sampling error in the test set, which is
    what the bootstrap measures.
    """
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    prob = clf.predict_proba(Xte)[:, 1]

    point = _score(yte, pred, prob)
    point["cm"] = confusion_matrix(yte, pred, labels=[0, 1]).tolist()

    rng = np.random.default_rng(seed)
    boots = []
    n = len(yte)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(set(yte[i])) < 2:
            continue
        boots.append(_score(yte[i], pred[i], prob[i]))

    for k in ("accuracy", "balanced_accuracy", "macro_f1", "auc"):
        v = [b[k] for b in boots if b[k] is not None]
        if v:
            point[k + "_lo"] = float(np.percentile(v, 2.5))
            point[k + "_hi"] = float(np.percentile(v, 97.5))
    return point


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--oct-track", default="oct")
    p.add_argument("--cfp-track", default="cfp_odir",
                   help="cfp_odir is the primary comparison: it has a true "
                        "healthy NORMAL class and a single acquisition "
                        "source")
    p.add_argument("--boot", type=int, default=1000,
                   help="bootstrap resamples of the test set for the "
                        "confidence interval")
    p.add_argument("--out", default="reports/crossmodal_probe.json")
    a = p.parse_args()

    print(f"\n{'='*72}\nCROSS-MODAL PROBE\n{'='*72}")
    print(f"  OCT track : {a.oct_track}")
    print(f"  CFP track : {a.cfp_track}")

    Xo, yo_raw, po = load_track(a.oct_track)
    Xc, yc_raw, pc = load_track(a.cfp_track)
    print(f"\n  OCT features : {Xo.shape}")
    print(f"  CFP features : {Xc.shape}")

    if Xo.shape[1] != Xc.shape[1]:
        sys.exit(f"feature dimensions differ ({Xo.shape[1]} vs "
                 f"{Xc.shape[1]}); both tracks must use the same backbone")

    # ---------------------------------------------------- 1. modality probe
    print(f"\n{'='*72}\n1. MODALITY SEPARABILITY\n{'='*72}")
    print("  Can a linear model tell OCT from fundus in the same features?")

    Xm = np.vstack([Xo, Xc])
    ym = np.concatenate([np.zeros(len(Xo)), np.ones(len(Xc))]).astype(int)
    rng = np.random.default_rng(Config.SEED)
    perm = rng.permutation(len(Xm))
    cut = int(0.7 * len(Xm))
    tr, te = perm[:cut], perm[cut:]

    mclf = LogisticRegression(max_iter=2000, class_weight="balanced")
    mclf.fit(Xm[tr], ym[tr])
    macc = float(mclf.score(Xm[te], ym[te]))
    mbase = float(max(np.mean(ym[te]), 1 - np.mean(ym[te])))
    print(f"\n  accuracy          : {macc:.4f}")
    print(f"  majority baseline : {mbase:.4f}")
    print(f"\n  For scale, NEH vs Kermany -- two OCT scanners -- was 0.996 "
          f"against\n  a 0.816 baseline. This is the same measurement across "
          f"modalities.")

    # ------------------------------------------------- 2. shared label space
    print(f"\n{'='*72}\n2. SHARED LABEL SPACE\n{'='*72}")
    yo = to_shared(a.oct_track, yo_raw)
    yc = to_shared(a.cfp_track, yc_raw)

    ko, kc = yo != None, yc != None          # noqa: E711  (object array)
    print(f"  OCT: {ko.sum():,} of {len(yo):,} images map "
          f"({sorted(set(yo_raw[ko]))} -> shared)")
    print(f"       dropped: {sorted(set(yo_raw[~ko]))}")
    print(f"  CFP: {kc.sum():,} of {len(yc):,} images map "
          f"({sorted(set(yc_raw[kc]))} -> shared)")
    if (~kc).any():
        print(f"       dropped: {sorted(set(yc_raw[~kc]))}")

    Xo2, yo2, po2 = Xo[ko], yo[ko], po[ko]
    Xc2, yc2, pc2 = Xc[kc], yc[kc], pc[kc]
    c2i = {c: i for i, c in enumerate(CLASSES)}
    yo2 = np.array([c2i[v] for v in yo2])
    yc2 = np.array([c2i[v] for v in yc2])

    for name, y in (("OCT", yo2), ("CFP", yc2)):
        n = np.bincount(y, minlength=2)
        print(f"  {name} shared classes: NORMAL {n[0]:,}  AMD {n[1]:,}")

    # -------------------------------------------------- 3. transfer, 2 ways
    results = {}
    for src, dst, Xs, ys, Xd, yd in (
        ("OCT", "CFP", Xo2, yo2, Xc2, yc2),
        ("CFP", "OCT", Xc2, yc2, Xo2, yo2),
    ):
        print(f"\n{'='*72}\n3. TRANSFER  {src} -> {dst}\n{'='*72}")
        r = fit_eval(Xs, ys, Xd, yd, n_boot=a.boot, seed=Config.SEED)

        base = float(max(np.mean(yd), 1 - np.mean(yd)))
        print(f"  trained on {len(Xs):,} {src} images, "
              f"tested on {len(Xd):,} {dst} images")
        print(f"  point estimate with 95% bootstrap interval "
              f"({a.boot} resamples of the test set)\n")
        for k, lbl in (("accuracy", "accuracy"),
                       ("balanced_accuracy", "balanced acc"),
                       ("macro_f1", "macro-F1"),
                       ("auc", "AUC")):
            if r.get(k) is None:
                print(f"    {lbl:<13} n/a")
            else:
                lo, hi = r.get(k + "_lo"), r.get(k + "_hi")
                ci = f"  [{lo:.4f}, {hi:.4f}]" if lo is not None else ""
                print(f"    {lbl:<13} {r[k]:.4f}{ci}")
        print(f"    {'baseline':<13} {base:.4f}   (majority class)")
        print(f"    {'chance AUC':<13} 0.5000")

        cm = np.array(r["cm"])
        print(f"\n  confusion (rows true / cols predicted):")
        print("    " + pd.DataFrame(cm, index=CLASSES,
                                    columns=CLASSES)
              .to_string().replace("\n", "\n    "))

        m_auc = r.get("auc")
        results[f"{src}_to_{dst}"] = {
            "n_train": int(len(Xs)), "n_test": int(len(Xd)),
            "majority_baseline": round(base, 4),
            **{k: (None if r.get(k) is None else round(r[k], 4))
               for k in ("accuracy", "balanced_accuracy", "macro_f1", "auc",
                         "auc_lo", "auc_hi", "macro_f1_lo", "macro_f1_hi")},
            "confusion": r["cm"],
        }

        if m_auc is not None:
            if m_auc < 0.55:
                print(f"\n  AUC {m_auc:.3f} is at or near chance: the frozen "
                      f"representation does\n  not carry AMD identity across "
                      f"this modality boundary.")
            elif m_auc < 0.65:
                print(f"\n  AUC {m_auc:.3f} is weakly above chance -- some "
                      f"signal transfers, but\n  far less than within "
                      f"modality.")
            else:
                print(f"\n  AUC {m_auc:.3f} is clearly above chance: disease "
                      f"identity does\n  transfer, and a joint model is "
                      f"worth building.")

    # ------------------------------------------------------ 4. within-modality
    print(f"\n{'='*72}\n4. WITHIN-MODALITY REFERENCE\n{'='*72}")
    print("  The same two-class task trained and tested inside one modality,")
    print("  so the transfer numbers above have something to be compared to.")
    for name, X, y, pat in (("OCT", Xo2, yo2, po2), ("CFP", Xc2, yc2, pc2)):
        # patient-disjoint 70/30 so this is not inflated by leakage
        pats = np.array(sorted(set(pat)))
        r = np.random.default_rng(Config.SEED)
        r.shuffle(pats)
        tr_p = set(pats[:int(0.7 * len(pats))])
        m = np.array([q in tr_p for q in pat])
        if len(set(y[m])) < 2 or len(set(y[~m])) < 2:
            print(f"  {name}: split left a single class; skipped")
            continue
        rr = fit_eval(X[m], y[m], X[~m], y[~m], n_boot=a.boot,
                      seed=Config.SEED)
        print(f"  {name}: AUC {rr['auc']:.4f}  macro-F1 {rr['macro_f1']:.4f} "
              f"({m.sum():,} train / {(~m).sum():,} test, patient-disjoint)")
        results[f"{name}_within"] = {k: (round(v, 4)
                                          if isinstance(v, float) else v)
                                     for k, v in rr.items() if k != "cm"}

    results["modality_probe"] = {"accuracy": round(macc, 4),
                                 "majority_baseline": round(mbase, 4)}
    results["_meta"] = {"oct_track": a.oct_track, "cfp_track": a.cfp_track,
                        "bootstrap": a.boot, "shared_classes": CLASSES,
                        "mapping": {k: SHARED[k] for k in
                                    (a.oct_track, a.cfp_track)}}

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  results -> {a.out}\n")


if __name__ == "__main__":
    main()