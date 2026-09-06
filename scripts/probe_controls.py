#!/usr/bin/env python3
"""
probe_controls.py -- is the 0.996 cohort probe a real effect or an artifact?

The concern is reasonable: a number that high on any task invites the
question of whether something is leaking. This script answers it with
controls rather than argument.

  1. LABEL SHUFFLE (the decisive one). Randomly permute the cohort labels
     and re-run the identical probe. If the pipeline is somehow inflating
     the score, it will inflate a meaningless target too. A real effect
     collapses to the majority baseline; an artifact does not.

  2. TRAIN vs TEST GAP. Overfitting means high on train, low on held-out.
     Reporting both settles whether that is what is happening. Note the
     probe is a linear model -- roughly 768 parameters against tens of
     thousands of samples -- so it has little capacity to memorise.

  3. LEARNING CURVE. Score against training-set size. A real, easily
     separable signal saturates almost immediately; a memorised one
     improves steadily with more data.

  4. WITHIN-KERMANY WIDTH PROBE. Kermany alone spans five export widths
     (512/768/1024/1536 x 496 and 512x512). Predicting width from features
     within a single cohort isolates a preprocessing component of the
     signal without involving NEH at all. This matters for what the 0.996
     may be attributed to: Kermany and NEH differ in device, resolution,
     file format, site, population AND labelling practice simultaneously,
     so the probe measures all of it together and none of it separately.
     "Cohort information" is supportable; "scanner bias" is not.

Usage:
    python probe_controls.py
    python probe_controls.py --shuffles 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

try:
    from scripts.config import Config
except ImportError:
    from config import Config


def load_track(track, splits=("train", "test")):
    t = Config.TRACKS[track]
    cache_dir = os.path.join(Config.BASE_DIR, "features", track)
    tag = f"{Config.MODEL_NAME}_{Config.IMAGE_SIZE}_{t['resize']}"
    man_path = os.path.join(Config.MANIFEST_DIR, t["manifest"])

    keep = ["split", "cohort", "source", "patient_key", "width", "height", "y_label"]
    man = pd.concat(
        [c for c in pd.read_csv(man_path, chunksize=20000,
                                usecols=lambda x: x in keep)],
        ignore_index=True)

    out = {}
    for sp in splits:
        f = os.path.join(cache_dir, f"{tag}_{sp}.npz")
        if not os.path.isfile(f):
            sys.exit(f"no cache for {track}/{sp}: {f}")
        d = np.load(f)
        sub = man[man["split"] == sp].reset_index(drop=True)
        rows = sub.loc[d["index"]].reset_index(drop=True)
        out[sp] = (d["features"], rows)
    return out


def fit_score(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    return float(clf.score(Xtr, ytr)), float(clf.score(Xte, yte))


def baseline(y):
    c = np.bincount(y, minlength=int(y.max()) + 1)
    return float(c.max() / c.sum())


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", default="oct")
    p.add_argument("--group-col", default=None,
                   help="column defining the groups; defaults to 'source' "
                        "when the manifest has more than one, else 'cohort'")
    p.add_argument("--shuffles", type=int, default=10)
    p.add_argument("--out", default="reports/probe_controls.json")
    a = p.parse_args()

    data = load_track(a.track)
    Xtr, mtr = data["train"]
    Xte, mte = data["test"]

    gcol = a.group_col
    if gcol is None:
        gcol = ("source" if ("source" in mtr.columns
                             and mtr["source"].nunique() > 1) else "cohort")
    if mtr[gcol].nunique() < 2:
        sys.exit(f"track {a.track!r} has a single {gcol}; there is no group "
                 f"probe to validate here.")
    groups = sorted(set(mtr[gcol]) | set(mte[gcol]))
    g2i = {g: i for i, g in enumerate(groups)}
    ytr = mtr[gcol].map(g2i).to_numpy()
    yte = mte[gcol].map(g2i).to_numpy()
    print(f"  grouping by '{gcol}': {groups}")

    print(f"\n{'='*72}\nPROBE CONTROLS  (track: {a.track})\n{'='*72}")
    print(f"  train {Xtr.shape}   test {Xte.shape}")
    print(f"  {gcol}: {dict(mtr[gcol].value_counts())}")

    res = {}

    # -------------------------------------------------- 1. real probe
    print(f"\n{'='*72}\n1. THE REAL PROBE\n{'='*72}")
    tr_acc, te_acc = fit_score(Xtr, ytr, Xte, yte)
    base = baseline(yte)
    print(f"  train accuracy    : {tr_acc:.4f}")
    print(f"  TEST  accuracy    : {te_acc:.4f}   <- held out, "
          f"patient-disjoint")
    print(f"  majority baseline : {base:.4f}")
    print(f"  train-test gap    : {tr_acc - te_acc:+.4f}")
    if abs(tr_acc - te_acc) < 0.02:
        print(f"\n  The gap is negligible. Overfitting means high on train "
              f"and LOW on\n  held-out data; that is not what this shows.")
    res["real"] = {"train": round(tr_acc, 4), "test": round(te_acc, 4),
                   "baseline": round(base, 4),
                   "gap": round(tr_acc - te_acc, 4)}

    # ----------------------------------------------- 2. label shuffle
    print(f"\n{'='*72}\n2. LABEL SHUFFLE CONTROL\n{'='*72}")
    print(f"  Same features, same model, same protocol -- cohort labels")
    print(f"  randomly permuted. If the pipeline inflates scores, it will")
    print(f"  inflate a meaningless target too.\n")
    rng = np.random.default_rng(Config.SEED)
    sh = []
    for i in range(a.shuffles):
        ys_tr = rng.permutation(ytr)
        ys_te = rng.permutation(yte)
        _, acc = fit_score(Xtr, ys_tr, Xte, ys_te)
        sh.append(acc)
        print(f"    shuffle {i+1:>2}: {acc:.4f}")
    m, s = float(np.mean(sh)), float(np.std(sh, ddof=1))
    print(f"\n  shuffled mean : {m:.4f} +/- {s:.4f}")
    print(f"  real          : {te_acc:.4f}")
    print(f"  baseline      : {base:.4f}")
    z = (te_acc - m) / s if s > 0 else float("inf")
    print(f"\n  separation    : {z:.1f} sd above the shuffled control")
    # the control passes when the shuffle collapses to (or below) the
    # majority baseline while the real labels stay clearly above it
    if m <= base + 0.05 and te_acc > base + 0.10 and z > 5:
        print(f"\n  VERDICT: the shuffled control collapses to roughly the")
        print(f"  majority baseline while the real labels score {te_acc:.3f}.")
        print(f"  The cohort signal is genuine, not a pipeline artifact.")
    else:
        print(f"\n  VERDICT: the shuffled control did NOT collapse. "
              f"Investigate\n  before trusting the real number.")
    res["shuffle"] = {"mean": round(m, 4), "sd": round(s, 4),
                      "n": a.shuffles, "z_vs_real": round(z, 1)}

    # --------------------------------------------- 3. learning curve
    print(f"\n{'='*72}\n3. LEARNING CURVE\n{'='*72}")
    print("  A signal this easy should saturate almost immediately.\n")
    curve = {}
    for n in (100, 500, 2000, 10000, len(Xtr)):
        n = min(n, len(Xtr))
        idx = rng.choice(len(Xtr), n, replace=False)
        if len(set(ytr[idx])) < 2:
            continue
        _, acc = fit_score(Xtr[idx], ytr[idx], Xte, yte)
        curve[n] = round(acc, 4)
        print(f"    {n:>6,} training images -> test {acc:.4f}")
    res["learning_curve"] = curve

    # ------------------------------------------ 4. within-Kermany width
    print(f"\n{'='*72}\n4. WITHIN-KERMANY WIDTH PROBE\n{'='*72}")
    print("  Kermany alone spans several export widths. Predicting width")
    print("  inside ONE cohort isolates a preprocessing component of the")
    print("  signal, with NEH not involved at all.\n")

    ktr = mtr["cohort"] == "kermany"
    kte = mte["cohort"] == "kermany"
    if ktr.sum() and "width" in mtr.columns:
        wtr_raw = mtr.loc[ktr, "width"].to_numpy()
        wte_raw = mte.loc[kte, "width"].to_numpy()
        widths = sorted(set(wtr_raw) | set(wte_raw))
        w2i = {w: i for i, w in enumerate(widths)}
        wtr = np.array([w2i[w] for w in wtr_raw])
        wte = np.array([w2i[w] for w in wte_raw])
        print(f"    widths present: {widths}")
        print(f"    train {ktr.sum():,}  test {kte.sum():,}")
        _, wacc = fit_score(Xtr[ktr.to_numpy()], wtr,
                            Xte[kte.to_numpy()], wte)
        wbase = baseline(wte)
        print(f"\n    width accuracy    : {wacc:.4f}")
        print(f"    majority baseline : {wbase:.4f}")
        if wacc > wbase + 0.15:
            print(f"\n    Export resolution is recoverable from the features")
            print(f"    within a single cohort. So part of the cross-cohort")
            print(f"    signal is preprocessing, not device.")
        res["kermany_width"] = {"accuracy": round(wacc, 4),
                                "baseline": round(wbase, 4),
                                "widths": [int(w) for w in widths]}
    else:
        print("    width column unavailable; skipped")

    # ------------------------------------------------------- summary
    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    print(f"  real probe (test)      {te_acc:.4f}")
    print(f"  shuffled control       {m:.4f} +/- {s:.4f}")
    print(f"  majority baseline      {base:.4f}")
    print(f"  train-test gap         {tr_acc - te_acc:+.4f}")
    print()
    print(f"  What this supports: the frozen representation carries strong")
    print(f"  COHORT information. Kermany and NEH differ in device, export")
    print(f"  resolution, file format, site, population and labelling all at")
    print(f"  once, so the probe measures their sum. 'Scanner bias' is not")
    print(f"  supportable from this design; 'cohort information' is.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  results -> {a.out}\n")


if __name__ == "__main__":
    main()