#!/usr/bin/env python3
"""
crossmodal_cv.py -- five-fold, group-disjoint cross-modal transfer, OCT <-> CFP.

crossmodal_probe.py answered the question once: one fit per direction and
a bootstrap interval over the test set. This repeats it over K folds, so
every number is a mean +/- sd across held-out groups, and it pairs each
cross-modal result with a within-modal result on the SAME test fold.

Each modality is split into K group-disjoint, class-stratified folds. In
fold k one model is fitted per modality and each is scored on both
modalities' held-out fold:

    OCT model (OCT folds != k)  ->  within_OCT   (OCT fold k)
                                ->  OCT_to_CFP   (CFP fold k)
    CFP model (CFP folds != k)  ->  within_CFP   (CFP fold k)
                                ->  CFP_to_OCT   (OCT fold k)

    delta OCT_to_CFP = within_CFP - OCT_to_CFP      same CFP test fold
    delta CFP_to_OCT = within_OCT - CFP_to_OCT      same OCT test fold

A delta is what is lost on that test fold by training on the other
modality instead of the right one. The test fold is identical on both
sides, so it is a paired comparison. Sign convention: positive = worse
when trained on the other modality (for log_loss that is cross minus
within; for every other metric within minus cross).

Losses: every model's log-loss is reported on its own training data and on
each test fold, next to accuracy, balanced accuracy, macro-F1, AUC,
sensitivity (AMD recall) and specificity (NORMAL recall).

Grouping uses group_key where the manifest has one (it links patients that
share duplicate images, so it is stricter than patient_key), else
patient_key. The shared two-class space and the dropped classes are the
ones defined in crossmodal_probe.py.

Fold scores are not independent -- the training sets overlap -- so the sd
describes spread across folds, not a standard error. A naive paired t-test
over folds overstates significance; use a corrected resampled test if a
p-value is required.

Task-matched variant (--oct-map amd_dme): OCT's negative class becomes DME
instead of NORMAL, mirroring HYAMD, whose CONTROL class is diabetic
retinopathy without AMD. DME exists only in Kermany, and Kermany vs NEH is
linearly separable at 0.996, so pair it with --oct-cohort kermany or the
task is confounded with cohort. --drop-mixed removes groups whose images
carry both classes after mapping. The defaults reproduce the original
AMD-vs-NORMAL runs exactly.

Sample-size control (--match-train images|groups): OCT has far more
training data than any CFP set here, so a within-modal gap could be data
volume rather than modality. With this flag, each fold's OCT TRAINING set
is subsampled to the CFP training size -- whole groups, class ratio kept,
--match-repeats random draws averaged -- while the OCT TEST fold stays
complete. If OCT still beats CFP on equal training data, the gap is not a
sample-size artifact.

Usage:
    python scripts/crossmodal_cv.py                           # HYAMD
    python scripts/crossmodal_cv.py --cfp-track cfp_odir      # healthy NORMAL
    python scripts/crossmodal_cv.py --cfp-track cfp_amdnet23
    python scripts/crossmodal_cv.py --oct-map amd_dme --oct-cohort kermany --drop-mixed
    python scripts/crossmodal_cv.py --oct-map amd_dme --oct-cohort kermany --drop-mixed --match-train images
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
                             confusion_matrix, f1_score, log_loss,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

try:
    from scripts.config import Config
    from scripts.crossmodal_probe import CLASSES, SHARED, to_shared
except ImportError:
    from config import Config
    from crossmodal_probe import CLASSES, SHARED, to_shared

METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "auc",
           "sensitivity", "specificity", "log_loss"]
SHORT = {"accuracy": "acc", "balanced_accuracy": "bal_acc",
         "macro_f1": "macro_F1", "auc": "AUC", "sensitivity": "sens",
         "specificity": "spec", "log_loss": "log_loss"}
LOWER_IS_BETTER = {"log_loss"}
CONDITIONS = ["train_OCT", "train_CFP", "within_OCT", "within_CFP",
              "OCT_to_CFP", "CFP_to_OCT"]
PAIRS = {"OCT_to_CFP": "within_CFP",      # scored on the same CFP fold
         "CFP_to_OCT": "within_OCT"}      # scored on the same OCT fold

# OCT label maps into the shared space. "NORMAL" is the shared token for
# the negative class (index 0); for amd_dme that negative class is DME.
OCT_MAPS = {
    "amd_normal": {"NORMAL": "NORMAL", "DRUSEN": "AMD", "CNV": "AMD"},
    "amd_dme":    {"DME": "NORMAL", "DRUSEN": "AMD", "CNV": "AMD"},
}


def load_track(track):
    """Cached features for every split, with the label and group per row."""
    if track not in Config.TRACKS:
        sys.exit(f"unknown track {track!r}; valid: {sorted(Config.TRACKS)}")
    t = Config.TRACKS[track]
    cache_dir = os.path.join(Config.BASE_DIR, "features", track)
    tag = f"{Config.MODEL_NAME}_{Config.IMAGE_SIZE}_{t['resize']}"
    man_path = os.path.join(Config.MANIFEST_DIR, t["manifest"])
    if not os.path.isfile(man_path):
        sys.exit(f"manifest not found: {man_path}")

    # chunked, few columns: the full read has exhausted the Windows commit
    # limit on this machine before
    filt = t.get("filter") or {}
    keep = {"split", "y_label", "patient_key", "group_key", "cohort", *filt}
    man = pd.concat([c for c in pd.read_csv(man_path, chunksize=20000,
                                            usecols=lambda x: x in keep)],
                    ignore_index=True)
    for col, vals in filt.items():
        man = man[man[col].isin(vals)]
    man = man.reset_index(drop=True)
    gcol = "group_key" if "group_key" in man.columns else "patient_key"

    X, y, g, c = [], [], [], []
    for sp in ("train", "val", "test"):
        f = os.path.join(cache_dir, f"{tag}_{sp}.npz")
        if not os.path.isfile(f):
            sys.exit(f"no cache for {track}/{sp}: {f}\n"
                     f"Run: python scripts/cache_features.py --track {track}")
        d = np.load(f)
        sub = man[man["split"] == sp].reset_index(drop=True)
        rows = sub.loc[d["index"]]
        X.append(d["features"])
        y.append(rows["y_label"].to_numpy())
        g.append(rows[gcol].astype(str).to_numpy())
        c.append(rows["cohort"].astype(str).to_numpy()
                 if "cohort" in rows.columns
                 else np.full(len(rows), track, dtype=object))
    return (np.concatenate(X), np.concatenate(y), np.concatenate(g), gcol,
            np.concatenate(c))


def to_binary(mapping, X, y_raw, g, coh):
    """Keep rows that map into the shared space; negative=0, AMD=1."""
    s = np.array([mapping.get(l) for l in y_raw], dtype=object)
    keep = s != None                                   # noqa: E711
    c2i = {name: i for i, name in enumerate(CLASSES)}
    y = np.array([c2i[v] for v in s[keep]], dtype=int)
    return X[keep], y, g[keep], coh[keep], sorted(set(y_raw[~keep]))


def mixed_mask(y, g):
    """Mask keeping only groups whose images all share one label."""
    n = pd.Series(y).groupby(pd.Series(g)).nunique()
    bad = set(n.index[n > 1])
    return np.array([x not in bad for x in g]), len(bad)


def assign_folds(y, g, k, seed):
    fid = np.full(len(y), -1)
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for i, (_, te) in enumerate(sgkf.split(np.zeros(len(y)), y, g)):
        fid[te] = i
    return fid


def fit(X, y):
    return LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, y)


def score(y, prob):
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    nan = float("nan")
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro",
                                   zero_division=0)),
        "auc": float(roc_auc_score(y, prob)) if len(set(y)) == 2 else nan,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else nan,
        "log_loss": float(log_loss(y, prob, labels=[0, 1])),
        "n": int(len(y)),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def mean_sd(v):
    v = np.asarray(v, dtype=float)
    return float(np.nanmean(v)), float(np.nanstd(v, ddof=1))


def match_mask(y, g, train_mask, target, by, rng):
    """
    Subsample the training rows to `target` units (images or groups) by
    drawing whole groups, keeping each class's share of those units.
    Test rows are never touched.
    """
    per = (pd.DataFrame({"g": g[train_mask], "y": y[train_mask]})
           .groupby("g")["y"].agg(["mean", "size"]))
    per["y"] = per["mean"].round().astype(int)
    unit = per["size"] if by == "images" else pd.Series(1, index=per.index)
    total = float(unit.sum())
    chosen = set()
    for _, sub in per.groupby("y"):
        quota = target * unit[sub.index].sum() / total
        acc = 0
        for gid in rng.permutation(sub.index.to_numpy()):
            if acc >= quota:
                break
            chosen.add(gid)
            acc += unit[gid]
    return train_mask & np.array([x in chosen for x in g])


def avg_scores(ds):
    """Average score dicts over repeats; confusion matrices are summed."""
    out = {}
    for k in ds[0]:
        v = [d[k] for d in ds]
        if k == "confusion":
            out[k] = np.sum(v, axis=0).tolist()
        elif k == "n":
            out[k] = int(round(float(np.mean(v))))
        else:
            out[k] = float(np.nanmean(v))
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--oct-track", default="oct")
    p.add_argument("--cfp-track", default="cfp_hyamd")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=Config.SEED)
    p.add_argument("--oct-map", default="amd_normal", choices=sorted(OCT_MAPS),
                   help="amd_normal: NORMAL vs DRUSEN+CNV (default); "
                        "amd_dme: DME vs DRUSEN+CNV, task-matched to HYAMD")
    p.add_argument("--oct-cohort", default=None,
                   help="restrict OCT to one cohort, e.g. kermany")
    p.add_argument("--drop-mixed", action="store_true",
                   help="drop groups whose images carry both classes "
                        "after mapping")
    p.add_argument("--match-train", choices=["images", "groups"],
                   default=None,
                   help="in each fold, subsample OCT training data to the "
                        "CFP training size (by images or by groups; whole "
                        "groups drawn, class ratio kept). Test folds are "
                        "never subsampled")
    p.add_argument("--match-repeats", type=int, default=5,
                   help="random subsamples per fold when --match-train is "
                        "set; OCT scores are averaged over them")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    tags = [t for t, on in ((a.oct_map, a.oct_map != "amd_normal"),
                            (a.oct_cohort, bool(a.oct_cohort)),
                            ("nomixed", a.drop_mixed),
                            (f"match{a.match_train}", bool(a.match_train)))
            if on]
    out = a.out or (f"reports/crossmodal_cv_{a.cfp_track}"
                    f"{''.join('_' + t for t in tags)}.json")
    oct_mapping = OCT_MAPS[a.oct_map]
    if a.cfp_track not in SHARED:
        sys.exit(f"no shared-space mapping for CFP track {a.cfp_track!r}")
    cfp_mapping = SHARED[a.cfp_track]
    neg = {"OCT": sorted(k for k, v in oct_mapping.items() if v == "NORMAL"),
           "CFP": sorted(k for k, v in cfp_mapping.items() if v == "NORMAL")}

    print(f"\n{'='*78}\nCROSS-MODAL TRANSFER, {a.folds}-FOLD GROUP-DISJOINT\n"
          f"{'='*78}")
    print(f"  OCT track : {a.oct_track}   map {a.oct_map}   "
          f"cohort {a.oct_cohort or 'all'}")
    print(f"  CFP track : {a.cfp_track}")
    print(f"  negative  : OCT {neg['OCT']}   CFP {neg['CFP']}")
    print(f"  mixed     : {'dropped' if a.drop_mixed else 'kept'}   "
          f"seed {a.seed}")
    print(f"  OCT train : "
          + (f"matched to CFP by {a.match_train}, {a.match_repeats} "
             f"repeats per fold" if a.match_train else "full"))

    Xo, yo_raw, go, gco, co = load_track(a.oct_track)
    Xc, yc_raw, gc, gcc, cc = load_track(a.cfp_track)
    if Xo.shape[1] != Xc.shape[1]:
        sys.exit(f"feature dimensions differ ({Xo.shape[1]} vs "
                 f"{Xc.shape[1]}); both tracks must use the same backbone")
    if a.oct_cohort:
        m = co == a.oct_cohort
        if not m.any():
            sys.exit(f"no OCT rows in cohort {a.oct_cohort!r}; "
                     f"available: {sorted(set(co))}")
        Xo, yo_raw, go, co = Xo[m], yo_raw[m], go[m], co[m]
    Xo, yo, go, co, drop_o = to_binary(oct_mapping, Xo, yo_raw, go, co)
    Xc, yc, gc, cc, drop_c = to_binary(cfp_mapping, Xc, yc_raw, gc, cc)

    mixed = {}
    keep_o, mixed["OCT"] = mixed_mask(yo, go)
    keep_c, mixed["CFP"] = mixed_mask(yc, gc)
    if a.drop_mixed:
        Xo, yo, go, co = Xo[keep_o], yo[keep_o], go[keep_o], co[keep_o]
        Xc, yc, gc, cc = Xc[keep_c], yc[keep_c], gc[keep_c], cc[keep_c]

    print(f"\n{'='*78}\nDATA IN THE SHARED SPACE  (negative=0, AMD=1)\n"
          f"{'='*78}")
    counts = {}
    for name, y, g, gcol, drop, coh in (("OCT", yo, go, gco, drop_o, co),
                                        ("CFP", yc, gc, gcc, drop_c, cc)):
        n = np.bincount(y, minlength=2)
        cohorts = {str(k): int(v)
                   for k, v in pd.Series(coh).value_counts().items()}
        counts[name] = {"images": int(len(y)), "groups": int(len(set(g))),
                        "negative": int(n[0]), "AMD": int(n[1]),
                        "negative_classes": neg[name], "cohorts": cohorts,
                        "grouped_by": gcol, "dropped_classes": drop,
                        "mixed_groups": mixed[name],
                        "mixed_dropped": bool(a.drop_mixed)}
        print(f"  {name}: {len(y):,} images, {len(set(g)):,} groups "
              f"(by {gcol})  {'+'.join(neg[name])} {n[0]:,}  AMD {n[1]:,}"
              f"   dropped {drop or 'none'}")
        print(f"       cohorts {cohorts}   mixed-label groups {mixed[name]} "
              f"({'dropped' if a.drop_mixed else 'kept'})")

    fo = assign_folds(yo, go, a.folds, a.seed)
    fc = assign_folds(yc, gc, a.folds, a.seed)

    print(f"\n{'='*78}\nFOLDS\n{'='*78}")
    for name, f, y, g in (("OCT", fo, yo, go), ("CFP", fc, yc, gc)):
        spans = pd.Series(f).groupby(pd.Series(g)).nunique()
        if (spans > 1).any():
            sys.exit(f"{name}: {int((spans > 1).sum())} groups span more "
                     f"than one fold")
        for k in range(a.folds):
            m = f == k
            print(f"  {name} fold {k}: {m.sum():>7,} images  "
                  f"{len(set(g[m])):>5,} groups  AMD share {y[m].mean():.3f}")
        print(f"  {name}: every group sits in exactly one fold   OK")

    print(f"\n{'='*78}\nPER FOLD  (AUC)\n{'='*78}")

    def prob(model, X):
        return model.predict_proba(X)[:, 1]

    rng = np.random.default_rng(a.seed)
    rows, oct_train_sizes = [], []
    for k in range(a.folds):
        otr, ote = fo != k, fo == k
        ctr, cte = fc != k, fc == k
        mc = fit(Xc[ctr], yc[ctr])

        if a.match_train:
            target = (int(ctr.sum()) if a.match_train == "images"
                      else len(set(gc[ctr])))
            reps = []
            for _ in range(a.match_repeats):
                m = match_mask(yo, go, otr, target, a.match_train, rng)
                mo = fit(Xo[m], yo[m])
                oct_train_sizes.append((int(m.sum()), len(set(go[m]))))
                reps.append({
                    "train_OCT": score(yo[m], prob(mo, Xo[m])),
                    "within_OCT": score(yo[ote], prob(mo, Xo[ote])),
                    "OCT_to_CFP": score(yc[cte], prob(mo, Xc[cte]))})
            oct_part = {c: avg_scores([x[c] for x in reps]) for c in reps[0]}
        else:
            mo = fit(Xo[otr], yo[otr])
            oct_part = {
                "train_OCT": score(yo[otr], prob(mo, Xo[otr])),
                "within_OCT": score(yo[ote], prob(mo, Xo[ote])),
                "OCT_to_CFP": score(yc[cte], prob(mo, Xc[cte]))}

        r = {"fold": k, **oct_part,
             "train_CFP": score(yc[ctr], prob(mc, Xc[ctr])),
             "within_CFP": score(yc[cte], prob(mc, Xc[cte])),
             "CFP_to_OCT": score(yo[ote], prob(mc, Xo[ote]))}
        rows.append(r)
        size = ""
        if a.match_train:
            recent = oct_train_sizes[-a.match_repeats:]
            size = (f"   OCT train ~{np.mean([s[0] for s in recent]):,.0f} "
                    f"img / {np.mean([s[1] for s in recent]):,.0f} groups "
                    f"vs CFP {int(ctr.sum()):,} / {len(set(gc[ctr]))}")
        print(f"  fold {k}:  within OCT {r['within_OCT']['auc']:.4f}   "
              f"within CFP {r['within_CFP']['auc']:.4f}   "
              f"OCT->CFP {r['OCT_to_CFP']['auc']:.4f}   "
              f"CFP->OCT {r['CFP_to_OCT']['auc']:.4f}{size}")

    summary = {c: {m: mean_sd([r[c][m] for r in rows]) for m in METRICS}
               for c in CONDITIONS}
    deltas = {}
    for cross, within in PAIRS.items():
        deltas[cross] = {}
        for m in METRICS:
            per = [(r[cross][m] - r[within][m]) if m in LOWER_IS_BETTER
                   else (r[within][m] - r[cross][m]) for r in rows]
            mu, sd = mean_sd(per)
            deltas[cross][m] = {"per_fold": [round(x, 4) for x in per],
                                "mean": round(mu, 4), "sd": round(sd, 4),
                                "folds_worse": int(sum(x > 0 for x in per))}

    width = 17
    head = "".join(f"{SHORT[m]:>{width}}" for m in METRICS)
    print(f"\n{'='*78}\nSUMMARY  mean +/- sd over {a.folds} folds\n{'='*78}")
    print(f"  {'condition':<12}{head}")
    for c in CONDITIONS:
        cells = "".join(f"{summary[c][m][0]:>{width-8}.4f}+/-"
                        f"{summary[c][m][1]:.3f}" for m in METRICS)
        print(f"  {c:<12}{cells}")

    print(f"\n{'='*78}\nDELTAS  (positive = worse when trained on the other "
          f"modality)\n{'='*78}")
    for cross, within in PAIRS.items():
        print(f"  {cross}  vs  {within}")
        for m in METRICS:
            d = deltas[cross][m]
            print(f"    {SHORT[m]:<9} {d['mean']:+.4f} +/- {d['sd']:.4f}   "
                  f"worse in {d['folds_worse']}/{a.folds} folds   "
                  f"per fold {d['per_fold']}")

    print(f"\n{'='*78}\nMODALITY PROBE  (can a linear model tell OCT from "
          f"CFP?)\n{'='*78}")
    ym = np.r_[np.zeros(len(yo), dtype=int), np.ones(len(yc), dtype=int)]
    gm = np.r_[np.char.add("oct:", go.astype(str)),
               np.char.add("cfp:", gc.astype(str))]
    fm = assign_folds(ym, gm, a.folds, a.seed)
    Xm = np.vstack([Xo, Xc])
    macc = []
    for k in range(a.folds):
        mm = fit(Xm[fm != k], ym[fm != k])
        macc.append(float(mm.score(Xm[fm == k], ym[fm == k])))
    mbase = float(max(ym.mean(), 1 - ym.mean()))
    mmu, msd = mean_sd(macc)
    print(f"  accuracy {mmu:.4f} +/- {msd:.4f}   majority baseline {mbase:.4f}")

    def rnd(d):
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in d.items()}

    res = {
        "summary": {c: {m: {"mean": round(summary[c][m][0], 4),
                            "sd": round(summary[c][m][1], 4)}
                        for m in METRICS} for c in CONDITIONS},
        "deltas": deltas,
        "per_fold": [{"fold": r["fold"], **{c: rnd(r[c]) for c in CONDITIONS}}
                     for r in rows],
        "modality_probe": {"per_fold": [round(x, 4) for x in macc],
                           "mean": round(mmu, 4), "sd": round(msd, 4),
                           "majority_baseline": round(mbase, 4)},
        "_meta": {"oct_track": a.oct_track, "cfp_track": a.cfp_track,
                  "folds": a.folds, "seed": a.seed, "data": counts,
                  "classes": CLASSES,
                  "mapping": {"oct": oct_mapping, "cfp": cfp_mapping},
                  "oct_map": a.oct_map, "oct_cohort": a.oct_cohort,
                  "drop_mixed": a.drop_mixed,
                  "match_train": a.match_train,
                  "match_repeats": a.match_repeats if a.match_train else None,
                  "oct_train_size_mean": (
                      {"images": round(float(np.mean(
                          [s[0] for s in oct_train_sizes])), 1),
                       "groups": round(float(np.mean(
                           [s[1] for s in oct_train_sizes])), 1)}
                      if oct_train_sizes else None),
                  "classifier": "LogisticRegression(max_iter=2000, "
                                "class_weight='balanced') on frozen "
                                "ConvNeXt features",
                  "delta_sign": "positive = worse when trained on the "
                                "other modality (log_loss: cross - within; "
                                "others: within - cross)",
                  "note": "fold scores share training data; sd is spread "
                          "across folds, not a standard error"},
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  results -> {out}\n")


if __name__ == "__main__":
    main()