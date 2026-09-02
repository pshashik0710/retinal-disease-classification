#!/usr/bin/env python3
"""
pool_manifests.py -- merge the NEH and Kermany manifests into one cohort and
produce a single patient-disjoint train/val/test split across the pool.

Both source manifests have already been audited independently:

  NEH      12,565 images / 437 patients   (duplicated eyes merged,
                                            cross-class conflicts excluded)
  Kermany  76,677 images / 4,656 patients (7,807 duplicates removed,
                                            466 cross-class conflicts dropped)

What this script does:

  1. Loads both, tags each row with its cohort, and checks that the
     patient keys cannot collide (NEH keys are CLASS_ID, Kermany keys
     are KER_ID).

  2. Optionally caps the number of images contributed per patient. Kermany
     is dominated by a few exhaustively scanned eyes -- CNV averages 40
     images/patient against NORMAL's 7.6, and one patient contributes 365
     against a median of 6. Without a cap those eyes dominate whichever
     split they land in and inflate the apparent class imbalance. Images
     are sampled EVENLY across each patient's B-scan range, not truncated,
     so the retained slices still span the volume.

  3. Optionally drops images whose aspect ratio is an outlier (Kermany
     contains 16 portrait 384x496 scans; everything else is landscape).

  4. Splits per class, grouped on group_key, so class proportions are
     preserved at the patient level and no patient appears in two splits.

  5. Verifies disjointness at patient, group, eye and file level, and
     reports the cohort x class x split breakdown -- including the
     cohort-confound warning that matters here, since DME comes only
     from Kermany.

Usage:
    python pool_manifests.py --neh manifests/neh_split.csv \
                             --kermany manifests/kermany_clean.csv \
                             --out manifests/pooled_split.csv \
                             --cap 50 --drop-ar-outliers
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

CLASSES = ["NORMAL", "DRUSEN", "CNV", "DME"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# columns every pooled row must carry
CORE = ["Directory", "cohort", "patient_key", "group_key", "eye_key",
        "Class", "Label", "y_label", "y", "Eye", "B-scan", "width", "height"]


def load_one(path, cohort, root_hint):
    if not os.path.isfile(path):
        sys.exit(f"ERROR: manifest not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        sys.exit(f"ERROR: {path} is empty")

    need = {"Directory", "patient_key", "y_label"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"ERROR: {path} missing columns: {sorted(missing)}")

    df = df.copy()
    df["cohort"] = cohort
    df["root_hint"] = root_hint

    if "group_key" not in df.columns:
        df["group_key"] = df["patient_key"]
    if "eye_key" not in df.columns:
        df["eye_key"] = df["patient_key"]
    for c in ("width", "height"):
        if c not in df.columns:
            df[c] = 0
    if "Eye" not in df.columns:
        df["Eye"] = "NA"
    if "B-scan" not in df.columns:
        df["B-scan"] = 0
    if "Class" not in df.columns:
        df["Class"] = df["y_label"]
    if "Label" not in df.columns:
        df["Label"] = df["y_label"]

    # drop any pre-existing split column: we are re-splitting the pool
    df = df.drop(columns=[c for c in ("split", "official_split")
                          if c in df.columns])

    df["y"] = df["y_label"].map(CLASS_TO_IDX)
    if df["y"].isna().any():
        bad = sorted(df.loc[df["y"].isna(), "y_label"].unique())
        sys.exit(f"ERROR: {path} has labels outside {CLASSES}: {bad}")

    return df[CORE + ["root_hint"]]


def cap_per_patient(df, cap, seed):
    """Keep at most `cap` images per patient, sampled evenly across the volume."""
    if not cap:
        return df, {}

    rng = np.random.default_rng(seed)
    keep_idx = []
    n_capped = 0

    for pk, sub in df.groupby("patient_key", sort=True):
        if len(sub) <= cap:
            keep_idx.extend(sub.index)
            continue
        n_capped += 1
        # order by B-scan where available so the sample spans the volume
        sub = sub.sort_values(["eye_key", "B-scan", "Directory"])
        # even stride across the ordered slices
        pick = np.linspace(0, len(sub) - 1, cap).round().astype(int)
        pick = np.unique(pick)
        keep_idx.extend(sub.index[pick])

    out = df.loc[sorted(keep_idx)].reset_index(drop=True)
    info = {"cap": cap, "patients_capped": n_capped,
            "images_before": len(df), "images_after": len(out)}
    return out, info


def summarise(df, title):
    print(f"\n{'='*72}\n{title}\n{'='*72}")
    print(f"  images   : {len(df):,}")
    print(f"  patients : {df['patient_key'].nunique():,}")
    print(f"  groups   : {df['group_key'].nunique():,}")
    print(f"  eyes     : {df['eye_key'].nunique():,}")

    print("\n  images per cohort x class:")
    ct = pd.crosstab(df["cohort"], df["y_label"])
    ct = ct.reindex(columns=[c for c in CLASSES if c in ct.columns], fill_value=0)
    ct["TOTAL"] = ct.sum(axis=1)
    print("    " + ct.to_string().replace("\n", "\n    "))

    print("\n  patients per cohort x class (a patient may span classes):")
    pt = df.drop_duplicates(["patient_key", "y_label"])
    ct2 = pd.crosstab(pt["cohort"], pt["y_label"])
    ct2 = ct2.reindex(columns=[c for c in CLASSES if c in ct2.columns],
                      fill_value=0)
    print("    " + ct2.to_string().replace("\n", "\n    "))

    vc = df["y_label"].value_counts()
    print(f"\n  class imbalance: {vc.max()/vc.min():.2f}:1 "
          f"({vc.idxmax()} {vc.max():,} vs {vc.idxmin()} {vc.min():,})")

    v = df.groupby("patient_key").size()
    print(f"  images per patient: mean {v.mean():.1f}, median {v.median():.0f}, "
          f"range {v.min()}-{v.max()}")


def cohort_confound_check(df):
    """Flag classes that come from only one cohort."""
    print(f"\n{'='*72}\nCOHORT CONFOUND CHECK\n{'='*72}")
    ct = pd.crosstab(df["y_label"], df["cohort"])
    single = []
    for cls in ct.index:
        nz = ct.loc[cls][ct.loc[cls] > 0]
        if len(nz) == 1:
            single.append((cls, nz.index[0], int(nz.iloc[0])))

    if not single:
        print("  every class draws on both cohorts               OK")
        return []

    for cls, coh, n in single:
        print(f"  {cls}: all {n:,} images come from '{coh}' only")
    print("\n  A model can reach high accuracy on such a class by recognising")
    print("  the COHORT (scanner, resolution, preprocessing) rather than the")
    print("  pathology. This will not show up in the headline accuracy.")
    print("  Mitigations, all cheap:")
    print("    1. train a cohort classifier (NEH vs Kermany) as a probe --")
    print("       if it is near-perfect the domain gap is large")
    print("    2. report per-class metrics broken down by cohort")
    print("    3. state the limitation explicitly in the paper")
    return [c for c, _, _ in single]


def split_pool(df, test_ratio, val_ratio, seed):
    """Per-class grouped split, so class proportions hold at patient level."""
    df = df.copy()
    df["split"] = "train"

    # a patient can span classes (Kermany); assign each patient to its
    # dominant class for the purposes of stratifying the split, so the
    # patient still moves as one unit
    dom = (df.groupby(["group_key", "y_label"]).size()
             .reset_index(name="n")
             .sort_values(["group_key", "n"], ascending=[True, False])
             .drop_duplicates("group_key")
             .set_index("group_key")["y_label"])
    df["_stratum"] = df["group_key"].map(dom)

    for cls in sorted(df["_stratum"].unique()):
        sub = df[df["_stratum"] == cls]
        groups = sub["group_key"].values

        gss = GroupShuffleSplit(n_splits=1, test_size=test_ratio,
                                random_state=seed)
        dev_rel, test_rel = next(gss.split(sub, groups=groups))

        dev = sub.iloc[dev_rel]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio,
                                 random_state=seed + 1)
        _, val_rel = next(gss2.split(dev, groups=dev["group_key"].values))

        df.loc[sub.index[test_rel], "split"] = "test"
        df.loc[dev.index[val_rel], "split"] = "val"

    return df.drop(columns=["_stratum"])


def verify(df):
    print(f"\n{'='*72}\nSPLIT VERIFICATION\n{'='*72}")

    print(f"\n  {'Split':<8}{'Images':>10}{'Patients':>11}{'Groups':>9}"
          f"{'Eyes':>8}{'%':>8}")
    print(f"  {'-'*54}")
    for s in ("train", "val", "test"):
        g = df[df["split"] == s]
        print(f"  {s:<8}{len(g):>10,}{g['patient_key'].nunique():>11}"
              f"{g['group_key'].nunique():>9}{g['eye_key'].nunique():>8}"
              f"{100*len(g)/len(df):>7.1f}%")
    print(f"  {'-'*54}")
    print(f"  {'TOTAL':<8}{len(df):>10,}{df['patient_key'].nunique():>11}"
          f"{df['group_key'].nunique():>9}{df['eye_key'].nunique():>8}"
          f"{100.0:>7.1f}%")

    print("\n  Intersections (must all be 0):")
    ok = True
    for col, name in (("patient_key", "patients"), ("group_key", "groups"),
                      ("eye_key", "eyes"), ("Directory", "files")):
        sets = {s: set(g[col]) for s, g in df.groupby("split")}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            n = len(sets.get(a, set()) & sets.get(b, set()))
            if n:
                ok = False
                print(f"    {name:<9} {a:<6} n {b:<6}: {n:>5}   *** LEAK ***")
        if all(len(sets.get(a, set()) & sets.get(b, set())) == 0
               for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            print(f"    {name:<9} disjoint across all splits        OK")

    print("\n  images per split x class:")
    ct = pd.crosstab(df["split"], df["y_label"])
    ct = ct.reindex(index=["train", "val", "test"],
                    columns=[c for c in CLASSES if c in ct.columns],
                    fill_value=0)
    print("    " + ct.to_string().replace("\n", "\n    "))

    print("\n  images per split x cohort:")
    ct2 = pd.crosstab(df["split"], df["cohort"])
    ct2 = ct2.reindex(index=["train", "val", "test"], fill_value=0)
    print("    " + ct2.to_string().replace("\n", "\n    "))

    print("\n  patients per split x cohort:")
    pt = df.drop_duplicates(["patient_key", "split"])
    ct3 = pd.crosstab(pt["split"], pt["cohort"])
    ct3 = ct3.reindex(index=["train", "val", "test"], fill_value=0)
    print("    " + ct3.to_string().replace("\n", "\n    "))

    print(f"\n  => {'PATIENT-DISJOINT' if ok else 'LEAKAGE PRESENT'}")
    return ok


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--neh", default="manifests/neh_split.csv")
    p.add_argument("--kermany", default="manifests/kermany_clean.csv")
    p.add_argument("--neh-root",
                   default="D:/datasets/neh/NEH_UT_2021RetinalOCTDataset")
    p.add_argument("--kermany-root", default="D:/datasets/kermany2018/OCT2017")
    p.add_argument("--out", default="manifests/pooled_split.csv")
    p.add_argument("--cap", type=int, default=50,
                   help="max images per patient (0 = no cap). Sampled evenly "
                        "across each volume, not truncated.")
    p.add_argument("--drop-ar-outliers", action="store_true",
                   help="drop images whose aspect ratio is below 1.0 "
                        "(portrait); Kermany has 16 such 384x496 scans.")
    p.add_argument("--test-ratio", type=float, default=0.20)
    p.add_argument("--val-ratio", type=float, default=0.20,
                   help="fraction of the remaining development set")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print(f"\n{'='*72}\nLOADING\n{'='*72}")
    neh = load_one(args.neh, "neh", args.neh_root)
    ker = load_one(args.kermany, "kermany", args.kermany_root)
    print(f"  NEH     : {len(neh):>7,} images  "
          f"{neh['patient_key'].nunique():>5} patients")
    print(f"  Kermany : {len(ker):>7,} images  "
          f"{ker['patient_key'].nunique():>5} patients")

    clash = set(neh["patient_key"]) & set(ker["patient_key"])
    if clash:
        sys.exit(f"ERROR: {len(clash)} patient keys collide between cohorts: "
                 f"{sorted(clash)[:5]}")
    print(f"  patient key collisions: 0                        OK")

    df = pd.concat([neh, ker], ignore_index=True)
    summarise(df, "POOLED (raw)")

    log = {"generated": datetime.now().isoformat(timespec="seconds"),
           "neh_manifest": args.neh, "kermany_manifest": args.kermany,
           "seed": args.seed, "test_ratio": args.test_ratio,
           "val_ratio_of_dev": args.val_ratio,
           "n_images_raw": len(df),
           "n_patients_raw": int(df["patient_key"].nunique())}

    # ------------------------------------------------ aspect-ratio filter
    if args.drop_ar_outliers:
        print(f"\n{'='*72}\nASPECT-RATIO FILTER\n{'='*72}")
        has_dim = (df["width"] > 0) & (df["height"] > 0)
        ar = df["width"] / df["height"].replace(0, np.nan)
        bad = has_dim & (ar < 1.0)
        if bad.any():
            dims = df.loc[bad].groupby(["width", "height"]).size()
            print(f"  dropping {int(bad.sum())} portrait images:")
            for (w, h), n in dims.items():
                print(f"    {w} x {h}  ({w/h:.3f})  {n}")
            df = df[~bad].reset_index(drop=True)
        else:
            print("  no portrait-orientation images found")
        log["ar_outliers_dropped"] = int(bad.sum())

    # -------------------------------------------------------------- cap
    if args.cap:
        print(f"\n{'='*72}\nPER-PATIENT CAP\n{'='*72}")
        before = len(df)
        df, capinfo = cap_per_patient(df, args.cap, args.seed)
        print(f"  cap: {args.cap} images per patient, sampled evenly "
              f"across each volume")
        print(f"  patients capped : {capinfo['patients_capped']:,}")
        print(f"  images          : {before:,} -> {len(df):,} "
              f"({before - len(df):,} dropped)")
        log["cap"] = capinfo
        summarise(df, "POOLED (after cap)")

    single_cohort_classes = cohort_confound_check(df)
    log["single_cohort_classes"] = single_cohort_classes

    df = split_pool(df, args.test_ratio, args.val_ratio, args.seed)
    ok = verify(df)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = CORE + ["root_hint", "split"]
    df[cols].to_csv(args.out, index=False)

    log.update({
        "n_images_final": len(df),
        "n_patients_final": int(df["patient_key"].nunique()),
        "n_groups_final": int(df["group_key"].nunique()),
        "patient_disjoint": bool(ok),
        "counts": {s: int((df["split"] == s).sum())
                   for s in ("train", "val", "test")},
        "class_counts": {k: int(v) for k, v in
                         df["y_label"].value_counts().items()},
        "cohort_counts": {k: int(v) for k, v in
                          df["cohort"].value_counts().items()},
    })
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n  Manifest -> {args.out}")
    print(f"  Metadata -> {meta_path}\n")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()