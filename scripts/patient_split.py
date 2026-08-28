#!/usr/bin/env python3
"""
patient_split.py — patient-disjoint train/val/test split for the NEH OCT dataset.

Writes a manifest CSV with a 'split' column added. No images are copied.

CRITICAL: 'Patient ID' is NOT unique across classes in this dataset.
    CNV IDs run 1..161, DRUSEN 1..160, NORMAL 1..120 -- they overlap completely.
    CNV patient 1 and DRUSEN patient 1 are DIFFERENT PEOPLE.
    The grouping key is therefore (Class, Patient ID), never 'Patient ID' alone.

Grouping is by PATIENT, not by eye: 113 of 441 patients contributed both eyes,
and two eyes of one person are correlated.

Usage:
    python patient_split.py --csv datasets/neh_meta/data_information.csv \
                            --out manifests/neh_split.csv
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

CLASSES = ["NORMAL", "DRUSEN", "CNV"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def load(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        sys.exit(f"ERROR: CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"Patient ID", "Class", "Eye", "B-scan", "Label", "Directory"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: CSV missing columns: {sorted(missing)}")

    if df.empty:
        sys.exit("ERROR: CSV is empty.")

    # The grouping key. See module docstring.
    df["patient_key"] = df["Class"] + "_" + df["Patient ID"].astype(str)
    df["eye_key"] = df["patient_key"] + "_" + df["Eye"]

    unknown = set(df["Class"]) - set(CLASSES)
    if unknown:
        sys.exit(f"ERROR: unexpected Class values: {sorted(unknown)}")
    unknown = set(df["Label"]) - set(CLASSES)
    if unknown:
        sys.exit(f"ERROR: unexpected Label values: {sorted(unknown)}")

    # The source CSV ships 12 paths listed twice (patients CNV/58, CNV/69,
    # DRUSEN/59, DRUSEN/150). Same patient, so no cross-split leakage, but
    # they would be sampled twice per epoch. Drop them and say so.
    n_before = len(df)
    df = df.drop_duplicates(subset="Directory", keep="first").reset_index(drop=True)
    if len(df) < n_before:
        print(f"  NOTE: dropped {n_before - len(df)} duplicate rows "
              f"from the source CSV ({n_before:,} -> {len(df):,}).")

    ext = df["Directory"].str.rsplit(".", n=1).str[-1].str.lower()
    print(f"  File extensions: {dict(ext.value_counts())}")

    return df


def summarise(df: pd.DataFrame, protocol: str) -> None:
    print(f"\n{'='*68}\nDATASET SUMMARY  (protocol: {protocol})\n{'='*68}")
    print(f"  B-scans        : {len(df):,}")
    print(f"  Patients       : {df['patient_key'].nunique()}")
    print(f"  Eyes (volumes) : {df['eye_key'].nunique()}")

    vol = df.groupby("eye_key").size()
    print(f"  B-scans/volume : mean {vol.mean():.1f}, "
          f"median {vol.median():.0f}, range {vol.min()}-{vol.max()}")

    print("\n  Patients per class:")
    pc = df.groupby("Class")["patient_key"].nunique()
    for c in CLASSES:
        print(f"    {c:<8} {pc.get(c, 0):>5}")

    print("\n  B-scans per training label:")
    lc = df["y_label"].value_counts()
    for c in CLASSES:
        print(f"    {c:<8} {lc.get(c, 0):>7,}")
    if len(lc) > 1:
        print(f"    imbalance ratio: {lc.max() / lc.min():.2f}:1")


def split(df: pd.DataFrame, test_ratio: float, val_ratio: float, seed: int):
    """
    Patient-disjoint split, performed independently WITHIN each class so that
    class proportions are preserved at the patient level (stratified by
    construction). Each patient belongs to exactly one class, so splitting
    per class cannot break disjointness.
    """
    df = df.copy()
    df["split"] = "train"

    for cls in sorted(df["Class"].unique()):
        sub = df[df["Class"] == cls]
        groups = sub["patient_key"].values

        gss = GroupShuffleSplit(n_splits=1, test_size=test_ratio,
                                random_state=seed)
        dev_rel, test_rel = next(gss.split(sub, groups=groups))

        dev = sub.iloc[dev_rel]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio,
                                 random_state=seed + 1)
        _, val_rel = next(gss2.split(dev, groups=dev["patient_key"].values))

        df.loc[sub.index[test_rel], "split"] = "test"
        df.loc[dev.index[val_rel], "split"] = "val"

    return df


def verify(df: pd.DataFrame) -> bool:
    """Assert patient-level disjointness. Prints the table for the paper."""
    print(f"\n{'='*68}\nSPLIT VERIFICATION\n{'='*68}")

    pats = {s: set(g["patient_key"]) for s, g in df.groupby("split")}
    eyes = {s: set(g["eye_key"]) for s, g in df.groupby("split")}

    print(f"\n  {'Split':<8}{'B-scans':>10}{'Patients':>11}{'Eyes':>8}{'%':>8}")
    print(f"  {'-'*45}")
    for s in ("train", "val", "test"):
        n = (df["split"] == s).sum()
        print(f"  {s:<8}{n:>10,}{len(pats.get(s, set())):>11}"
              f"{len(eyes.get(s, set())):>8}{100*n/len(df):>7.1f}%")
    print(f"  {'-'*45}")
    print(f"  {'TOTAL':<8}{len(df):>10,}{df['patient_key'].nunique():>11}"
          f"{df['eye_key'].nunique():>8}{100.0:>7.1f}%")

    print("\n  Patient-level intersections (must all be 0):")
    ok = True
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        n = len(pats.get(a, set()) & pats.get(b, set()))
        flag = "OK" if n == 0 else "*** LEAK ***"
        if n:
            ok = False
        print(f"    {a:<6} n {b:<6}: {n:>4}   {flag}")

    print("\n  Class distribution per split (B-scans):")
    ct = pd.crosstab(df["split"], df["y_label"])
    ct = ct.reindex(index=["train", "val", "test"],
                    columns=[c for c in CLASSES if c in ct.columns])
    print("    " + ct.to_string().replace("\n", "\n    "))

    print("\n  Class distribution per split (patients):")
    pt = df.drop_duplicates("patient_key")
    ct2 = pd.crosstab(pt["split"], pt["Class"])
    ct2 = ct2.reindex(index=["train", "val", "test"],
                      columns=[c for c in CLASSES if c in ct2.columns])
    print("    " + ct2.to_string().replace("\n", "\n    "))

    print(f"\n  => {'PATIENT-DISJOINT' if ok else 'LEAKAGE PRESENT'}")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="datasets/neh_meta/data_information.csv")
    p.add_argument("--out", default="manifests/neh_split.csv")
    p.add_argument("--protocol", choices=["all", "worstcase"], default="worstcase",
                   help="'worstcase' keeps only B-scans where Label == Class "
                        "(12,649 images); 'all' keeps every B-scan labelled by "
                        "its own Label (16,822 images).")
    p.add_argument("--test-ratio", type=float, default=0.20)
    p.add_argument("--val-ratio", type=float, default=0.20,
                   help="fraction of the remaining development set")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = load(args.csv)

    if args.protocol == "worstcase":
        df = df[df["Class"] == df["Label"]].copy()
        df["y_label"] = df["Label"]
    else:
        df["y_label"] = df["Label"]

    if df.empty:
        sys.exit("ERROR: no rows left after protocol filter.")

    df["y"] = df["y_label"].map(CLASS_TO_IDX)

    summarise(df, args.protocol)
    df = split(df, args.test_ratio, args.val_ratio, args.seed)
    ok = verify(df)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = ["Directory", "patient_key", "eye_key", "Class", "Label",
            "y_label", "y", "Eye", "B-scan", "split"]
    df[cols].to_csv(args.out, index=False)

    meta = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source_csv": args.csv,
        "protocol": args.protocol,
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "val_ratio_of_dev": args.val_ratio,
        "grouping_key": "(Class, Patient ID) -- IDs are not unique across classes",
        "n_images": int(len(df)),
        "n_patients": int(df["patient_key"].nunique()),
        "n_eyes": int(df["eye_key"].nunique()),
        "patient_disjoint": bool(ok),
        "counts": {s: int((df["split"] == s).sum()) for s in ("train", "val", "test")},
    }
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Manifest -> {args.out}")
    print(f"  Metadata -> {meta_path}\n")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()