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


def apply_groups(df: pd.DataFrame, groups_csv, exclude_conflicted: bool):
    """
    Attach a 'group_key' used for splitting. Without --groups this is just
    patient_key. With it, patients linked by byte-identical images share a
    group_key, so a duplicated eye recorded under two patient IDs cannot end
    up in two different splits.
    """
    if not groups_csv:
        df["group_key"] = df["patient_key"]
        if exclude_conflicted:
            sys.exit("ERROR: --exclude-conflicted requires --groups")
        return df

    if not os.path.isfile(groups_csv):
        sys.exit(f"ERROR: groups file not found: {groups_csv}")

    g = pd.read_csv(groups_csv)
    for col in ("patient_key", "group_key"):
        if col not in g.columns:
            sys.exit(f"ERROR: {groups_csv} missing column '{col}'")

    mapping = dict(zip(g["patient_key"], g["group_key"]))
    unmapped = set(df["patient_key"]) - set(mapping)
    if unmapped:
        sys.exit(f"ERROR: {len(unmapped)} patients absent from {groups_csv}: "
                 f"{sorted(unmapped)[:5]}")

    df["group_key"] = df["patient_key"].map(mapping)
    n_merged = df["patient_key"].nunique() - df["group_key"].nunique()
    if n_merged:
        print(f"  Grouping: {df['patient_key'].nunique()} patients -> "
              f"{df['group_key'].nunique()} independent groups "
              f"({n_merged} merged as duplicates).")

    if exclude_conflicted:
        if "conflicted" not in g.columns:
            sys.exit(f"ERROR: {groups_csv} has no 'conflicted' column")
        bad = set(g.loc[g["conflicted"].astype(bool), "patient_key"])
        if bad:
            n0 = len(df)
            df = df[~df["patient_key"].isin(bad)].reset_index(drop=True)
            print(f"  Excluded {len(bad)} patients with cross-class duplicate "
                  f"images ({n0 - len(df)} B-scans dropped): "
                  f"{sorted(bad)}")
        else:
            print("  No conflicted patients to exclude.")

    return df


def summarise(df: pd.DataFrame, protocol: str) -> None:
    print(f"\n{'='*68}\nDATASET SUMMARY  (protocol: {protocol})\n{'='*68}")
    print(f"  B-scans        : {len(df):,}")
    print(f"  Patients       : {df['patient_key'].nunique()}")
    if "group_key" in df.columns and df["group_key"].nunique() != df["patient_key"].nunique():
        print(f"  Split groups   : {df['group_key'].nunique()} "
              f"(duplicated patients merged)")
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

    return df


def verify(df: pd.DataFrame) -> bool:
    """Assert patient-level disjointness. Prints the table for the paper."""
    print(f"\n{'='*68}\nSPLIT VERIFICATION\n{'='*68}")

    pats = {s: set(g["patient_key"]) for s, g in df.groupby("split")}
    eyes = {s: set(g["eye_key"]) for s, g in df.groupby("split")}
    grps = {s: set(g["group_key"]) for s, g in df.groupby("split")}

    print(f"\n  {'Split':<8}{'B-scans':>10}{'Patients':>11}{'Eyes':>8}{'%':>8}")
    print(f"  {'-'*45}")
    for s in ("train", "val", "test"):
        n = (df["split"] == s).sum()
        print(f"  {s:<8}{n:>10,}{len(pats.get(s, set())):>11}"
              f"{len(eyes.get(s, set())):>8}{100*n/len(df):>7.1f}%")
    print(f"  {'-'*45}")
    print(f"  {'TOTAL':<8}{len(df):>10,}{df['patient_key'].nunique():>11}"
          f"{df['eye_key'].nunique():>8}{100.0:>7.1f}%")

    print("\n  Intersections (must all be 0):")
    ok = True
    for name, d in (("patients", pats), ("groups", grps)):
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            n = len(d.get(a, set()) & d.get(b, set()))
            flag = "OK" if n == 0 else "*** LEAK ***"
            if n:
                ok = False
            print(f"    {name:<9} {a:<6} n {b:<6}: {n:>4}   {flag}")

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
    p.add_argument("--groups", default=None,
                   help="optional patient_groups.csv from dedup_patients.py. "
                        "Links patients that share identical images so "
                        "duplicated eyes cannot straddle a split.")
    p.add_argument("--exclude-conflicted", action="store_true",
                   help="drop patients whose images are byte-identical to "
                        "another patient of a DIFFERENT class (contradictory "
                        "labels). Requires --groups.")
    args = p.parse_args()

    groups_csv_used = args.groups
    df = load(args.csv)
    df = apply_groups(df, args.groups, args.exclude_conflicted)

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
    cols = ["Directory", "patient_key", "group_key", "eye_key", "Class", "Label",
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
        "n_split_groups": int(df["group_key"].nunique()),
        "groups_file": groups_csv_used,
        "excluded_conflicted": bool(args.exclude_conflicted),
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