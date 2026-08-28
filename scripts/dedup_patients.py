#!/usr/bin/env python3
"""
dedup_patients.py -- find duplicated images in the NEH dataset and link the
patients they connect, so that duplicated eyes cannot straddle a split.

The published dataset contains eyes entered more than once:

  * same class, different patient ID   e.g. CNV/26  == CNV/147
        -> one eye recorded twice. Two 'patients' that are one person.
           If they land in different splits, that is train/test contamination
           that patient-level grouping alone will NOT catch, because the two
           records have different patient IDs.

  * different class, different ID      e.g. CNV/22  == DRUSEN/69
        -> the same pixels carry two different diagnoses. Beyond the split
           problem, the labels are contradictory: the model would be taught
           to call one image both CNV and DRUSEN.

This script hashes every image, groups patients connected by identical
content (union-find), and writes a patient -> group mapping. Feed that to
patient_split.py with --groups so linked patients are split together.

Usage:
    python dedup_patients.py \
        --root "D:/datasets/neh/NEH_UT_2021RetinalOCTDataset" \
        --csv datasets/neh_meta/data_information.csv \
        --out manifests/patient_groups.csv
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

import pandas as pd


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic: smaller string becomes the root
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def file_hash(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--csv", default="datasets/neh_meta/data_information.csv")
    p.add_argument("--out", default="manifests/patient_groups.csv")
    args = p.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"ERROR: root not found: {args.root}")
    if not os.path.isfile(args.csv):
        sys.exit(f"ERROR: csv not found: {args.csv}")

    df = pd.read_csv(args.csv)
    df = df.drop_duplicates(subset="Directory", keep="first").reset_index(drop=True)
    df["patient_key"] = df["Class"] + "_" + df["Patient ID"].astype(str)
    df["eye_key"] = df["patient_key"] + "_" + df["Eye"]

    print(f"\n  manifest rows : {len(df):,}")
    print(f"  patients      : {df['patient_key'].nunique()}")
    print(f"  eyes          : {df['eye_key'].nunique()}")
    print(f"\n  hashing {len(df):,} images (a few minutes)...")

    by_hash = defaultdict(list)
    missing = 0
    for i, row in enumerate(df.itertuples(index=False), 1):
        path = os.path.join(args.root, row.Directory.replace("/", os.sep))
        if not os.path.isfile(path):
            missing += 1
            continue
        by_hash[file_hash(path)].append(
            (row.Directory, row.patient_key, row.eye_key, row.Class))
        if i % 2000 == 0:
            print(f"    {i:,} / {len(df):,}")

    if missing:
        print(f"  WARNING: {missing} manifest paths not found on disk")

    dup_groups = {h: v for h, v in by_hash.items() if len(v) > 1}

    print(f"\n{'='*70}\nDUPLICATE CONTENT\n{'='*70}")
    print(f"  duplicate image groups : {len(dup_groups)}")
    print(f"  images involved        : {sum(len(v) for v in dup_groups.values())}")

    # ------------------------------------------------ link patients
    uf = UnionFind()
    for pk in df["patient_key"].unique():
        uf.find(pk)

    cross_class = []
    for h, members in dup_groups.items():
        pks = {m[1] for m in members}
        classes = {m[3] for m in members}
        if len(pks) > 1:
            base = sorted(pks)[0]
            for other in sorted(pks)[1:]:
                uf.union(base, other)
        if len(classes) > 1:
            cross_class.append(members)

    groups = defaultdict(set)
    for pk in df["patient_key"].unique():
        groups[uf.find(pk)].add(pk)

    linked = {root: members for root, members in groups.items()
              if len(members) > 1}

    print(f"\n  patients linked by identical content: "
          f"{sum(len(v) for v in linked.values())} in {len(linked)} groups")
    if linked:
        print()
        for root, members in sorted(linked.items()):
            classes = sorted({m.split("_")[0] for m in members})
            tag = "  <-- CROSS-CLASS, contradictory labels" if len(classes) > 1 else ""
            print(f"    {sorted(members)}{tag}")

    # --------------------------------------------- cross-class detail
    conflicted = set()
    if cross_class:
        print(f"\n{'='*70}\nCROSS-CLASS DUPLICATES\n{'='*70}")
        print(f"  {len(cross_class)} image groups carry more than one class label.")
        pairs = defaultdict(int)
        for members in cross_class:
            key = tuple(sorted({m[1] for m in members}))
            pairs[key] += 1
            conflicted.update(key)
        print("\n  affected patient pairs (and how many images each shares):")
        for key, n in sorted(pairs.items()):
            print(f"    {' == '.join(key)}   {n} identical images")
        print(f"\n  These patients cannot be assigned a single trustworthy label.")
        print(f"  Recommend excluding them: pass --exclude-conflicted to "
              f"patient_split.py")

    # ------------------------------------------------------- write out
    rows = []
    for pk in sorted(df["patient_key"].unique()):
        root = uf.find(pk)
        rows.append({
            "patient_key": pk,
            "group_key": root,
            "n_linked": len(groups[root]),
            "conflicted": pk in conflicted,
        })
    out = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    meta = {
        "root": args.root,
        "source_csv": args.csv,
        "n_patients": int(df["patient_key"].nunique()),
        "n_groups": int(out["group_key"].nunique()),
        "n_linked_patients": int((out["n_linked"] > 1).sum()),
        "n_conflicted_patients": int(out["conflicted"].sum()),
        "duplicate_image_groups": len(dup_groups),
        "cross_class_image_groups": len(cross_class),
    }
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"  patients                : {meta['n_patients']}")
    print(f"  independent groups      : {meta['n_groups']}")
    print(f"  patients merged         : {meta['n_linked_patients']}")
    print(f"  patients with conflicts : {meta['n_conflicted_patients']}")
    print(f"\n  Mapping  -> {args.out}")
    print(f"  Metadata -> {meta_path}\n")


if __name__ == "__main__":
    main()