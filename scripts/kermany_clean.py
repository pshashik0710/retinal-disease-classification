#!/usr/bin/env python3
"""
kermany_clean.py -- turn the raw Kermany OCT2017 tree into a manifest that
matches the NEH schema, so both cohorts can be pooled and split together.

The published OCT2017 release has several defects this script records and
removes:

  * the official train/val/test split leaks: 546 of 609 test patients (89.7%)
    and 21 of 25 val patients also appear in train. The official split is
    therefore DISCARDED here -- a new patient-disjoint split is made later,
    over the pooled cohorts, by pool_manifests.py.

  * 19.2% of patients appear under more than one class folder. Per the agreed
    protocol, each IMAGE keeps its own folder label (labels are per-B-scan),
    while SPLITTING is still done per patient, so every slice from one patient
    lands in one split regardless of its label.

  * byte-identical images occur across the official splits. All duplicate
    content is detected by hashing and reduced to a single copy.

  * five distinct image dimensions are present (512x496, 768x496, 512x512,
    1536x496, 1024x496), i.e. aspect ratios from 1.00 to 3.10. This script
    records each image's dimensions in the manifest so the transform pipeline
    can preserve aspect ratio rather than forcing a square.

Filenames encode the patient: CNV-1016042-1.jpeg -> class CNV, patient 1016042.

Usage:
    python kermany_clean.py --root "D:/datasets/kermany2018/OCT2017" \
                            --out manifests/kermany_clean.csv
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd
from PIL import Image

EXTS = {".jpeg", ".jpg", ".png", ".tif", ".tiff", ".bmp"}
FNAME = re.compile(r"^([A-Za-z]+)-(\d+)-(\d+)$")
CLASSES = ["NORMAL", "DRUSEN", "CNV", "DME"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def file_hash(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def scan(root):
    """Walk the official train/val/test tree; parse class and patient."""
    rows, unparsed = [], []
    for split in ("train", "val", "test"):
        sd = os.path.join(root, split)
        if not os.path.isdir(sd):
            print(f"  WARNING: no '{split}' directory under root")
            continue
        for cls in sorted(os.listdir(sd)):
            cd = os.path.join(sd, cls)
            if not os.path.isdir(cd):
                continue
            for fn in sorted(os.listdir(cd)):
                stem, ext = os.path.splitext(fn)
                if ext.lower() not in EXTS:
                    continue
                m = FNAME.match(stem)
                if not m:
                    unparsed.append(f"{split}/{cls}/{fn}")
                    continue
                fcls, pid, scan_no = m.groups()
                rows.append({
                    "Directory": f"{split}/{cls}/{fn}",
                    "official_split": split,
                    "folder_class": cls.upper(),
                    "filename_class": fcls.upper(),
                    "patient_id": pid,
                    "bscan": int(scan_no),
                })
    return pd.DataFrame(rows), unparsed


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="manifests/kermany_clean.csv")
    p.add_argument("--classes", nargs="+", default=CLASSES,
                   help="class space to keep (default: all four)")
    p.add_argument("--no-dims", action="store_true",
                   help="skip reading image dimensions (much faster)")
    p.add_argument("--hash-sample", type=int, default=0,
                   help="0 = hash every image (recommended). A positive value "
                        "hashes only that many, which will MISS duplicates.")
    args = p.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"ERROR: root not found: {args.root}")

    keep = [c.upper() for c in args.classes]
    unknown = set(keep) - set(CLASSES)
    if unknown:
        sys.exit(f"ERROR: unknown classes {sorted(unknown)}; "
                 f"valid: {CLASSES}")

    log = {"root": args.root, "classes_kept": keep}

    # -------------------------------------------------------------- scan
    print(f"\n{'='*70}\n1. SCANNING\n{'='*70}")
    df, unparsed = scan(args.root)
    if df.empty:
        sys.exit("ERROR: no images found. Check --root.")

    print(f"  images found  : {len(df):,}")
    print(f"  patients      : {df['patient_id'].nunique():,}")
    print(f"  classes       : {sorted(df['folder_class'].unique())}")
    log["n_images_raw"] = len(df)
    log["n_patients_raw"] = int(df["patient_id"].nunique())

    if unparsed:
        print(f"  WARNING: {len(unparsed)} filenames did not parse; dropped")
        for u in unparsed[:5]:
            print(f"      {u}")
    log["n_unparsed"] = len(unparsed)

    mism = df[df["folder_class"] != df["filename_class"]]
    if len(mism):
        print(f"  WARNING: {len(mism)} files where folder class != filename "
              f"class; dropped")
        print(mism[["Directory", "folder_class", "filename_class"]]
              .head(5).to_string(index=False))
        df = df[df["folder_class"] == df["filename_class"]].copy()
    log["n_class_mismatch"] = int(len(mism))

    # ------------------------------------------------ official split audit
    print(f"\n{'='*70}\n2. OFFICIAL SPLIT AUDIT  (recorded, then discarded)\n{'='*70}")
    sets = {s: set(g["patient_id"]) for s, g in df.groupby("official_split")}
    overlaps = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in sets and b in sets:
            n = len(sets[a] & sets[b])
            pct = 100 * n / len(sets[b]) if sets[b] else 0.0
            overlaps[f"{a}_vs_{b}"] = {"n_patients": n, "pct_of_second": round(pct, 1)}
            print(f"  {a:<6} n {b:<6}: {n:>5} patients "
                  f"({pct:.1f}% of {b})")
    print("\n  The official split is NOT patient-disjoint and is discarded.")
    print("  A new split is made over the pooled cohorts by pool_manifests.py.")
    log["official_split_overlap"] = overlaps

    # ------------------------------------------------- multi-class patients
    print(f"\n{'='*70}\n3. MULTI-CLASS PATIENTS\n{'='*70}")
    ncls = df.groupby("patient_id")["folder_class"].nunique()
    multi = ncls[ncls > 1]
    print(f"  patients under >1 class: {len(multi):,} of "
          f"{df['patient_id'].nunique():,} ({100*len(multi)/df['patient_id'].nunique():.1f}%)")
    if len(multi):
        combos = Counter()
        for pid in multi.index:
            combos[tuple(sorted(df[df.patient_id == pid]["folder_class"].unique()))] += 1
        print("\n  label combinations:")
        for combo, n in combos.most_common(10):
            print(f"    {' + '.join(combo):<28} {n:>5} patients")
    print("\n  PROTOCOL: each image keeps its own folder label (labels are")
    print("  per-B-scan). Splitting remains per patient, so all slices from")
    print("  one patient land in the same split regardless of their labels.")
    log["n_multiclass_patients"] = int(len(multi))

    # ------------------------------------------------------ class filter
    print(f"\n{'='*70}\n4. CLASS FILTER\n{'='*70}")
    n0 = len(df)
    df = df[df["folder_class"].isin(keep)].copy()
    print(f"  keeping {keep}")
    print(f"  {n0:,} -> {len(df):,} images "
          f"({n0 - len(df):,} dropped)")
    if df.empty:
        sys.exit("ERROR: no images left after class filter.")
    log["n_images_after_class_filter"] = len(df)

    # -------------------------------------------------------- duplicates
    print(f"\n{'='*70}\n5. DUPLICATE DETECTION\n{'='*70}")
    paths = df["Directory"].tolist()
    n_hash = len(paths) if args.hash_sample == 0 else min(args.hash_sample, len(paths))
    if n_hash < len(paths):
        print(f"  WARNING: hashing only {n_hash:,} of {len(paths):,} "
              f"-- duplicates WILL be missed. Use --hash-sample 0.")
        step = max(1, len(paths) // n_hash)
        subset = paths[::step]
    else:
        subset = paths
    print(f"  hashing {len(subset):,} images (this takes a while)...")

    by_hash = defaultdict(list)
    for i, rel in enumerate(subset, 1):
        full = os.path.join(args.root, rel.replace("/", os.sep))
        try:
            by_hash[file_hash(full)].append(rel)
        except Exception as e:
            print(f"    unreadable: {rel} ({e})")
        if i % 10000 == 0:
            print(f"    {i:,} / {len(subset):,}")

    dups = {h: v for h, v in by_hash.items() if len(v) > 1}
    n_dup_imgs = sum(len(v) - 1 for v in dups.values())
    print(f"\n  duplicate content groups : {len(dups):,}")
    print(f"  redundant copies         : {n_dup_imgs:,}")

    if dups:
        cross_split, cross_class, cross_patient = 0, 0, 0
        meta = df.set_index("Directory")
        for v in dups.values():
            sp = {meta.loc[x, "official_split"] for x in v}
            cl = {meta.loc[x, "folder_class"] for x in v}
            pa = {meta.loc[x, "patient_id"] for x in v}
            cross_split += len(sp) > 1
            cross_class += len(cl) > 1
            cross_patient += len(pa) > 1
        print(f"    spanning official splits : {cross_split:,}")
        print(f"    spanning classes         : {cross_class:,}")
        print(f"    spanning patients        : {cross_patient:,}")

        ex = list(dups.values())[:5]
        print("\n  examples:")
        for v in ex:
            print(f"    {v}")

        
        # Same-class duplicates: keep one copy. Identical images would
        # otherwise be oversampled -- the model would see them repeatedly
        # each epoch and weight them accordingly in the loss.
        #
        # Cross-class duplicates: the same pixels are filed under two
        # different diagnoses, so NO copy carries a trustworthy label.
        # Dropping all copies is the defensible choice; keeping whichever
        # path happens to sort first would pick a label by filename order
        # rather than by evidence.

        drop = set()
        n_conflict_imgs = 0
        conflict_patients = set()
        conflict_combos = Counter()

        for v in dups.values():
            cl = {meta.loc[x, "folder_class"] for x in v}
            if len(cl) > 1:
                drop.update(v)
                n_conflict_imgs += len(v)
                conflict_patients.update(meta.loc[x, "patient_id"] for x in v)
                conflict_combos[tuple(sorted(cl))] += 1
            else:
                drop.update(sorted(v)[1:])

        if n_conflict_imgs:
            print(f"\n  CROSS-CLASS CONFLICTS")
            print(f"    {n_conflict_imgs:,} images from "
                  f"{len(conflict_patients):,} patients carry contradictory "
                  f"labels.")
            print(f"    All copies dropped -- the label is unresolvable.")
            for combo, n in conflict_combos.most_common():
                print(f"      {' / '.join(combo):<24} {n:>5} groups")

        df = df[~df["Directory"].isin(drop)].copy()
        print(f"\n  removed {len(drop):,} images total "
              f"-> {len(df):,} remaining")

        log["n_crossclass_images_dropped"] = n_conflict_imgs
        log["n_crossclass_patients"] = len(conflict_patients)
        log["crossclass_combos"] = {"/".join(k): v
                                    for k, v in conflict_combos.items()}

    log["n_duplicate_groups"] = len(dups)
    log["n_duplicates_removed"] = int(n_dup_imgs)
    log["hashed_all"] = args.hash_sample == 0

    # -------------------------------------------------------- dimensions
    if not args.no_dims:
        print(f"\n{'='*70}\n6. IMAGE DIMENSIONS\n{'='*70}")
        print(f"  reading dimensions for {len(df):,} images...")
        widths, heights = [], []
        for i, rel in enumerate(df["Directory"], 1):
            full = os.path.join(args.root, rel.replace("/", os.sep))
            try:
                with Image.open(full) as im:
                    w, h = im.size
            except Exception:
                w = h = 0
            widths.append(w)
            heights.append(h)
            if i % 20000 == 0:
                print(f"    {i:,} / {len(df):,}")
        df["width"] = widths
        df["height"] = heights

        dims = Counter(zip(widths, heights))
        print(f"\n  distinct dimensions: {len(dims)}")
        for (w, h), n in dims.most_common():
            print(f"    {w:>5} x {h:<5}  {n:>7,}   AR {w/h:.3f}")
        ars = sorted({round(w / h, 3) for w, h in dims if h})
        print(f"\n  aspect ratios: {ars}")
        print("  NOTE: forcing these to a square distorts anatomy "
              "non-uniformly.\n        The transform pipeline must resize "
              "the short side and crop,\n        or pad, rather than "
              "resize((N, N)).")
        log["dimensions"] = {f"{w}x{h}": n for (w, h), n in dims.most_common()}
    else:
        df["width"] = 0
        df["height"] = 0
        print("\n  (dimension pass skipped)")

    # ------------------------------------------------------ build manifest
    print(f"\n{'='*70}\n7. MANIFEST\n{'='*70}")
    out = pd.DataFrame({
        "Directory": df["Directory"],
        "cohort": "kermany",
        "patient_key": "KER_" + df["patient_id"],
        "eye_key": "KER_" + df["patient_id"],   # eye not encoded in filenames
        "group_key": "KER_" + df["patient_id"],
        "Class": df["folder_class"],            # patient-level dx unavailable
        "Label": df["folder_class"],            # per-image label = folder
        "y_label": df["folder_class"],
        "y": df["folder_class"].map(CLASS_TO_IDX),
        "Eye": "NA",
        "B-scan": df["bscan"],
        "width": df["width"],
        "height": df["height"],
        "official_split": df["official_split"],
    }).reset_index(drop=True)

    print(f"  images   : {len(out):,}")
    print(f"  patients : {out['patient_key'].nunique():,}")
    print("\n  images per class:")
    for c, n in out["y_label"].value_counts().items():
        print(f"    {c:<8} {n:>8,}")
    vc = out["y_label"].value_counts()
    print(f"    imbalance ratio: {vc.max()/vc.min():.2f}:1")

    print("\n  patients per class (a patient may count in several):")
    pc = out.groupby("y_label")["patient_key"].nunique()
    for c in keep:
        if c in pc:
            print(f"    {c:<8} {pc[c]:>8,}")

    v = out.groupby("patient_key").size()
    print(f"\n  images per patient: mean {v.mean():.1f}, "
          f"median {v.median():.0f}, range {v.min()}-{v.max()}")
    top = v.nlargest(3)
    print(f"  largest patients: "
          f"{', '.join(f'{k}={n}' for k, n in top.items())}")
    if v.max() > 200:
        print("  NOTE: a few patients contribute very many scans and will "
              "dominate\n        whichever split they land in. Consider "
              "capping per patient.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    log["n_images_final"] = len(out)
    log["n_patients_final"] = int(out["patient_key"].nunique())
    log["class_counts"] = {k: int(v) for k, v in
                           out["y_label"].value_counts().items()}
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n  Manifest -> {args.out}")
    print(f"  Metadata -> {meta_path}\n")


if __name__ == "__main__":
    main()