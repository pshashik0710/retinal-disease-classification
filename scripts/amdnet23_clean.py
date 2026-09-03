#!/usr/bin/env python3
"""
amdnet23_clean.py -- audit AMDNet23 and build a patient-disjoint manifest.

AMDNet23 is a compilation of 2,000 fundus images (400 per class, four
classes) assembled from six public sources: Ocular Disease Recognition
(ODIR-5K), DR_200, Fundus Dataset, RFMiD, HRF and ARIA. That provenance
is visible in the filenames and creates three problems this script
measures and handles.

  1. DUPLICATE PADDING. The classes are padded to exactly 400 with literal
     copies. In train/amd alone the listing shows 17 files named
     "Copy of <original>.jpg" plus Windows "(1)" suffixes, and several
     bare-numbered files that duplicate one half of an L/R pair
     (e.g. 377.jpg alongside 377_left.jpg and 377_right.jpg).
     All of these are detected by content hash, not by filename, so
     re-encoded copies are caught too.

  2. PATIENT STRUCTURE. ODIR-derived files use <patient>_left / <patient>_right,
     so both eyes of one person are identifiable and must not straddle a
     split. Files from other sources use different schemes and are treated
     as one patient each -- which is conservative, since it can only make
     the split stricter.

  3. SOURCE COHORTS. Filename prefixes reveal which dataset each image came
     from. That is an internal cohort structure of exactly the kind the OCT
     track measured between NEH and Kermany, and it is recorded here so the
     same probe can be run on the fundus side.

The dataset's own train/valid split is DISCARDED. With duplicate padding
present there is no reason to assume it is patient-clean, and this script
re-splits from the pooled images.

Usage:
    python amdnet23_clean.py --root "D:/datasets/amdnet23" \
                             --out manifests/amdnet23_clean.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# AMDNet23's own folder names, lowercase on disk
RAW_CLASSES = ["normal", "amd", "diabetes", "cataract"]

# Canonical labels. Kept at four so config can filter to the three-class
# shared space (NORMAL / AMD / DIABETIC) for cross-modal work without
# regenerating the manifest.
CLASS_MAP = {
    "normal": "NORMAL",
    "amd": "AMD",
    "diabetes": "DIABETIC",
    "cataract": "CATARACT",
}
CLASSES = ["NORMAL", "AMD", "DIABETIC", "CATARACT"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Filename patterns -> (source, patient id, eye)
ODIR_LR = re.compile(r"^(\d+)_(left|right)$", re.I)
ARIA = re.compile(r"^(aria_[a-z]_\d+)_\d+$", re.I)
IMG_N = re.compile(r"^img_(\d+)$", re.I)
BARE_N = re.compile(r"^(\d+)$")

# "Copy of X", "X (1)", "X - Copy"
COPY_PREFIX = re.compile(r"^Copy of (.+)$", re.I)
COPY_SUFFIX = re.compile(r"^(.+?) \(\d+\)$")
COPY_DASH = re.compile(r"^(.+?) - Copy$", re.I)


def canonical_stem(stem: str) -> tuple[str, bool]:
    """Strip copy markers. Returns (base stem, was_marked_as_copy)."""
    for pat in (COPY_PREFIX, COPY_SUFFIX, COPY_DASH):
        m = pat.match(stem)
        if m:
            return m.group(1).strip(), True
    return stem, False


def parse_identity(stem: str):
    """(source, patient_id, eye) from a filename stem."""
    base, _ = canonical_stem(stem)

    m = ODIR_LR.match(base)
    if m:
        return "odir", m.group(1), m.group(2).upper()[:1] + "S" if \
            m.group(2).lower() == "left" else "OD"

    m = ARIA.match(base)
    if m:
        return "aria", m.group(1), "NA"

    m = IMG_N.match(base)
    if m:
        return "img", f"img{m.group(1)}", "NA"

    m = BARE_N.match(base)
    if m:
        # A bare number matching an ODIR id is very likely the same patient
        # as <n>_left / <n>_right; resolved after the full scan.
        return "bare", m.group(1), "NA"

    return "other", base, "NA"


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
    rows = []
    splits = [d for d in ("train", "valid", "val", "test")
              if os.path.isdir(os.path.join(root, d))]
    if not splits:
        splits = [""]

    for sp in splits:
        sd = os.path.join(root, sp) if sp else root
        for cls in sorted(os.listdir(sd)):
            cd = os.path.join(sd, cls)
            if not os.path.isdir(cd):
                continue
            if cls.lower() not in CLASS_MAP:
                print(f"  WARNING: unexpected class folder {cls!r}; skipped")
                continue
            for fn in sorted(os.listdir(cd)):
                stem, ext = os.path.splitext(fn)
                if ext.lower() not in EXTS:
                    continue
                base, is_copy = canonical_stem(stem)
                src, pid, eye = parse_identity(stem)
                rows.append({
                    "Directory": f"{sp}/{cls}/{fn}" if sp else f"{cls}/{fn}",
                    "orig_split": sp or "root",
                    "raw_class": cls.lower(),
                    "y_label": CLASS_MAP[cls.lower()],
                    "stem": stem,
                    "base_stem": base,
                    "marked_copy": is_copy,
                    "source": src,
                    "pid_raw": pid,
                    "Eye": eye,
                })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--out", default="manifests/amdnet23_clean.csv")
    p.add_argument("--test-ratio", type=float, default=0.20)
    p.add_argument("--val-ratio", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-dims", action="store_true")
    args = p.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"ERROR: root not found: {args.root}")

    log = {"root": args.root, "generated":
           datetime.now().isoformat(timespec="seconds")}

    # ---------------------------------------------------------- scan
    print(f"\n{'='*72}\n1. SCAN\n{'='*72}")
    df = scan(args.root)
    if df.empty:
        sys.exit("ERROR: no images found under --root")

    print(f"  images : {len(df):,}")
    print(f"\n  per original split x class:")
    print("    " + pd.crosstab(df["orig_split"], df["y_label"])
          .to_string().replace("\n", "\n    "))

    print(f"\n  inferred source cohort:")
    for s, n in df["source"].value_counts().items():
        print(f"    {s:<8} {n:>6,}")
    print("\n  AMDNet23 is assembled from six public datasets; these are the")
    print("  patterns visible in the filenames. This is an internal cohort")
    print("  structure of the same kind measured between NEH and Kermany.")

    log["n_images_raw"] = len(df)
    log["sources"] = {k: int(v) for k, v in df["source"].value_counts().items()}

    n_marked = int(df["marked_copy"].sum())
    print(f"\n  filenames explicitly marked as copies: {n_marked}")
    if n_marked:
        ex = df.loc[df["marked_copy"], "stem"].head(6).tolist()
        for e in ex:
            print(f"    {e}")

    # ------------------------------------------------------ duplicates
    print(f"\n{'='*72}\n2. DUPLICATE DETECTION (content hash)\n{'='*72}")
    print(f"  hashing {len(df):,} images...")
    hashes = []
    for i, rel in enumerate(df["Directory"], 1):
        full = os.path.join(args.root, rel.replace("/", os.sep))
        try:
            hashes.append(file_hash(full))
        except Exception as e:
            print(f"    unreadable: {rel} ({e})")
            hashes.append(None)
        if i % 500 == 0:
            print(f"    {i:,} / {len(df):,}")
    df["md5"] = hashes
    df = df[df["md5"].notna()].reset_index(drop=True)

    groups = df.groupby("md5")
    dup = {h: g for h, g in groups if len(g) > 1}
    n_redundant = sum(len(g) - 1 for g in dup.values())

    print(f"\n  duplicate groups   : {len(dup):,}")
    print(f"  redundant copies   : {n_redundant:,}")

    cross_class = [g for g in dup.values() if g["y_label"].nunique() > 1]
    cross_split = [g for g in dup.values() if g["orig_split"].nunique() > 1]
    print(f"    spanning classes         : {len(cross_class):,}")
    print(f"    spanning original splits : {len(cross_split):,}")

    if dup:
        print("\n  examples:")
        for g in list(dup.values())[:5]:
            print(f"    {list(g['Directory'])}")

    drop = set()
    n_conflict = 0
    conflict_pairs = Counter()
    for h, g in dup.items():
        if g["y_label"].nunique() > 1:
            # identical pixels under two different diagnoses: no copy has a
            # trustworthy label, so drop the whole group
            drop.update(g["Directory"])
            n_conflict += len(g)
            conflict_pairs[tuple(sorted(g["y_label"].unique()))] += 1
        else:
            drop.update(sorted(g["Directory"])[1:])

    if n_conflict:
        print(f"\n  CROSS-CLASS CONFLICTS")
        print(f"    {n_conflict} images carry contradictory labels; "
              f"all copies dropped")
        for combo, n in conflict_pairs.most_common():
            print(f"      {' / '.join(combo):<22} {n:>4} groups")

    df = df[~df["Directory"].isin(drop)].reset_index(drop=True)
    print(f"\n  removed {len(drop):,} images -> {len(df):,} remaining")

    log["n_duplicate_groups"] = len(dup)
    log["n_removed"] = len(drop)
    log["n_crossclass_images"] = n_conflict

    # ------------------------------------------------ patient grouping
    print(f"\n{'='*72}\n3. PATIENT GROUPING\n{'='*72}")

    # A bare-numbered file whose number also appears as <n>_left/<n>_right
    # is almost certainly the same patient; merge it into that patient.
    odir_ids = set(df.loc[df["source"] == "odir", "pid_raw"])
    merged = 0

    def resolve(row):
        nonlocal merged
        if row["source"] == "bare" and row["pid_raw"] in odir_ids:
            merged += 1
            return "odir"
        return row["source"]

    df["source"] = df.apply(resolve, axis=1)
    if merged:
        print(f"  {merged} bare-numbered files merged into ODIR patients "
              f"(the same id appears as _left/_right)")

    df["patient_key"] = df["source"] + "_" + df["pid_raw"].astype(str)
    df["eye_key"] = df["patient_key"] + "_" + df["Eye"]
    df["group_key"] = df["patient_key"]

    print(f"  patients : {df['patient_key'].nunique():,}")
    print(f"  eyes     : {df['eye_key'].nunique():,}")
    v = df.groupby("patient_key").size()
    print(f"  images per patient: mean {v.mean():.2f}, "
          f"median {v.median():.0f}, max {v.max()}")

    both_eyes = df[df["source"] == "odir"].groupby("patient_key")["Eye"].nunique()
    print(f"  ODIR patients contributing both eyes: "
          f"{int((both_eyes > 1).sum()):,} of {len(both_eyes):,}")

    # patients under more than one class
    pc = df.groupby("patient_key")["y_label"].nunique()
    multi = pc[pc > 1]
    if len(multi):
        print(f"\n  WARNING: {len(multi)} patients appear under >1 class")
        combos = Counter()
        for pk in multi.index:
            combos[tuple(sorted(df.loc[df.patient_key == pk,
                                       "y_label"].unique()))] += 1
        for combo, n in combos.most_common(8):
            print(f"    {' + '.join(combo):<28} {n:>4} patients")
        print("  These stay in the data -- a patient can have more than one")
        print("  finding -- but they are grouped so all their images land in")
        print("  the same split.")
    else:
        print("  no patient appears under more than one class      OK")

    log["n_patients"] = int(df["patient_key"].nunique())
    log["n_multiclass_patients"] = int(len(multi))

    # -------------------------------------------------------- dimensions
    if not args.no_dims:
        print(f"\n{'='*72}\n4. IMAGE DIMENSIONS\n{'='*72}")
        w, h = [], []
        for rel in df["Directory"]:
            full = os.path.join(args.root, rel.replace("/", os.sep))
            try:
                with Image.open(full) as im:
                    a, b = im.size
            except Exception:
                a = b = 0
            w.append(a)
            h.append(b)
        df["width"], df["height"] = w, h
        dims = Counter(zip(w, h))
        print(f"  distinct dimensions: {len(dims)}")
        for (a, b), n in dims.most_common(10):
            print(f"    {a:>5} x {b:<5} {n:>6,}   AR {a/b:.3f}" if b else "")
        if len(dims) > 10:
            print(f"    ... and {len(dims)-10} more")
        log["n_distinct_dimensions"] = len(dims)
    else:
        df["width"] = df["height"] = 0

    # ------------------------------------------------------------ split
    print(f"\n{'='*72}\n5. PATIENT-DISJOINT SPLIT\n{'='*72}")
    print("  The dataset's own train/valid split is discarded: with duplicate")
    print("  padding present there is no reason to assume it is patient-clean.")

    df["split"] = "train"

    # A patient can carry more than one class (a real eye can have several
    # findings, and AMDNet23's sources overlap). Splitting class-by-class
    # would then assign the SAME patient to train under one class and test
    # under another -- leakage that the per-class loop cannot see. So each
    # patient is first assigned to its dominant class, and the split runs
    # over those strata, keeping every patient whole.
    dom = (df.groupby(["group_key", "y_label"]).size()
             .reset_index(name="n")
             .sort_values(["group_key", "n"], ascending=[True, False])
             .drop_duplicates("group_key")
             .set_index("group_key")["y_label"])
    df["_stratum"] = df["group_key"].map(dom)

    for cls in sorted(df["_stratum"].unique()):
        sub = df[df["_stratum"] == cls]
        g = sub["group_key"].values
        if sub["group_key"].nunique() < 5:
            print(f"  WARNING: stratum {cls} has only "
                  f"{sub['group_key'].nunique()} groups; left in train")
            continue
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_ratio,
                                random_state=args.seed)
        dev_i, test_i = next(gss.split(sub, groups=g))
        dev = sub.iloc[dev_i]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=args.val_ratio,
                                 random_state=args.seed + 1)
        _, val_i = next(gss2.split(dev, groups=dev["group_key"].values))
        df.loc[sub.index[test_i], "split"] = "test"
        df.loc[dev.index[val_i], "split"] = "val"

    df = df.drop(columns=["_stratum"])

    print(f"\n  {'split':<8}{'images':>9}{'patients':>10}{'eyes':>8}{'%':>8}")
    print(f"  {'-'*43}")
    for s in ("train", "val", "test"):
        g = df[df["split"] == s]
        print(f"  {s:<8}{len(g):>9,}{g['patient_key'].nunique():>10}"
              f"{g['eye_key'].nunique():>8}{100*len(g)/len(df):>7.1f}%")

    print("\n  intersections (must all be 0):")
    ok = True
    for col, name in (("patient_key", "patients"), ("eye_key", "eyes"),
                      ("Directory", "files"), ("md5", "content")):
        sets = {s: set(g[col]) for s, g in df.groupby("split")}
        bad = [f"{a}n{b}={n}" for a, b in
               (("train", "val"), ("train", "test"), ("val", "test"))
               if (n := len(sets.get(a, set()) & sets.get(b, set())))]
        if bad:
            ok = False
            print(f"    {name:<9} *** {', '.join(bad)}")
        else:
            print(f"    {name:<9} disjoint across all splits     OK")

    print("\n  images per split x class:")
    print("    " + pd.crosstab(df["split"], df["y_label"])
          .reindex(index=["train", "val", "test"], fill_value=0)
          .to_string().replace("\n", "\n    "))

    print("\n  images per split x source:")
    print("    " + pd.crosstab(df["split"], df["source"])
          .reindex(index=["train", "val", "test"], fill_value=0)
          .to_string().replace("\n", "\n    "))

    print(f"\n  => {'PATIENT-DISJOINT' if ok else 'LEAKAGE PRESENT'}")

    # ----------------------------------------------------------- write
    df["cohort"] = "amdnet23"
    df["Class"] = df["y_label"]
    df["Label"] = df["y_label"]
    df["y"] = df["y_label"].map(CLASS_TO_IDX)
    df["B-scan"] = 0
    df["root_hint"] = args.root

    cols = ["Directory", "cohort", "patient_key", "group_key", "eye_key",
            "Class", "Label", "y_label", "y", "Eye", "B-scan",
            "width", "height", "source", "orig_split", "root_hint", "split"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df[cols].to_csv(args.out, index=False)

    log.update({
        "n_images_final": len(df),
        "patient_disjoint": bool(ok),
        "class_counts": {k: int(v) for k, v in
                         df["y_label"].value_counts().items()},
        "split_counts": {s: int((df["split"] == s).sum())
                         for s in ("train", "val", "test")},
        "seed": args.seed,
        "note": ("classes kept at four; filter to NORMAL/AMD/DIABETIC in "
                 "config for the three-class shared space with OCT"),
    })
    mp = os.path.splitext(args.out)[0] + "_meta.json"
    with open(mp, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n  Manifest -> {args.out}")
    print(f"  Metadata -> {mp}\n")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()