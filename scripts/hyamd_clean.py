#!/usr/bin/env python3
"""
hyamd_clean.py -- audit HYAMD and build a patient-disjoint manifest.

HYAMD (Hillel Yaffe Medical Center, 2021-2024) is the best-documented
cohort in this project: 1,560 fundus images from 325 patients, all
1960x1934 from a single Topcon DRI OCT Triton at 45 degrees, with labels
set by full clinical ophthalmic evaluation supported by OCT and OCT
angiography rather than by reading the photograph.

Labels:
    0  control: diabetic retinopathy WITHOUT AMD
    1  early AMD
    2  intermediate-to-late AMD

Three properties of the release this script handles explicitly.

  FILENAME MAPPING. image_id in labels.csv does not equal the filename.
  The first image of an eye drops its counter and later ones carry a
  trailing underscore:

      <id>_<side>        -> <id>_<side>.png
      <id>_<side>_1      -> <id>_<side>.png
      <id>_<side>_k      -> <id>_<side>_{k-1}_.png     for k >= 2

  Verified to cover all 1,560 rows with no collisions and no orphan files.

  SIDE TYPOS. Some filenames use D or E where the side is L or R -- e.g.
  patient 002456631 has 002456631_D*.png while every one of that patient's
  label rows says side = L, and the side column contains only L (813) and
  R (747). The letter in the filename is therefore treated as part of the
  path, and laterality is taken from the side column.

  LONGITUDINAL LABELS. Imaging spans 2021-2024 with repeat visits, and the
  authors relabelled an eye when its stage changed. 12 patients therefore
  appear under more than one AMD label. That is the design working, not an
  error -- but it means the split must group by patient and stratify on a
  dominant label, or the same person lands in two splits.

CLASS BALANCE WARNING. Early AMD (label 1) has 108 images from only 20
patients. After a patient-disjoint split that leaves roughly 3-4 patients
in validation and test, which is too few to select a model on. Use
--task binary for the headline result and treat --task staging as a
secondary analysis with that limitation stated.

Usage:
    python hyamd_clean.py --root "D:/datasets/hyamd" \
                          --out manifests/hyamd_binary.csv --task binary
    python hyamd_clean.py --root "D:/datasets/hyamd" \
                          --out manifests/hyamd_staging.csv --task staging
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

FNAME = re.compile(r"^(.+_[LRDE])(?:_(\d+))?$")

TASKS = {
    # task -> {raw AMD label: class name}, and the class order
    "binary": ({0: "CONTROL", 1: "AMD", 2: "AMD"},
               ["CONTROL", "AMD"]),
    "staging": ({0: "CONTROL", 1: "AMD_EARLY", 2: "AMD_LATE"},
                ["CONTROL", "AMD_EARLY", "AMD_LATE"]),
}


def image_id_to_file(image_id: str):
    """See module docstring. Returns a filename or None if unparseable."""
    m = FNAME.match(image_id)
    if not m:
        return None
    base, k = m.group(1), m.group(2)
    if k is None or int(k) == 1:
        return base + ".png"
    return f"{base}_{int(k) - 1}_.png"


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
    p.add_argument("--out", default="manifests/hyamd_binary.csv")
    p.add_argument("--task", choices=sorted(TASKS), default="binary")
    p.add_argument("--test-ratio", type=float, default=0.20)
    p.add_argument("--val-ratio", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-hash", action="store_true",
                   help="skip duplicate detection (faster)")
    p.add_argument("--no-dims", action="store_true")
    args = p.parse_args()

    lab_path = os.path.join(args.root, "labels", "labels.csv")
    img_dir = os.path.join(args.root, "Images")
    for pth in (lab_path, img_dir):
        if not os.path.exists(pth):
            sys.exit(f"ERROR: not found: {pth}")

    label_map, classes = TASKS[args.task]
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    log = {"root": args.root, "task": args.task, "classes": classes,
           "generated": datetime.now().isoformat(timespec="seconds")}

    # ------------------------------------------------------------ load
    print(f"\n{'='*72}\n1. LOAD AND JOIN\n{'='*72}")
    d = pd.read_csv(lab_path)
    files = set(os.listdir(img_dir))
    print(f"  label rows    : {len(d):,}")
    print(f"  files on disk : {len(files):,}")

    d["filename"] = [image_id_to_file(i) for i in d["image_id"]]
    unmapped = d["filename"].isna().sum()
    if unmapped:
        sys.exit(f"ERROR: {unmapped} image_ids do not match the naming "
                 f"pattern; the join rule needs revisiting")

    missing = [f for f in d["filename"] if f not in files]
    if missing:
        sys.exit(f"ERROR: {len(missing)} labelled images are absent from "
                 f"{img_dir}, e.g. {missing[:3]}")

    if d["filename"].duplicated().any():
        n = int(d["filename"].duplicated().sum())
        sys.exit(f"ERROR: {n} label rows map to the same file; the join "
                 f"rule is ambiguous")

    orphans = files - set(d["filename"])
    print(f"  joined        : {len(d):,} / {len(d):,}   "
          f"orphan files: {len(orphans)}      OK")

    # ---------------------------------------------------- side typos
    fname_side = d["filename"].str.extract(r"_([LRDE])[_.]")[0]
    odd = d[fname_side != d["side"]]
    if len(odd):
        print(f"\n  {len(odd)} filenames carry a side letter that disagrees "
              f"with the side column:")
        for _, r in odd.head(4).iterrows():
            print(f"    {r['filename']:<28} side column says {r['side']}")
        print("  The side column is authoritative (it contains only L and R);")
        print("  the filename letter is treated as part of the path.")
    log["n_side_typos"] = int(len(odd))

    # ---------------------------------------------------------- labels
    print(f"\n{'='*72}\n2. LABELS  (task: {args.task})\n{'='*72}")
    print("  raw AMD label   images   patients")
    for k in sorted(d["AMD"].unique()):
        sub = d[d["AMD"] == k]
        print(f"    {k}  {'':<12} {len(sub):>6,}   "
              f"{sub['patient_id'].nunique():>7}")

    d["y_label"] = d["AMD"].map(label_map)
    d["y"] = d["y_label"].map(cls_to_idx)
    if d["y"].isna().any():
        sys.exit("ERROR: unmapped AMD values")

    print(f"\n  mapped to {args.task}:")
    for c in classes:
        sub = d[d["y_label"] == c]
        print(f"    {c:<12} {len(sub):>6,} images   "
              f"{sub['patient_id'].nunique():>4} patients")

    vc = d["y_label"].value_counts()
    print(f"    imbalance {vc.max()/vc.min():.2f}:1")

    pc = d.groupby("y_label")["patient_id"].nunique()
    thin = [c for c in classes if pc.get(c, 0) < 40]
    if thin:
        print(f"\n  WARNING: {thin} have fewer than 40 patients. After a")
        print(f"  patient-disjoint split that is a handful of patients per")
        print(f"  fold -- metrics for those classes will be unstable and")
        print(f"  should be reported with that caveat, not used for model")
        print(f"  selection.")
    log["class_patients"] = {k: int(v) for k, v in pc.items()}

    # --------------------------------------------------- longitudinal
    multi = d.groupby("patient_id")["y_label"].nunique()
    n_multi = int((multi > 1).sum())
    print(f"\n  patients under more than one label: {n_multi}")
    if n_multi:
        print("  HYAMD is longitudinal (2021-2024) and eyes were relabelled")
        print("  when their stage changed, so this is expected. Those")
        print("  patients are grouped and stratified on their dominant")
        print("  label so they cannot straddle a split.")
    log["n_multilabel_patients"] = n_multi

    # ------------------------------------------------------ duplicates
    if not args.no_hash:
        print(f"\n{'='*72}\n3. DUPLICATE DETECTION\n{'='*72}")
        print(f"  hashing {len(d):,} images...")
        hs = []
        for i, fn in enumerate(d["filename"], 1):
            try:
                hs.append(file_hash(os.path.join(img_dir, fn)))
            except Exception as e:
                print(f"    unreadable: {fn} ({e})")
                hs.append(None)
            if i % 300 == 0:
                print(f"    {i:,} / {len(d):,}")
        d["md5"] = hs
        d = d[d["md5"].notna()].reset_index(drop=True)

        groups = {h: g for h, g in d.groupby("md5") if len(g) > 1}
        print(f"\n  duplicate groups : {len(groups)}")
        if groups:
            cross_label = [g for g in groups.values()
                           if g["y_label"].nunique() > 1]
            cross_pat = [g for g in groups.values()
                         if g["patient_id"].nunique() > 1]
            print(f"    spanning labels   : {len(cross_label)}")
            print(f"    spanning patients : {len(cross_pat)}")
            drop = set()
            for g in groups.values():
                if g["y_label"].nunique() > 1:
                    drop.update(g["filename"])          # unresolvable label
                else:
                    drop.update(sorted(g["filename"])[1:])
            for g in list(groups.values())[:3]:
                print(f"    {list(g['filename'])}")
            d = d[~d["filename"].isin(drop)].reset_index(drop=True)
            print(f"\n  removed {len(drop)} -> {len(d):,} images")
            log["n_duplicates_removed"] = len(drop)
        else:
            print("  none                                        OK")
            log["n_duplicates_removed"] = 0
    else:
        d["md5"] = ""

    # ------------------------------------------------------ dimensions
    if not args.no_dims:
        print(f"\n{'='*72}\n4. DIMENSIONS\n{'='*72}")
        w, h = [], []
        for fn in d["filename"]:
            try:
                with Image.open(os.path.join(img_dir, fn)) as im:
                    a, b = im.size
            except Exception:
                a = b = 0
            w.append(a)
            h.append(b)
        d["width"], d["height"] = w, h
        dims = Counter(zip(w, h))
        print(f"  distinct dimensions: {len(dims)}")
        for (a, b), n in dims.most_common(6):
            print(f"    {a:>5} x {b:<5} {n:>6,}   AR {a/b:.3f}")
        if len(dims) == 1:
            print("  Uniform acquisition -- a single scanner and export "
                  "pipeline.\n  Compare AMDNet23, which has 751 distinct "
                  "sizes across 1,849 images.")
        log["n_distinct_dimensions"] = len(dims)
    else:
        d["width"] = d["height"] = 0

    # ----------------------------------------------------------- keys
    d["patient_key"] = "HY_" + d["patient_id"].astype(str)
    d["eye_key"] = d["patient_key"] + "_" + d["side"]
    d["group_key"] = d["patient_key"]

    print(f"\n{'='*72}\n5. PATIENT-DISJOINT SPLIT\n{'='*72}")
    print(f"  patients : {d['patient_key'].nunique():,}")
    print(f"  eyes     : {d['eye_key'].nunique():,}")
    v = d.groupby("patient_key").size()
    print(f"  images per patient: mean {v.mean():.1f}, "
          f"median {v.median():.0f}, max {v.max()}")
    both = d.groupby("patient_key")["side"].nunique()
    print(f"  contributing both eyes: {int((both > 1).sum())} of {len(both)}")

    # dominant label per patient, so a relabelled patient stays whole
    dom = (d.groupby(["group_key", "y_label"]).size()
             .reset_index(name="n")
             .sort_values(["group_key", "n"], ascending=[True, False])
             .drop_duplicates("group_key")
             .set_index("group_key")["y_label"])
    d["_stratum"] = d["group_key"].map(dom)

    d["split"] = "train"
    for cls in sorted(d["_stratum"].unique()):
        sub = d[d["_stratum"] == cls]
        n_groups = sub["group_key"].nunique()
        if n_groups < 5:
            print(f"  WARNING: stratum {cls} has {n_groups} patients; "
                  f"left entirely in train")
            continue
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_ratio,
                                random_state=args.seed)
        dev_i, test_i = next(gss.split(sub, groups=sub["group_key"].values))
        dev = sub.iloc[dev_i]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=args.val_ratio,
                                 random_state=args.seed + 1)
        _, val_i = next(gss2.split(dev, groups=dev["group_key"].values))
        d.loc[sub.index[test_i], "split"] = "test"
        d.loc[dev.index[val_i], "split"] = "val"
    d = d.drop(columns=["_stratum"])

    print(f"\n  {'split':<8}{'images':>9}{'patients':>10}{'eyes':>8}{'%':>8}")
    print(f"  {'-'*43}")
    for s in ("train", "val", "test"):
        g = d[d["split"] == s]
        print(f"  {s:<8}{len(g):>9,}{g['patient_key'].nunique():>10}"
              f"{g['eye_key'].nunique():>8}{100*len(g)/len(d):>7.1f}%")

    print("\n  intersections (must all be 0):")
    ok = True
    checks = [("patient_key", "patients"), ("eye_key", "eyes"),
              ("filename", "files")]
    if not args.no_hash:
        checks.append(("md5", "content"))
    for col, name in checks:
        sets = {s: set(g[col]) for s, g in d.groupby("split")}
        bad = [f"{a}n{b}={n}" for a, b in
               (("train", "val"), ("train", "test"), ("val", "test"))
               if (n := len(sets.get(a, set()) & sets.get(b, set())))]
        if bad:
            ok = False
            print(f"    {name:<9} *** {', '.join(bad)}")
        else:
            print(f"    {name:<9} disjoint across all splits     OK")

    print("\n  images per split x class:")
    print("    " + pd.crosstab(d["split"], d["y_label"])
          .reindex(index=["train", "val", "test"],
                   columns=classes, fill_value=0)
          .to_string().replace("\n", "\n    "))

    print("\n  patients per split x class:")
    pt = d.drop_duplicates(["patient_key", "y_label"])
    print("    " + pd.crosstab(pt["split"], pt["y_label"])
          .reindex(index=["train", "val", "test"],
                   columns=classes, fill_value=0)
          .to_string().replace("\n", "\n    "))

    # demographics -- HYAMD is the only cohort here that has them
    print("\n  demographics (enables a fairness check no other cohort "
          "supports):")
    print(f"    sex : {dict(d['sex'].value_counts())}")
    print(f"    age : {d['age'].min()}-{d['age'].max()}, "
          f"mean {d['age'].mean():.1f}")
    print("    " + pd.crosstab(d["y_label"], d["sex"])
          .to_string().replace("\n", "\n    "))

    print(f"\n  => {'PATIENT-DISJOINT' if ok else 'LEAKAGE PRESENT'}")

    # ---------------------------------------------------------- write
    d["Directory"] = "Images/" + d["filename"]
    d["cohort"] = "hyamd"
    d["Class"] = d["y_label"]
    d["Label"] = d["y_label"]
    d["Eye"] = d["side"]
    d["B-scan"] = 0
    d["root_hint"] = args.root

    cols = ["Directory", "cohort", "patient_key", "group_key", "eye_key",
            "Class", "Label", "y_label", "y", "Eye", "B-scan",
            "width", "height", "sex", "age", "AMD", "root_hint", "split"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    d[cols].to_csv(args.out, index=False)

    log.update({
        "n_images": int(len(d)),
        "n_patients": int(d["patient_key"].nunique()),
        "n_eyes": int(d["eye_key"].nunique()),
        "patient_disjoint": bool(ok),
        "class_counts": {k: int(v) for k, v in
                         d["y_label"].value_counts().items()},
        "split_counts": {s: int((d["split"] == s).sum())
                         for s in ("train", "val", "test")},
        "seed": args.seed,
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