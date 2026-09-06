#!/usr/bin/env python3
"""
subsample_match.py -- build OCT subsets matched to each CFP cohort's
patient count, so the modality comparison is not confounded by sample size.

The headline comparison is OCT 0.840 vs fundus 0.607, but OCT has 66,454
images from 5,093 patients against fundus's 3,409 from 1,745. A reviewer
will reasonably ask whether the gap is modality or data volume. This
script answers it by matching patient counts and re-running.

  OCT pool: 1,745 patients drawn from 5,093
     |
     +-- 1,420 patients  <->  AMDNet23  (1,420 patients / 1,849 images)
     +--   879 patients  <->  ODIR      (  879 patients / 1,286 images)
     +--   325 patients  <->  HYAMD     (  325 patients / 1,560 images)

The three subsets are NESTED -- 325 is a subset of 879 is a subset of
1,420 -- so a difference between them is sample size alone rather than
sample size plus a different draw.

1,745 is the true CFP patient union: ODIR's 879 patients are already
counted inside AMDNet23's 1,420, so summing all three (2,624) would
double-count them.

Labels are collapsed to the shared binary space used by the cross-modal
probe:

    NORMAL  <- NORMAL
    AMD     <- DRUSEN + CNV
    (DME dropped -- no fundus counterpart in these cohorts)

Split assignments are INHERITED from pooled_split.csv, so the verified
patient-disjointness carries over and no new split is computed.

Nothing on disk is modified. The source manifest is read only, and each
subset is written as a new file.

Usage:
    python subsample_match.py
    python subsample_match.py --cap 5      # also cap images per patient
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from scripts.config import Config
except ImportError:
    from config import Config

# Collapse to the shared space. DME has no fundus counterpart here.
COLLAPSE = {"NORMAL": "NORMAL", "DRUSEN": "AMD", "CNV": "AMD"}
CLASSES = ["NORMAL", "AMD"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Patient counts of each fundus cohort, and the manifest each comes from.
TARGETS = [
    ("amdnet23", "amdnet23_clean.csv", None),
    ("odir",     "amdnet23_clean.csv", {"source": ["odir"]}),
    ("hyamd",    "hyamd_binary.csv",   None),
]


def read_manifest(path, cols=None):
    """Chunked read -- the full manifest has exhausted memory on this box."""
    if not os.path.isfile(path):
        sys.exit(f"manifest not found: {path}")
    return pd.concat(
        [c for c in pd.read_csv(path, chunksize=20000,
                                usecols=(lambda x: x in cols) if cols else None)],
        ignore_index=True)


def cfp_patient_counts():
    """Actual patient counts per fundus cohort, read from the manifests."""
    out = {}
    for name, man, filt in TARGETS:
        p = os.path.join(Config.MANIFEST_DIR, man)
        d = read_manifest(p, ["patient_key", "y_label", "source", "split"])
        if filt:
            for col, vals in filt.items():
                if col not in d.columns:
                    sys.exit(f"{man} has no column {col!r}")
                d = d[d[col].isin(vals)]
        out[name] = {
            "patients": int(d["patient_key"].nunique()),
            "images": int(len(d)),
            "manifest": man,
            "filter": filt,
        }
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=None,
                   help="OCT manifest (default: the oct track's)")
    p.add_argument("--out-dir", default=None,
                   help="where to write the subset manifests "
                        "(default: the project's manifests directory)")
    p.add_argument("--cap", type=int, default=0,
                   help="optional images-per-patient cap, sampled evenly "
                        "across each volume. An OCT volume is 20-30 B-scans "
                        "while a fundus camera gives one photograph, so "
                        "matching patients still leaves OCT with more "
                        "images. Use this for a sensitivity check that "
                        "matches both.")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    src = a.source or os.path.join(Config.MANIFEST_DIR,
                                   Config.TRACKS["oct"]["manifest"])
    a.out_dir = a.out_dir or Config.MANIFEST_DIR
    os.makedirs(a.out_dir, exist_ok=True)

    print(f"\n{'='*72}\nPATIENT-MATCHED OCT SUBSETS\n{'='*72}")
    print(f"  source (read only): {src}")

    # ------------------------------------------------ fundus patient counts
    print(f"\n{'='*72}\n1. FUNDUS COHORT SIZES\n{'='*72}")
    cfp = cfp_patient_counts()
    for k, v in cfp.items():
        print(f"  {k:<10} {v['patients']:>6,} patients  "
              f"{v['images']:>6,} images  "
              f"({v['images']/v['patients']:.1f} per patient)")

    union = cfp["hyamd"]["patients"] + cfp["amdnet23"]["patients"]
    print(f"\n  ODIR is a SUBSET of AMDNet23, so the distinct fundus patient")
    print(f"  union is HYAMD + AMDNet23 = {union:,}, not the sum of all three")
    print(f"  ({sum(v['patients'] for v in cfp.values()):,}, which "
          f"double-counts ODIR).")

    # -------------------------------------------------------- load OCT
    print(f"\n{'='*72}\n2. SOURCE OCT MANIFEST\n{'='*72}")
    oct_df = read_manifest(src)
    print(f"  {len(oct_df):,} images  "
          f"{oct_df['patient_key'].nunique():,} patients")

    n0 = len(oct_df)
    oct_df = oct_df[oct_df["y_label"].isin(COLLAPSE)].copy()
    oct_df["y_label"] = oct_df["y_label"].map(COLLAPSE)
    oct_df["Class"] = oct_df["y_label"]
    oct_df["Label"] = oct_df["y_label"]
    oct_df["y"] = oct_df["y_label"].map(CLASS_TO_IDX)
    print(f"  collapsed to {CLASSES}: {n0:,} -> {len(oct_df):,} images "
          f"(DME dropped)")

    # a patient can span classes; assign each to its dominant one so the
    # proportional draw is well defined and the patient stays whole
    dom = (oct_df.groupby(["patient_key", "y_label"]).size()
           .reset_index(name="n")
           .sort_values(["patient_key", "n"], ascending=[True, False])
           .drop_duplicates("patient_key")
           .set_index("patient_key")["y_label"])
    print(f"  patients: {len(dom):,}  "
          f"({dict(dom.value_counts())} by dominant class)")

    # ---------------------------------------------- nested patient draw
    print(f"\n{'='*72}\n3. NESTED DRAW\n{'='*72}")
    sizes = sorted(((v["patients"], k) for k, v in cfp.items()),
                   reverse=True)
    pool_n = min(union, len(dom))
    print(f"  pool: {pool_n:,} patients drawn from {len(dom):,}")
    if pool_n < union:
        print(f"  NOTE: OCT has fewer patients than the fundus union; "
              f"pool capped at {pool_n:,}")

    rng = np.random.default_rng(a.seed)

    def draw(candidates, n, proportions):
        """Draw n patients from candidates, matching class proportions."""
        chosen = []
        by_class = {c: np.array([p for p in candidates if dom[p] == c])
                    for c in CLASSES}
        for c in CLASSES:
            want = int(round(n * proportions.get(c, 0)))
            have = by_class[c]
            take = min(want, len(have))
            if take < want:
                print(f"    NOTE: wanted {want} {c} patients, only "
                      f"{len(have)} available")
            chosen.extend(rng.choice(have, take, replace=False))
        # top up if rounding left us short
        short = n - len(chosen)
        if short > 0:
            taken = set(chosen)
            rest = np.array([p for p in candidates if p not in taken])
            if len(rest):
                chosen.extend(rng.choice(rest, min(short, len(rest)),
                                         replace=False))
        return sorted(chosen)

    # pool draw uses OCT's own class balance
    own = dom.value_counts(normalize=True).to_dict()
    pool = draw(list(dom.index), pool_n, own)
    print(f"  pool drawn: {len(pool):,} patients")

    written = {}
    remaining = pool
    for n, name in sizes:                      # largest first, nested
        v = cfp[name]
        # match the fundus cohort's class proportions where they exist in
        # the shared space
        cman = read_manifest(os.path.join(Config.MANIFEST_DIR, v["manifest"]),
                             ["patient_key", "y_label", "source"])
        if v["filter"]:
            for col, vals in v["filter"].items():
                cman = cman[cman[col].isin(vals)]
        cpat = cman.drop_duplicates("patient_key")
        # fundus labels map onto the shared space where they can
        # Map the fundus cohort's labels into the shared space. HYAMD uses
        # CONTROL/AMD, the AMDNet23 family uses NORMAL/AMD/DIABETIC/CATARACT;
        # only the first two of each have a shared-space counterpart.
        share_map = {"NORMAL": "NORMAL", "CONTROL": "NORMAL", "AMD": "AMD"}
        cshare = cpat["y_label"].map(share_map).dropna()
        prop = cshare.value_counts(normalize=True).to_dict()
        if not prop or set(prop) != set(CLASSES):
            print(f"    NOTE: {name} maps to {sorted(prop)} in the shared "
                  f"space; using OCT's own balance instead")
            prop = own
        print(f"    target proportions: "
              f"{ {k: round(v, 3) for k, v in prop.items()} }")

        n_take = min(n, len(remaining))
        sel = draw(remaining, n_take, prop)
        sub = oct_df[oct_df["patient_key"].isin(sel)].copy()

        if a.cap:
            keep = []
            for pk, g in sub.groupby("patient_key"):
                if len(g) <= a.cap:
                    keep.extend(g.index)
                else:
                    g = g.sort_values(["eye_key", "B-scan"])
                    pick = np.unique(np.linspace(0, len(g) - 1,
                                                 a.cap).round().astype(int))
                    keep.extend(g.index[pick])
            sub = sub.loc[sorted(keep)]

        out = os.path.join(a.out_dir, f"oct_match_{name}.csv")
        sub.to_csv(out, index=False)

        print(f"\n  oct_match_{name}")
        print(f"    target        : {name} = {v['patients']:,} patients / "
              f"{v['images']:,} images")
        print(f"    OCT subset    : {sub['patient_key'].nunique():,} patients"
              f" / {len(sub):,} images "
              f"({len(sub)/max(sub['patient_key'].nunique(),1):.1f} per "
              f"patient)")
        cc = sub["y_label"].value_counts()
        print(f"    classes       : {dict(cc)}")
        sp = sub["split"].value_counts()
        print(f"    splits        : {dict(sp)}")

        # inherited disjointness must still hold
        bad = []
        for col in ("patient_key", "group_key", "Directory"):
            if col not in sub.columns:
                continue
            s = {k: set(g[col]) for k, g in sub.groupby("split")}
            for x, y in (("train", "val"), ("train", "test"), ("val", "test")):
                if len(s.get(x, set()) & s.get(y, set())):
                    bad.append(f"{col}:{x}n{y}")
        print(f"    disjointness  : "
              f"{'OK (inherited)' if not bad else '*** ' + ','.join(bad)}")

        written[name] = {
            "file": out,
            "oct_patients": int(sub["patient_key"].nunique()),
            "oct_images": int(len(sub)),
            "target_patients": v["patients"],
            "target_images": v["images"],
            "classes": {k: int(x) for k, x in cc.items()},
            "splits": {k: int(x) for k, x in sp.items()},
            "disjoint": not bad,
        }
        remaining = sel                       # nest the next one inside this

    # ------------------------------------------------------------ notes
    print(f"\n{'='*72}\nNOTES\n{'='*72}")
    print("  Patient counts are matched; image counts are not, and cannot")
    print("  be without discarding data: an OCT acquisition produces 20-30")
    print("  B-scans of one eye while a fundus camera produces a single")
    print("  photograph. That difference is intrinsic to the modalities.")
    if a.cap:
        print(f"  --cap {a.cap} was applied, so this run matches images too.")
    else:
        print("  Re-run with --cap to match images as a sensitivity check.")
    print("\n  The source manifest was not modified.")

    meta = {"generated": datetime.now().isoformat(timespec="seconds"),
            "source": src, "seed": a.seed, "cap": a.cap,
            "shared_classes": CLASSES, "collapse": COLLAPSE,
            "cfp_cohorts": cfp, "fundus_patient_union": union,
            "subsets": written,
            "nested": "each subset is contained in the next larger one"}
    mp = os.path.join(a.out_dir, "oct_match_meta.json")
    with open(mp, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  metadata -> {mp}\n")


if __name__ == "__main__":
    main()