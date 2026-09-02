#!/usr/bin/env python3
"""
Build a manifest for Kermany OCT2017 matching the NEH manifest schema,
so the same verification and split tooling applies to both datasets.

Filenames encode the patient: CNV-1016042-1.jpeg -> class CNV, patient 1016042.
"""

import argparse, os, re, sys
import pandas as pd

EXTS = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}
PAT = re.compile(r"^([A-Za-z]+)-(\d+)-(\d+)$")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True,
                   help=r"e.g. D:\datasets\kermany2018\OCT2017")
    p.add_argument("--out", default="manifests/kermany_official.csv")
    args = p.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"ERROR: root not found: {args.root}")

    rows, unparsed = [], []
    for split in ("train", "val", "test"):
        sd = os.path.join(args.root, split)
        if not os.path.isdir(sd):
            print(f"  WARNING: no '{split}' directory")
            continue
        for cls in sorted(os.listdir(sd)):
            cd = os.path.join(sd, cls)
            if not os.path.isdir(cd):
                continue
            for fn in sorted(os.listdir(cd)):
                stem, ext = os.path.splitext(fn)
                if ext.lower() not in EXTS:
                    continue
                m = PAT.match(stem)
                if not m:
                    unparsed.append(f"{split}/{cls}/{fn}")
                    continue
                fcls, pid, scan = m.groups()
                rows.append({
                    "Directory": f"{split}/{cls}/{fn}",
                    "Patient ID": pid,
                    "Class": cls,
                    "Label": cls,
                    "y_label": cls,
                    "Eye": "NA",           # not encoded in Kermany filenames
                    "B-scan": int(scan),
                    "patient_key": f"P{pid}",   # globally unique, unlike NEH
                    "eye_key": f"P{pid}",
                    "group_key": f"P{pid}",
                    "split": split,
                    "filename_class": fcls,
                })

    if not rows:
        sys.exit("ERROR: no images found. Check --root.")

    df = pd.DataFrame(rows)

    if unparsed:
        print(f"  WARNING: {len(unparsed)} filenames did not match the "
              f"CLASS-PATIENT-SCAN pattern:")
        for u in unparsed[:10]:
            print(f"      {u}")

    mism = df[df["Class"].str.upper() != df["filename_class"].str.upper()]
    if len(mism):
        print(f"  WARNING: {len(mism)} files whose folder class disagrees "
              f"with the class in the filename")
        print(mism[["Directory", "Class", "filename_class"]].head(10).to_string(index=False))

    classes = sorted(df["Class"].unique())
    df["y"] = df["Class"].map({c: i for i, c in enumerate(classes)})

    print(f"\n  images   : {len(df):,}")
    print(f"  patients : {df['patient_key'].nunique():,}")
    print(f"  classes  : {classes}")
    print(f"\n  images per split x class:")
    print("    " + pd.crosstab(df['split'], df['Class']).to_string().replace("\n", "\n    "))
    print(f"\n  patients per split x class:")
    pt = df.drop_duplicates(["patient_key", "split", "Class"])
    print("    " + pd.crosstab(pt['split'], pt['Class']).to_string().replace("\n", "\n    "))

    print(f"\n  PATIENT OVERLAP BETWEEN OFFICIAL SPLITS:")
    sets = {s: set(g["patient_key"]) for s, g in df.groupby("split")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a in sets and b in sets:
            n = len(sets[a] & sets[b])
            pct = 100 * n / len(sets[b]) if sets[b] else 0
            flag = "OK" if n == 0 else f"*** {pct:.1f}% of {b} patients seen in {a} ***"
            print(f"    {a:<6} n {b:<6}: {n:>5}   {flag}")

    n_multi = df.groupby("patient_key")["Class"].nunique()
    multi = n_multi[n_multi > 1]
    if len(multi):
        print(f"\n  WARNING: {len(multi)} patients appear under MORE THAN ONE class")
        for pk in list(multi.index)[:10]:
            print(f"      {pk}: {sorted(df[df.patient_key == pk]['Class'].unique())}")

    v = df.groupby("patient_key").size()
    print(f"\n  scans per patient: mean {v.mean():.1f}, median {v.median():.0f}, "
          f"range {v.min()}-{v.max()}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.drop(columns=["filename_class"]).to_csv(args.out, index=False)
    print(f"\n  Manifest -> {args.out}\n")


if __name__ == "__main__":
    main()