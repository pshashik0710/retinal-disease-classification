#!/usr/bin/env python3
"""
verify_data_integrity.py -- read-only audit of the NEH dataset against its manifest.

Checks, in order:
  1. every path in the manifest exists on disk
  2. every image on disk is referenced by the manifest (no orphans)
  3. every referenced image opens, and reports its size / mode
  4. byte-level duplicates (same content, different path)
  5. patient/eye/file disjointness across splits
  6. class distribution per split, at image AND patient level

Writes nothing except an optional report. Exits non-zero if any check fails,
so it can gate a training run.

Usage:
    python verify_data_integrity.py \
        --root "D:/datasets/neh/NEH_UT_2021RetinalOCTDataset" \
        --manifest manifests/neh_split.csv \
        --report reports/integrity_neh.txt
"""

import argparse
import hashlib
import os
import sys
from collections import Counter, defaultdict

import pandas as pd
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class Report:
    """Collects output so it can be printed and optionally written to disk."""

    def __init__(self):
        self.lines = []
        self.failures = []

    def __call__(self, s=""):
        print(s)
        self.lines.append(s)

    def head(self, title):
        self("")
        self("=" * 70)
        self(title)
        self("=" * 70)

    def fail(self, s):
        self(f"  *** FAIL: {s}")
        self.failures.append(s)

    def save(self, path):
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")
        print(f"\n  Report -> {path}")


def scan_disk(root):
    """Every image file under root, as paths relative to root, forward-slashed."""
    found = set()
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                rel = os.path.relpath(os.path.join(dirpath, fn), root)
                found.add(rel.replace("\\", "/"))
    return found


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True,
                   help="dataset root containing CNV/ DRUSEN/ NORMAL/")
    p.add_argument("--manifest", default="manifests/neh_split.csv")
    p.add_argument("--report", default="reports/integrity_neh.txt")
    p.add_argument("--hash-sample", type=int, default=2000,
                   help="how many images to hash for duplicate detection "
                        "(0 = all; hashing all takes a few minutes)")
    p.add_argument("--no-open", action="store_true",
                   help="skip opening images (faster, less thorough)")
    args = p.parse_args()

    r = Report()

    # ---------------------------------------------------------------- inputs
    r.head("INPUTS")
    if not os.path.isdir(args.root):
        print(f"ERROR: root not found: {args.root}")
        sys.exit(2)
    if not os.path.isfile(args.manifest):
        print(f"ERROR: manifest not found: {args.manifest}")
        sys.exit(2)

    df = pd.read_csv(args.manifest)
    r(f"  root     : {args.root}")
    r(f"  manifest : {args.manifest}  ({len(df):,} rows)")

    if df.empty:
        r.fail("manifest is empty")
        r.save(args.report)
        sys.exit(1)

    # -------------------------------------------------- manifest vs disk
    r.head("1. MANIFEST PATHS vs DISK")
    wanted = set(df["Directory"].str.replace("\\", "/", regex=False))
    on_disk = scan_disk(args.root)

    r(f"  referenced by manifest : {len(wanted):,}")
    r(f"  image files on disk    : {len(on_disk):,}")

    missing = sorted(wanted - on_disk)
    orphans = sorted(on_disk - wanted)

    if missing:
        r.fail(f"{len(missing)} manifest paths do NOT exist on disk")
        for m in missing[:20]:
            r(f"      {m}")
        if len(missing) > 20:
            r(f"      ... and {len(missing) - 20} more")
    else:
        r("  all manifest paths exist                      OK")

    if orphans:
        r(f"  NOTE: {len(orphans)} images on disk are not in the manifest")
        for o in orphans[:20]:
            r(f"      {o}")
        if len(orphans) > 20:
            r(f"      ... and {len(orphans) - 20} more")
        by_class = Counter(o.split("/")[0] for o in orphans)
        r(f"      by top-level folder: {dict(by_class)}")
    else:
        r("  no orphan files on disk                       OK")

    present = sorted(wanted & on_disk)

    # ------------------------------------------------------ readability
    r.head("2. IMAGE READABILITY")
    if args.no_open:
        r("  skipped (--no-open)")
        sizes, modes, unreadable = Counter(), Counter(), []
    else:
        sizes, modes, unreadable = Counter(), Counter(), []
        step = max(1, len(present) // 3000) if len(present) > 3000 else 1
        sample = present[::step]
        r(f"  opening {len(sample):,} of {len(present):,} images "
          f"(every {step}{'st' if step == 1 else 'th'})")
        for rel in sample:
            try:
                with Image.open(os.path.join(args.root, rel)) as im:
                    im.verify()
                with Image.open(os.path.join(args.root, rel)) as im:
                    sizes[im.size] += 1
                    modes[im.mode] += 1
            except Exception as e:
                unreadable.append((rel, str(e)))

        if unreadable:
            r.fail(f"{len(unreadable)} images failed to open")
            for rel, e in unreadable[:10]:
                r(f"      {rel}: {e}")
        else:
            r("  all sampled images open cleanly               OK")

        r(f"\n  colour modes: {dict(modes)}")
        r(f"  distinct dimensions: {len(sizes)}")
        for dim, n in sizes.most_common(10):
            r(f"      {dim[0]:>5} x {dim[1]:<5}  {n:>6}")
        if len(sizes) > 10:
            r(f"      ... and {len(sizes) - 10} more")
        if len(sizes) > 1:
            ars = {round(w / h, 3) for w, h in sizes}
            r(f"  aspect ratios present: {sorted(ars)[:8]}"
              f"{' ...' if len(ars) > 8 else ''}")
            r("  NOTE: non-square resize will distort these. Resize the short "
              "side and crop, or pad, rather than forcing a square.")

    # --------------------------------------------------------- duplicates
    r.head("3. BYTE-LEVEL DUPLICATES")
    n_hash = len(present) if args.hash_sample == 0 else min(args.hash_sample,
                                                            len(present))
    step = max(1, len(present) // n_hash) if n_hash else 1
    sample = present[::step]
    r(f"  hashing {len(sample):,} of {len(present):,} images")

    hashes = defaultdict(list)
    for rel in sample:
        try:
            with open(os.path.join(args.root, rel), "rb") as f:
                hashes[hashlib.md5(f.read()).hexdigest()].append(rel)
        except Exception:
            pass

    dups = {h: v for h, v in hashes.items() if len(v) > 1}
    if dups:
        r(f"  found {len(dups)} duplicate content groups:")
        for v in list(dups.values())[:10]:
            r(f"      {v}")
        cross = [v for v in dups.values()
                 if len({x.split('/')[0] for x in v}) > 1]
        if cross:
            r.fail(f"{len(cross)} duplicate groups span different CLASSES")
            for v in cross[:5]:
                r(f"      {v}")
    else:
        r("  no byte-identical duplicates in sample        OK")

    # ------------------------------------------------------ split hygiene
    if "split" in df.columns:
        r.head("4. SPLIT DISJOINTNESS")
        keys = [("patient_key", "patients"), ("eye_key", "eyes"),
                ("Directory", "files")]
        for col, label in keys:
            if col not in df.columns:
                continue
            sets = {s: set(g[col]) for s, g in df.groupby("split")}
            bad = []
            for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
                if a in sets and b in sets:
                    n = len(sets[a] & sets[b])
                    if n:
                        bad.append(f"{a}n{b}={n}")
            if bad:
                r.fail(f"{label}: overlap between splits -> {', '.join(bad)}")
            else:
                r(f"  {label:<10} disjoint across all splits         OK")

        r.head("5. CLASS DISTRIBUTION")
        r("\n  B-scans:")
        ct = pd.crosstab(df["split"], df.get("y_label", df["Label"]))
        r("    " + ct.to_string().replace("\n", "\n    "))

        if "patient_key" in df.columns:
            r("\n  Patients:")
            pt = df.drop_duplicates("patient_key")
            ct2 = pd.crosstab(pt["split"], pt["Class"])
            r("    " + ct2.to_string().replace("\n", "\n    "))

            r("\n  B-scans per volume:")
            if "eye_key" in df.columns:
                v = df.groupby("eye_key").size()
                r(f"    mean {v.mean():.1f}, median {v.median():.0f}, "
                  f"range {v.min()}-{v.max()}")
                r("    NOTE: adjacent B-scans from one eye are near-identical. "
                  "An image-level\n          split would place siblings in both "
                  "train and test.")

    # -------------------------------------------------------------- verdict
    r.head("VERDICT")
    if r.failures:
        r(f"  {len(r.failures)} CHECK(S) FAILED:")
        for f in r.failures:
            r(f"    - {f}")
        r("\n  Do not train until these are resolved.")
    else:
        r("  ALL CHECKS PASSED")

    r.save(args.report)
    sys.exit(1 if r.failures else 0)


if __name__ == "__main__":
    main()