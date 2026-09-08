#!/usr/bin/env python3
"""
translate.py -- apply a trained CycleGAN generator to a manifest and write
the translated images plus a manifest that points at them.

The harmonisation experiment asks whether mapping both cohorts into one
appearance reduces the cohort signal without destroying the disease signal.
To measure that, every image the downstream classifier sees must have gone
through the same treatment, so this script translates ALL splits -- train,
val and test.

That is not a leak. The GAN was fitted on training images only (asserted in
train_cyclegan.py), and translating a held-out image with a frozen
generator is a fixed preprocessing function, exactly like a resize. What
would be a leak is fitting the generator on test images, which did not
happen.

DIRECTION. --to-domain names the appearance everything is mapped INTO:

    --to-domain neh       Kermany images -> NEH appearance via G_AB
                          NEH images copied unchanged
    --to-domain kermany   NEH images -> Kermany appearance via G_BA
                          Kermany images copied unchanged
    --to-domain both      each cohort translated to the other; doubles the
                          data and is NOT the harmonisation experiment

Images are written at the GAN's training resolution. The classifier's own
transforms resize from there, so this is one resize more than the baseline
pipeline applies. The comparison stays fair because the baseline is
re-measured on identically resized images -- see --write-resized-baseline.

Usage:
    python translate.py --checkpoint /kaggle/working/harmonization/latest.pth \
        --to-domain neh \
        --out-images /kaggle/working/translated \
        --out-manifest manifests/pooled_translated_neh.csv \
        --data-root "kermany=/kaggle/input/.../OCT2017 " \
        --data-root "neh=/kaggle/input/.../neh-cleaned-12565"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))

from config import Config                       # noqa: E402
from models import ResNetGenerator              # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TranslateSet(Dataset):
    """Deterministic loading: resize the short side, centre crop, to [-1,1]."""

    def __init__(self, rows, roots, size):
        self.rows = rows.reset_index(drop=True)
        self.roots = roots
        self.size = size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        full = os.path.join(self.roots[r["cohort"]],
                            r["Directory"].replace("/", os.sep))
        with Image.open(full) as im:
            img = im.convert("RGB")
        w, h = img.size
        s = self.size / min(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))),
                         Image.BICUBIC)
        a = np.asarray(img, dtype=np.float32) / 255.0
        H, W = a.shape[:2]
        top, left = (H - self.size) // 2, (W - self.size) // 2
        a = a[max(0, top):max(0, top) + self.size,
              max(0, left):max(0, left) + self.size]
        if a.shape[0] != self.size or a.shape[1] != self.size:
            a = np.pad(a, ((0, max(0, self.size - a.shape[0])),
                           (0, max(0, self.size - a.shape[1])),
                           (0, 0)))[:self.size, :self.size]
        t = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)
        return t * 2.0 - 1.0, i


def save_image(tensor, path):
    """tanh output back to 8-bit."""
    a = ((tensor.detach().cpu().float() + 1.0) / 2.0).clamp(0, 1)
    a = (a.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(a).save(path)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--to-domain", required=True,
                   choices=["kermany", "neh", "both"])
    p.add_argument("--manifest", default=None)
    p.add_argument("--out-images", required=True)
    p.add_argument("--out-manifest", required=True)
    p.add_argument("--data-root", action="append", default=None,
                   metavar="COHORT=PATH")
    p.add_argument("--splits", nargs="+",
                   default=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--image-size", type=int, default=None,
                   help="defaults to the size the checkpoint was trained at")
    p.add_argument("--write-resized-baseline", default=None,
                   metavar="DIR",
                   help="also write every image resized but NOT translated, "
                        "so the baseline can be re-measured after the same "
                        "resize. Without this, any difference could be the "
                        "resize rather than the translation.")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    roots = dict(Config.DATA_ROOTS)
    for spec in (a.data_root or []):
        if "=" not in spec:
            sys.exit(f"--data-root expects COHORT=PATH, got {spec!r}")
        k, v = spec.split("=", 1)
        roots[k] = v

    if not os.path.isfile(a.checkpoint):
        sys.exit(f"checkpoint not found: {a.checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    targs = ck.get("args", {})
    size = a.image_size or targs.get("image_size", 128)
    blocks = targs.get("blocks", 6)
    trained_domains = ck.get("domains", ["kermany", "neh"])

    print(f"\n{'='*72}\nTRANSLATE\n{'='*72}")
    print(f"  checkpoint  : {a.checkpoint}")
    print(f"  trained to  : epoch {ck.get('epoch','?')}, "
          f"iteration {ck.get('gstep','?')}")
    print(f"  domains     : {trained_domains[0]} (A) <-> "
          f"{trained_domains[1]} (B)")
    print(f"  image size  : {size}   blocks {blocks}")
    print(f"  device      : {device}")
    print(f"  to-domain   : {a.to_domain}")

    G_AB = ResNetGenerator(num_residual_blocks=blocks).to(device).eval()
    G_BA = ResNetGenerator(num_residual_blocks=blocks).to(device).eval()
    G_AB.load_state_dict(ck["G_AB"])
    G_BA.load_state_dict(ck["G_BA"])

    dom_A, dom_B = trained_domains

    manifest = a.manifest or os.path.join(Config.MANIFEST_DIR,
                                          Config.TRACKS["oct"]["manifest"])
    df = pd.concat([c for c in pd.read_csv(manifest, chunksize=20000)],
                   ignore_index=True)
    df = df[df["split"].isin(a.splits)].reset_index(drop=True)
    if a.limit:
        df = df.groupby("split", group_keys=False).head(a.limit)
    print(f"\n  manifest    : {manifest}")
    print(f"  rows        : {len(df):,}  "
          f"({dict(df['split'].value_counts())})")
    print(f"  cohorts     : {dict(df['cohort'].value_counts())}")

    print(f"\n  NOTE: all splits are translated. The generator was fitted on")
    print(f"  training images only; applying it to held-out images with")
    print(f"  frozen weights is a fixed preprocessing function, not a leak.")

    # which cohort gets translated, and by which generator
    if a.to_domain == "both":
        plan = {dom_A: ("G_AB", G_AB), dom_B: ("G_BA", G_BA)}
    elif a.to_domain == dom_B:
        plan = {dom_A: ("G_AB", G_AB)}
    elif a.to_domain == dom_A:
        plan = {dom_B: ("G_BA", G_BA)}
    else:
        sys.exit(f"--to-domain {a.to_domain!r} is not one of the trained "
                 f"domains {trained_domains}")

    print(f"\n  plan:")
    for coh in sorted(df["cohort"].unique()):
        if coh in plan:
            print(f"    {coh:<10} -> translated by {plan[coh][0]}")
        else:
            print(f"    {coh:<10} -> resized only (already the target "
                  f"appearance)")

    os.makedirs(a.out_images, exist_ok=True)
    rows_out = []
    t0 = time.time()

    for coh, sub in df.groupby("cohort"):
        gen = plan.get(coh, (None, None))[1]
        tag = "translated" if gen is not None else "resized"
        ds = TranslateSet(sub, roots, size)
        ld = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        num_workers=a.workers)

        print(f"\n  {coh}: {len(sub):,} images ({tag})")
        done = 0
        for x, idx in ld:
            x = x.to(device)
            with torch.no_grad():
                y = gen(x) if gen is not None else x

            for j, k in enumerate(idx.tolist()):
                r = ds.rows.iloc[k]
                rel = os.path.join(coh, r["Directory"].replace("/", os.sep))
                rel = os.path.splitext(rel)[0] + ".png"
                save_image(y[j], os.path.join(a.out_images, rel))

                if a.write_resized_baseline:
                    save_image(x[j], os.path.join(a.write_resized_baseline,
                                                  rel))

                nr = r.to_dict()
                nr["Directory"] = rel.replace(os.sep, "/")
                nr["cohort"] = "translated"
                nr["source_cohort"] = coh
                nr["translated"] = gen is not None
                nr["width"] = nr["height"] = size
                rows_out.append(nr)

            done += len(idx)
            if done % (a.batch_size * 20) < a.batch_size or done == len(sub):
                el = time.time() - t0
                print(f"    {done:>7,}/{len(sub):,}  "
                      f"{done/max(el,1e-9):5.1f} img/s  "
                      f"elapsed {timedelta(seconds=int(el))}", flush=True)

    out = pd.DataFrame(rows_out)
    out["root_hint"] = a.out_images
    os.makedirs(os.path.dirname(a.out_manifest) or ".", exist_ok=True)
    out.to_csv(a.out_manifest, index=False)

    # the split assignment is inherited, so patient-disjointness carries over
    print(f"\n{'='*72}\nVERIFY\n{'='*72}")
    ok = True
    for col in ("patient_key", "group_key", "Directory"):
        if col not in out.columns:
            continue
        s = {k: set(g[col]) for k, g in out.groupby("split")}
        bad = [f"{x}n{y}" for x, y in
               (("train", "val"), ("train", "test"), ("val", "test"))
               if len(s.get(x, set()) & s.get(y, set()))]
        if bad:
            ok = False
            print(f"  {col:<12} *** {', '.join(bad)}")
        else:
            print(f"  {col:<12} disjoint across splits (inherited)   OK")

    missing = sum(1 for d in out["Directory"]
                  if not os.path.isfile(os.path.join(a.out_images,
                                                     d.replace("/", os.sep))))
    print(f"  files on disk  {len(out) - missing:,}/{len(out):,} "
          f"{'OK' if missing == 0 else f'*** {missing} MISSING'}")
    ok = ok and missing == 0

    meta = {"generated": datetime.now().isoformat(timespec="seconds"),
            "checkpoint": a.checkpoint,
            "checkpoint_epoch": ck.get("epoch"),
            "to_domain": a.to_domain, "image_size": size, "blocks": blocks,
            "trained_domains": trained_domains,
            "source_manifest": manifest, "splits": a.splits,
            "n_images": int(len(out)),
            "n_translated": int(out["translated"].sum()),
            "n_resized_only": int((~out["translated"]).sum()),
            "seconds": round(time.time() - t0, 1),
            "note": ("generator fitted on training images only; applying it "
                     "to val/test with frozen weights is preprocessing, not "
                     "leakage")}
    with open(os.path.splitext(a.out_manifest)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  images    -> {a.out_images}")
    print(f"  manifest  -> {a.out_manifest}")
    if a.write_resized_baseline:
        print(f"  baseline  -> {a.write_resized_baseline}")
        print(f"              (same resize, no translation -- compare "
              f"against this,\n               not against the original "
              f"pipeline, or the resize\n               confounds the "
              f"result)")
    print()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()