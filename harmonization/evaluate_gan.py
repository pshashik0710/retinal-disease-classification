#!/usr/bin/env python3
"""
evaluate_gan.py -- is the trained generator worth building a conclusion on?

Run this BEFORE any downstream claim. If the translations are poor, or the
generator memorised its training images, then a change in the cohort probe
or the classifier says nothing about harmonisation.

Four measurements:

  1. FID between real and translated images, computed in the same frozen
     ConvNeXt feature space the classification experiments use. This is not
     the standard Inception-based FID and its absolute value is NOT
     comparable to published numbers -- it is used here to compare
     conditions within this project (real-vs-real of the target cohort
     gives the floor, real-vs-untranslated the ceiling). Reported as
     "ConvNeXt-FID" throughout so it is never mistaken for the usual metric.

  2. Cycle-consistency L1 on held-out images. The training loss is measured
     on images the generator was fitted to; this measures it on images it
     was not.

  3. Memorisation. For each translated image, the nearest real training
     image in feature space. A generator that copied its training data will
     produce near-zero distances. Compared against a real-to-real baseline
     so the number has a scale.

  4. Intensity and structure statistics before and after, so a translation
     that merely rescales brightness is visible as such.

The memorisation check reuses the feature caches already built for the
classification experiments, so it costs nothing extra.

Usage:
    python evaluate_gan.py \
        --checkpoint /kaggle/working/harmonization/latest.pth \
        --translated-manifest manifests/pooled_translated_neh.csv \
        --translated-root /kaggle/working/translated \
        --data-root "kermany=/kaggle/input/.../OCT2017 " \
        --data-root "neh=/kaggle/input/.../neh-cleaned-12565"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from scipy import linalg

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))

from config import Config                                    # noqa: E402
from models import ConvNeXtTiny, ResNetGenerator             # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGENET_MEAN = np.array(Config.IMAGENET_MEAN, dtype=np.float32)
IMAGENET_STD = np.array(Config.IMAGENET_STD, dtype=np.float32)


class EvalSet(Dataset):
    """Loads for the feature extractor: resize, crop, ImageNet-normalise."""

    def __init__(self, paths, size=224, gan_space=False):
        self.paths = paths
        self.size = size
        self.gan_space = gan_space          # [-1,1] instead of ImageNet

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        w, h = img.size
        s = self.size / min(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))),
                         Image.BICUBIC)
        a = np.asarray(img, dtype=np.float32) / 255.0
        H, W = a.shape[:2]
        t, l = (H - self.size) // 2, (W - self.size) // 2
        a = a[max(0, t):max(0, t) + self.size, max(0, l):max(0, l) + self.size]
        if a.shape[0] != self.size or a.shape[1] != self.size:
            a = np.pad(a, ((0, max(0, self.size - a.shape[0])),
                           (0, max(0, self.size - a.shape[1])),
                           (0, 0)))[:self.size, :self.size]
        if self.gan_space:
            a = a * 2.0 - 1.0
        else:
            a = (a - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)


@torch.no_grad()
def features(model, paths, device, batch=32, workers=2, size=224):
    ds = EvalSet(paths, size=size)
    ld = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    out = []
    for x in ld:
        out.append(model.forward_features(x.to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 768), np.float32)


def frechet(f1, f2, eps=1e-6):
    """Frechet distance between two Gaussians fitted to the feature sets."""
    if len(f1) < 2 or len(f2) < 2:
        return None
    m1, m2 = f1.mean(0), f2.mean(0)
    s1 = np.cov(f1, rowvar=False)
    s2 = np.cov(f2, rowvar=False)
    diff = m1 - m2
    with np.errstate(all="ignore"):
        covmean = linalg.sqrtm(s1.dot(s2))
    if not np.isfinite(covmean).all():
        offset = np.eye(s1.shape[0]) * eps
        with np.errstate(all="ignore"):
            covmean = linalg.sqrtm((s1 + offset).dot(s2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2)
                 - 2 * np.trace(covmean))


def nearest_neighbour(query, reference, chunk=256):
    """
    Smallest cosine distance from each query feature to any reference
    feature. Used to detect a generator that copied its training images.
    """
    if len(query) == 0 or len(reference) == 0:
        return np.array([])
    q = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-9)
    r = reference / (np.linalg.norm(reference, axis=1, keepdims=True) + 1e-9)
    best = np.full(len(q), np.inf, dtype=np.float32)
    for i in range(0, len(q), chunk):
        sim = q[i:i + chunk] @ r.T
        best[i:i + chunk] = 1.0 - sim.max(axis=1)
    return best


def intensity_stats(paths, n=400):
    step = max(1, len(paths) // n)
    means, stds = [], []
    for p in paths[::step]:
        with Image.open(p) as im:
            a = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
        means.append(a.mean())
        stds.append(a.std())
    return {"mean": round(float(np.mean(means)), 4),
            "sd": round(float(np.mean(stds)), 4)}


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--translated-manifest", required=True)
    p.add_argument("--translated-root", required=True)
    p.add_argument("--source-manifest", default=None)
    p.add_argument("--data-root", action="append", default=None,
                   metavar="COHORT=PATH")
    p.add_argument("--n-fid", type=int, default=2000,
                   help="images per group for ConvNeXt-FID")
    p.add_argument("--n-cycle", type=int, default=300,
                   help="held-out images for the cycle-consistency check")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="reports/gan_evaluation.json")
    a = p.parse_args()

    roots = dict(Config.DATA_ROOTS)
    for spec in (a.data_root or []):
        k, v = spec.split("=", 1)
        roots[k] = v

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(Config.SEED)

    src_man = a.source_manifest or os.path.join(
        Config.MANIFEST_DIR, Config.TRACKS["oct"]["manifest"])
    src = pd.concat([c for c in pd.read_csv(src_man, chunksize=20000)],
                    ignore_index=True)
    tr = pd.read_csv(a.translated_manifest)

    ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    dom_A, dom_B = ck.get("domains", ["kermany", "neh"])

    print(f"\n{'='*72}\nGAN EVALUATION\n{'='*72}")
    print(f"  checkpoint : {a.checkpoint}  "
          f"(epoch {ck.get('epoch','?')}, iter {ck.get('gstep','?')})")
    print(f"  device     : {device}")
    print(f"  translated : {len(tr):,} images")

    print(f"\n  loading {Config.MODEL_NAME} for feature extraction...")
    fx = ConvNeXtTiny(num_classes=Config.NUM_CLASSES,
                      model_name=Config.MODEL_NAME,
                      pretrained=Config.PRETRAINED).to(device).eval()

    def abs_src(sub):
        return [os.path.join(roots[c], d.replace("/", os.sep))
                for c, d in zip(sub["cohort"], sub["Directory"])]

    def abs_tr(sub):
        return [os.path.join(a.translated_root, d.replace("/", os.sep))
                for d in sub["Directory"]]

    def sample(seq, n):
        seq = list(seq)
        if len(seq) <= n:
            return seq
        idx = rng.choice(len(seq), n, replace=False)
        return [seq[i] for i in idx]

    res = {}

    # ------------------------------------------------- 1. ConvNeXt-FID
    print(f"\n{'='*72}\n1. ConvNeXt-FID\n{'='*72}")
    print("  Frechet distance in the frozen ConvNeXt feature space used by")
    print("  the classification experiments. NOT Inception-FID: the absolute")
    print("  value is not comparable to published numbers, only to the other")
    print("  rows here.\n")

    real = {c: sample(abs_src(src[(src.cohort == c) &
                                  (src.split == "train")]), a.n_fid)
            for c in (dom_A, dom_B)}
    trans = {c: sample(abs_tr(tr[(tr.get("source_cohort", tr.cohort) == c) &
                                 (tr.split == "train")]), a.n_fid)
             for c in (dom_A, dom_B)}

    feats = {}
    for k, paths in [(f"real_{dom_A}", real[dom_A]),
                     (f"real_{dom_B}", real[dom_B]),
                     (f"trans_{dom_A}", trans[dom_A]),
                     (f"trans_{dom_B}", trans[dom_B])]:
        if not paths:
            continue
        t0 = time.time()
        feats[k] = features(fx, paths, device, a.batch_size, a.workers)
        print(f"    {k:<16} {len(paths):>6,} images  "
              f"{time.time()-t0:5.1f}s")

    # a real-vs-real split of the target cohort gives the floor: two
    # samples of the same distribution, so anything at or near this is
    # indistinguishable from real
    rb = feats.get(f"real_{dom_B}")
    floor = None
    if rb is not None and len(rb) > 20:
        h = len(rb) // 2
        floor = frechet(rb[:h], rb[h:])

    pairs = []
    if floor is not None:
        pairs.append((f"real_{dom_B} vs itself (floor)", floor))
    for name, f1, f2 in [
        (f"real_{dom_A} vs real_{dom_B} (untranslated gap)",
         feats.get(f"real_{dom_A}"), feats.get(f"real_{dom_B}")),
        (f"trans_{dom_A} vs real_{dom_B} (after translation)",
         feats.get(f"trans_{dom_A}"), feats.get(f"real_{dom_B}")),
        (f"trans_{dom_A} vs real_{dom_A} (drift from source)",
         feats.get(f"trans_{dom_A}"), feats.get(f"real_{dom_A}")),
    ]:
        if f1 is not None and f2 is not None:
            pairs.append((name, frechet(f1, f2)))

    print()
    for name, v in pairs:
        print(f"    {name:<48} {v:10.2f}" if v is not None
              else f"    {name:<48}        n/a")
    res["convnext_fid"] = {n: (round(v, 2) if v is not None else None)
                           for n, v in pairs}

    gap = dict(pairs).get(f"real_{dom_A} vs real_{dom_B} (untranslated gap)")
    after = dict(pairs).get(f"trans_{dom_A} vs real_{dom_B} "
                            f"(after translation)")
    if gap and after:
        red = 100 * (1 - after / gap)
        print(f"\n    distance to the target cohort fell {red:.1f}% "
              f"({gap:.1f} -> {after:.1f})")
        res["fid_reduction_pct"] = round(red, 1)

    # ------------------------------------------ 2. cycle consistency
    print(f"\n{'='*72}\n2. CYCLE CONSISTENCY, HELD-OUT IMAGES\n{'='*72}")
    print("  The training loss is measured on images the generator was")
    print("  fitted to. This is the same quantity on images it was not.\n")

    targs = ck.get("args", {})
    size = targs.get("image_size", 128)
    G_AB = ResNetGenerator(
        num_residual_blocks=targs.get("blocks", 6)).to(device).eval()
    G_BA = ResNetGenerator(
        num_residual_blocks=targs.get("blocks", 6)).to(device).eval()
    G_AB.load_state_dict(ck["G_AB"])
    G_BA.load_state_dict(ck["G_BA"])

    cyc = {}
    for coh, fwd, back in ((dom_A, G_AB, G_BA), (dom_B, G_BA, G_AB)):
        for split in ("train", "test"):
            paths = sample(abs_src(src[(src.cohort == coh) &
                                       (src.split == split)]), a.n_cycle)
            if not paths:
                continue
            ds = EvalSet(paths, size=size, gan_space=True)
            ld = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                            num_workers=a.workers)
            tot, n = 0.0, 0
            with torch.no_grad():
                for x in ld:
                    x = x.to(device)
                    tot += torch.abs(back(fwd(x)) - x).mean().item() * len(x)
                    n += len(x)
            cyc[f"{coh}_{split}"] = round(tot / n, 4)
            print(f"    {coh:<10} {split:<6} L1 {tot/n:.4f}  ({n} images)")

    for coh in (dom_A, dom_B):
        a_, b_ = cyc.get(f"{coh}_train"), cyc.get(f"{coh}_test")
        if a_ and b_:
            print(f"    {coh:<10} train->test gap {b_ - a_:+.4f}"
                  + ("   (generalises)" if abs(b_ - a_) < 0.02 else
                     "   (worse on held-out images)"))
    res["cycle_l1"] = cyc

    # ---------------------------------------------- 3. memorisation
    print(f"\n{'='*72}\n3. MEMORISATION\n{'='*72}")
    print("  Nearest real training image for each translated image, by")
    print("  cosine distance in feature space. A generator that copied its")
    print("  training data produces near-zero distances.\n")

    ta = feats.get(f"trans_{dom_A}")
    ra = feats.get(f"real_{dom_A}")
    rbf = feats.get(f"real_{dom_B}")
    if ta is not None and rbf is not None:
        d_gen = nearest_neighbour(ta, rbf)
        # baseline: how close are real target images to each other? the
        # generator's distances should be comparable, not far smaller
        h = len(rbf) // 2
        d_real = nearest_neighbour(rbf[:h], rbf[h:])
        print(f"    translated -> nearest real {dom_B}: "
              f"median {np.median(d_gen):.4f}  min {d_gen.min():.4f}")
        print(f"    real {dom_B} -> nearest other real {dom_B}: "
              f"median {np.median(d_real):.4f}  min {d_real.min():.4f}")
        ratio = float(np.median(d_gen) / max(np.median(d_real), 1e-9))
        print(f"    ratio {ratio:.2f}")
        if ratio < 0.5:
            print(f"\n    Translated images sit MUCH closer to real training")
            print(f"    images than real images sit to each other. That is")
            print(f"    the signature of memorisation -- treat any downstream")
            print(f"    gain as suspect.")
        else:
            print(f"\n    Distances are comparable to the real-to-real")
            print(f"    baseline; no evidence of copying.")
        res["memorisation"] = {
            "median_translated_to_real": round(float(np.median(d_gen)), 4),
            "median_real_to_real": round(float(np.median(d_real)), 4),
            "ratio": round(ratio, 3),
            "min_translated_to_real": round(float(d_gen.min()), 4)}

    # -------------------------------------------- 4. intensity stats
    print(f"\n{'='*72}\n4. INTENSITY AND CONTRAST\n{'='*72}")
    print("  If the translation only rescaled brightness, it shows here.\n")
    st = {}
    for label, paths in [(f"real {dom_A}", real[dom_A]),
                         (f"real {dom_B}", real[dom_B]),
                         (f"translated {dom_A}", trans[dom_A])]:
        if paths:
            st[label] = intensity_stats(paths)
            print(f"    {label:<22} mean {st[label]['mean']:.4f}  "
                  f"sd {st[label]['sd']:.4f}")
    res["intensity"] = st

    # ------------------------------------------------------- verdict
    print(f"\n{'='*72}\nVERDICT\n{'='*72}")
    notes = []
    if res.get("fid_reduction_pct", 0) > 30:
        notes.append("translation moved the source cohort substantially "
                     "toward the target distribution")
    elif res.get("fid_reduction_pct") is not None:
        notes.append(f"translation moved the distribution only "
                     f"{res['fid_reduction_pct']:.0f}% of the way")
    if res.get("memorisation", {}).get("ratio", 1) < 0.5:
        notes.append("MEMORISATION SUSPECTED")
    else:
        notes.append("no memorisation signature")
    for n in notes:
        print(f"  - {n}")
    print(f"\n  These are necessary conditions, not sufficient ones. A good")
    print(f"  ConvNeXt-FID says the translation looks like the target; it")
    print(f"  does not say the pathology survived. That is what the cohort")
    print(f"  probe and the classifier measure.")

    res["_meta"] = {"generated": datetime.now().isoformat(timespec="seconds"),
                    "checkpoint": a.checkpoint,
                    "checkpoint_epoch": ck.get("epoch"),
                    "domains": [dom_A, dom_B],
                    "n_fid": a.n_fid, "n_cycle": a.n_cycle,
                    "fid_note": ("ConvNeXt features, not Inception; not "
                                 "comparable to published FID values")}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  results -> {a.out}\n")


if __name__ == "__main__":
    main()