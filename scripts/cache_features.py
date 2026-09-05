#!/usr/bin/env python3
"""
cache_features.py -- run the frozen backbone over every image once and store
the pooled feature vectors.

Why: measured on this machine (ConvNeXt-Tiny, batch 16, 224px, CPU),

    full fine-tune    ~32.6 s/batch  ->  ~8   h per epoch over 42,838 images
    frozen backbone   ~14.5 s/batch  ->  ~3.5 h per epoch

A frozen backbone recomputes identical features every epoch, so 30 epochs
costs 100+ hours for no new information. Caching moves that to a single
pass: one traversal of the data (hours), then head training on 768-d
vectors (seconds per epoch). Identical result, and it makes the ablations,
the cohort probe and any hyperparameter search affordable.

This is the standard linear-probe protocol and is reported as such.

Caveat: cached features are computed WITHOUT augmentation, because an
augmented image yields different features each epoch and could not be
cached. If augmentation turns out to matter for this task, use
TRAIN_MODE="linear_probe" (frozen backbone, images each epoch) or
"finetune" instead. The comparison is worth one paragraph in the paper.

Output per split: an .npz holding
    features  (N, 768) float32
    labels    (N,)     int64
    index     (N,)     int64   row index into the manifest, so any
                               prediction can be traced back to its
                               patient, cohort and file

Usage:
    python cache_features.py                    # all splits
    python cache_features.py --splits train
    python cache_features.py --limit 200        # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import timedelta

import numpy as np
import torch

try:
    from scripts.config import Config
    from scripts.models import ConvNeXtTiny
    from scripts.dataset import (load_manifest, get_split, OCTDataset,
                                 get_eval_transforms, make_loader)
except ImportError:
    from config import Config
    from models import ConvNeXtTiny
    from dataset import (load_manifest, get_split, OCTDataset,
                         get_eval_transforms, make_loader)


def human(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


@torch.no_grad()
def cache_split(model, df, split, args):
    out_path = Config.feature_cache_path(split)

    if os.path.isfile(out_path) and not args.overwrite:
        d = np.load(out_path)
        print(f"  {split:<6} already cached: {d['features'].shape} "
              f"-> {out_path}")
        print(f"         (pass --overwrite to redo)")
        return None

    sub = get_split(df, split)
    if args.limit:
        sub = sub.iloc[:args.limit]

    ds = OCTDataset(sub, get_eval_transforms(), return_meta=True)
    loader = make_loader(ds, shuffle=False, batch_size=args.batch_size)

    n = len(ds)
    feats = np.zeros((n, model.num_features), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    index = np.zeros(n, dtype=np.int64)

    print(f"\n  {split}: {n:,} images, {sub['patient_key'].nunique():,} "
          f"patients")

    start = time.time()
    seen = 0
    for bi, (x, y, idx) in enumerate(loader):
        x = x.to(Config.DEVICE, non_blocking=False)
        f = model.forward_features(x).cpu().numpy()

        b = len(y)
        feats[seen:seen + b] = f
        labels[seen:seen + b] = y.numpy()
        index[seen:seen + b] = idx.numpy()
        seen += b

        if bi % max(1, args.print_every) == 0 or seen == n:
            el = time.time() - start
            rate = seen / el if el > 0 else 0
            eta = (n - seen) / rate if rate > 0 else 0
            print(f"    {seen:>7,}/{n:,}  {100*seen/n:5.1f}%  "
                  f"{rate:6.1f} img/s  elapsed {human(el)}  "
                  f"eta {human(eta)}", flush=True)

    assert seen == n, f"processed {seen} of {n}"
    elapsed = time.time() - start

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, features=feats, labels=labels, index=index)
    size_mb = os.path.getsize(out_path) / 1e6

    print(f"  {split:<6} done in {human(elapsed)}  "
          f"({n/elapsed:.1f} img/s)  -> {out_path}  ({size_mb:.0f} MB)")

    return {
        "split": split, "n": int(n), "seconds": round(elapsed, 1),
        "images_per_sec": round(n / elapsed, 2),
        "file": out_path, "size_mb": round(size_mb, 1),
        "feature_dim": int(model.num_features),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="only this many images per split (smoke test)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--print-every", type=int, default=20,
                    help="progress line every N batches")
    ap.add_argument("--track", default=None,
                    help="override Config.TRACK for this run, so the run "
                         "and its output paths cannot disagree")
    args = ap.parse_args()

    if args.track:
        if args.track not in Config.TRACKS:
            sys.exit(f"unknown track {args.track!r}; "
                     f"valid: {sorted(Config.TRACKS)}")

        _t = Config.TRACKS[args.track]
        Config.TRACK = args.track
        Config.POOLED_MANIFEST = os.path.join(Config.MANIFEST_DIR,
                                              _t["manifest"])
        Config.DATA_ROOTS = _t["roots"]
        Config.CLASSES = _t["classes"]
        Config.CLASS_TO_IDX = {c: i for i, c in enumerate(Config.CLASSES)}
        Config.NUM_CLASSES = len(Config.CLASSES)
        Config.RESIZE_STRATEGY = _t["resize"]
        Config.TRACK_NOTE = _t["note"]
        Config.MANIFEST_FILTER = _t.get("filter")
        Config.FEATURE_CACHE_DIR = os.path.join(Config.BASE_DIR, "features",
                                                args.track)


    args.batch_size = args.batch_size or Config.BATCH_SIZE

    problems = Config.validate(check_data=True)
    if problems:
        print("\n  CONFIGURATION PROBLEMS:")
        for p in problems:
            print(f"    - {p}")
        sys.exit(1)

    torch.set_num_threads(Config.TORCH_NUM_THREADS)
    torch.manual_seed(Config.SEED)

    print()
    print(Config.summary())
    print(f"\n  cache dir : {Config.FEATURE_CACHE_DIR}")
    print(f"  batch     : {args.batch_size}")

    print(f"\n  loading {Config.MODEL_NAME} "
          f"(pretrained={Config.PRETRAINED})...")
    model = ConvNeXtTiny(
        num_classes=Config.NUM_CLASSES,
        dropout=Config.DROPOUT,
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
    ).to(Config.DEVICE)
    model.eval()
    print(f"  feature dim: {model.num_features}")

    df = load_manifest()

    results = []
    t0 = time.time()
    for split in args.splits:
        r = cache_split(model, df, split, args)
        if r:
            results.append(r)

    if results:
        meta = {
            "model": Config.MODEL_NAME,
            "pretrained": Config.PRETRAINED,
            "image_size": Config.IMAGE_SIZE,
            "resize_strategy": Config.RESIZE_STRATEGY,
            "augmentation": "none (eval transforms)",
            "device": Config.DEVICE,
            "torch_threads": Config.TORCH_NUM_THREADS,
            "batch_size": args.batch_size,
            "manifest": Config.POOLED_MANIFEST,
            "classes": Config.CLASSES,
            "total_seconds": round(time.time() - t0, 1),
            "splits": results,
        }
        mp = os.path.join(Config.FEATURE_CACHE_DIR, "cache_meta.json")
        with open(mp, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n  total {human(time.time() - t0)}")
        print(f"  metadata -> {mp}")

    print("\n  Next: train the head on the cached features.\n")


if __name__ == "__main__":
    main()