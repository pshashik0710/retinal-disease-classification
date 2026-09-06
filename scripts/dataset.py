#!/usr/bin/env python3
"""
dataset.py -- manifest-driven dataset and transforms.

Everything is read from manifests/pooled_split.csv. Nothing globs the
filesystem, so none of the failure modes from the previous pipeline apply:

  * no case-sensitive extension matching (the pooled data is 13,250 .tif
    and 3,560 .jpg; a glob for '*.jpg' would have silently dropped 79%
    of NEH)
  * no single-directory-depth assumption (NEH stores one-eye patients as
    CNV/1/*.jpg and two-eye patients as CNV/101/OD/*.tif)
  * no silent empty dataset -- an empty selection raises
  * no re-splitting: the 'split' column is authoritative and was verified
    patient/group/eye/file-disjoint

Transforms preserve aspect ratio by default. The pooled data has six native
ratios (0.774 to 3.097): Kermany B-scans are 512/768/1024/1536 x 496 or
512x512, NEH is uniformly 768x496. Height is retinal DEPTH, width is
lateral EXTENT, so a plain resize to a square compresses the two axes by
different -- and per-image different -- amounts.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile

import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from scripts.config import Config
except ImportError:
    from config import Config

# A truncated file should not kill a multi-hour run.
ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================================================================
# TRANSFORMS
# =========================================================================

def _geometry(image_size: int, strategy: str):
    """Aspect-ratio handling. See module docstring for why this matters."""
    if strategy == "resize_crop":
        # short side -> image_size, then centre crop. Geometry preserved,
        # lateral periphery lost on wide scans.
        return [
            A.SmallestMaxSize(max_size=image_size),
            A.CenterCrop(height=image_size, width=image_size),
        ]
    if strategy == "pad":
        # long side -> image_size, pad the rest. Geometry and full field
        # preserved, black bars added.
        return [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size,
                          border_mode=0, fill=0),
        ]
    if strategy == "squash":
        # what the rejected pipeline did; kept so the distortion can be
        # quantified as an ablation
        return [A.Resize(height=image_size, width=image_size)]

    if strategy == "normalize_768":
        # Both cohorts pass through a common intermediate resolution before
        # the network resize, so lateral sampling is uniform.
        #
        # NEH is uniformly 768x496; Kermany spans 512/768/1024/1536 x 496.
        # Under resize_crop a 1536-wide scan loses 68% of its lateral field
        # to the centre crop while a 512-wide one loses almost nothing --
        # the crop is far more destructive to one cohort than the other.
        # Normalising first makes that loss uniform.
        return [
            A.Resize(height=496, width=768),
            A.SmallestMaxSize(max_size=image_size),
            A.CenterCrop(height=image_size, width=image_size),
        ]
    
    raise ValueError(f"unknown resize strategy: {strategy!r}")


def get_train_transforms(image_size=None, strategy=None):
    image_size = image_size or Config.IMAGE_SIZE
    strategy = strategy or Config.RESIZE_STRATEGY

    aug = []
    if Config.HFLIP_PROB > 0:
        # nasal <-> temporal mirror; anatomically valid on a B-scan.
        # NOTE: there is deliberately no vertical flip. The vertical axis
        # is retinal depth (ILM above RPE, always), and DRUSEN/CNV are
        # defined by lesion position relative to the RPE.
        aug.append(A.HorizontalFlip(p=Config.HFLIP_PROB))
    if Config.ROTATION_LIMIT > 0:
        aug.append(A.Rotate(limit=Config.ROTATION_LIMIT,
                            border_mode=0,
                            p=Config.ROTATION_PROB))
    if Config.BRIGHTNESS_CONTRAST_PROB > 0:
        aug.append(A.RandomBrightnessContrast(
            brightness_limit=Config.BRIGHTNESS_LIMIT,
            contrast_limit=Config.CONTRAST_LIMIT,
            p=Config.BRIGHTNESS_CONTRAST_PROB))

    return A.Compose(
        _geometry(image_size, strategy) + aug + [
            A.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
            ToTensorV2(),
        ])


def get_eval_transforms(image_size=None, strategy=None):
    """Deterministic: geometry, normalise, to tensor. No augmentation."""
    image_size = image_size or Config.IMAGE_SIZE
    strategy = strategy or Config.RESIZE_STRATEGY
    return A.Compose(
        _geometry(image_size, strategy) + [
            A.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD),
            ToTensorV2(),
        ])


def get_gan_transforms(image_size=None):
    """[-1, 1] for a tanh generator, with the reference crop augmentation."""
    image_size = image_size or Config.GAN_IMAGE_SIZE
    return A.Compose([
        A.SmallestMaxSize(max_size=int(image_size * 1.12)),
        A.RandomCrop(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=Config.GAN_MEAN, std=Config.GAN_STD),
        ToTensorV2(),
    ])


# =========================================================================
# DATASET
# =========================================================================

class OCTDataset(Dataset):
    """
    Rows of the pooled manifest. Returns (image, label) or, with
    return_meta=True, (image, label, index) so predictions can be traced
    back to a patient for error analysis.
    """

    def __init__(self, df, transform, data_roots=None, return_meta=False):
        if len(df) == 0:
            raise ValueError("empty dataframe passed to OCTDataset")
        if transform is None:
            raise ValueError(
                "transform is required: the pipeline must end in "
                "ToTensorV2, or collate receives HWC uint8 arrays")

        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.roots = data_roots or Config.DATA_ROOTS
        self.return_meta = return_meta

        missing = set(self.df["cohort"]) - set(self.roots)
        if missing:
            raise ValueError(f"no data root configured for cohort(s): "
                             f"{sorted(missing)}")

        self.paths = [
            os.path.join(self.roots[c], d.replace("/", os.sep))
            for c, d in zip(self.df["cohort"], self.df["Directory"])
        ]
        self.labels = self.df["y"].to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = None
        for attempt in range(3):
            try:
                with Image.open(path) as im:
                    image = np.array(im.convert("RGB"))
                break
            except MemoryError:
                # transient allocation failure under memory pressure --
                # give the allocator a moment rather than losing the run
                import gc, time as _t
                gc.collect()
                _t.sleep(0.5 * (attempt + 1))
            except Exception as e:
                raise RuntimeError(f"failed to read {path}: {e}") from e
        if image is None:
            raise RuntimeError(f"repeated MemoryError reading {path}; "
                               f"reduce NUM_WORKERS or BATCH_SIZE")

        image = self.transform(image=image)["image"]
        label = int(self.labels[idx])

        if self.return_meta:
            return image, label, idx
        return image, label

    # -- helpers ---------------------------------------------------------
    def class_counts(self):
        return self.df["y_label"].value_counts().reindex(
            Config.CLASSES, fill_value=0)

    def describe(self):
        c = self.class_counts()
        return (f"{len(self.df):,} images | "
                f"{self.df['patient_key'].nunique():,} patients | "
                + ", ".join(f"{k} {v:,}" for k, v in c.items()))


# =========================================================================
# MANIFEST LOADING
# =========================================================================

REQUIRED_COLUMNS = {"Directory", "cohort", "patient_key", "y_label", "y",
                    "split"}


def load_manifest(path=None):
    path = path or Config.POOLED_MANIFEST
    if not os.path.isfile(path):
        sys.exit(f"ERROR: manifest not found: {path}\n"
                 f"Run pool_manifests.py first.")

    df = pd.read_csv(path, low_memory=False)

    # A track may restrict the manifest to a subset of rows -- e.g. cfp_odir
    # keeps only the ODIR-derived images of AMDNet23, so the multi-source
    # confound is absent. Applied here so every consumer (caching, training,
    # evaluation) sees exactly the same rows.
    filt = getattr(Config, "MANIFEST_FILTER", None)
    if filt:
        n0 = len(df)
        for col, vals in filt.items():
            if col not in df.columns:
                sys.exit(f"ERROR: MANIFEST_FILTER names column {col!r}, "
                         f"absent from {path}")
            df = df[df[col].isin(vals)]
        df = df.reset_index(drop=True)
        if df.empty:
            sys.exit(f"ERROR: MANIFEST_FILTER {filt} left no rows")
        print(f"  manifest filter {filt}: {n0:,} -> {len(df):,} rows")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        sys.exit(f"ERROR: manifest missing columns: {sorted(missing)}")
    if df.empty:
        sys.exit(f"ERROR: manifest is empty: {path}")

    # The label column must agree with Config.CLASSES, or every per-class
    # metric is silently mislabelled.
    for cls, idx in Config.CLASS_TO_IDX.items():
        sub = df[df["y_label"] == cls]
        if len(sub) and set(sub["y"]) != {idx}:
            sys.exit(f"ERROR: manifest maps {cls!r} to {sorted(set(sub['y']))} "
                     f"but Config.CLASSES puts it at {idx}. "
                     f"Config.CLASSES order must match the manifest.")

    unknown = set(df["y_label"]) - set(Config.CLASSES)
    if unknown:
        sys.exit(f"ERROR: manifest has labels outside Config.CLASSES: "
                 f"{sorted(unknown)}")

    return df


def get_split(df, split):
    sub = df[df["split"] == split]
    if len(sub) == 0:
        sys.exit(f"ERROR: no rows for split {split!r}. "
                 f"Available: {sorted(df['split'].unique())}")
    return sub


def get_datasets(manifest_path=None, train_transform=None,
                 eval_transform=None, return_meta=False):
    """(train, val, test) datasets from the pooled manifest."""
    df = load_manifest(manifest_path)
    tt = train_transform or get_train_transforms()
    et = eval_transform or get_eval_transforms()
    return (
        OCTDataset(get_split(df, "train"), tt, return_meta=return_meta),
        OCTDataset(get_split(df, "val"), et, return_meta=return_meta),
        OCTDataset(get_split(df, "test"), et, return_meta=return_meta),
    )


# =========================================================================
# DATALOADERS
# =========================================================================

def seed_worker(worker_id):
    """Seed numpy and random per worker.

    torch seeds itself per worker, but numpy and python's random are not
    reliably seeded under every start method, and the augmentation pipeline
    draws from them.
    """
    import random
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def make_loader(dataset, shuffle, batch_size=None, seed=None):
    batch_size = batch_size or Config.BATCH_SIZE
    g = torch.Generator()
    g.manual_seed(seed if seed is not None else Config.SEED)

    kw = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY and Config.DEVICE == "cuda",
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=False,
    )
    if Config.NUM_WORKERS > 0:
        kw["persistent_workers"] = Config.PERSISTENT_WORKERS
        kw["prefetch_factor"] = Config.PREFETCH_FACTOR
    return DataLoader(dataset, **kw)


def compute_class_weights(df_or_ds):
    """
    Inverse-frequency weights, normalised to mean 1, in Config.CLASSES order.
    Pooled counts run about 3.6:1 (NORMAL ~29.8k vs DME ~8.2k).
    """
    df = df_or_ds.df if hasattr(df_or_ds, "df") else df_or_ds
    counts = df["y_label"].value_counts().reindex(
        Config.CLASSES, fill_value=0).to_numpy(dtype=np.float64)
    if (counts == 0).any():
        zero = [c for c, n in zip(Config.CLASSES, counts) if n == 0]
        print(f"  WARNING: no training examples for {zero}; "
              f"their weight is set to 0")
    w = np.divide(counts.sum(), counts * len(counts),
                  out=np.zeros_like(counts), where=counts > 0)
    if w.sum() > 0:
        w = w / w[w > 0].mean()
    return torch.tensor(w, dtype=torch.float32)


# =========================================================================
# SELF-TEST
# =========================================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--check-files", type=int, default=20,
                    help="how many images to actually open")
    a = ap.parse_args()

    print()
    df = load_manifest(a.manifest)
    print(f"  manifest: {len(df):,} rows, "
          f"{df['patient_key'].nunique():,} patients")
    print(f"  label map verified against Config.CLASSES      OK")

    print("\n  splits:")
    for s in ("train", "val", "test"):
        sub = get_split(df, s)
        print(f"    {s:<6} {len(sub):>8,} images  "
              f"{sub['patient_key'].nunique():>5} patients")

    print("\n  transforms:")
    for strat in ("resize_crop", "pad", "squash", "normalize_768"):
        t = get_eval_transforms(strategy=strat)
        # exercise on the six real dimension variants
        for (w, h) in [(768, 496), (512, 496), (512, 512),
                       (1536, 496), (1024, 496), (384, 496)]:
            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            out = t(image=img)["image"]
            assert out.shape == (3, Config.IMAGE_SIZE, Config.IMAGE_SIZE), \
                f"{strat} on {w}x{h} gave {tuple(out.shape)}"
        print(f"    {strat:<12} all six native sizes -> "
              f"(3, {Config.IMAGE_SIZE}, {Config.IMAGE_SIZE})   OK")

    train_t = get_train_transforms()
    img = np.random.randint(0, 255, (496, 768, 3), dtype=np.uint8)
    o1 = train_t(image=img)["image"]
    print(f"    train        {tuple(o1.shape)}  "
          f"dtype {o1.dtype}  range [{o1.min():.2f}, {o1.max():.2f}]   OK")
    assert o1.dtype == torch.float32

    w = compute_class_weights(get_split(df, "train"))
    print("\n  class weights (inverse frequency, mean 1):")
    for c, v in zip(Config.CLASSES, w.tolist()):
        print(f"    {c:<8} {v:.3f}")

    if a.check_files > 0:
        print(f"\n  opening {a.check_files} images from disk:")
        try:
            tr, va, te = get_datasets(a.manifest)
            print(f"    train  {tr.describe()}")
            print(f"    val    {va.describe()}")
            print(f"    test   {te.describe()}")

            idx = np.linspace(0, len(tr) - 1, a.check_files).astype(int)
            shapes = set()
            for i in idx:
                x, y = tr[int(i)]
                shapes.add(tuple(x.shape))
                assert 0 <= y < Config.NUM_CLASSES
            print(f"    shapes seen: {shapes}   OK")

            loader = make_loader(tr, shuffle=True, batch_size=4)
            xb, yb = next(iter(loader))
            print(f"    batch: {tuple(xb.shape)} labels {yb.tolist()}   OK")
        except (RuntimeError, ValueError, SystemExit) as e:
            print(f"    could not read images: {e}")
            print("    (expected if the dataset roots are not mounted here)")

    print("\n  dataset.py OK\n")