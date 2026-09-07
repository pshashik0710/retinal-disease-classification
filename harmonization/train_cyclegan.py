#!/usr/bin/env python3
"""
train_cyclegan.py -- CycleGAN harmonisation between the Kermany and NEH
OCT cohorts.

MOTIVATION, from this project's own measurements. A linear probe on frozen
ConvNeXt features identifies which cohort a B-scan came from with 0.996
accuracy on held-out patients (majority baseline 0.816; label-shuffle
control 0.520 +/- 0.003, 140 sd separation; train-test gap 0.003). That
signal survives every preprocessing strategy tried (0.996 / 0.999 / 0.997 /
0.996 for resize_crop / squash / pad / normalize_768), so it is not
geometric. Within Kermany alone, export width is recoverable at 0.898
(baseline 0.659), so acquisition preprocessing contributes substantially.

The question this experiment asks is deliberately two-sided:

    can image translation reduce that cohort signal WITHOUT destroying
    the disease signal?

Both axes are measured against existing five-seed baselines:

    cohort probe        0.996
    4-class macro-F1    0.840 +/- 0.004

    probe down, F1 held      -> harmonisation preserved pathology
    probe down, F1 down      -> over-harmonised; pathology went too
    probe unchanged          -> the GAN did not learn the domain

A negative result is reported as a result. The experiment is not designed
to produce an improvement.

TRAINING DATA. Only rows with split == "train" are used, asserted at load
time. If the GAN saw validation or test images, translating those images
later and re-evaluating the classifier would be contaminated.

DOMAIN BALANCE. Kermany contributes ~34,800 training images against NEH's
~8,000. Iterating max(len(A), len(B)) as the reference does would show each
NEH image four times per epoch and make each epoch 4x longer. Domains are
therefore balanced by subsampling the larger one at the PATIENT level,
re-drawn each epoch so Kermany's full diversity is still seen over the run.

Usage:
    # verify the whole loop on CPU in about a minute
    python train_cyclegan.py --smoke

    # real run (Kaggle GPU)
    python train_cyclegan.py --epochs 60 --image-size 128 --batch-size 4

    # resumes automatically from the latest checkpoint in --out
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

# the harmonisation experiment reuses the classification pipeline's config
# and model definitions, which live one directory up
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))

from config import Config                      # noqa: E402
from models import ResNetGenerator, PatchGANDiscriminator   # noqa: E402
from image_pool import ImagePool               # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =========================================================================
# DATA
# =========================================================================

class DomainDataset(Dataset):
    """
    Unpaired images from one cohort, normalised to [-1, 1] for the tanh
    generator. Deliberately minimal: resize, random crop, horizontal flip.
    Vertical flip is never applied -- the vertical axis of a B-scan is
    retinal depth, and ILM is always superior to RPE.
    """

    def __init__(self, paths, roots, image_size, train=True):
        if not paths:
            raise ValueError("empty path list")
        self.paths = paths
        self.roots = roots
        self.size = image_size
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        cohort, rel = self.paths[i]
        full = os.path.join(self.roots[cohort], rel.replace("/", os.sep))
        with Image.open(full) as im:
            img = im.convert("RGB")

        # resize the short side a little above the target, then crop
        load = int(self.size * 1.12) if self.train else self.size
        w, h = img.size
        scale = load / min(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.BICUBIC)

        a = np.asarray(img, dtype=np.float32) / 255.0
        H, W = a.shape[:2]
        if self.train:
            top = np.random.randint(0, max(1, H - self.size + 1))
            left = np.random.randint(0, max(1, W - self.size + 1))
        else:
            top, left = (H - self.size) // 2, (W - self.size) // 2
        a = a[top:top + self.size, left:left + self.size]

        if a.shape[0] != self.size or a.shape[1] != self.size:
            pad_h = self.size - a.shape[0]
            pad_w = self.size - a.shape[1]
            a = np.pad(a, ((0, max(0, pad_h)), (0, max(0, pad_w)), (0, 0)))
            a = a[:self.size, :self.size]

        if self.train and np.random.rand() < 0.5:
            a = a[:, ::-1]

        t = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)
        return t * 2.0 - 1.0            # [0,1] -> [-1,1] for tanh


def load_domains(manifest, roots, seed):
    """Training-split paths per cohort, with the patient key for balancing."""
    keep = ["Directory", "cohort", "patient_key", "split"]
    df = pd.concat([c for c in pd.read_csv(
        manifest, chunksize=20000, usecols=lambda x: x in keep)],
        ignore_index=True)

    n_all = len(df)
    df = df[df["split"] == "train"].reset_index(drop=True)
    # the GAN must never see val or test: translating those images later
    # and re-evaluating the classifier would be contaminated
    assert (df["split"] == "train").all(), "non-train rows leaked through"
    print(f"  manifest {n_all:,} rows -> {len(df):,} training rows")

    out = {}
    for coh, g in df.groupby("cohort"):
        out[coh] = {
            "paths": list(zip(g["cohort"], g["Directory"])),
            "patients": g["patient_key"].to_numpy(),
        }
        print(f"    {coh:<10} {len(g):>7,} images  "
              f"{g['patient_key'].nunique():>5} patients")
    if len(out) != 2:
        sys.exit(f"expected exactly 2 cohorts, found {sorted(out)}")
    return out


def balanced_epoch_paths(domains, rng):
    """
    Equal-sized draws from both domains for one epoch. The larger domain is
    subsampled at the PATIENT level so no patient is split across the draw,
    and it is re-drawn every epoch so its full diversity is seen over the
    run rather than a fixed subset.
    """
    a, b = sorted(domains)
    n = min(len(domains[a]["paths"]), len(domains[b]["paths"]))

    out = {}
    for coh in (a, b):
        paths = domains[coh]["paths"]
        pats = domains[coh]["patients"]
        if len(paths) <= n:
            out[coh] = paths
            continue
        order = rng.permutation(np.unique(pats))
        keep, taken = [], 0
        for p in order:
            idx = np.where(pats == p)[0]
            keep.extend(idx)
            taken += len(idx)
            if taken >= n:
                break
        out[coh] = [paths[i] for i in keep[:n]]
    return out[a], out[b], a, b


# =========================================================================
# TRAINING
# =========================================================================

def lr_lambda(epoch, n_epochs, decay_start):
    """Constant, then linear decay to zero -- the reference schedule."""
    if epoch < decay_start:
        return 1.0
    span = max(1, n_epochs - decay_start)
    return max(0.0, 1.0 - (epoch - decay_start) / span)


def atomic_save(obj, path):
    """
    Write to a temporary file, then rename over the target.

    A checkpoint half-written when the process dies is unreadable, and the
    run restarts from nothing. Kaggle ends a session at 12 hours without
    warning, and the odds of that landing inside a multi-hundred-megabyte
    write are not negligible over a long run. Rename is atomic on the same
    filesystem, so the target is either the old checkpoint or the new one,
    never a partial file.
    """
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_grid(tensors, path, n=4):
    """Small qualitative grid: real / fake / reconstructed, both directions."""
    rows = []
    for t in tensors:
        t = t[:n].detach().cpu().float()
        t = ((t + 1.0) / 2.0).clamp(0, 1)       # tanh -> [0,1]
        rows.append(torch.cat(list(t), dim=2))
    grid = torch.cat(rows, dim=1)
    arr = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=None)
    p.add_argument("--out", default=None, help="checkpoint / sample dir")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--decay-start", type=int, default=None,
                   help="epoch at which LR begins its linear decay to zero "
                        "(default: half of --epochs)")
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--blocks", type=int, default=6,
                   help="residual blocks: 9 for 256px, 6 for 128px")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda-cycle", type=float, default=10.0)
    p.add_argument("--lambda-identity", type=float, default=5.0)
    p.add_argument("--pool-size", type=int, default=50)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--save-every", type=int, default=500,
                   help="checkpoint every N iterations. Kaggle sessions end "
                        "at 12 hours without warning.")
    p.add_argument("--sample-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="tiny CPU run that exercises the whole loop, "
                        "including checkpoint and resume")
    a = p.parse_args()

    if a.smoke:
        # smoke defaults, but anything the user passed explicitly wins --
        # otherwise "--smoke --epochs 6" would silently run 2 epochs
        given = {x.lstrip("-").replace("-", "_")
                 for x in sys.argv if x.startswith("--")}
        for k, v in (("epochs", 2), ("image_size", 64), ("batch_size", 2),
                     ("blocks", 2), ("pool_size", 8), ("save_every", 5),
                     ("sample_every", 5), ("workers", 0)):
            if k not in given:
                setattr(a, k, v)

    a.decay_start = a.decay_start or a.epochs // 2
    manifest = a.manifest or os.path.join(Config.MANIFEST_DIR,
                                          Config.TRACKS["oct"]["manifest"])
    out_dir = a.out or os.path.join(Config.BASE_DIR, "outputs",
                                    "harmonization")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "latest.pth")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if device == "cpu":
        torch.set_num_threads(Config.TORCH_NUM_THREADS)

    print(f"\n{'='*72}\nCYCLEGAN HARMONISATION\n{'='*72}")
    print(f"  device      : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})"
             if device == "cuda" else ""))
    print(f"  mode        : {'SMOKE TEST' if a.smoke else 'full training'}")
    print(f"  image size  : {a.image_size}   batch {a.batch_size}   "
          f"blocks {a.blocks}")
    print(f"  epochs      : {a.epochs}  (LR decays from {a.decay_start})")
    print(f"  lambdas     : cycle {a.lambda_cycle}  "
          f"identity {a.lambda_identity}")
    print(f"  pool        : {a.pool_size}")
    print(f"  output      : {out_dir}")

    # ------------------------------------------------------------ data
    print(f"\n{'='*72}\nDATA\n{'='*72}")
    domains = load_domains(manifest, Config.DATA_ROOTS, a.seed)
    names = sorted(domains)
    print(f"\n  translating {names[0]} <-> {names[1]}")
    print(f"  domains are balanced per epoch by subsampling the larger one")
    print(f"  at the patient level, re-drawn each epoch")

    # ----------------------------------------------------------- models
    G_AB = ResNetGenerator(num_residual_blocks=a.blocks).to(device)
    G_BA = ResNetGenerator(num_residual_blocks=a.blocks).to(device)
    D_A = PatchGANDiscriminator().to(device)
    D_B = PatchGANDiscriminator().to(device)

    print(f"\n  generator params     : "
          f"{sum(x.numel() for x in G_AB.parameters()):,} each")
    print(f"  discriminator params : "
          f"{sum(x.numel() for x in D_A.parameters()):,} each")
    print(f"  PatchGAN receptive field: {D_A.get_receptive_field()}")

    crit_gan = nn.MSELoss()          # LSGAN
    crit_cyc = nn.L1Loss()
    crit_idt = nn.L1Loss()

    opt_G = torch.optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()),
                             lr=a.lr, betas=(0.5, 0.999))
    opt_DA = torch.optim.Adam(D_A.parameters(), lr=a.lr, betas=(0.5, 0.999))
    opt_DB = torch.optim.Adam(D_B.parameters(), lr=a.lr, betas=(0.5, 0.999))

    sched = [torch.optim.lr_scheduler.LambdaLR(
        o, lambda e: lr_lambda(e, a.epochs, a.decay_start))
        for o in (opt_G, opt_DA, opt_DB)]

    pool_A = ImagePool(a.pool_size, seed=a.seed)
    pool_B = ImagePool(a.pool_size, seed=a.seed + 1)

    # ---------------------------------------------------------- resume
    start_epoch, gstep = 0, 0
    if os.path.isfile(ckpt_path):
        print(f"\n  resuming from {ckpt_path}")
        try:
            ck = torch.load(ckpt_path, map_location=device,
                            weights_only=False)
        except Exception as e:
            # should not happen now that saves are atomic, but a checkpoint
            # from an older run, or one copied while being written, can
            # still be unreadable. Fall back rather than crash.
            print(f"  checkpoint unreadable ({e.__class__.__name__}); "
                  f"looking for an epoch snapshot")
            snaps = sorted(f for f in os.listdir(out_dir)
                           if f.startswith("generators_ep"))
            if not snaps:
                sys.exit("no usable checkpoint; delete latest.pth to start "
                         "over, or restore one from an earlier session")
            sys.exit(f"latest.pth is corrupt. The most recent usable "
                     f"snapshot is {snaps[-1]}, but it holds generators "
                     f"only -- discriminators and optimizer state are lost. "
                     f"Delete latest.pth and restart, or resume manually.")
        G_AB.load_state_dict(ck["G_AB"])
        G_BA.load_state_dict(ck["G_BA"])
        D_A.load_state_dict(ck["D_A"])
        D_B.load_state_dict(ck["D_B"])
        opt_G.load_state_dict(ck["opt_G"])
        opt_DA.load_state_dict(ck["opt_DA"])
        opt_DB.load_state_dict(ck["opt_DB"])
        for s, sd in zip(sched, ck["sched"]):
            s.load_state_dict(sd)
        # a cold pool on resume would reintroduce the oscillation the pool
        # exists to prevent, so its contents are checkpointed too
        pool_A.load_state_dict(ck["pool_A"])
        pool_B.load_state_dict(ck["pool_B"])
        start_epoch, gstep = ck["epoch"], ck["gstep"]
        print(f"  epoch {start_epoch}, iteration {gstep}, "
              f"pools {len(pool_A)}/{len(pool_B)}")

    log_path = os.path.join(out_dir, "losses.csv")
    if not os.path.isfile(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "iter", "loss_G", "loss_D_A", "loss_D_B",
                 "loss_gan", "loss_cycle", "loss_identity", "lr", "seconds"])

    rng = np.random.default_rng(a.seed)
    for _ in range(start_epoch):
        rng.permutation(4)                     # keep the draw reproducible

    print(f"\n{'='*72}\nTRAINING\n{'='*72}")
    t_start = time.time()
    na, nb = sorted(domains)          # defined even if the loop body never runs

    if start_epoch >= a.epochs:
        print(f"  checkpoint is already at epoch {start_epoch} of "
              f"{a.epochs}; nothing to do.")
        print(f"  Increase --epochs to continue, or delete {ckpt_path} "
              f"to start over.")

    for epoch in range(start_epoch, a.epochs):
        pa, pb, na, nb = balanced_epoch_paths(domains, rng)
        ds_A = DomainDataset(pa, Config.DATA_ROOTS, a.image_size)
        ds_B = DomainDataset(pb, Config.DATA_ROOTS, a.image_size)

        if a.smoke:
            ds_A.paths = ds_A.paths[:8]
            ds_B.paths = ds_B.paths[:8]

        ld_A = DataLoader(ds_A, batch_size=a.batch_size, shuffle=True,
                          num_workers=a.workers, drop_last=True)
        ld_B = DataLoader(ds_B, batch_size=a.batch_size, shuffle=True,
                          num_workers=a.workers, drop_last=True)
        n_iter = min(len(ld_A), len(ld_B))
        if n_iter == 0:
            sys.exit("no full batches; reduce --batch-size")

        print(f"\n  epoch {epoch+1}/{a.epochs}   "
              f"{na} {len(pa):,} <-> {nb} {len(pb):,}   "
              f"{n_iter} iterations   lr {opt_G.param_groups[0]['lr']:.2e}")

        it_A, it_B = iter(ld_A), iter(ld_B)
        ep_t = time.time()

        for i in range(n_iter):
            real_A = next(it_A).to(device)
            real_B = next(it_B).to(device)

            # ---- generators ----------------------------------------
            opt_G.zero_grad()

            # identity: G_AB applied to a B image should leave it alone
            loss_idt = (crit_idt(G_AB(real_B), real_B) +
                        crit_idt(G_BA(real_A), real_A)) * a.lambda_identity

            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)

            loss_gan = (crit_gan(D_B(fake_B), torch.ones_like(D_B(fake_B))) +
                        crit_gan(D_A(fake_A), torch.ones_like(D_A(fake_A))))

            rec_A = G_BA(fake_B)
            rec_B = G_AB(fake_A)
            loss_cyc = (crit_cyc(rec_A, real_A) +
                        crit_cyc(rec_B, real_B)) * a.lambda_cycle

            loss_G = loss_gan + loss_cyc + loss_idt
            loss_G.backward()
            opt_G.step()

            # ---- discriminators ------------------------------------
            # detach, and draw from the history buffer
            opt_DA.zero_grad()
            pred_real = D_A(real_A)
            pred_fake = D_A(pool_A.query(fake_A.detach()))
            loss_DA = 0.5 * (crit_gan(pred_real, torch.ones_like(pred_real)) +
                             crit_gan(pred_fake, torch.zeros_like(pred_fake)))
            loss_DA.backward()
            opt_DA.step()

            opt_DB.zero_grad()
            pred_real = D_B(real_B)
            pred_fake = D_B(pool_B.query(fake_B.detach()))
            loss_DB = 0.5 * (crit_gan(pred_real, torch.ones_like(pred_real)) +
                             crit_gan(pred_fake, torch.zeros_like(pred_fake)))
            loss_DB.backward()
            opt_DB.step()

            gstep += 1

            if i % max(1, n_iter // 10) == 0 or i == n_iter - 1:
                el = time.time() - ep_t
                rate = (i + 1) / el if el else 0
                print(f"    {i+1:>5}/{n_iter}  G {loss_G.item():7.3f}  "
                      f"D_A {loss_DA.item():6.3f}  D_B {loss_DB.item():6.3f}  "
                      f"cyc {loss_cyc.item():6.3f}  "
                      f"{rate:5.2f} it/s", flush=True)
                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow([
                        epoch, gstep, round(loss_G.item(), 4),
                        round(loss_DA.item(), 4), round(loss_DB.item(), 4),
                        round(loss_gan.item(), 4), round(loss_cyc.item(), 4),
                        round(loss_idt.item(), 4),
                        opt_G.param_groups[0]["lr"],
                        round(time.time() - t_start, 1)])

            if gstep % a.sample_every == 0:
                save_grid([real_A, fake_B, rec_A, real_B, fake_A, rec_B],
                          os.path.join(out_dir, "samples",
                                       f"iter_{gstep:07d}.png"),
                          n=min(4, a.batch_size))

            if gstep % a.save_every == 0:
                atomic_save({
                    "G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(),
                    "D_A": D_A.state_dict(), "D_B": D_B.state_dict(),
                    "opt_G": opt_G.state_dict(),
                    "opt_DA": opt_DA.state_dict(),
                    "opt_DB": opt_DB.state_dict(),
                    "sched": [s.state_dict() for s in sched],
                    "pool_A": pool_A.state_dict(),
                    "pool_B": pool_B.state_dict(),
                    "epoch": epoch, "gstep": gstep,
                    "domains": [na, nb], "args": vars(a),
                }, ckpt_path)

        for s in sched:
            s.step()

        # end-of-epoch checkpoint, plus a numbered copy every 10 epochs
        atomic_save({
            "G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(),
            "D_A": D_A.state_dict(), "D_B": D_B.state_dict(),
            "opt_G": opt_G.state_dict(), "opt_DA": opt_DA.state_dict(),
            "opt_DB": opt_DB.state_dict(),
            "sched": [s.state_dict() for s in sched],
            "pool_A": pool_A.state_dict(), "pool_B": pool_B.state_dict(),
            "epoch": epoch + 1, "gstep": gstep,
            "domains": [na, nb], "args": vars(a),
        }, ckpt_path)
        if (epoch + 1) % 10 == 0 or epoch + 1 == a.epochs:
            atomic_save({"G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(),
                        "epoch": epoch + 1, "domains": [na, nb],
                        "args": vars(a)},
                       os.path.join(out_dir, f"generators_ep{epoch+1:03d}.pth"))

        print(f"    epoch done in {timedelta(seconds=int(time.time()-ep_t))}")

    total = time.time() - t_start
    meta = {"finished": datetime.now().isoformat(timespec="seconds"),
            "device": device, "domains": [na, nb], "epochs": a.epochs,
            "iterations": gstep, "seconds": round(total, 1),
            "args": vars(a), "manifest": manifest,
            "note": ("trained on split == 'train' only; val and test were "
                     "never seen by the GAN")}
    with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*72}")
    print(f"  {gstep:,} iterations in {timedelta(seconds=int(total))}")
    print(f"  checkpoint : {ckpt_path}")
    print(f"  samples    : {os.path.join(out_dir, 'samples')}")
    print(f"  losses     : {log_path}")
    if a.smoke:
        print(f"\n  Smoke test passed. Delete {out_dir} before real training,")
        print(f"  or the run will resume from these 64px weights.")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()