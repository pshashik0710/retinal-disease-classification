#!/usr/bin/env python3
"""
image_pool.py -- history buffer of generated images for the discriminator.

Without this, the discriminator only ever sees the generator's newest
output. That makes the adversarial game non-stationary in a way that
oscillates: the generator can cycle between a few modes, "fooling" a
discriminator that has already forgotten the previous mode. The reference
CycleGAN keeps a buffer of 50 past outputs and shows the discriminator a
mixture of new and old, which damps the oscillation.

Its absence is a documented cause of instability and was one of the
missing pieces in the previous implementation in this project.

Behaviour, matching the reference:

    while the buffer is not full   -> store the image, return it
    once full, with p = 0.5        -> return a random stored image and
                                      replace it with the new one
                   otherwise       -> return the new image, store nothing

Usage:
    pool = ImagePool(50)
    fake_B = G_A2B(real_A)
    loss_D = criterion(D_B(pool.query(fake_B.detach())), target)

Always query with detached tensors. The buffer holds images across
iterations, so a retained graph would leak memory and produce gradients
through a generator state that no longer exists.
"""

from __future__ import annotations

import random

import torch


class ImagePool:
    """Buffer of previously generated images."""

    def __init__(self, pool_size: int = 50, seed: int | None = None):
        """
        pool_size: number of images to retain. 0 disables the buffer and
                   query() becomes a passthrough, which is useful for an
                   ablation showing what the buffer contributes.
        seed:      optional, for reproducible buffer behaviour. The pool's
                   randomness is separate from the model's, so seeding it
                   independently keeps a run repeatable without disturbing
                   weight initialisation.
        """
        if pool_size < 0:
            raise ValueError(f"pool_size must be >= 0, got {pool_size}")
        self.pool_size = pool_size
        self.images: list[torch.Tensor] = []
        self.rng = random.Random(seed) if seed is not None else random

    def __len__(self) -> int:
        return len(self.images)

    def query(self, images: torch.Tensor) -> torch.Tensor:
        """
        Return a batch for the discriminator: a mixture of the images just
        generated and images generated earlier.

        images: (B, C, H, W), detached.
        """
        if self.pool_size == 0:
            return images

        if images.dim() != 4:
            raise ValueError(f"expected (B, C, H, W), got {tuple(images.shape)}")
        if images.requires_grad:
            raise ValueError(
                "query() received a tensor that requires grad. Pass "
                "fake.detach() -- the buffer holds images across iterations, "
                "so a retained graph would leak memory and backpropagate "
                "into a generator state that no longer exists.")

        out = []
        for image in images:
            image = image.unsqueeze(0)

            if len(self.images) < self.pool_size:
                self.images.append(image.clone())
                out.append(image)
                continue

            if self.rng.uniform(0, 1) > 0.5:
                # return a stored image and replace it with this one
                i = self.rng.randint(0, self.pool_size - 1)
                stored = self.images[i].clone()
                self.images[i] = image.clone()
                out.append(stored)
            else:
                # return the current image, leave the buffer alone
                out.append(image)

        return torch.cat(out, dim=0)

    # -- checkpointing ---------------------------------------------------
    def state_dict(self) -> dict:
        """
        Buffer contents, so a resumed run continues with the same history
        rather than an empty pool. On Kaggle, where a session can die at
        any point in a 12-hour window, resuming with a cold buffer would
        reintroduce exactly the instability the buffer exists to prevent.
        """
        return {"pool_size": self.pool_size,
                "images": [im.cpu() for im in self.images]}

    def load_state_dict(self, state: dict) -> None:
        if state.get("pool_size") != self.pool_size:
            raise ValueError(
                f"checkpoint pool_size {state.get('pool_size')} does not "
                f"match this pool's {self.pool_size}")
        self.images = [im.clone() for im in state["images"]]


# =========================================================================
# SELF-TEST
# =========================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    ok = True

    def check(label, cond):
        global ok
        print(f"  {label:<52} {'OK' if cond else '*** FAIL'}")
        ok = ok and cond

    print(f"\n{'='*62}\nImagePool\n{'='*62}")

    # shape preserved
    pool = ImagePool(50, seed=0)
    x = torch.randn(4, 3, 64, 64)
    y = pool.query(x)
    check(f"shape preserved {tuple(x.shape)}", y.shape == x.shape)

    # fills, then stops growing
    pool = ImagePool(10, seed=0)
    for _ in range(5):
        pool.query(torch.randn(4, 3, 8, 8))
    check(f"buffer caps at pool_size (len={len(pool)})", len(pool) == 10)

    # while filling, output is the input unchanged
    pool = ImagePool(100, seed=0)
    x = torch.randn(4, 3, 8, 8)
    check("passthrough while filling", torch.equal(pool.query(x), x))

    # once full, roughly half the outputs come from history. Sampled over
    # a large batch: with only a handful of draws the binomial variance is
    # wide enough that all-new or all-history is unremarkable.
    # the pool must exceed the probe batch, or the probes themselves
    # displace the history and there is nothing left to return
    pool = ImagePool(500, seed=1)
    for _ in range(150):
        pool.query(torch.randn(4, 3, 8, 8))
    n_probe = 100
    probe = torch.full((n_probe, 3, 8, 8), 99.0)
    out = pool.query(probe)
    from_history = int((out.flatten(1) != 99.0).any(dim=1).sum())
    frac = from_history / n_probe
    check(f"mixes history once full ({from_history}/{n_probe} = "
          f"{frac:.0%}, expect ~50%)", 0.35 < frac < 0.65)

    # pool_size 0 disables
    pool = ImagePool(0)
    x = torch.randn(4, 3, 8, 8)
    check("pool_size=0 is a passthrough",
          torch.equal(pool.query(x), x) and len(pool) == 0)

    # rejects attached tensors
    pool = ImagePool(10)
    try:
        pool.query(torch.randn(2, 3, 8, 8, requires_grad=True))
        check("rejects tensors requiring grad", False)
    except ValueError:
        check("rejects tensors requiring grad", True)

    # rejects wrong rank
    try:
        pool.query(torch.randn(3, 8, 8))
        check("rejects non-4D input", False)
    except ValueError:
        check("rejects non-4D input", True)

    # checkpoint round-trip
    a = ImagePool(16, seed=7)
    for _ in range(8):
        a.query(torch.randn(4, 3, 8, 8))
    b = ImagePool(16, seed=7)
    b.load_state_dict(a.state_dict())
    check(f"checkpoint round-trip (len={len(b)})",
          len(b) == len(a)
          and all(torch.equal(p, q) for p, q in zip(a.images, b.images)))

    # mismatched size is refused
    try:
        ImagePool(8).load_state_dict(a.state_dict())
        check("refuses mismatched pool_size", False)
    except ValueError:
        check("refuses mismatched pool_size", True)

    # seeded pools agree
    p1, p2 = ImagePool(8, seed=3), ImagePool(8, seed=3)
    same = True
    for _ in range(6):
        x = torch.randn(2, 3, 4, 4)
        same &= torch.equal(p1.query(x), p2.query(x))
    check("seeded pools are reproducible", same)

    # stored images are copies, not views
    pool = ImagePool(4, seed=0)
    x = torch.zeros(1, 3, 4, 4)
    pool.query(x)
    x.fill_(7.0)
    check("stores clones, not references", float(pool.images[0].max()) == 0.0)

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}\n")
    raise SystemExit(0 if ok else 1)