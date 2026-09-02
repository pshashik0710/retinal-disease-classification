#!/usr/bin/env python3
"""
models.py -- classifier and CycleGAN architectures.

Corrections against the previous version, each verified empirically:

  * ConvNeXtTiny is now parameterised (num_classes, dropout, model name,
    pretrained) instead of reading Config at construction time. The old
    version could not be instantiated with a different class count without
    editing config.py.

  * ConvNeXtTiny exposes forward_features() so a two-stream or fusion model
    can take 768-d features directly, rather than reaching in and replacing
    .classifier with nn.Identity() and hoping the attribute exists.

  * freeze_backbone() / unfreeze_backbone() replace the ad-hoc parameter
    loops in the training scripts. Those loops froze self.model and then
    "unfroze the head" via self.model.head -- but with timm's num_classes=0
    the head's fc is Identity, so that touched only 1,536 LayerNorm
    parameters and never the actual classifier, which lives in
    self.classifier.

  * PatchGANDiscriminator now really is the 70x70 PatchGAN. The previous
    version used four stride-2 blocks, giving a 94x94 receptive field and a
    15x15 output map, while config.py declared PATCHGAN_SIZE = 70 and
    losses.py tested against 30x30. The reference design uses three
    stride-2 blocks and one stride-1 block; that is what is implemented
    here, and get_receptive_field() reports the true value.

  * Generator dropout defaults to 0.0. The reference CycleGAN uses no
    dropout in the residual blocks for unpaired translation; the previous
    default of 0.5 in all nine blocks adds substantial noise.

  * Weight initialisation follows the reference: normal(0, 0.02) for conv
    and transposed-conv layers. Its absence is a known source of GAN
    instability.

  * num_residual_blocks is a real argument rather than a hardcoded default
    that silently ignored Config.RESIDUAL_BLOCKS.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    import timm
except ImportError:  # pragma: no cover
    timm = None


# =========================================================================
# CLASSIFIER
# =========================================================================

class ConvNeXtTiny(nn.Module):
    """
    timm backbone with a fresh classifier head.

    The backbone is created with num_classes=0, so it returns pooled
    features (768 for convnext_tiny). The head here is a plain
    Dropout + Linear on top of that.
    """

    def __init__(
        self,
        num_classes: int = 4,
        dropout: float = 0.3,
        model_name: str = "convnext_tiny",
        pretrained: bool = True,
        in_chans: int = 3,
    ):
        super().__init__()

        if timm is None:
            raise ImportError("timm is required: pip install timm")

        self.model_name = model_name
        self.num_classes = num_classes

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
        )

        self.num_features = self.model.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.num_features, num_classes),
        )

    # -- features only, for fusion / probing / feature caching ------------
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled backbone features, shape (B, num_features)."""
        return self.model(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.model(x))

    # -- explicit freezing, so the training loop does not guess -----------
    def freeze_backbone(self) -> None:
        """Freeze the timm backbone. The classifier head stays trainable."""
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True

    def unfreeze_backbone(self) -> None:
        for p in self.parameters():
            p.requires_grad = True

    def param_groups(self, backbone_lr: float, head_lr: float):
        """
        Two learning rates: a small one for pretrained weights, a larger one
        for the new head. Pass straight to an optimizer.

        Rebuild this after unfreezing -- an optimizer built while the
        backbone was frozen holds no reference to those parameters, so
        unfreezing alone will not start training them. Groups with no
        trainable parameters are omitted, since some optimizers object.
        """
        groups = []
        backbone = [p for p in self.model.parameters() if p.requires_grad]
        if backbone:
            groups.append({"params": backbone, "lr": backbone_lr})
        head = [p for p in self.classifier.parameters() if p.requires_grad]
        if head:
            groups.append({"params": head, "lr": head_lr})
        if not groups:
            raise ValueError("no trainable parameters; call "
                             "unfreeze_backbone() or check freezing logic")
        return groups

    def trainable_parameter_count(self) -> tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return train, total


# =========================================================================
# WEIGHT INIT (reference CycleGAN)
# =========================================================================

def init_weights(net: nn.Module, gain: float = 0.02) -> nn.Module:
    """normal(0, gain) on conv / transposed-conv / affine-norm weights."""
    def _init(m):
        cn = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in cn or "Linear" in cn):
            nn.init.normal_(m.weight.data, 0.0, gain)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "InstanceNorm" in cn or "BatchNorm" in cn:
            if getattr(m, "weight", None) is not None:
                nn.init.normal_(m.weight.data, 1.0, gain)
            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias.data, 0.0)

    net.apply(_init)
    return net


# =========================================================================
# CYCLEGAN GENERATOR
# =========================================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """
    ResNet generator, 9 residual blocks by default (the reference setting
    for 256x256 inputs; use 6 for 128x128). Output is tanh, so images must
    be normalised to [-1, 1] on the way in and denormalised with (x+1)/2
    on the way out.
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_channels: int = 3,
        num_residual_blocks: int = 9,
        base_filters: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_residual_blocks = num_residual_blocks
        self.dropout = dropout

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, base_filters, kernel_size=7),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True),
        ]

        c = base_filters
        for _ in range(2):                       # downsample x4
            layers += [
                nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(c * 2),
                nn.ReLU(inplace=True),
            ]
            c *= 2

        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(c, dropout=dropout))

        for _ in range(2):                       # upsample x4
            layers += [
                nn.ConvTranspose2d(c, c // 2, kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.InstanceNorm2d(c // 2),
                nn.ReLU(inplace=True),
            ]
            c //= 2

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(c, output_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)
        init_weights(self)

    def forward(self, x):
        return self.model(x)


# =========================================================================
# CYCLEGAN DISCRIMINATOR
# =========================================================================

class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN (n_layers=3).

    Layout: C64(s2) - C128(s2) - C256(s2) - C512(s1) - conv(s1).
    Note the FOURTH block is stride 1. Making it stride 2 -- as the previous
    version did -- gives a 94x94 receptive field and a 15x15 output map for
    a 256x256 input, not the documented 70x70 / 30x30.
    """

    def __init__(
        self,
        input_channels: int = 3,
        base_filters: int = 64,
        n_layers: int = 3,
    ):
        super().__init__()
        self.n_layers = n_layers

        def block(i, o, stride, normalize=True):
            layers = [nn.Conv2d(i, o, kernel_size=4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(o))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = block(input_channels, base_filters, 2, normalize=False)

        c = base_filters
        for n in range(1, n_layers):             # stride-2 blocks
            layers += block(c, min(c * 2, 512), 2)
            c = min(c * 2, 512)

        layers += block(c, min(c * 2, 512), 1)   # stride-1 block
        c = min(c * 2, 512)

        layers.append(nn.Conv2d(c, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)
        init_weights(self)

    def forward(self, x):
        return self.model(x)

    def get_receptive_field(self) -> int:
        """Actual receptive field in pixels, computed from the layer stack."""
        convs = []
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                convs.append((m.kernel_size[0], m.stride[0]))
        r = 1
        for k, s in reversed(convs):
            r = (r - 1) * s + k
        return r


# =========================================================================
# SELF-TEST
# =========================================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--num-classes", type=int, default=4)
    ap.add_argument("--gan-size", type=int, default=256)
    ap.add_argument("--pretrained", action="store_true",
                    help="download ImageNet weights (needs network)")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\ndevice: {device}   torch: {torch.__version__}\n")
    ok = True

    # ---- classifier -----------------------------------------------------
    print("=" * 66)
    print("ConvNeXtTiny")
    print("=" * 66)
    clf = ConvNeXtTiny(num_classes=a.num_classes,
                       pretrained=a.pretrained).to(device)
    x = torch.randn(2, 3, a.image_size, a.image_size, device=device)

    feats = clf.forward_features(x)
    out = clf(x)
    print(f"  features {tuple(feats.shape)}  ->  logits {tuple(out.shape)}")
    assert feats.shape == (2, clf.num_features), "bad feature shape"
    assert out.shape == (2, a.num_classes), "bad logit shape"

    tr, tot = clf.trainable_parameter_count()
    print(f"  all trainable      : {tr:,} / {tot:,}")
    clf.freeze_backbone()
    tr, tot = clf.trainable_parameter_count()
    print(f"  backbone frozen    : {tr:,} / {tot:,}  (classifier only)")
    assert tr == sum(p.numel() for p in clf.classifier.parameters()), \
        "freeze_backbone left the wrong parameters trainable"
    clf.unfreeze_backbone()
    tr, _ = clf.trainable_parameter_count()
    print(f"  backbone unfrozen  : {tr:,} / {tot:,}")
    assert tr == tot

    clf.freeze_backbone()
    g = clf.param_groups(backbone_lr=1e-5, head_lr=1e-4)
    print(f"  frozen  -> {len(g)} group(s): "
          + ", ".join(f"{sum(p.numel() for p in x['params']):,} @ {x['lr']}"
                      for x in g))
    assert len(g) == 1, "frozen backbone should yield one group"
    clf.unfreeze_backbone()
    g = clf.param_groups(backbone_lr=1e-5, head_lr=1e-4)
    print(f"  unfrozen-> {len(g)} group(s): "
          + ", ".join(f"{sum(p.numel() for p in x['params']):,} @ {x['lr']}"
                      for x in g))
    assert len(g) == 2, "unfrozen backbone should yield two groups"

    # backward pass actually runs
    clf(x).sum().backward()
    print("  backward pass      : OK")

    # ---- generator ------------------------------------------------------
    print("\n" + "=" * 66)
    print("ResNetGenerator")
    print("=" * 66)
    gen = ResNetGenerator(num_residual_blocks=9).to(device)
    xg = torch.randn(1, 3, a.gan_size, a.gan_size, device=device)
    fake = gen(xg)
    print(f"  in {tuple(xg.shape)} -> out {tuple(fake.shape)}")
    assert fake.shape == xg.shape, "generator must preserve spatial size"
    print(f"  output range       : [{fake.min():.3f}, {fake.max():.3f}] "
          f"(tanh, expect within [-1, 1])")
    assert fake.min() >= -1.001 and fake.max() <= 1.001
    print(f"  residual blocks    : {gen.num_residual_blocks}")
    print(f"  dropout            : {gen.dropout}")
    print(f"  parameters         : {sum(p.numel() for p in gen.parameters()):,}")

    # ---- discriminator --------------------------------------------------
    print("\n" + "=" * 66)
    print("PatchGANDiscriminator")
    print("=" * 66)
    disc = PatchGANDiscriminator().to(device)
    pred = disc(fake)
    rf = disc.get_receptive_field()
    print(f"  in {tuple(fake.shape)} -> out {tuple(pred.shape)}")
    print(f"  receptive field    : {rf}x{rf}")
    print(f"  parameters         : {sum(p.numel() for p in disc.parameters()):,}")
    if rf != 70:
        print(f"  *** expected 70, got {rf}")
        ok = False
    if a.gan_size == 256 and tuple(pred.shape) != (1, 1, 30, 30):
        print(f"  *** expected (1,1,30,30) for 256px input, "
              f"got {tuple(pred.shape)}")
        ok = False

    # ---- gradient flow through the full GAN step ------------------------
    print("\n" + "=" * 66)
    print("GAN gradient flow")
    print("=" * 66)
    g2 = ResNetGenerator(num_residual_blocks=2).to(device)   # small, fast
    d2 = PatchGANDiscriminator().to(device)
    xs = torch.randn(1, 3, 128, 128, device=device)
    f = g2(xs)
    loss_g = nn.MSELoss()(d2(f), torch.ones_like(d2(f)))
    loss_g.backward()
    gn = sum(p.grad.abs().sum().item() for p in g2.parameters()
             if p.grad is not None)
    print(f"  generator grad norm: {gn:.4f}  "
          f"{'OK' if gn > 0 else '*** NO GRADIENT'}")
    if gn <= 0:
        ok = False

    d2.zero_grad()
    loss_d = nn.MSELoss()(d2(f.detach()), torch.zeros_like(d2(f.detach())))
    loss_d.backward()
    dn = sum(p.grad.abs().sum().item() for p in d2.parameters()
             if p.grad is not None)
    print(f"  discrim.  grad norm: {dn:.4f}  "
          f"{'OK' if dn > 0 else '*** NO GRADIENT'}")
    if dn <= 0:
        ok = False

    print("\n" + "=" * 66)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    print("=" * 66 + "\n")
    raise SystemExit(0 if ok else 1)