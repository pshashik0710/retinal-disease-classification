# losses.py

import torch
import torch.nn as nn

from scripts.config import Config


# =========================================================
# ADVERSARIAL LOSS
# =========================================================

class AdversarialLoss(nn.Module):
    """
    GAN Loss

    Implements:
    L_GAN(G,D)

    Uses:
    - MSELoss (LSGAN style)
    - More stable than BCE for CycleGAN
    """

    def __init__(self):

        super().__init__()

        self.loss = nn.MSELoss()

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(
        self,
        prediction,
        target_is_real
    ):

        if target_is_real:

            target_tensor = torch.ones_like(prediction)

        else:

            target_tensor = torch.zeros_like(prediction)

        return self.loss(
            prediction,
            target_tensor
        )


# =========================================================
# CYCLE CONSISTENCY LOSS
# =========================================================

class CycleConsistencyLoss(nn.Module):
    """
    Cycle Consistency Loss

    Implements:

    ||F(G(x)) - x||1
    +
    ||G(F(y)) - y||1

    Preserves:
    - retinal structures
    - lesions
    - vessel consistency
    """

    def __init__(self):

        super().__init__()

        self.loss = nn.L1Loss()

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(
        self,
        reconstructed,
        real
    ):

        return self.loss(
            reconstructed,
            real
        )


# =========================================================
# IDENTITY LOSS
# =========================================================

class IdentityLoss(nn.Module):
    """
    Identity Loss

    Implements:

    ||G(y) - y||1
    +
    ||F(x) - x||1

    Important for:
    - medical image preservation
    - anatomical consistency
    - preventing hallucinations
    """

    def __init__(self):

        super().__init__()

        self.loss = nn.L1Loss()

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(
        self,
        same_image,
        real_image
    ):

        return self.loss(
            same_image,
            real_image
        )


# =========================================================
# FULL CYCLEGAN LOSS
# =========================================================

class CycleGANLoss:
    """
    Combines:
    - adversarial loss
    - cycle consistency loss
    - identity loss
    """

    def __init__(self):

        # ---------------------------------------------
        # INDIVIDUAL LOSSES
        # ---------------------------------------------

        self.gan_loss = AdversarialLoss()

        self.cycle_loss = CycleConsistencyLoss()

        self.identity_loss = IdentityLoss()

        # ---------------------------------------------
        # WEIGHTS
        # ---------------------------------------------

        self.lambda_cycle = Config.LAMBDA_CYCLE

        self.lambda_identity = Config.LAMBDA_IDENTITY

    # =====================================================
    # GENERATOR LOSS
    # =====================================================

    def generator_loss(
        self,

        # discriminator outputs
        pred_fake_B,
        pred_fake_A,

        # reconstructed images
        recov_A,
        recov_B,

        # real images
        real_A,
        real_B,

        # identity images
        same_A,
        same_B
    ):

        # ---------------------------------------------
        # ADVERSARIAL LOSS
        # ---------------------------------------------

        loss_GAN_A2B = self.gan_loss(
            pred_fake_B,
            True
        )

        loss_GAN_B2A = self.gan_loss(
            pred_fake_A,
            True
        )

        loss_GAN = (
            loss_GAN_A2B +
            loss_GAN_B2A
        )

        # ---------------------------------------------
        # CYCLE LOSS
        # ---------------------------------------------

        loss_cycle_A = self.cycle_loss(
            recov_A,
            real_A
        )

        loss_cycle_B = self.cycle_loss(
            recov_B,
            real_B
        )

        loss_cycle = (
            loss_cycle_A +
            loss_cycle_B
        )

        # ---------------------------------------------
        # IDENTITY LOSS
        # ---------------------------------------------

        loss_identity_A = self.identity_loss(
            same_A,
            real_A
        )

        loss_identity_B = self.identity_loss(
            same_B,
            real_B
        )

        loss_identity = (
            loss_identity_A +
            loss_identity_B
        )

        # ---------------------------------------------
        # TOTAL GENERATOR LOSS
        # ---------------------------------------------

        total_generator_loss = (

            loss_GAN +

            self.lambda_cycle * loss_cycle +

            self.lambda_identity * loss_identity
        )

        return {
            "total_generator_loss": total_generator_loss,

            "loss_GAN": loss_GAN,

            "loss_cycle": loss_cycle,

            "loss_identity": loss_identity
        }

    # =====================================================
    # DISCRIMINATOR LOSS
    # =====================================================

    def discriminator_loss(
        self,
        pred_real,
        pred_fake
    ):

        loss_real = self.gan_loss(
            pred_real,
            True
        )

        loss_fake = self.gan_loss(
            pred_fake,
            False
        )

        total_discriminator_loss = (
            loss_real +
            loss_fake
        ) * 0.5

        return total_discriminator_loss


# =========================================================
# MAIN TEST
# =========================================================

if __name__ == "__main__":

    device = Config.DEVICE

    criterion = CycleGANLoss()

    # ---------------------------------------------
    # FAKE TEST TENSORS
    # ---------------------------------------------

    pred_fake = torch.randn(
        2,
        1,
        30,
        30
    ).to(device)

    pred_real = torch.randn(
        2,
        1,
        30,
        30
    ).to(device)

    real_img = torch.randn(
        2,
        3,
        256,
        256
    ).to(device)

    fake_img = torch.randn(
        2,
        3,
        256,
        256
    ).to(device)

    # ---------------------------------------------
    # TEST DISCRIMINATOR LOSS
    # ---------------------------------------------

    d_loss = criterion.discriminator_loss(
        pred_real,
        pred_fake
    )

    print("\nDiscriminator Loss:")
    print(d_loss.item())

    # ---------------------------------------------
    # TEST GENERATOR LOSS
    # ---------------------------------------------

    g_losses = criterion.generator_loss(

        pred_fake_B=pred_fake,
        pred_fake_A=pred_fake,

        recov_A=fake_img,
        recov_B=fake_img,

        real_A=real_img,
        real_B=real_img,

        same_A=fake_img,
        same_B=fake_img
    )

    print("\nGenerator Losses:")

    for key, value in g_losses.items():

        print(f"{key}: {value.item()}")

    print("\nLoss functions initialized successfully.\n")