# train_cyclegan.py

import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from scripts.config import Config

from cyclegan_dataset import CycleGANDataset
from cyclegan_transforms import get_transforms

from scripts.models import ResNetGenerator
from scripts.models import PatchGANDiscriminator

from losses import CycleGANLoss


# =========================================================
# DEVICE
# =========================================================

device = Config.DEVICE

print(f"\nUsing Device: {device}\n")


# =========================================================
# DIRECTORIES
# =========================================================

CHECKPOINT_DIR = "checkpoints"

GENERATED_DIR = "generated_images"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

os.makedirs(GENERATED_DIR, exist_ok=True)


# =========================================================
# CHECKPOINT FILES
# =========================================================

LATEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "latest_cyclegan_checkpoint.pth"
)

BEST_GENERATOR_A2B = os.path.join(
    CHECKPOINT_DIR,
    "G_A2B.pth"
)

BEST_GENERATOR_B2A = os.path.join(
    CHECKPOINT_DIR,
    "G_B2A.pth"
)

BEST_DISCRIMINATOR_A = os.path.join(
    CHECKPOINT_DIR,
    "D_A.pth"
)

BEST_DISCRIMINATOR_B = os.path.join(
    CHECKPOINT_DIR,
    "D_B.pth"
)


# =========================================================
# TRANSFORMS
# =========================================================

transform = get_transforms(
    image_size=Config.IMAGE_SIZE
)


# =========================================================
# DATASET
# =========================================================

dataset = CycleGANDataset(

    root_dir="/home/user24/retinal_project/dataset",

    mode="train",

    transform=transform,

    image_size=Config.IMAGE_SIZE
)


# =========================================================
# DATALOADER
# =========================================================

train_loader = DataLoader(

    dataset,

    batch_size=Config.CYCLEGAN_BATCH_SIZE,

    shuffle=True,

    num_workers=4,

    pin_memory=True,

    persistent_workers=True
)


# =========================================================
# INITIALIZE MODELS
# =========================================================

# Fundus → OCT
G_A2B = ResNetGenerator().to(device)

# OCT → Fundus
G_B2A = ResNetGenerator().to(device)

# Fundus Discriminator
D_A = PatchGANDiscriminator().to(device)

# OCT Discriminator
D_B = PatchGANDiscriminator().to(device)


# =========================================================
# OPTIMIZERS
# =========================================================

optimizer_G = torch.optim.Adam(

    list(G_A2B.parameters()) +
    list(G_B2A.parameters()),

    lr=Config.CYCLEGAN_LR,

    betas=(0.5, 0.999)
)

optimizer_D_A = torch.optim.Adam(

    D_A.parameters(),

    lr=Config.CYCLEGAN_LR,

    betas=(0.5, 0.999)
)

optimizer_D_B = torch.optim.Adam(

    D_B.parameters(),

    lr=Config.CYCLEGAN_LR,

    betas=(0.5, 0.999)
)


# =========================================================
# LOSS FUNCTION
# =========================================================

criterion = CycleGANLoss()


# =========================================================
# RESUME TRAINING
# =========================================================

start_epoch = 0

start_batch = 0

best_G_loss = float("inf")

if os.path.exists(LATEST_CHECKPOINT):

    print("\n========================================")
    print("Resuming Previous CycleGAN Training...")
    print("========================================\n")

    checkpoint = torch.load(

        LATEST_CHECKPOINT,

        map_location=device,

        weights_only=False
    )

    # -----------------------------------------------------
    # LOAD MODELS
    # -----------------------------------------------------

    G_A2B.load_state_dict(
        checkpoint["G_A2B"]
    )

    G_B2A.load_state_dict(
        checkpoint["G_B2A"]
    )

    D_A.load_state_dict(
        checkpoint["D_A"]
    )

    D_B.load_state_dict(
        checkpoint["D_B"]
    )

    # -----------------------------------------------------
    # LOAD OPTIMIZERS
    # -----------------------------------------------------

    optimizer_G.load_state_dict(
        checkpoint["optimizer_G"]
    )

    optimizer_D_A.load_state_dict(
        checkpoint["optimizer_D_A"]
    )

    optimizer_D_B.load_state_dict(
        checkpoint["optimizer_D_B"]
    )

    # -----------------------------------------------------
    # LOAD TRAINING INFO
    # -----------------------------------------------------

    start_epoch = checkpoint["epoch"]

    start_batch = checkpoint.get(
        "batch_idx",
        0
    ) + 10

    best_G_loss = checkpoint["best_G_loss"]

    print(f"Resumed From Epoch: {start_epoch+1}")

    print(f"Resumed From Batch: {start_batch}")

    print(
        f"Best Generator Loss: "
        f"{best_G_loss:.4f}\n"
    )

else:

    print("\nStarting Fresh CycleGAN Training...\n")


# =========================================================
# TRAINING LOOP
# =========================================================

try:

    for epoch in range(

        start_epoch,

        Config.CYCLEGAN_EPOCHS
    ):

        epoch_start_time = time.time()

        G_A2B.train()
        G_B2A.train()

        D_A.train()
        D_B.train()

        running_G_loss = 0.0
        running_D_A_loss = 0.0
        running_D_B_loss = 0.0

        # =================================================
        # BATCH LOOP
        # =================================================

        for batch_idx, batch in enumerate(train_loader):

            # -------------------------------------------------
            # SKIP COMPLETED BATCHES
            # -------------------------------------------------

            if (
                epoch == start_epoch
                and
                batch_idx < start_batch
            ):

                continue

            # =================================================
            # LOAD REAL IMAGES
            # =================================================

            real_A = batch["X"].to(device)

            real_B = batch["Y"].to(device)

            # =================================================
            # TRAIN GENERATORS
            # =================================================

            optimizer_G.zero_grad()

            # ---------------------------------------------
            # IDENTITY LOSS
            # ---------------------------------------------

            same_B = G_A2B(real_B)

            same_A = G_B2A(real_A)

            # ---------------------------------------------
            # FORWARD TRANSLATION
            # ---------------------------------------------

            fake_B = G_A2B(real_A)

            fake_A = G_B2A(real_B)

            # ---------------------------------------------
            # CYCLE RECONSTRUCTION
            # ---------------------------------------------

            recov_A = G_B2A(fake_B)

            recov_B = G_A2B(fake_A)

            # ---------------------------------------------
            # DISCRIMINATOR PREDICTIONS
            # ---------------------------------------------

            pred_fake_B = D_B(fake_B)

            pred_fake_A = D_A(fake_A)

            # ---------------------------------------------
            # GENERATOR LOSSES
            # ---------------------------------------------

            generator_losses = criterion.generator_loss(

                pred_fake_B=pred_fake_B,

                pred_fake_A=pred_fake_A,

                recov_A=recov_A,

                recov_B=recov_B,

                real_A=real_A,

                real_B=real_B,

                same_A=same_A,

                same_B=same_B
            )

            loss_G = generator_losses[
                "total_generator_loss"
            ]

            loss_G.backward()

            optimizer_G.step()

            # =================================================
            # TRAIN DISCRIMINATOR A
            # =================================================

            optimizer_D_A.zero_grad()

            pred_real_A = D_A(real_A)

            pred_fake_A = D_A(
                fake_A.detach()
            )

            loss_D_A = criterion.discriminator_loss(

                pred_real_A,

                pred_fake_A
            )

            loss_D_A.backward()

            optimizer_D_A.step()

            # =================================================
            # TRAIN DISCRIMINATOR B
            # =================================================

            optimizer_D_B.zero_grad()

            pred_real_B = D_B(real_B)

            pred_fake_B = D_B(
                fake_B.detach()
            )

            loss_D_B = criterion.discriminator_loss(

                pred_real_B,

                pred_fake_B
            )

            loss_D_B.backward()

            optimizer_D_B.step()

            # =================================================
            # UPDATE LOSSES
            # =================================================

            running_G_loss += loss_G.item()

            running_D_A_loss += loss_D_A.item()

            running_D_B_loss += loss_D_B.item()

            # =====================================================
            # SAVE BATCH CHECKPOINT
            # =====================================================

            if batch_idx % 10 == 0:

                torch.save({

                    "epoch": epoch,

                    "batch_idx": batch_idx,

                    "best_G_loss": best_G_loss,

                    "G_A2B": G_A2B.state_dict(),

                    "G_B2A": G_B2A.state_dict(),

                    "D_A": D_A.state_dict(),

                    "D_B": D_B.state_dict(),

                    "optimizer_G":
                        optimizer_G.state_dict(),

                    "optimizer_D_A":
                        optimizer_D_A.state_dict(),

                    "optimizer_D_B":
                        optimizer_D_B.state_dict()

                }, LATEST_CHECKPOINT)

            # =================================================
            # PRINT STATUS
            # =================================================

            if batch_idx % 10 == 0:

                print(

                    f"Epoch "
                    f"[{epoch+1}/{Config.CYCLEGAN_EPOCHS}] "

                    f"Batch "
                    f"[{batch_idx}/{len(train_loader)}] "

                    f"G Loss: {loss_G.item():.4f} "

                    f"D_A Loss: {loss_D_A.item():.4f} "

                    f"D_B Loss: {loss_D_B.item():.4f}"
                )

        # =====================================================
        # EPOCH AVERAGES
        # =====================================================

        avg_G_loss = (

            running_G_loss /
            len(train_loader)
        )

        avg_D_A_loss = (

            running_D_A_loss /
            len(train_loader)
        )

        avg_D_B_loss = (

            running_D_B_loss /
            len(train_loader)
        )

        epoch_time = time.time() - epoch_start_time

        # =====================================================
        # SAVE SAMPLE IMAGES
        # =====================================================

        save_image(

            fake_B[:4],

            os.path.join(
                GENERATED_DIR,
                f"fake_B_epoch_{epoch+1}.png"
            ),

            normalize=True
        )

        save_image(

            fake_A[:4],

            os.path.join(
                GENERATED_DIR,
                f"fake_A_epoch_{epoch+1}.png"
            ),

            normalize=True
        )

        # =====================================================
        # SAVE BEST MODELS
        # =====================================================

        if avg_G_loss < best_G_loss:

            best_G_loss = avg_G_loss

            torch.save(

                G_A2B.state_dict(),

                BEST_GENERATOR_A2B
            )

            torch.save(

                G_B2A.state_dict(),

                BEST_GENERATOR_B2A
            )

            torch.save(

                D_A.state_dict(),

                BEST_DISCRIMINATOR_A
            )

            torch.save(

                D_B.state_dict(),

                BEST_DISCRIMINATOR_B
            )

            print("\nBest CycleGAN Models Saved.\n")

        # =====================================================
        # SAVE LATEST CHECKPOINT
        # =====================================================

        torch.save({

            "epoch": epoch,

            "batch_idx": batch_idx,

            "best_G_loss": best_G_loss,

            "G_A2B": G_A2B.state_dict(),

            "G_B2A": G_B2A.state_dict(),

            "D_A": D_A.state_dict(),

            "D_B": D_B.state_dict(),

            "optimizer_G": optimizer_G.state_dict(),

            "optimizer_D_A": optimizer_D_A.state_dict(),

            "optimizer_D_B": optimizer_D_B.state_dict()

        }, LATEST_CHECKPOINT)

        # =====================================================
        # EPOCH SUMMARY
        # =====================================================

        print("\n=================================================")

        print(f"Epoch {epoch+1} Completed")

        print(
            f"Generator Loss       : "
            f"{avg_G_loss:.4f}"
        )

        print(
            f"Discriminator A Loss : "
            f"{avg_D_A_loss:.4f}"
        )

        print(
            f"Discriminator B Loss : "
            f"{avg_D_B_loss:.4f}"
        )

        print(
            f"Epoch Time           : "
            f"{epoch_time/60:.2f} minutes"
        )

        print("=================================================\n")


# =========================================================
# CTRL + C SAFE EXIT
# =========================================================

except KeyboardInterrupt:

    print("\n========================================")
    print("Training Interrupted.")
    print("Saving Emergency Checkpoint...")
    print("========================================\n")

    torch.save({

        "epoch": epoch,

        "batch_idx": batch_idx,

        "best_G_loss": best_G_loss,

        "G_A2B": G_A2B.state_dict(),

        "G_B2A": G_B2A.state_dict(),

        "D_A": D_A.state_dict(),

        "D_B": D_B.state_dict(),

        "optimizer_G": optimizer_G.state_dict(),

        "optimizer_D_A": optimizer_D_A.state_dict(),

        "optimizer_D_B": optimizer_D_B.state_dict()

    }, LATEST_CHECKPOINT)

    print("\nCheckpoint Saved Successfully.\n")


# =========================================================
# TRAINING COMPLETE
# =========================================================

print("\n========================================")

print("CycleGAN Training Completed Successfully.")

print("========================================\n")