# generate_synthetic_images.py

import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from scripts.config import Config

from cyclegan_dataset import CycleGANDataset
from cyclegan_transforms import get_transforms

from scripts.models import ResNetGenerator


# =========================================================
# DEVICE
# =========================================================

device = Config.DEVICE

print(f"\nUsing Device: {device}\n")


# =========================================================
# OUTPUT DIRECTORIES
# =========================================================

synthetic_oct_dir = os.path.join(
    "generated_images",
    "synthetic_OCT"
)

synthetic_fundus_dir = os.path.join(
    "generated_images",
    "synthetic_Fundus"
)

os.makedirs(synthetic_oct_dir, exist_ok=True)

os.makedirs(synthetic_fundus_dir, exist_ok=True)


# =========================================================
# CHECKPOINT DIRECTORY
# =========================================================

CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

GENERATION_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "synthetic_generation_checkpoint.pth"
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

loader = DataLoader(

    dataset,

    batch_size=1,

    shuffle=False,

    num_workers=4,

    pin_memory=True,

    persistent_workers=True
)


# =========================================================
# LOAD GENERATORS
# =========================================================

# Fundus → OCT
G_A2B = ResNetGenerator().to(device)

# OCT → Fundus
G_B2A = ResNetGenerator().to(device)


# =========================================================
# LOAD TRAINED WEIGHTS
# =========================================================

print("Loading Trained CycleGAN Generators...\n")

G_A2B.load_state_dict(

    torch.load(

        "checkpoints/G_A2B.pth",

        map_location=device,

        weights_only=False
    )
)

G_B2A.load_state_dict(

    torch.load(

        "checkpoints/G_B2A.pth",

        map_location=device,

        weights_only=False
    )
)

print("Generators Loaded Successfully.\n")


# =========================================================
# EVALUATION MODE
# =========================================================

G_A2B.eval()

G_B2A.eval()


# =========================================================
# RESUME GENERATION
# =========================================================

start_idx = 0

if os.path.exists(GENERATION_CHECKPOINT):

    print("========================================")
    print("Resuming Synthetic Image Generation...")
    print("========================================\n")

    checkpoint = torch.load(

        GENERATION_CHECKPOINT,

        map_location=device,

        weights_only=False
    )

    start_idx = checkpoint.get(
        "last_generated_idx",
        0
    ) + 1

    print(
        f"Resuming From Image Index: "
        f"{start_idx}\n"
    )

else:

    print("Starting Fresh Synthetic Generation...\n")


# =========================================================
# GENERATION LOOP
# =========================================================

print("========================================")
print("Generating Synthetic Images...")
print("========================================\n")

start_time = time.time()

try:

    with torch.no_grad():

        for idx, batch in enumerate(loader):

            # -------------------------------------------------
            # SKIP GENERATED IMAGES
            # -------------------------------------------------

            if idx < start_idx:

                continue

            # -------------------------------------------------
            # LOAD REAL IMAGES
            # -------------------------------------------------

            real_A = batch["X"].to(device)

            real_B = batch["Y"].to(device)

            # -------------------------------------------------
            # GENERATE SYNTHETIC IMAGES
            # -------------------------------------------------

            fake_B = G_A2B(real_A)

            fake_A = G_B2A(real_B)

            # -------------------------------------------------
            # DENORMALIZE [-1,1] → [0,1]
            # -------------------------------------------------

            fake_B = (fake_B + 1) / 2

            fake_A = (fake_A + 1) / 2

            # -------------------------------------------------
            # SAVE SYNTHETIC OCT
            # -------------------------------------------------

            save_image(

                fake_B,

                os.path.join(

                    synthetic_oct_dir,

                    f"synthetic_OCT_{idx+1}.png"
                )
            )

            # -------------------------------------------------
            # SAVE SYNTHETIC FUNDUS
            # -------------------------------------------------

            save_image(

                fake_A,

                os.path.join(

                    synthetic_fundus_dir,

                    f"synthetic_Fundus_{idx+1}.png"
                )
            )

            # =====================================================
            # SAVE GENERATION CHECKPOINT
            # =====================================================

            if idx % 100 == 0:

                torch.save({

                    "last_generated_idx": idx,

                    "timestamp": time.time()

                }, GENERATION_CHECKPOINT)

            # -------------------------------------------------
            # PRINT PROGRESS
            # -------------------------------------------------

            if idx % 100 == 0:

                elapsed = (
                    time.time() - start_time
                ) / 60

                print(

                    f"Generated "
                    f"{idx+1}/{len(loader)} images | "

                    f"Elapsed Time: "
                    f"{elapsed:.2f} mins"
                )


# =========================================================
# CTRL + C SAFE EXIT
# =========================================================

except KeyboardInterrupt:

    print("\n========================================")
    print("Generation Interrupted.")
    print("Saving Recovery Checkpoint...")
    print("========================================\n")

    torch.save({

        "last_generated_idx": idx,

        "timestamp": time.time()

    }, GENERATION_CHECKPOINT)

    print("Checkpoint Saved Successfully.\n")


# =========================================================
# COMPLETION MESSAGE
# =========================================================

total_time = (
    time.time() - start_time
) / 60

print("\n========================================")

print("Synthetic Image Generation Completed.")

print(f"Total Time: {total_time:.2f} minutes")

print("\nImages Saved In:")

print(f"-> {synthetic_oct_dir}")

print(f"-> {synthetic_fundus_dir}")

print("========================================\n")