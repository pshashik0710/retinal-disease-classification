# train_two_stream.py

import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import time

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder

from PIL import Image

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from scripts.config import Config

from scripts.models import ConvNeXtTiny


# =========================================================
# DEVICE
# =========================================================

device = Config.DEVICE

print(f"\nUsing Device: {device}\n")

# =========================================================
# HYPERPARAMETERS
# =========================================================

NUM_CLASSES = 4

NUM_EPOCHS = 8

BATCH_SIZE = 32

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

DROPOUT_RATE = 0.6

LABEL_SMOOTHING = 0.2

NUM_WORKERS = 4

IMAGE_SIZE = 224

FREEZE_EPOCHS = 6

EARLY_STOPPING_PATIENCE = 4

RANDOM_ROTATION = 10


# =========================================================
# DIRECTORIES
# =========================================================

CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_two_stream_model.pth"
)

LATEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "latest_two_stream_checkpoint.pth"
)


# =========================================================
# TRANSFORMS
# =========================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(
         RANDOM_ROTATION
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# CUSTOM TWO-STREAM DATASET
# =========================================================

class TwoStreamDataset(Dataset):

    def __init__(

        self,

        real_root,

        synthetic_root,

        transform=None
    ):

        self.real_dataset = ImageFolder(
            real_root
        )

        self.synthetic_root = synthetic_root

        self.transform = transform

    def __len__(self):

        return len(self.real_dataset)

    def __getitem__(self, idx):

        real_path, label = (
            self.real_dataset.samples[idx]
        )

        filename = os.path.basename(
            real_path
        )

        synthetic_path = os.path.join(

            self.synthetic_root,

            f"synthetic_OCT_{idx+1}.png"
        )

        real_image = Image.open(
            real_path
        ).convert("RGB")

        synthetic_image = Image.open(
            synthetic_path
        ).convert("RGB")

        if self.transform:

            real_image = self.transform(
                real_image
            )

            synthetic_image = self.transform(
                synthetic_image
            )

        return (

            real_image,

            synthetic_image,

            label
        )


# =========================================================
# DATASET PATHS
# =========================================================

REAL_DATASET_PATH = (
    "/home/user24/retinal_project/datasets/processed/oct2017/train"
)

SYNTHETIC_DATASET_PATH = (
    "/home/user24/retinal_project/generated_images/synthetic_OCT"
)


# =========================================================
# DATASET
# =========================================================

train_dataset = TwoStreamDataset(

    real_root=REAL_DATASET_PATH,

    synthetic_root=SYNTHETIC_DATASET_PATH,

    transform=transform
)

# =========================================================
# VALIDATION DATASET
# =========================================================

VAL_DATASET_PATH = (
    "/home/user24/retinal_project/datasets/processed/oct2017/val"
)

val_dataset = ImageFolder(

    VAL_DATASET_PATH,

    transform=transform
)

# =========================================================
# DATALOADER
# =========================================================

train_loader = DataLoader(

    train_dataset,

    batch_size= BATCH_SIZE,

    shuffle=True,

    num_workers= NUM_WORKERS,

    pin_memory=True,

    persistent_workers=True
)

# =========================================================
# VALIDATION LOADER
# =========================================================

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=True
)

# =========================================================
# TWO-STREAM MODEL
# =========================================================

# =========================================================
# TWO-STREAM MODEL
# =========================================================

class TwoStreamFusionModel(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        # -------------------------------------------------
        # SHARED BACKBONE
        # -------------------------------------------------

        self.backbone = ConvNeXtTiny()

        # -------------------------------------------------
        # REMOVE FINAL CLASSIFIER
        # -------------------------------------------------

        if hasattr(self.backbone, "classifier"):

            # classifier is Sequential
            last_layer = list(
                self.backbone.classifier.children()
            )[-1]

            feature_dim = last_layer.in_features

            self.backbone.classifier = nn.Identity()

        elif hasattr(self.backbone, "fc"):

            feature_dim = (
                self.backbone.fc.in_features
            )

            self.backbone.fc = nn.Identity()

        else:

            feature_dim = 1000

        
        # -------------------------------------------------
        # FUSION CLASSIFIER
        # -------------------------------------------------

        self.fusion_layer = nn.Sequential(

            nn.Linear(
                feature_dim * 2,
                512
            ),

            nn.ReLU(),

            nn.Dropout(DROPOUT_RATE),

            nn.Linear(
                512,
                num_classes
            )
        )

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(

        self,

        real_image,

        synthetic_image
    ):

        # -------------------------------------------------
        # FEATURE EXTRACTION
        # -------------------------------------------------

        feat_real = self.backbone(
            real_image
        )

        feat_syn = self.backbone(
            synthetic_image
        )

        # -------------------------------------------------
        # FEATURE FUSION
        # -------------------------------------------------

        fused = torch.cat(

            [feat_real, feat_syn],

            dim=1
        )

        # -------------------------------------------------
        # CLASSIFICATION
        # -------------------------------------------------

        output = self.fusion_layer(
            fused
        )

        return output


# =========================================================
# MODEL
# =========================================================

model = TwoStreamFusionModel().to(device)

# =========================================================
# FREEZE BACKBONE INITIALLY
# =========================================================

for param in model.backbone.parameters():

    param.requires_grad = False

# =========================================================
# LOSS FUNCTION
# =========================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing= LABEL_SMOOTHING
)


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# =========================================================
# LR SCHEDULER
# =========================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

    optimizer,

    T_max=10
)


# =========================================================
# RESUME TRAINING
# =========================================================

start_epoch = 0

best_accuracy = 0.0

if os.path.exists(LATEST_CHECKPOINT):

    print("\n========================================")
    print("Resuming Two-Stream Training...")
    print("========================================\n")

    checkpoint = torch.load(

        LATEST_CHECKPOINT,

        map_location=device,

        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    start_epoch = checkpoint["epoch"] + 1

    best_accuracy = checkpoint[
        "best_accuracy"
    ]

    print(
        f"Resumed From Epoch: "
        f"{start_epoch}"
    )

    print(
        f"Best Accuracy: "
        f"{best_accuracy:.4f}\n"
    )

else:

    print(
        "\nStarting Fresh Two-Stream Training...\n"
    )


# =========================================================
# TRAINING LOOP
# =========================================================

try:

    for epoch in range(

        start_epoch,

        NUM_EPOCHS
    ):
        
        # =========================================================
        # UNFREEZE BACKBONE
        # =========================================================

        if epoch == FREEZE_EPOCHS:

            print("\nUnfreezing ConvNeXt Backbone...\n")

            for param in model.backbone.parameters():

                param.requires_grad = True

        model.train()

        epoch_start_time = time.time()

        running_loss = 0.0

        all_labels = []

        all_preds = []

        # =====================================================
        # BATCH LOOP
        # =====================================================

        for batch_idx, (

            real_images,

            synthetic_images,

            labels

        ) in enumerate(train_loader):

            real_images = real_images.to(
                device
            )

            synthetic_images = synthetic_images.to(
                device
            )

            labels = labels.to(
                device
            )

            # =================================================
            # ZERO GRADIENTS
            # =================================================

            optimizer.zero_grad()

            # =================================================
            # FORWARD PASS
            # =================================================

            outputs = model(

                real_images,

                synthetic_images
            )

            # =================================================
            # LOSS
            # =================================================

            loss = criterion(

                outputs,

                labels
            )

            # =================================================
            # BACKPROP
            # =================================================

            loss.backward()

            optimizer.step()

            # =================================================
            # METRICS
            # =================================================

            preds = torch.argmax(

                outputs,

                dim=1
            )

            all_labels.extend(

                labels.cpu().numpy()
            )

            all_preds.extend(

                preds.cpu().numpy()
            )

            running_loss += loss.item()

            # =====================================================
            # PRINT STATUS
            # =====================================================

            if batch_idx % 10 == 0:

                print(

                    f"Epoch "
                    f"[{epoch+1}/{NUM_EPOCHS}] "

                    f"Batch "
                    f"[{batch_idx}/{len(train_loader)}] "

                    f"Loss: {loss.item():.4f}"
                )

        # =====================================================
        # SCHEDULER STEP
        # =====================================================

        scheduler.step()

        # =====================================================
        # VALIDATION
        # =====================================================

        model.eval()

        val_labels = []

        val_preds = []

        with torch.no_grad():
            for val_images, val_labels_batch in val_loader:

                val_images = val_images.to(device)

                val_labels_batch = val_labels_batch.to(device)

            outputs = model(

                val_images,

                val_images
            )

            preds = torch.argmax(

                outputs,

                dim=1
            )

            val_labels.extend(

              val_labels_batch.cpu().numpy()
            )

            val_preds.extend(

                preds.cpu().numpy()
            
            )
        
        val_accuracy = accuracy_score(

            val_labels,

            val_preds
        )

        val_precision = precision_score(

            val_labels,

            val_preds,

            average="weighted"
        )

        val_recall = recall_score(

            val_labels,

            val_preds,

            average="weighted"
        )

        val_f1 = f1_score(

            val_labels,

            val_preds,

            average="weighted"
        )


        # =====================================================
        # METRICS
        # =====================================================

        epoch_loss = (

            running_loss /
            len(train_loader)
        )

        accuracy = accuracy_score(

            all_labels,

            all_preds
        )

        precision = precision_score(

            all_labels,

            all_preds,

            average="weighted"
        )

        recall = recall_score(

            all_labels,

            all_preds,

            average="weighted"
        )

        f1 = f1_score(

            all_labels,

            all_preds,

            average="weighted"
        )

        epoch_time = (
            time.time() -
            epoch_start_time
        ) / 60

        # =====================================================
        # SAVE BEST MODEL
        # =====================================================

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            torch.save(

                model.state_dict(),

                BEST_MODEL_PATH
            )

            print(
                "\nBest Two-Stream Model Saved.\n"
            )

        # =====================================================
        # SAVE CHECKPOINT
        # =====================================================

        torch.save({

            "epoch": epoch,

            "best_accuracy": best_accuracy,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict()

        }, LATEST_CHECKPOINT)

        # =====================================================
        # EPOCH SUMMARY
        # =====================================================

        print("\n========================================")

        print(
            f"Epoch {epoch+1} Completed"
        )

        print(
            f"Train Loss              : "
            f"{epoch_loss:.4f}"
        )

        print(
            f"Train Accuracy          : "
            f"{accuracy:.4f}"
        )

        print(
            f"Validation Accuracy     : "
            f"{val_accuracy:.4f}"
        )

        print(
            f"Train Precision         : "
            f"{precision:.4f}"
        )

        print(
            f"Validation Precision    : "
            f"{val_precision:.4f}"
        )

        print(
            f"Train Recall            : "
            f"{recall:.4f}"
        )

        print(
            f"Validation Recall       : "
            f"{val_recall:.4f}"
        )

        print(
            f"Train F1-score          : "
            f"{f1:.4f}"
        )

        print(
            f"Validation F1-score     : "
            f"{val_f1:.4f}"
        )

        print(
            f"Epoch Time              : "
            f"{epoch_time:.2f} mins"
        )

        print("========================================\n")


# =========================================================
# CTRL + C SAFE EXIT
# =========================================================

except KeyboardInterrupt:

    print("\n========================================")
    print("Training Interrupted.")
    print("Saving Recovery Checkpoint...")
    print("========================================\n")

    torch.save({

        "epoch": epoch,

        "best_accuracy": best_accuracy,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict()

    }, LATEST_CHECKPOINT)

    print(
        "Checkpoint Saved Successfully.\n"
    )


# =========================================================
# TRAINING COMPLETE
# =========================================================

print("\n========================================")

print("Two-Stream Training Completed.")

print("========================================\n")