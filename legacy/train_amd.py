import os
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torch.amp import (
    autocast,
    GradScaler
)

from dataset import (
    RetinalDataset,
    get_amd_datasets
)

from transforms import (
    get_train_transforms,
    get_valid_transforms
)

from models import ConvNeXtTiny

from config import Config

from seed_utils import set_seed

from early_stopping import EarlyStopping


# =========================================================
# SET SEED
# =========================================================

set_seed(Config.SEED)


# =========================================================
# DEVICE
# =========================================================

device = Config.DEVICE


# =========================================================
# RESUME TRAINING
# =========================================================

start_fold = 0

start_epoch = 0

best_score = 0.0


# =========================================================
# LOAD AMD DATASET PATHS
# =========================================================

(
    train_paths,
    val_paths,
    test_paths,
    class_to_idx
) = get_amd_datasets()


# =========================================================
# COMBINE TRAIN + VAL FOR CV
# =========================================================

all_train_paths = train_paths + val_paths

all_labels = []

for path in all_train_paths:

    class_name = os.path.basename(
        os.path.dirname(path)
    )

    all_labels.append(
        class_to_idx[class_name]
    )

all_labels = np.array(all_labels)


# =========================================================
# STRATIFIED K-FOLD
# =========================================================

skf = StratifiedKFold(
    n_splits=Config.NUM_FOLDS,
    shuffle=True,
    random_state=Config.SEED
)


# =========================================================
# METRIC STORAGE
# =========================================================

fold_results = []


# =========================================================
# TRAINING LOOP
# =========================================================

for fold, (train_idx, val_idx) in enumerate(

    skf.split(
        all_train_paths,
        all_labels
    )

):

    # -----------------------------------------------------
    # SKIP COMPLETED FOLDS
    # -----------------------------------------------------

    if fold < start_fold:
        continue

    print("\n====================================")
    print(f"AMD FOLD {fold + 1}/{Config.NUM_FOLDS}")
    print("====================================\n")

    # -----------------------------------------------------
    # SPLIT PATHS
    # -----------------------------------------------------

    fold_train_paths = [
        all_train_paths[i]
        for i in train_idx
    ]

    fold_val_paths = [
        all_train_paths[i]
        for i in val_idx
    ]

    # -----------------------------------------------------
    # DATASETS
    # -----------------------------------------------------

    train_dataset = RetinalDataset(
        fold_train_paths,
        class_to_idx,
        transform=get_train_transforms()
    )

    val_dataset = RetinalDataset(
        fold_val_paths,
        class_to_idx,
        transform=get_valid_transforms()
    )

    # -----------------------------------------------------
    # DATALOADERS
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=Config.PERSISTENT_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        persistent_workers=Config.PERSISTENT_WORKERS
    )

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = ConvNeXtTiny().to(device)

    # -----------------------------------------------------
    # FREEZE BACKBONE
    # -----------------------------------------------------

    for param in model.model.parameters():

        param.requires_grad = False

    # -----------------------------------------------------
    # UNFREEZE CLASSIFIER HEAD
    # -----------------------------------------------------

    for param in model.model.head.parameters():

        param.requires_grad = True

    # -----------------------------------------------------
    # LOSS
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1
    )

    # -----------------------------------------------------
    # OPTIMIZER
    # -----------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        betas=Config.BETAS
    )

    # -----------------------------------------------------
    # SCHEDULER
    # -----------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.MIN_LR
    )

    # -----------------------------------------------------
    # MIXED PRECISION
    # -----------------------------------------------------

    scaler = GradScaler("cuda")

    # -----------------------------------------------------
    # RESUME CHECKPOINT
    # -----------------------------------------------------

    if (
        os.path.exists(Config.RESUME_CHECKPOINT)
        and fold == start_fold
    ):

        print("\nResuming Previous Training...\n")

        checkpoint = torch.load(
            Config.RESUME_CHECKPOINT,
            map_location=device
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

        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )

        start_fold = checkpoint["fold"]

        start_epoch = checkpoint["epoch"] + 1

        best_score = checkpoint["best_score"]

        print(
            f"Resumed from Fold {start_fold+1}, "
            f"Epoch {start_epoch}"
        )

    # -----------------------------------------------------
    # CHECKPOINT PATH
    # -----------------------------------------------------

    checkpoint_path = os.path.join(
        Config.CHECKPOINT_DIR,
        f"amd_fold_{fold + 1}.pth"
    )

    # -----------------------------------------------------
    # EARLY STOPPING
    # -----------------------------------------------------

    early_stopper = EarlyStopping(
        patience=Config.PATIENCE,
        mode="max",
        checkpoint_path=checkpoint_path
    )

    # =====================================================
    # EPOCH LOOP
    # =====================================================

    epoch_begin = (
        start_epoch
        if fold == start_fold
        else 0
    )

    for epoch in range(
        epoch_begin,
        Config.EPOCHS
    ):

        # =================================================
        # GRADUAL UNFREEZING
        # =================================================

        if epoch == 5:

            print("\nUnfreezing ConvNeXt Backbone...\n")

            for param in model.model.parameters():

                param.requires_grad = True

        print(
            f"\nAMD Fold [{fold+1}] "
            f"Epoch [{epoch+1}/{Config.EPOCHS}]"
        )

        # =================================================
        # TRAINING
        # =================================================

        model.train()

        train_loss = 0.0

        for images, labels in tqdm(train_loader):

            images = images.to(device)

            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast(
                device_type="cuda",
                enabled=Config.USE_AMP
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # =================================================
        # VALIDATION
        # =================================================

        model.eval()

        val_loss = 0.0

        all_preds = []

        all_probs = []

        all_targets = []

        with torch.no_grad():

            for images, labels in tqdm(val_loader):

                images = images.to(device)

                labels = labels.to(device)

                with autocast(
                    device_type="cuda",
                    enabled=Config.USE_AMP
                ):

                    outputs = model(images)

                    loss = criterion(
                        outputs,
                        labels
                    )

                probs = torch.softmax(
                    outputs.float(),
                    dim=1
                )

                preds = torch.argmax(
                    probs,
                    dim=1
                )

                val_loss += loss.item()

                all_preds.extend(
                    preds.cpu().numpy()
                )

                all_probs.append(
                    probs.cpu().numpy()
                )

                all_targets.extend(
                    labels.cpu().numpy()
                )

        val_loss /= len(val_loader)

        # =================================================
        # METRICS
        # =================================================

        all_probs = np.concatenate(all_probs, axis=0)

        accuracy = accuracy_score(
            all_targets,
            all_preds
        )

        precision = precision_score(
            all_targets,
            all_preds,
            average="macro"
        )

        recall = recall_score(
            all_targets,
            all_preds,
            average="macro"
        )

        macro_f1 = f1_score(
            all_targets,
            all_preds,
            average="macro"
        )

        roc_auc = roc_auc_score(
            all_targets,
            all_probs,
            multi_class="ovr"
        )

        # =================================================
        # PRINT METRICS
        # =================================================

        print(f"\nTrain Loss : {train_loss:.4f}")
        print(f"Val Loss   : {val_loss:.4f}")

        print(f"Accuracy   : {accuracy:.4f}")
        print(f"Precision  : {precision:.4f}")
        print(f"Recall     : {recall:.4f}")
        print(f"Macro F1   : {macro_f1:.4f}")
        print(f"ROC-AUC    : {roc_auc:.4f}")

        # =================================================
        # SCHEDULER
        # =================================================

        scheduler.step(macro_f1)

        # =================================================
        # SAVE RESUME CHECKPOINT
        # =================================================

        torch.save({

            "fold": fold,

            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "scaler_state_dict":
                scaler.state_dict(),

            "best_score": macro_f1

        }, Config.RESUME_CHECKPOINT)

        # =================================================
        # EARLY STOPPING
        # =================================================

        early_stopper(
            macro_f1,
            model
        )

        if early_stopper.early_stop:

            print("\nEarly stopping triggered.\n")

            break

    # =====================================================
    # STORE FOLD RESULTS
    # =====================================================

    fold_results.append({

        "fold": fold + 1,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "macro_f1": macro_f1,

        "roc_auc": roc_auc

    })


# =========================================================
# FINAL CV RESULTS
# =========================================================

results_df = pd.DataFrame(fold_results)

print("\n====================================")
print("AMD CROSS VALIDATION RESULTS")
print("====================================\n")

print(results_df)

print("\n====================================")
print("AMD MEAN CV METRICS")
print("====================================\n")

print(results_df.mean(numeric_only=True))


# =========================================================
# SAVE METRICS
# =========================================================

metrics_path = os.path.join(
    Config.METRICS_DIR,
    "amd_cv_results.csv"
)

results_df.to_csv(
    metrics_path,
    index=False
)

print(f"\nMetrics saved to:\n{metrics_path}\n")