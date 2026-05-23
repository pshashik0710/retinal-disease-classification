# test_evaluate_two_stream.py

import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import time
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_fscore_support

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder

from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score
)

from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import seaborn as sns

from scripts.models import ConvNeXtTiny
from scripts.config import Config


# =========================================================
# DEVICE
# =========================================================

device = Config.DEVICE

print(f"\nUsing Device: {device}\n")


# =========================================================
# HYPERPARAMETERS
# =========================================================

NUM_CLASSES = 4

BATCH_SIZE = 32

NUM_WORKERS = 4

IMAGE_SIZE = 224


# =========================================================
# PATHS
# =========================================================

MODEL_PATH = (
    "checkpoints/best_two_stream_model.pth"
)

TEST_REAL_PATH = (
    "/home/user24/retinal_project/datasets/processed/oct2017/test"
)

TEST_SYNTHETIC_PATH = (
    "/home/user24/retinal_project/generated_images/synthetic_OCT"
)

RESULTS_DIR = "two_stream_results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================
# TRANSFORMS
# =========================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# DATASET
# =========================================================

class TwoStreamTestDataset(Dataset):

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
# TEST DATASET
# =========================================================

test_dataset = TwoStreamTestDataset(

    real_root=TEST_REAL_PATH,

    synthetic_root=TEST_SYNTHETIC_PATH,

    transform=transform
)


# =========================================================
# TEST LOADER
# =========================================================

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=NUM_WORKERS,

    pin_memory=True
)


# =========================================================
# TWO-STREAM MODEL
# =========================================================

class TwoStreamFusionModel(nn.Module):

    def __init__(self, num_classes=4):

        super().__init__()

        self.backbone = ConvNeXtTiny()

        if hasattr(self.backbone, "classifier"):

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

        self.fusion_layer = nn.Sequential(

            nn.Linear(
                feature_dim * 2,
                512
            ),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(
                512,
                num_classes
            )
        )

    def forward(

        self,

        real_image,

        synthetic_image
    ):

        feat_real = self.backbone(
            real_image
        )

        feat_syn = self.backbone(
            synthetic_image
        )

        fused = torch.cat(

            [feat_real, feat_syn],

            dim=1
        )

        output = self.fusion_layer(
            fused
        )

        return output


# =========================================================
# LOAD MODEL
# =========================================================

model = TwoStreamFusionModel().to(device)

model.load_state_dict(

    torch.load(

        MODEL_PATH,

        map_location=device,

        weights_only=False
    )
)

model.eval()

print("Best Two-Stream Model Loaded.\n")


# =========================================================
# EVALUATION
# =========================================================

all_labels = []

all_preds = []

all_probs = []

start_time = time.time()

with torch.no_grad():

    for batch_idx, (

        real_images,

        synthetic_images,

        labels

    ) in enumerate(test_loader):

        real_images = real_images.to(device)

        synthetic_images = synthetic_images.to(device)

        labels = labels.to(device)

        outputs = model(

            real_images,

            synthetic_images
        )

        probs = torch.softmax(
            outputs,
            dim=1
        )

        preds = torch.argmax(
            probs,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_probs.extend(
            probs.cpu().numpy()
        )

        if batch_idx % 10 == 0:

            print(

                f"Processed "
                f"{batch_idx}/{len(test_loader)} batches"
            )


# =========================================================
# CONVERT TO NUMPY
# =========================================================

all_labels = np.array(all_labels)

all_preds = np.array(all_preds)

all_probs = np.array(all_probs)


# =========================================================
# METRICS
# =========================================================

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

labels_bin = label_binarize(

    all_labels,

    classes=[0, 1, 2, 3]
)

roc_auc = roc_auc_score(

    labels_bin,

    all_probs,

    multi_class="ovr",

    average="weighted"
)


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n========================================")
print("FINAL TEST RESULTS")
print("========================================\n")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print(
    f"\nEvaluation Time : "
    f"{(time.time()-start_time)/60:.2f} mins"
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

class_names = test_dataset.real_dataset.classes

report = classification_report(

    all_labels,

    all_preds,

    target_names=class_names,

    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "classification_report.csv"
    )
)

print("\nClassification Report Saved.")

# =========================================================
# CLASS-WISE METRICS TABLE
# =========================================================

class_precision, class_recall, class_f1, _ = (

    precision_recall_fscore_support(

        all_labels,

        all_preds,

        average=None,

        zero_division=0
    )
)

# ---------------------------------------------------------
# CLASS-WISE ACCURACY
# ---------------------------------------------------------

class_accuracy = []

for i in range(NUM_CLASSES):

    idx = (all_labels == i)

    if np.sum(idx) > 0:

        acc = np.mean(

            all_preds[idx] == i
        )

    else:

        acc = 0.0

    class_accuracy.append(acc)

# ---------------------------------------------------------
# STANDARD DEVIATION
# ---------------------------------------------------------

class_std = []

for i in range(NUM_CLASSES):

    idx = (all_labels == i)

    if np.sum(idx) > 0:

        std = np.std(

            all_probs[idx, i]
        )

    else:

        std = 0.0

    class_std.append(std)

# ---------------------------------------------------------
# CREATE TABLE
# ---------------------------------------------------------

table_df = pd.DataFrame({

    "Class": class_names,

    "Accuracy (Sim.)": np.round(
        class_accuracy, 4
    ),

    "Precision": np.round(
        class_precision, 4
    ),

    "Recall": np.round(
        class_recall, 4
    ),

    "F1-score": np.round(
        class_f1, 4
    ),

    "STD": [

        f"±{x:.4f}" for x in class_std
    ]
})

# ---------------------------------------------------------
# OVERALL ROW
# ---------------------------------------------------------

overall_row = pd.DataFrame({

    "Class": ["Overall (Weighted)"],

    "Accuracy (Sim.)": [

        round(accuracy, 4)
    ],

    "Precision": [

        round(precision, 4)
    ],

    "Recall": [

        round(recall, 4)
    ],

    "F1-score": [

        round(f1, 4)
    ],

    "STD": [

        f"±{np.mean(class_std):.4f}"
    ]
})

table_df = pd.concat(

    [table_df, overall_row],

    ignore_index=True
)

# ---------------------------------------------------------
# PRINT TABLE
# ---------------------------------------------------------

print("\n========================================")
print("CLASS-WISE PERFORMANCE TABLE")
print("========================================\n")

print(table_df.to_string(index=False))

# ---------------------------------------------------------
# SAVE CSV
# ---------------------------------------------------------

table_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "classwise_performance_table.csv"
    ),

    index=False
)

print(
    "\nClass-wise Performance Table Saved."
)

# ---------------------------------------------------------
# SAVE TABLE IMAGE
# ---------------------------------------------------------

fig, ax = plt.subplots(

    figsize=(12, 3)
)

ax.axis("off")

table = ax.table(

    cellText=table_df.values,

    colLabels=table_df.columns,

    loc="center",

    cellLoc="center"
)

table.auto_set_font_size(False)

table.set_fontsize(10)

table.scale(1.2, 1.5)

plt.tight_layout()

plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "classwise_performance_table.png"
    ),

    dpi=300,

    bbox_inches="tight"
)

plt.close()

print(
    "Class-wise Table Image Saved."
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(

    all_labels,

    all_preds
)

plt.figure(figsize=(10, 8))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=class_names,

    yticklabels=class_names
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Two-Stream Confusion Matrix")

plt.tight_layout()

plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    )
)

plt.close()

print("Confusion Matrix Saved.")


# =========================================================
# ROC CURVES
# =========================================================

from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(10, 8))

for i in range(NUM_CLASSES):

    fpr, tpr, _ = roc_curve(

        labels_bin[:, i],

        all_probs[:, i]
    )

    roc_auc_class = auc(fpr, tpr)

    plt.plot(

        fpr,

        tpr,

        label=(
            f"{class_names[i]} "
            f"(AUC={roc_auc_class:.3f})"
        )
    )

plt.plot([0, 1], [0, 1], "k--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curves")

plt.legend()

plt.tight_layout()

plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "roc_curves.png"
    )
)

plt.close()

print("ROC Curves Saved.")


# =========================================================
# PR CURVES
# =========================================================

plt.figure(figsize=(10, 8))

for i in range(NUM_CLASSES):

    precision_curve, recall_curve, _ = precision_recall_curve(

        labels_bin[:, i],

        all_probs[:, i]
    )

    ap = average_precision_score(

        labels_bin[:, i],

        all_probs[:, i]
    )

    plt.plot(

        recall_curve,

        precision_curve,

        label=(
            f"{class_names[i]} "
            f"(AP={ap:.3f})"
        )
    )

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision-Recall Curves")

plt.legend()

plt.tight_layout()

plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "precision_recall_curves.png"
    )
)

plt.close()

print("Precision-Recall Curves Saved.")


# =========================================================
# CLASS-WISE F1 SCORES
# =========================================================

class_f1_scores = f1_score(

    all_labels,

    all_preds,

    average=None
)

plt.figure(figsize=(8, 6))

plt.bar(

    class_names,

    class_f1_scores
)

plt.ylim(0, 1)

plt.ylabel("F1-score")

plt.title("Class-wise F1 Scores")

for i, score in enumerate(class_f1_scores):

    plt.text(

        i,

        score + 0.01,

        f"{score:.4f}",

        ha="center"
    )

plt.tight_layout()

plt.savefig(

    os.path.join(
        RESULTS_DIR,
        "classwise_f1_scores.png"
    )
)

plt.close()

print("F1-score Plot Saved.")


# =========================================================
# SAVE FINAL METRICS
# =========================================================

metrics_df = pd.DataFrame({

    "Metric": [

        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ],

    "Value": [

        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

metrics_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "final_metrics.csv"
    ),

    index=False
)

print("\nFinal Metrics Saved.")


# =========================================================
# COMPLETE
# =========================================================

print("\n========================================")
print("Two-Stream Evaluation Completed.")
print("========================================\n")