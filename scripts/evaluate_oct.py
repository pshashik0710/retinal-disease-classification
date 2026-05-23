# =========================================================
# FILE: ~/retinal_project/scripts/evaluate_oct.py
# =========================================================

import os

from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from sklearn.preprocessing import label_binarize

from dataset import RetinalDataset
from transforms import get_valid_transforms
from models import ConvNeXtTiny
from config import Config


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing Device: {device}")


# =========================================================
# CLASS NAMES
# =========================================================

CLASS_NAMES = Config.OCT_CLASSES
NUM_CLASSES = len(CLASS_NAMES)


# =========================================================
# LOAD TEST DATASET
# =========================================================

test_image_paths = glob(
    os.path.join(
        Config.OCT_TEST_DIR,
        "*",
        "*"
    )
)

class_to_idx = {
    class_name: idx
    for idx, class_name in enumerate(Config.OCT_CLASSES)
}

test_dataset = RetinalDataset(
    image_paths=test_image_paths,
    class_to_idx=class_to_idx,
    transform=get_valid_transforms()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    num_workers=Config.NUM_WORKERS,
    pin_memory=True,
)

print(f"\nTotal Test Images: {len(test_dataset)}")


# =========================================================
# LOAD BEST MODEL
# =========================================================

CHECKPOINT_PATH = os.path.join(
    Config.CHECKPOINT_DIR,
    "fold_5.pth"
)

model = ConvNeXtTiny()

model.load_state_dict(
    torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )
)

model.to(device)
model.eval()

print(f"\nLoaded Checkpoint: {CHECKPOINT_PATH}")


# =========================================================
# EVALUATION
# =========================================================

all_preds = []
all_probs = []
all_targets = []

test_loss = 0.0

criterion = torch.nn.CrossEntropyLoss()

print("\nStarting Evaluation...\n")

with torch.no_grad():

    for images, labels in tqdm(test_loader):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        test_loss += loss.item()

        probs = F.softmax(outputs.float(), dim=1)

        preds = torch.argmax(probs, dim=1)

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_targets.extend(
            labels.cpu().numpy()
        )

        all_probs.append(
            probs.cpu().numpy()
        )


all_probs = np.concatenate(all_probs, axis=0)

avg_test_loss = test_loss / len(test_loader)


# =========================================================
# FINAL TEST ACCURACY
# =========================================================

test_accuracy = accuracy_score(
    all_targets,
    all_preds
)

print("\n======================================")
print("FINAL TEST RESULTS")
print("======================================")

print(f"Test Loss     : {avg_test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

report = classification_report(
    all_targets,
    all_preds,
    target_names=CLASS_NAMES,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

report_save_path = os.path.join(
    Config.METRICS_DIR,
    "oct_classification_report.csv"
)

report_df.to_csv(report_save_path)

print("\nClassification Report Saved.")


# =========================================================
# PRIMARY CLASSIFICATION TABLE
# =========================================================

table_rows = []

for idx, class_name in enumerate(CLASS_NAMES):

    y_true = np.array(all_targets) == idx
    y_pred = np.array(all_preds) == idx

    class_acc = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    std = np.std(y_pred.astype(np.float32))

    table_rows.append([
        class_name,
        round(class_acc, 4),
        round(precision, 4),
        round(recall, 4),
        round(f1, 4),
        f"±{std:.4f}"
    ])


overall_f1 = f1_score(
    all_targets,
    all_preds,
    average="weighted"
)

overall_acc = accuracy_score(
    all_targets,
    all_preds
)

overall_std = np.std(
    np.array(all_preds).astype(np.float32)
)

table_rows.append([
    "Overall (Weighted)",
    round(overall_acc, 4),
    "N/A",
    "N/A",
    round(overall_f1, 4),
    f"±{overall_std:.4f}"
])


table_df = pd.DataFrame(
    table_rows,
    columns=[
        "Class",
        "Accuracy (Sim.)",
        "Precision",
        "Recall",
        "F1-score",
        "STD"
    ]
)

table_save_path = os.path.join(
    Config.METRICS_DIR,
    "oct_primary_metrics.csv"
)

table_df.to_csv(
    table_save_path,
    index=False
)

print("\nPrimary Metrics Table Saved.\n")
print(table_df)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    all_targets,
    all_preds
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("OCT Confusion Matrix")

cm_save_path = os.path.join(
    Config.PLOTS_DIR,
    "oct_confusion_matrix.png"
)

plt.savefig(
    cm_save_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Confusion Matrix Saved.")


# =========================================================
# ROC CURVES
# =========================================================

y_true_bin = label_binarize(
    all_targets,
    classes=np.arange(NUM_CLASSES)
)

plt.figure(figsize=(8, 6))

for i in range(NUM_CLASSES):

    fpr, tpr, _ = roc_curve(
        y_true_bin[:, i],
        all_probs[:, i]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{CLASS_NAMES[i]} (AUC={roc_auc:.4f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (OCT)")
plt.legend()

roc_save_path = os.path.join(
    Config.PLOTS_DIR,
    "oct_roc_curves.png"
)

plt.savefig(
    roc_save_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("ROC Curves Saved.")


# =========================================================
# PRECISION-RECALL CURVES
# =========================================================

plt.figure(figsize=(8, 6))

for i in range(NUM_CLASSES):

    precision_curve, recall_curve, _ = precision_recall_curve(
        y_true_bin[:, i],
        all_probs[:, i]
    )

    ap_score = average_precision_score(
        y_true_bin[:, i],
        all_probs[:, i]
    )

    plt.plot(
        recall_curve,
        precision_curve,
        label=f"{CLASS_NAMES[i]} (AP={ap_score:.4f})"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves (OCT)")
plt.legend()

pr_save_path = os.path.join(
    Config.PLOTS_DIR,
    "oct_precision_recall_curves.png"
)

plt.savefig(
    pr_save_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Precision-Recall Curves Saved.")


# =========================================================
# F1-SCORE BAR PLOT
# =========================================================

class_f1_scores = []

for idx in range(NUM_CLASSES):

    class_f1 = f1_score(
        np.array(all_targets) == idx,
        np.array(all_preds) == idx
    )

    class_f1_scores.append(class_f1)

plt.figure(figsize=(8, 6))

sns.barplot(
    x=CLASS_NAMES,
    y=class_f1_scores
)

plt.ylim(0, 1)

plt.ylabel("F1-score")
plt.title("Class-wise F1-score (OCT)")

f1_plot_path = os.path.join(
    Config.PLOTS_DIR,
    "oct_f1_scores.png"
)

plt.savefig(
    f1_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("F1-score Plot Saved.")


# =========================================================
# SAVE FINAL SUMMARY
# =========================================================

summary = {
    "test_accuracy": test_accuracy,
    "weighted_f1": overall_f1,
    "test_loss": avg_test_loss,
}

summary_df = pd.DataFrame([summary])

summary_save_path = os.path.join(
    Config.METRICS_DIR,
    "oct_final_summary.csv"
)

summary_df.to_csv(
    summary_save_path,
    index=False
)

print("\n======================================")
print("EVALUATION COMPLETED SUCCESSFULLY")
print("======================================")

print("\nSaved Outputs:")

print(f"\nMetrics:")
print(f"- {report_save_path}")
print(f"- {table_save_path}")
print(f"- {summary_save_path}")

print(f"\nPlots:")
print(f"- {cm_save_path}")
print(f"- {roc_save_path}")
print(f"- {pr_save_path}")
print(f"- {f1_plot_path}")