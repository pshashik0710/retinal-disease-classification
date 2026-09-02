import os
import numpy as np
import pandas as pd

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataset import RetinalDataset

from transforms import get_valid_transforms

from models import ConvNeXtTiny

from config import Config


# =========================================================
# DEVICE
# =========================================================

device = Config.DEVICE

print(f"\nUsing Device: {device}")


# =========================================================
# AMD CLASS NAMES
# =========================================================

class_names = Config.AMD_CLASSES

num_classes = len(class_names)


# =========================================================
# LOAD TEST DATA
# =========================================================

test_image_paths = glob(
    os.path.join(
        Config.AMD_TEST_DIR,
        "*",
        "*"
    )
)

class_to_idx = {
    class_name: idx
    for idx, class_name in enumerate(
        class_names
    )
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
    pin_memory=Config.PIN_MEMORY
)

print(f"\nTotal Test Images: {len(test_dataset)}")


# =========================================================
# LOAD MODEL
# =========================================================

model = ConvNeXtTiny().to(device)

checkpoint_path = os.path.join(
    Config.CHECKPOINT_DIR,
    "amd_fold_5.pth"
)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device
    )
)

print(f"\nLoaded Checkpoint: {checkpoint_path}")

model.eval()


# =========================================================
# LOSS FUNCTION
# =========================================================

criterion = nn.CrossEntropyLoss()


# =========================================================
# EVALUATION
# =========================================================

all_preds = []

all_probs = []

all_targets = []

test_loss = 0.0

print("\nStarting Evaluation...\n")

with torch.no_grad():

    for images, labels in tqdm(test_loader):

        images = images.to(device)

        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        probs = torch.softmax(
            outputs,
            dim=1
        )

        preds = torch.argmax(
            probs,
            dim=1
        )

        test_loss += loss.item()

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_probs.append(
            probs.cpu().numpy()
        )

        all_targets.extend(
            labels.cpu().numpy()
        )

test_loss /= len(test_loader)

all_probs = np.concatenate(
    all_probs,
    axis=0
)


# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    all_targets,
    all_preds
)

precision = precision_score(
    all_targets,
    all_preds,
    average="weighted"
)

recall = recall_score(
    all_targets,
    all_preds,
    average="weighted"
)

f1 = f1_score(
    all_targets,
    all_preds,
    average="weighted"
)

roc_auc = roc_auc_score(
    all_targets,
    all_probs,
    multi_class="ovr"
)


# =========================================================
# PRINT RESULTS
# =========================================================

print("\n======================================")
print("FINAL AMD TEST RESULTS")
print("======================================")

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"Weighted F1   : {f1:.4f}")
print(f"ROC-AUC       : {roc_auc:.4f}")


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

report = classification_report(
    all_targets,
    all_preds,
    target_names=class_names,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

classification_report_path = os.path.join(
    Config.METRICS_DIR,
    "amd_classification_report.csv"
)

report_df.to_csv(
    classification_report_path
)

print("\nClassification Report Saved.")


# =========================================================
# PRIMARY METRICS TABLE
# =========================================================

primary_metrics = []

for class_name in class_names:

    class_metrics = report[class_name]

    primary_metrics.append({

        "Class": class_name,

        "Accuracy (Sim.)":
            round(accuracy, 4),

        "Precision":
            round(class_metrics["precision"], 4),

        "Recall":
            round(class_metrics["recall"], 4),

        "F1-score":
            round(class_metrics["f1-score"], 4),

        "STD":
            f"±{np.std(all_probs):.4f}"

    })

primary_metrics.append({

    "Class": "Overall (Weighted)",

    "Accuracy (Sim.)":
        round(accuracy, 4),

    "Precision": "N/A",

    "Recall": "N/A",

    "F1-score":
        round(f1, 4),

    "STD":
        f"±{np.std(all_probs):.4f}"

})

primary_df = pd.DataFrame(
    primary_metrics
)

primary_metrics_path = os.path.join(
    Config.METRICS_DIR,
    "amd_primary_metrics.csv"
)

primary_df.to_csv(
    primary_metrics_path,
    index=False
)

print("\nPrimary Metrics Table Saved.\n")

print(primary_df)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    all_targets,
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

plt.title("AMD Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

confusion_matrix_path = os.path.join(
    Config.PLOTS_DIR,
    "amd_confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
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
    classes=range(num_classes)
)

plt.figure(figsize=(10, 8))

for i in range(num_classes):

    fpr, tpr, _ = roc_curve(
        y_true_bin[:, i],
        all_probs[:, i]
    )

    roc_score = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{class_names[i]} (AUC={roc_score:.4f})"
    )

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("AMD ROC Curves")

plt.legend()

roc_plot_path = os.path.join(
    Config.PLOTS_DIR,
    "amd_roc_curves.png"
)

plt.savefig(
    roc_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("ROC Curves Saved.")


# =========================================================
# PRECISION-RECALL CURVES
# =========================================================

plt.figure(figsize=(10, 8))

for i in range(num_classes):

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            y_true_bin[:, i],
            all_probs[:, i]
        )
    )

    ap_score = average_precision_score(
        y_true_bin[:, i],
        all_probs[:, i]
    )

    plt.plot(
        recall_curve,
        precision_curve,
        label=f"{class_names[i]} (AP={ap_score:.4f})"
    )

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("AMD Precision-Recall Curves")

plt.legend()

pr_curve_path = os.path.join(
    Config.PLOTS_DIR,
    "amd_precision_recall_curves.png"
)

plt.savefig(
    pr_curve_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Precision-Recall Curves Saved.")


# =========================================================
# F1-SCORE BAR PLOT
# =========================================================

f1_scores = []

for class_name in class_names:

    f1_scores.append(
        report[class_name]["f1-score"]
    )

plt.figure(figsize=(8, 6))

bars = plt.bar(
    class_names,
    f1_scores
)

for bar, score in zip(bars, f1_scores):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{score:.4f}",
        ha="center",
        va="bottom"
    )

plt.ylim(0, 1)

plt.ylabel("F1-score")

plt.title("AMD Class-wise F1 Scores")

f1_plot_path = os.path.join(
    Config.PLOTS_DIR,
    "amd_f1_scores.png"
)

plt.savefig(
    f1_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("F1-score Plot Saved.")


# =========================================================
# FINAL SUMMARY CSV
# =========================================================

summary_df = pd.DataFrame([{

    "Test Loss": test_loss,

    "Accuracy": accuracy,

    "Precision": precision,

    "Recall": recall,

    "Weighted F1": f1,

    "ROC-AUC": roc_auc

}])

summary_path = os.path.join(
    Config.METRICS_DIR,
    "amd_final_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# =========================================================
# COMPLETED
# =========================================================

print("\n======================================")
print("AMD EVALUATION COMPLETED SUCCESSFULLY")
print("======================================")

print("\nSaved Outputs:")

print("\nMetrics:")
print(
    f"- {classification_report_path}"
)
print(
    f"- {primary_metrics_path}"
)
print(
    f"- {summary_path}"
)

print("\nPlots:")
print(
    f"- {confusion_matrix_path}"
)
print(
    f"- {roc_plot_path}"
)
print(
    f"- {pr_curve_path}"
)
print(
    f"- {f1_plot_path}"
)