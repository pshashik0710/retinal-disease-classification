#!/usr/bin/env python3
"""
train.py -- train the classifier head on cached ConvNeXt features.

Run cache_features.py first. Each epoch here is a pass over 768-d vectors,
so it takes seconds rather than the ~3.5 hours a frozen-backbone epoch over
images would cost on this CPU.

Four things the previous pipeline got wrong, fixed here and worth naming
because they are the reasons its reported numbers were not trustworthy:

  1. It appended the LAST epoch's metrics to the results table while
     EarlyStopping saved the BEST epoch's weights. With patience 6 that
     meant reporting a score from six epochs of degradation past the model
     it actually kept. Here every metric written out comes from the best
     epoch, and the checkpoint holds exactly those weights.

  2. It selected the best model on TRAINING accuracy in the two-stream
     script. Selection here is on validation macro-F1 only.

  3. Its validation loop was mis-indented so only the final batch was
     evaluated -- roughly 32 images. Here validation runs over the whole
     split, and the count is printed so a silent truncation would show.

  4. It never evaluated the held-out test set at all. Here the test set is
     touched exactly once, at the end, using the restored best checkpoint,
     and never influences selection.

Also reported, because DME images come only from the Kermany cohort:
per-cohort metrics and an optional cohort probe measuring how separable
the two datasets are in feature space.

Usage:
    python train.py
    python train.py --experiment lr3e-4 --head-lr 3e-4
    python train.py --no-class-weights
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support, f1_score,
                             roc_auc_score, confusion_matrix,
                             classification_report, roc_curve,
                             precision_recall_curve, average_precision_score)

try:
    from scripts.config import Config
    from scripts.dataset import load_manifest, get_split
except ImportError:
    from config import Config
    from dataset import load_manifest, get_split


# =========================================================================
# DATA
# =========================================================================

def load_cached(split):
    path = Config.feature_cache_path(split)
    if not os.path.isfile(path):
        sys.exit(f"ERROR: no cached features for {split!r} at {path}\n"
                 f"Run:  python scripts/cache_features.py")
    d = np.load(path)
    X = d["features"].astype(np.float32)
    y = d["labels"].astype(np.int64)
    idx = d["index"].astype(np.int64)
    if len(X) == 0:
        sys.exit(f"ERROR: {path} is empty")
    return X, y, idx


def class_weights_from(y, n_classes):
    """Inverse frequency, mean 1, in Config.CLASSES order."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = np.divide(counts.sum(), counts * n_classes,
                  out=np.zeros_like(counts), where=counts > 0)
    if (w > 0).any():
        w = w / w[w > 0].mean()
    return torch.tensor(w, dtype=torch.float32)


# =========================================================================
# METRICS
# =========================================================================

def compute_metrics(y_true, y_pred, y_prob, n_classes):
    labels = list(range(n_classes))
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)

    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(p.mean()),
        "recall_macro": float(r.mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels,
                                      average="weighted", zero_division=0)),
        "per_class": {
            Config.CLASSES[i]: {
                "precision": float(p[i]), "recall": float(r[i]),
                "f1": float(f1[i]), "support": int(sup[i]),
            } for i in labels
        },
    }

    # ROC-AUC needs every class present in y_true; say so rather than
    # silently emitting a number computed over a subset
    present = sorted(set(y_true.tolist()))
    if len(present) == n_classes:
        try:
            if n_classes == 2:
                # binary: roc_auc_score wants the positive-class score
                m["roc_auc_ovr_macro"] = float(roc_auc_score(
                    y_true, y_prob[:, 1]))
            else:
                m["roc_auc_ovr_macro"] = float(roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="macro",
                    labels=labels))
        except ValueError as e:
            m["roc_auc_ovr_macro"] = None
            m["roc_auc_note"] = f"roc_auc_score error: {e}"
    else:
        m["roc_auc_ovr_macro"] = None
        m["roc_auc_note"] = (f"only {len(present)} of {n_classes} classes "
                             f"present in y_true")
    return m


def fmt(m):
    auc = m.get("roc_auc_ovr_macro")
    auc_s = f"{auc:.4f}" if auc is not None else "n/a"
    return (f"acc {m['accuracy']:.4f}  bal {m['balanced_accuracy']:.4f}  "
            f"macroF1 {m['macro_f1']:.4f}  AUC {auc_s}")


# =========================================================================
# TRAIN / EVAL
# =========================================================================

def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_loss, n = 0.0, 0
    preds, probs, trues = [], [], []

    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(yb)
            n += len(yb)
            pr = torch.softmax(out.float(), dim=1)
            probs.append(pr.detach().cpu().numpy())
            preds.append(pr.argmax(1).detach().cpu().numpy())
            trues.append(yb.detach().cpu().numpy())

    return (total_loss / n,
            np.concatenate(trues), np.concatenate(preds),
            np.concatenate(probs), n)


# =========================================================================
# REPORTING
# =========================================================================

def save_confusion(y_true, y_pred, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred,
                          labels=list(range(Config.NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(Config.NUM_CLASSES))
    ax.set_yticks(range(Config.NUM_CLASSES))
    ax.set_xticklabels(Config.CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(Config.CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cm


def save_curves(y_true, y_prob, roc_path, pr_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import label_binarize

    labels = list(range(Config.NUM_CLASSES))
    yb = label_binarize(y_true, classes=labels)
    # label_binarize collapses a 2-class problem to a single column; expand
    # it back so the per-class loops below work for binary and multiclass
    if yb.shape[1] == 1:
        yb = np.hstack([1 - yb, yb])


    fig, ax = plt.subplots(figsize=(7, 6))
    for i in labels:
        if yb[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(yb[:, i], y_prob[:, i])
        ax.plot(fpr, tpr,
                label=f"{Config.CLASSES[i]} (AUC={np.trapezoid(tpr, fpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC, one-vs-rest")
    ax.legend()
    fig.tight_layout()
    fig.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for i in labels:
        if yb[:, i].sum() == 0:
            continue
        pr, rc, _ = precision_recall_curve(yb[:, i], y_prob[:, i])
        ap = average_precision_score(yb[:, i], y_prob[:, i])
        ax.plot(rc, pr, label=f"{Config.CLASSES[i]} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall, one-vs-rest")
    ax.legend()
    fig.tight_layout()
    fig.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def per_cohort_report(y_true, y_pred, idx, split, paths, group_col="cohort"):
    """
    Break the test metrics down by source cohort.

    This matters because DME images come only from Kermany: a model can
    score well on that class by recognising the scanner rather than the
    pathology, and a single pooled number would hide it.
    """
    df = load_manifest()
    sub = get_split(df, split).reset_index(drop=True)
    cohorts = sub.loc[idx, group_col].to_numpy()

    rows = []
    for coh in sorted(set(cohorts)):
        mask = cohorts == coh
        if mask.sum() == 0:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        labels = list(range(Config.NUM_CLASSES))
        p, r, f1, sup = precision_recall_fscore_support(
            yt, yp, labels=labels, average=None, zero_division=0)
        for i in labels:
            rows.append({
                "cohort": coh, "class": Config.CLASSES[i],
                "n": int(sup[i]),
                "precision": round(float(p[i]), 4) if sup[i] else None,
                "recall": round(float(r[i]), 4) if sup[i] else None,
                "f1": round(float(f1[i]), 4) if sup[i] else None,
            })
        # Average over classes PRESENT in this cohort only. NEH has no DME,
        # and including a zero for an absent class would understate its
        # macro-F1 by a quarter.
        present = [i for i in labels if sup[i] > 0]
        rows.append({
            "cohort": coh, "class": f"MACRO ({len(present)} cls)",
            "n": int(mask.sum()),
            "precision": round(float(p[present].mean()), 4),
            "recall": round(float(r[present].mean()), 4),
            "f1": round(float(f1[present].mean()), 4),
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(paths["metrics"], "per_cohort.csv"), index=False)
    return out


def cohort_probe(Xtr, Xte, idx_tr, idx_te, group_col="cohort"):
    """
    How separable are the source groups in feature space? A near-perfect
    score means the domain gap is large -- which matters most when a class
    exists in only one group.

    Multi-group problems are scored as multiclass accuracy against the
    majority baseline.
    """

    from sklearn.linear_model import LogisticRegression

    df = load_manifest()
    if group_col not in df.columns or df[group_col].nunique() < 2:
        return None

    tr = get_split(df, "train").reset_index(drop=True)
    te = get_split(df, "test").reset_index(drop=True)
    groups = sorted(df[group_col].unique())
    g2i = {g: i for i, g in enumerate(groups)}
    ytr = tr.loc[idx_tr, group_col].map(g2i).to_numpy()
    yte = te.loc[idx_te, group_col].map(g2i).to_numpy()

    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return None

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    acc = float(clf.score(Xte, yte))
    counts = np.bincount(yte, minlength=len(groups))
    base = float(counts.max() / counts.sum())
    return {"group_column": group_col, "groups": groups,
            "accuracy": round(acc, 4), "majority_baseline": round(base, 4)}

# =========================================================================
# MAIN
# =========================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--head-lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--no-cohort-probe", action="store_true")
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--track", default=None,
                    help="override Config.TRACK for this run")
    a = ap.parse_args()

    if a.track:
        if a.track not in Config.TRACKS:
            sys.exit(f"unknown track {a.track!r}; "
                     f"valid: {sorted(Config.TRACKS)}")
        _t = Config.TRACKS[a.track]
        Config.TRACK = a.track
        Config.POOLED_MANIFEST = os.path.join(Config.MANIFEST_DIR,
                                              _t["manifest"])
        Config.DATA_ROOTS = _t["roots"]
        Config.CLASSES = _t["classes"]
        Config.CLASS_TO_IDX = {c: i for i, c in enumerate(Config.CLASSES)}
        Config.NUM_CLASSES = len(Config.CLASSES)
        Config.RESIZE_STRATEGY = _t["resize"]
        Config.TRACK_NOTE = _t["note"]
        Config.MANIFEST_FILTER = _t.get("filter")
        Config.FEATURE_CACHE_DIR = os.path.join(Config.BASE_DIR, "features",
                                                a.track)

    name = a.experiment or Config.EXPERIMENT_NAME
    epochs = a.epochs or Config.EPOCHS
    lr = a.head_lr or Config.HEAD_LR
    bs = a.batch_size or Config.FEATURE_BATCH_SIZE
    wd = a.weight_decay if a.weight_decay is not None else Config.WEIGHT_DECAY
    do = a.dropout if a.dropout is not None else Config.DROPOUT
    seed = a.seed or Config.SEED
    use_w = Config.USE_CLASS_WEIGHTS and not a.no_class_weights

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(Config.TORCH_NUM_THREADS)
    device = Config.DEVICE

    paths = Config.paths(name)

    print(f"\n{'='*70}\nCACHED LINEAR PROBE\n{'='*70}")
    print(f"  experiment    : {name}")
    print(f"  device        : {device}")
    print(f"  epochs        : {epochs}   batch {bs}   lr {lr}   wd {wd}")
    print(f"  dropout       : {do}   class weights: {use_w}")
    print(f"  seed          : {seed}")
    print(f"  output        : {paths['root']}")

    Xtr, ytr, itr = load_cached("train")
    Xva, yva, iva = load_cached("val")
    Xte, yte, ite = load_cached("test")

    print(f"\n  train {len(Xtr):>7,} x {Xtr.shape[1]}")
    print(f"  val   {len(Xva):>7,}")
    print(f"  test  {len(Xte):>7,}")

    print("\n  class distribution (train / val / test):")
    for i, c in enumerate(Config.CLASSES):
        print(f"    {c:<8} {int((ytr==i).sum()):>7,} "
              f"{int((yva==i).sum()):>7,} {int((yte==i).sum()):>7,}")

    # -- model: a linear head on frozen features -------------------------
    model = nn.Sequential(
        nn.Dropout(do),
        nn.Linear(Xtr.shape[1], Config.NUM_CLASSES),
    ).to(device)

    weights = None
    if use_w:
        weights = class_weights_from(ytr, Config.NUM_CLASSES).to(device)
        print("\n  class weights: " +
              ", ".join(f"{c} {w:.3f}"
                        for c, w in zip(Config.CLASSES, weights.tolist())))

    criterion = nn.CrossEntropyLoss(weight=weights,
                                    label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=wd, betas=Config.BETAS)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=Config.SCHEDULER_MODE,
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE, min_lr=Config.MIN_LR)

    g = torch.Generator(); g.manual_seed(seed)
    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=bs, shuffle=True, generator=g)
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)),
        batch_size=bs, shuffle=False)
    te_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)),
        batch_size=bs, shuffle=False)

    # -- training loop ---------------------------------------------------
    print(f"\n{'='*70}\nTRAINING\n{'='*70}")
    print(f"  {'ep':>3} {'train':>8} {'val':>8} {'acc':>7} {'bal':>7} "
          f"{'macroF1':>8} {'lr':>9}")

    best = {"macro_f1": -1.0, "epoch": -1}
    best_state = None
    history = []
    bad = 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        trl, *_ = run_epoch(model, tr_loader, criterion, optimizer, device)
        vl, vt, vp, vpr, vn = run_epoch(model, va_loader, criterion,
                                        None, device)

        # a silent truncation of validation -- the previous pipeline's
        # mis-indented loop evaluated only the last batch -- would show up
        # here as a wrong count
        assert vn == len(Xva), f"validated {vn} of {len(Xva)}"

        m = compute_metrics(vt, vp, vpr, Config.NUM_CLASSES)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(m["macro_f1"])

        history.append({"epoch": ep, "train_loss": trl, "val_loss": vl,
                        "lr": cur_lr, **{k: v for k, v in m.items()
                                         if k != "per_class"}})

        star = ""
        if m["macro_f1"] > best["macro_f1"]:
            best = {**m, "epoch": ep, "val_loss": vl, "train_loss": trl}
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
            star = "  *"
        else:
            bad += 1

        print(f"  {ep:>3} {trl:>8.4f} {vl:>8.4f} {m['accuracy']:>7.4f} "
              f"{m['balanced_accuracy']:>7.4f} {m['macro_f1']:>8.4f} "
              f"{cur_lr:>9.2e}{star}")

        if bad >= Config.PATIENCE:
            print(f"\n  early stopping: no improvement for "
                  f"{Config.PATIENCE} epochs")
            break

    elapsed = time.time() - t0
    print(f"\n  trained {len(history)} epochs in {elapsed:.1f}s "
          f"({elapsed/max(len(history),1):.2f}s/epoch)")
    print(f"  BEST epoch {best['epoch']}: {fmt(best)}")
    print(f"  (metrics reported are from this epoch, and the saved "
          f"checkpoint holds exactly these weights)")

    # -- restore best, then evaluate the test set ONCE --------------------
    model.load_state_dict(best_state)
    ckpt = os.path.join(paths["checkpoints"], "head_best.pth")
    torch.save({"state_dict": best_state, "epoch": best["epoch"],
                "val_macro_f1": best["macro_f1"],
                "config": {"lr": lr, "dropout": do, "weight_decay": wd,
                           "batch_size": bs, "seed": seed,
                           "class_weights": use_w,
                           "classes": Config.CLASSES}}, ckpt)

    print(f"\n{'='*70}\nTEST  (held out; evaluated once, from the best "
          f"checkpoint)\n{'='*70}")
    _, tt, tp, tpr, tn = run_epoch(model, te_loader, criterion, None, device)
    assert tn == len(Xte)
    tm = compute_metrics(tt, tp, tpr, Config.NUM_CLASSES)
    print(f"  {fmt(tm)}")

    print("\n  per class:")
    print(f"    {'class':<8} {'prec':>7} {'recall':>7} {'f1':>7} {'n':>7}")
    for c in Config.CLASSES:
        d = tm["per_class"][c]
        print(f"    {c:<8} {d['precision']:>7.4f} {d['recall']:>7.4f} "
              f"{d['f1']:>7.4f} {d['support']:>7,}")

    cm = save_confusion(tt, tp,
                        os.path.join(paths["plots"], "confusion_test.png"),
                        f"Test confusion ({name})")
    print("\n  confusion matrix (rows true, cols predicted):")
    print("    " + pd.DataFrame(cm, index=Config.CLASSES,
                                columns=Config.CLASSES)
          .to_string().replace("\n", "\n    "))

    save_curves(tt, tpr, os.path.join(paths["plots"], "roc_test.png"),
                os.path.join(paths["plots"], "pr_test.png"))

    # -- cohort breakdown -------------------------------------------------
    df_all = load_manifest()
    group_col = "source" if ("source" in df_all.columns
                             and df_all["source"].nunique() > 1) else "cohort"
    n_groups = df_all[group_col].nunique()

    if n_groups > 1:
        print(f"\n{'='*70}\nPER-{group_col.upper()} BREAKDOWN\n{'='*70}")
        print(f"  {Config.TRACK_NOTE}\n")
        pc = per_cohort_report(tt, tp, ite, "test", paths, group_col)
        print("    " + pc.to_string(index=False).replace("\n", "\n    "))
    else:
        print(f"\n{'='*70}\nPER-COHORT BREAKDOWN\n{'='*70}")
        print(f"  Single cohort ({df_all['cohort'].iloc[0]}); no breakdown "
              f"to report.")
        print(f"  {Config.TRACK_NOTE}")
        pc = None

    probe = None
    if Config.RUN_COHORT_PROBE and not a.no_cohort_probe and n_groups > 1:
        print(f"\n{'='*70}\n{group_col.upper()} PROBE\n{'='*70}")
        probe = cohort_probe(Xtr, Xte, itr, ite, group_col)
        if probe:
            print(f"  {' vs '.join(map(str, probe['groups']))} from the same "
                  f"features: {probe['accuracy']:.4f} "
                  f"(majority baseline {probe['majority_baseline']:.4f})")
            if probe["accuracy"] > 0.95:
                print(f"  Near-perfectly separable. Report this, and treat "
                      f"any class confined\n  to one {group_col} as partly "
                      f"{group_col}-driven.")

    # -- write everything out ---------------------------------------------
    pd.DataFrame(history).to_csv(
        os.path.join(paths["metrics"], "history.csv"), index=False)

    rep = classification_report(tt, tp,
                                labels=list(range(Config.NUM_CLASSES)),
                                target_names=Config.CLASSES,
                                output_dict=True, zero_division=0)
    pd.DataFrame(rep).transpose().to_csv(
        os.path.join(paths["metrics"], "classification_report_test.csv"))

    pd.DataFrame(cm, index=Config.CLASSES, columns=Config.CLASSES).to_csv(
        os.path.join(paths["metrics"], "confusion_test.csv"))

    summary = {
        "experiment": name,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "protocol": "cached linear probe (frozen ConvNeXt features)",
        "hyperparams": {"epochs_run": len(history), "max_epochs": epochs,
                        "lr": lr, "batch_size": bs, "weight_decay": wd,
                        "dropout": do, "label_smoothing":
                        Config.LABEL_SMOOTHING, "class_weights": use_w,
                        "seed": seed},
        "data": {"manifest": Config.POOLED_MANIFEST,
                 "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
                 "n_test": int(len(Xte)), "classes": Config.CLASSES},
        "best_epoch": best["epoch"],
        "val_at_best": {k: v for k, v in best.items() if k != "per_class"},
        "test": tm,
        "cohort_probe": probe,
        "seconds": round(elapsed, 1),
    }
    with open(os.path.join(paths["metrics"], "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  checkpoint : {ckpt}")
    print(f"  metrics    : {paths['metrics']}")
    print(f"  plots      : {paths['plots']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()