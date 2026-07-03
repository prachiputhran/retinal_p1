"""
evaluate.py
Final evaluation on the held-out IDRiD test set: kappa, confusion matrices,
ROC-AUC (one-vs-rest), and misclassification analysis.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    cohen_kappa_score, confusion_matrix, roc_auc_score,
    classification_report
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

import config
from dataset import IDRiDMultiLabelDataset, get_val_transforms
from model import MultiLabelRetinalNet


@torch.no_grad()
def get_predictions(model, loader, device=config.DEVICE):
    model.eval()
    dr_probs_all, dme_probs_all = [], []
    dr_true_all, dme_true_all = [], []
    image_ids = []

    for batch in loader:
        images = batch["image"].to(device)
        dr_logits, dme_logits = model(images)

        dr_probs_all.append(torch.softmax(dr_logits, 1).cpu().numpy())
        dme_probs_all.append(torch.softmax(dme_logits, 1).cpu().numpy())
        dr_true_all.extend(batch["dr_label"].numpy())
        dme_true_all.extend(batch["dme_label"].numpy())
        image_ids.extend(batch["image_id"])

    return (np.concatenate(dr_probs_all), np.array(dr_true_all),
            np.concatenate(dme_probs_all), np.array(dme_true_all), image_ids)


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def evaluate_task(y_true, probs, class_names, task_name):
    y_pred = probs.argmax(axis=1)
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    try:
        auc = roc_auc_score(y_true_bin, probs, average="macro", multi_class="ovr")
    except ValueError:
        auc = float("nan")  # can happen if a class is absent from the test fold

    print(f"\n--- {task_name} ---")
    print(f"Quadratic Weighted Kappa: {kappa:.4f}")
    print(f"Macro ROC-AUC (OvR): {auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    plot_confusion_matrix(
        y_true, y_pred, class_names, f"{task_name} Confusion Matrix",
        os.path.join(config.OUTPUT_DIR, f"{task_name.lower().replace(' ', '_')}_confusion_matrix.png")
    )
    return {"kappa": kappa, "auc": auc, "y_pred": y_pred}


def find_misclassified(y_true, y_pred, image_ids, n=10):
    """Returns image IDs where prediction disagrees with ground truth — good report material."""
    mis_idx = np.where(np.array(y_true) != np.array(y_pred))[0]
    mis_ids = [(image_ids[i], y_true[i], y_pred[i]) for i in mis_idx[:n]]
    return mis_ids


def main(checkpoint_path: str):
    test_ds = IDRiDMultiLabelDataset(
        csv_path=config.IDRID_LABELS_TEST,
        image_dir=config.IDRID_IMAGE_DIR_TEST,
        transform=get_val_transforms(),
    )
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    model = MultiLabelRetinalNet().to(config.DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))

    dr_probs, dr_true, dme_probs, dme_true, image_ids = get_predictions(model, test_loader)

    dr_class_names = [f"DR-{i}" for i in range(config.NUM_DR_CLASSES)]
    dme_class_names = [f"DME-{i}" for i in range(config.NUM_DME_CLASSES)]

    dr_results = evaluate_task(dr_true, dr_probs, dr_class_names, "DR Grade")
    dme_results = evaluate_task(dme_true, dme_probs, dme_class_names, "DME Risk")

    print("\n--- Sample Misclassifications (DR) ---")
    for img_id, true_l, pred_l in find_misclassified(dr_true, dr_results["y_pred"], image_ids):
        print(f"{img_id}: true={true_l}, predicted={pred_l}")


if __name__ == "__main__":
    best_checkpoint = os.path.join(config.CHECKPOINT_DIR, "fold_0_best.pt")  # pick your best fold
    main(best_checkpoint)