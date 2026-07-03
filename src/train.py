"""
train.py
Stratified K-Fold training loop for the multi-label retinal model.
"""

import os
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

from src import config
from dataset import IDRiDMultiLabelDataset, get_train_transforms, get_val_transforms, compute_class_weights
from model import MultiLabelRetinalNet, MultiTaskLoss


def set_seed(seed: int = config.SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer=None, device=config.DEVICE):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, dr_loss_sum, dme_loss_sum = 0.0, 0.0, 0.0
    all_dr_preds, all_dr_true = [], []
    all_dme_preds, all_dme_true = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, leave=False):
            images = batch["image"].to(device)
            dr_labels = batch["dr_label"].to(device)
            dme_labels = batch["dme_label"].to(device)

            if is_train:
                optimizer.zero_grad()

            dr_logits, dme_logits = model(images)
            loss, dr_l, dme_l = criterion(dr_logits, dme_logits, dr_labels, dme_labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            dr_loss_sum += dr_l * images.size(0)
            dme_loss_sum += dme_l * images.size(0)

            all_dr_preds.extend(dr_logits.argmax(1).cpu().numpy())
            all_dr_true.extend(dr_labels.cpu().numpy())
            all_dme_preds.extend(dme_logits.argmax(1).cpu().numpy())
            all_dme_true.extend(dme_labels.cpu().numpy())

    n = len(loader.dataset)
    dr_kappa = cohen_kappa_score(all_dr_true, all_dr_preds, weights="quadratic")
    dme_kappa = cohen_kappa_score(all_dme_true, all_dme_preds, weights="quadratic")

    return {
        "loss": total_loss / n,
        "dr_loss": dr_loss_sum / n,
        "dme_loss": dme_loss_sum / n,
        "dr_kappa": dr_kappa,
        "dme_kappa": dme_kappa,
    }


def train_one_fold(fold_idx, train_idx, val_idx, full_df, image_dir, dr_weights, dme_weights):
    print(f"\n{'='*20} FOLD {fold_idx + 1}/{config.N_FOLDS} {'='*20}")

    # Write fold-specific CSVs (Dataset class reads from CSV path)
    fold_dir = os.path.join(config.PROCESSED_DATA_DIR, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    train_csv = os.path.join(fold_dir, "train.csv")
    val_csv = os.path.join(fold_dir, "val.csv")
    full_df.iloc[train_idx].to_csv(train_csv, index=False)
    full_df.iloc[val_idx].to_csv(val_csv, index=False)

    train_ds = IDRiDMultiLabelDataset(train_csv, image_dir, transform=get_train_transforms())
    val_ds = IDRiDMultiLabelDataset(val_csv, image_dir, transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                               num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=True)

    model = MultiLabelRetinalNet().to(config.DEVICE)
    criterion = MultiTaskLoss(dr_class_weights=dr_weights.to(config.DEVICE),
                               dme_class_weights=dme_weights.to(config.DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_score = -np.inf
    best_state = None
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer=None)

        # Combined score: average of DR and DME quadratic weighted kappa
        combined_score = (val_metrics["dr_kappa"] + val_metrics["dme_kappa"]) / 2
        scheduler.step(combined_score)

        print(f"Epoch {epoch+1}/{config.EPOCHS} | "
              f"train_loss={train_metrics['loss']:.4f} | "
              f"val_dr_kappa={val_metrics['dr_kappa']:.4f} | "
              f"val_dme_kappa={val_metrics['dme_kappa']:.4f} | "
              f"combined={combined_score:.4f}")

        if combined_score > best_score:
            best_score = combined_score
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"fold_{fold_idx}_best.pt")
    torch.save(best_state, ckpt_path)
    print(f"Fold {fold_idx+1} best combined kappa: {best_score:.4f} -> saved to {ckpt_path}")
    return best_score


def main():
    set_seed()
    print(f"Using device: {config.DEVICE}")
    full_df = pd.read_csv(config.IDRID_LABELS_TRAIN)
    full_df = full_df.dropna(subset=[config.COL_DR_GRADE, config.COL_DME_GRADE]).reset_index(drop=True)

    dr_weights = compute_class_weights(config.IDRID_LABELS_TRAIN, config.COL_DR_GRADE, config.NUM_DR_CLASSES)
    dme_weights = compute_class_weights(config.IDRID_LABELS_TRAIN, config.COL_DME_GRADE, config.NUM_DME_CLASSES)

    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    fold_scores = []

    # Stratify on DR grade (primary task, and the harder/more imbalanced one)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(full_df, full_df[config.COL_DR_GRADE])):
        score = train_one_fold(fold_idx, train_idx, val_idx, full_df,
                                config.IDRID_IMAGE_DIR_TRAIN, dr_weights, dme_weights)
        fold_scores.append(score)

    print(f"\n{'='*50}")
    print(f"Mean combined kappa across {config.N_FOLDS} folds: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")


if __name__ == "__main__":
    main()