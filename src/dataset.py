"""
dataset.py
Custom PyTorch Dataset for IDRiD with fundus-specific preprocessing:
circular crop, CLAHE, Ben Graham normalization.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


# ---------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------

def crop_black_border(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """Remove the black background surrounding the circular fundus image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]


def ben_graham_preprocess(img: np.ndarray, sigma_x: int = 10) -> np.ndarray:
    """
    Ben Graham's method — winner of the original Kaggle DR competition.
    Subtracts a locally-blurred version to boost small-lesion visibility
    (microaneurysms, hemorrhages).
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma_x)
    return cv2.addWeighted(img, 4, blurred, -4, 128)


def apply_clahe_green_channel(img: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """CLAHE on the green channel — lesions show highest contrast there."""
    b, g, r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    g_eq = clahe.apply(g)
    return cv2.merge([b, g_eq, r])


def preprocess_fundus_image(img_bgr: np.ndarray, size: int = config.IMG_SIZE) -> np.ndarray:
    img = crop_black_border(img_bgr)
    img = cv2.resize(img, (size, size))
    img = apply_clahe_green_channel(img)
    img = ben_graham_preprocess(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# ---------------------------------------------------------------------------
# AUGMENTATION
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(size: int = config.IMG_SIZE) -> A.Compose:
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5),
        # Keep color jitter conservative — color carries diagnostic meaning
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02, p=0.3),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(size: int = config.IMG_SIZE) -> A.Compose:
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# DATASET CLASS
# ---------------------------------------------------------------------------

class IDRiDMultiLabelDataset(Dataset):
    """Loads IDRiD fundus images with joint DR grade + DME risk labels."""

    def __init__(self, csv_path: str, image_dir: str, transform: A.Compose = None,
                 preprocess: bool = True, image_ext: str = ".jpg"):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        self.preprocess = preprocess
        self.image_ext = image_ext
        self.df = self.df.dropna(subset=[config.COL_DR_GRADE, config.COL_DME_GRADE]).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row[config.COL_IMAGE_ID]).strip()

        img_path = os.path.join(self.image_dir, image_id + self.image_ext)
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, image_id)  # csv may include ext already

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image at {img_path}")

        img = preprocess_fundus_image(img_bgr) if self.preprocess else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(image=img)["image"]
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float()

        dr_label = int(row[config.COL_DR_GRADE])
        dme_label = int(row[config.COL_DME_GRADE])

        return {
            "image": img,
            "dr_label": torch.tensor(dr_label, dtype=torch.long),
            "dme_label": torch.tensor(dme_label, dtype=torch.long),
            "image_id": image_id,
        }


def compute_class_weights(csv_path: str, label_col: str, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights to counter DR/DME class imbalance."""
    df = pd.read_csv(csv_path)
    counts = df[label_col].value_counts().sort_index()
    counts = counts.reindex(range(num_classes), fill_value=1)
    weights = 1.0 / counts.values
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float)


if __name__ == "__main__":
    ds = IDRiDMultiLabelDataset(
        csv_path=config.IDRID_LABELS_TRAIN,
        image_dir=config.IDRID_IMAGE_DIR_TRAIN,
        transform=get_train_transforms(),
    )
    print(f"Dataset size: {len(ds)}")
    sample = ds[0]
    print(f"Image shape: {sample['image'].shape}, DR: {sample['dr_label']}, DME: {sample['dme_label']}")