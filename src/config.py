"""
config.py
Central configuration for the Retinal Disease Multi-Label Classification project.
"""

import os
import torch

# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

IDRID_IMAGE_DIR_TRAIN = os.path.join(RAW_DATA_DIR, "B. Disease Grading", "1. Original Images", "a. Training Set")
IDRID_IMAGE_DIR_TEST = os.path.join(RAW_DATA_DIR, "B. Disease Grading", "1. Original Images", "b. Testing Set")
IDRID_LABELS_TRAIN = os.path.join(
    RAW_DATA_DIR, "B. Disease Grading", "2. Groundtruths",
    "a. IDRiD_Disease Grading_Training Labels.csv"
)
IDRID_LABELS_TEST = os.path.join(
    RAW_DATA_DIR, "B. Disease Grading", "2. Groundtruths",
    "b. IDRiD_Disease Grading_Testing Labels.csv"
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
GRADCAM_DIR = os.path.join(OUTPUT_DIR, "gradcam_images")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

for d in [PROCESSED_DATA_DIR, CHECKPOINT_DIR, GRADCAM_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# -------------------------------------------------------------------
# DATA
# -------------------------------------------------------------------
IMG_SIZE = 224          # use 380 if BACKBONE == "efficientnet_b4"
NUM_DR_CLASSES = 5      # DR grade 0-4
NUM_DME_CLASSES = 3     # DME risk 0-2

# Verify these against your actual CSV headers before running anything
COL_IMAGE_ID = "Image name"
COL_DR_GRADE = "Retinopathy grade"
COL_DME_GRADE = "Risk of macular edema "

# -------------------------------------------------------------------
# TRAINING
# -------------------------------------------------------------------
BACKBONE = "resnet50"        # "resnet50" or "efficientnet_b4"
PRETRAINED = True
BATCH_SIZE = 8
NUM_WORKERS = 0
EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 1e-5
N_FOLDS = 5
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DR_LOSS_WEIGHT = 1.0
DME_LOSS_WEIGHT = 0.7
PATIENCE = 8   # early stopping

# -------------------------------------------------------------------
# GRAD-CAM
# -------------------------------------------------------------------
GRADCAM_TARGET_LAYER = "layer4"