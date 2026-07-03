"""
gradcam_utils.py
Grad-CAM explainability for both the DR and DME heads independently.
"""

import os
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src import config
from dataset import preprocess_fundus_image, IMAGENET_MEAN, IMAGENET_STD
from model import MultiLabelRetinalNet


class DRHeadWrapper(torch.nn.Module):
    """Wraps the model so Grad-CAM sees only the DR head's output."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        dr_logits, _ = self.model(x)
        return dr_logits


class DMEHeadWrapper(torch.nn.Module):
    """Wraps the model so Grad-CAM sees only the DME head's output."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, dme_logits = self.model(x)
        return dme_logits


def load_and_prepare_image(img_path: str, size: int = config.IMG_SIZE):
    img_bgr = cv2.imread(img_path)
    processed = preprocess_fundus_image(img_bgr, size=size)  # RGB, uint8

    rgb_float = processed.astype(np.float32) / 255.0  # for show_cam_on_image

    normalized = (rgb_float - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0)

    return tensor, rgb_float


def generate_gradcam(model, image_path: str, task: str, target_class: int, save_path: str):
    """
    task: "dr" or "dme"
    target_class: which class index to explain (e.g. the predicted class)
    """
    wrapper = DRHeadWrapper(model) if task == "dr" else DMEHeadWrapper(model)
    wrapper.eval()

    target_layers = [model.get_target_layer()]
    cam = GradCAM(model=wrapper, target_layers=target_layers)

    input_tensor, rgb_float = load_and_prepare_image(image_path)
    input_tensor = input_tensor.to(config.DEVICE)

    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Saved Grad-CAM ({task}, class {target_class}) -> {save_path}")
    return grayscale_cam


def generate_comparison_grid(model, image_path: str, dr_class: int, dme_class: int, save_path: str):
    """
    Side-by-side DR-head vs DME-head Grad-CAM for the same image — this is
    the figure that supports the 'DR and DME attend to different regions'
    finding for your report.
    """
    import matplotlib.pyplot as plt

    _, rgb_float = load_and_prepare_image(image_path)

    dr_cam = generate_gradcam(model, image_path, "dr", dr_class,
                               save_path.replace(".png", "_dr.png"))
    dme_cam = generate_gradcam(model, image_path, "dme", dme_class,
                                save_path.replace(".png", "_dme.png"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb_float); axes[0].set_title("Original (preprocessed)"); axes[0].axis("off")
    axes[1].imshow(rgb_float); axes[1].imshow(dr_cam, cmap="jet", alpha=0.5)
    axes[1].set_title(f"DR Head Attention (class {dr_class})"); axes[1].axis("off")
    axes[2].imshow(rgb_float); axes[2].imshow(dme_cam, cmap="jet", alpha=0.5)
    axes[2].set_title(f"DME Head Attention (class {dme_class})"); axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved comparison grid -> {save_path}")


def batch_gradcam_by_severity(model, csv_path: str, image_dir: str, n_per_grade: int = 2):
    """
    Generates a Grad-CAM grid across DR severity levels 0-4 to show how
    attention shifts as disease progresses — a strong report figure.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)

    for grade in range(config.NUM_DR_CLASSES):
        subset = df[df[config.COL_DR_GRADE] == grade].head(n_per_grade)
        for _, row in subset.iterrows():
            image_id = str(row[config.COL_IMAGE_ID]).strip()
            img_path = os.path.join(image_dir, image_id + ".jpg")
            if not os.path.exists(img_path):
                continue
            save_path = os.path.join(config.GRADCAM_DIR, f"dr_grade{grade}_{image_id}.png")
            generate_gradcam(model, img_path, "dr", target_class=grade, save_path=save_path)


if __name__ == "__main__":
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "fold_0_best.pt")
    model = MultiLabelRetinalNet().to(config.DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))

    batch_gradcam_by_severity(model, config.IDRID_LABELS_TEST, config.IDRID_IMAGE_DIR_TEST)