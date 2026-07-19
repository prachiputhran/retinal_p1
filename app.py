import os
import urllib.request
import torch
import streamlit as st
from PIL import Image

# Imports from src
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.append("src")

import config
from model import MultiLabelRetinalNet
from dataset import preprocess_fundus_image, IMAGENET_MEAN, IMAGENET_STD
from gradcam_utils import DRHeadWrapper, DMEHeadWrapper
from model import MultiLabelRetinalNet
import config

import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.dataset import preprocess_fundus_image, IMAGENET_MEAN, IMAGENET_STD
from src.gradcam_utils import DRHeadWrapper, DMEHeadWrapper
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

st.set_page_config(
    page_title="Retinal Disease Classification",
    layout="wide"
)
DR_LABELS = ["No DR", "Mild DR", "Moderate DR", "Severe DR", "Proliferative DR"]
DME_LABELS = ["No DME Risk", "Mild DME Risk", "Severe DME Risk"]


def prepare_tensor(rgb_uint8: np.ndarray):
    rgb_float = rgb_uint8.astype(np.float32) / 255.0
    normalized = (rgb_float - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, rgb_float


def run_gradcam(model, task, tensor, target_class, rgb_float):
    wrapper = DRHeadWrapper(model) if task == "dr" else DMEHeadWrapper(model)
    wrapper.eval()
    target_layers = [model.get_target_layer()]
    cam = GradCAM(model=wrapper, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=tensor.to(config.DEVICE),
                         targets=[ClassifierOutputTarget(target_class)])[0]
    visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    return visualization
CHECKPOINT_URL = "https://huggingface.co/PrachiPuthran/retinal-xai/resolve/main/fold_4_best.pt"
CHECKPOINT_PATH = "outputs/checkpoints/fold_4_best.pt"


@st.cache_resource
def load_model():
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        with st.spinner("Downloading model weights..."):
            urllib.request.urlretrieve(
                CHECKPOINT_URL,
                CHECKPOINT_PATH
            )

    model = MultiLabelRetinalNet().to(config.DEVICE)

    state_dict = torch.load(
        CHECKPOINT_PATH,
        map_location=config.DEVICE
    )

    model.load_state_dict(state_dict)
    model.eval()
    return model


# Load model
try:
    model = load_model()
    st.success("Model training completed successfully.")
except Exception as e:
    st.error(f"Failed to load model: {e}")


st.title("Retinal Disease Classification using Deep Learning and XAI")

st.write("""
This application predicts:

- Diabetic Retinopathy (DR) Grade
- Diabetic Macular Edema (DME) Risk

using a deep learning model trained on the IDRiD dataset.
""")

st.header("Project Results")

col1, col2 = st.columns(2)

with col1:
    st.subheader("DR Confusion Matrix")
    st.image("assets/dr_confusion_matrix.png")

with col2:
    st.subheader("DME Confusion Matrix")
    st.image("assets/dme_confusion_matrix.png")

st.subheader("Grad-CAM Severity Grid")
st.image("assets/gradcam_grid.png")

st.subheader("Example Misclassification")
st.image("assets/misclassification_example.png")
st.header("Try It Yourself")
st.write("Upload a retinal fundus image to get live predictions with Grad-CAM explainability.")

uploaded_file = st.file_uploader("Upload a fundus image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Couldn't read that image — try a different file.")
    else:
        with st.spinner("Running inference..."):
            processed_rgb = preprocess_fundus_image(img_bgr)
            tensor, rgb_float = prepare_tensor(processed_rgb)

            with torch.no_grad():
                dr_logits, dme_logits = model(tensor.to(config.DEVICE))
                dr_probs = torch.softmax(dr_logits, 1).cpu().numpy()[0]
                dme_probs = torch.softmax(dme_logits, 1).cpu().numpy()[0]

            dr_pred = int(dr_probs.argmax())
            dme_pred = int(dme_probs.argmax())

            dr_cam_img = run_gradcam(model, "dr", tensor, dr_pred, rgb_float)
            dme_cam_img = run_gradcam(model, "dme", tensor, dme_pred, rgb_float)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Original Upload", use_container_width=True)
        with col2:
            st.image(dr_cam_img, caption=f"DR Grad-CAM: {DR_LABELS[dr_pred]}", use_container_width=True)
        with col3:
            st.image(dme_cam_img, caption=f"DME Grad-CAM: {DME_LABELS[dme_pred]}", use_container_width=True)

        st.subheader("Prediction Confidence")
        bar_col1, bar_col2 = st.columns(2)

        with bar_col1:
            st.markdown("**DR Grade**")
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(DR_LABELS, dr_probs, color="#4C72B0")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            st.pyplot(fig)

        with bar_col2:
            st.markdown("**DME Risk**")
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(DME_LABELS, dme_probs, color="#DD8452")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            st.pyplot(fig)

        st.caption(
            "⚠️ Research demo only. Not validated for clinical use. "
            "Trained on 413 images — predictions should not be interpreted as medical advice."
        )
else:
    st.info("Upload a fundus image above to see live predictions and explainability maps.")
    st.caption("No sample handy? IDRiD test images work well — download from the [IDRiD website](https://idrid.grand-challenge.org/).")