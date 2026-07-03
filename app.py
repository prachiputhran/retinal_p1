import os
import urllib.request
import torch
import streamlit as st
from PIL import Image

# Imports from src
from src import config
from src.model import MultiLabelRetinalNet

st.set_page_config(
    page_title="Retinal Disease Classification",
    layout="wide"
)

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
