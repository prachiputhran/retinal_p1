import os
import urllib.request
import torch
import streamlit as st

CHECKPOINT_URL = "https://huggingface.co/PrachiPuthran/retinal-xai/resolve/main/fold_4_best.pt"
CHECKPOINT_PATH = "outputs/checkpoints/fold_4_best.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
        with st.spinner("Downloading model weights..."):
            urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)

    model = MultiLabelRetinalNet().to(config.DEVICE)
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=config.DEVICE)
    )
    model.eval()
    return model