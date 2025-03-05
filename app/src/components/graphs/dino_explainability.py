import matplotlib.pyplot as plt
import cv2
import numpy as np
import streamlit as st
import os
import sys
from typing import Literal

APP_PATH = os.path.dirname(os.path.abspath(sys.argv[0]))#get dirname of ./app.py

def show_viridis_cm():
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    plt.imshow(gradient, aspect='auto', cmap='viridis')
    plt.axis('off')
    plt.gcf().set_size_inches(5, 1)
    st.pyplot(plt)
    plt.clf()

def load_image(path: str, group: str, img_type: Literal["AI", "WEB", "FINAL"], visualization: Literal["lines", "attention"]) -> np.ndarray:
    path = os.path.join(APP_PATH, "src", "static", "dino", group, f'{path}_{img_type}_{visualization}.png')
    if path:
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return None

@st.dialog("Dino results with visualized attention maps and matched features", width="large")
def explain_dino(img_index):
    similarities = st.session_state.get("similarities")
    groups = similarities.groupby("Final_Submission")
    group_key = sorted(
        list(groups.groups.keys()),
        key=lambda img: [int(img.split("_")[0]), int(img.split("_")[1].split(".")[0])] #sort by group idx and image num
        )[img_index]
    image_similarities = groups.get_group(group_key)
    final_image_src = image_similarities["Final_Submission"].values[0]
    final_filename = final_image_src[:final_image_src.find('.')] if '.' in final_image_src else final_image_src
    
    group, photo = group_key[:group_key.find('_')], group_key[group_key.find('_') + 1:]
    final_filename = photo[:photo.find('.')] if '.' in photo else photo

    final_attention_image = load_image(final_filename, group, "FINAL", "attention")
    inspiration_attention_image = None
    inspiration_lines_image = None

    col1, col2 = st.columns(2)
    with col1:
        if st.button("AI"):
            inspiration_attention_image = load_image(final_filename, group, "AI", "attention")
            inspiration_lines_image = load_image(final_filename, group, "AI", "lines")
    with col2:
        if st.button("WEB"):
            inspiration_attention_image = load_image(final_filename, group, "WEB", "attention")
            inspiration_lines_image = load_image(final_filename, group, "WEB", "lines")
    
    st.image(final_attention_image, caption="Final image with attention mask")
    if inspiration_attention_image is not None:
        st.image(inspiration_attention_image, caption="Inspiration image with attention mask")
    if inspiration_lines_image is not None:
        st.image(inspiration_lines_image, caption="Similar features marked across images")
        st.subheader("Viridis Colormap used to express the magnitude of similarity.")
        show_viridis_cm()