import streamlit as st

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import cv2
import os
import re

from src.lib.data import load_images, load_images_from_path, read_csv
from src.components.graphs.group_similarity import group_similarity
from src.components.graphs.groups_similarity import groups_similarity
from src.components.graphs.image_similarity import image_similarity

if "similarities" not in st.session_state:
    st.session_state["similarities"] = read_csv("all_similarities_with_srcs.csv")

if not (current_group_index := st.session_state.get("group_index")):
    st.session_state["group_index"] = 0
    current_group_index = st.session_state["group_index"]

if not (current_image_index := st.session_state.get("image_index")):
    st.session_state["image_index"] = 0
    current_image_index = st.session_state["image_index"]

groupes = list(range(1, 28))
images_indexes = st.session_state.get("similarities")["Final_Submission"].value_counts()

per_group_tab, all_group_tab, per_image_tab = st.tabs(["Per Group", "All Groups", "Per Image"])

with per_image_tab:
    back_button, next_button = st.columns(2)

    with back_button:
        if st.button("Previous Image"):
            if current_image_index > 0:
                st.session_state["image_index"] = current_image_index - 1
            else:
                st.session_state["image_index"] = len(images_indexes) - 1

    with next_button:
        if st.button("Next Image"):
            if current_image_index < len(images_indexes) - 1:
                st.session_state["image_index"] = current_image_index + 1
            else:
                st.session_state["image_index"] = 0
    current_image_index = st.session_state["image_index"]
    st.pyplot(image_similarity(current_image_index))

with per_group_tab:
    back_button, next_button = st.columns(2)

    with back_button:
        if st.button("Back"):
            if current_group_index > 0:
                st.session_state["group_index"] = current_group_index - 1
            else:
                st.session_state["group_index"] = len(groupes) - 1

    with next_button:
        if st.button("Next"):
            if current_group_index < len(groupes) - 1:
                st.session_state["group_index"] = current_group_index + 1
            else:
                st.session_state["group_index"] = 0

    current_group_index = st.session_state["group_index"]
    st.pyplot(group_similarity(groupes[current_group_index]))
    

with all_group_tab:
    st.pyplot(groups_similarity())