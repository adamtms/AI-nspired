import streamlit as st

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import cv2
import os
import re


@st.cache_data
def load_images_from_path(path, number):
    images = []
    srcs = []
    pattern = rf"^{number}(?!\d)"

    for filename in os.listdir(path):
        if re.match(pattern, filename):
            img = cv2.imread(os.path.join(path, filename))
            if img is not None:
                images.append(img)
                srcs.append(filename)

    return images, srcs


@st.cache_data
def load_images(number):
    final_images = []
    final_srcs = []

    final_path = f"data/final_submissions/{number}/"
    web_path = "data/web/"
    ai_path = "data/ai/"

    if not os.path.exists(final_path):
        print(f"The final submissions path '{final_path}' does not exist.")
        return None, None, None

    for filename in os.listdir(final_path):
        img = cv2.imread(os.path.join(final_path, filename))
        if img is not None:
            final_images.append(img)
            final_srcs.append(f"{number}_{filename}")

    web_images, web_srcs = load_images_from_path(web_path, number)
    ai_images, ai_srcs = load_images_from_path(ai_path, number)

    if not final_images:
        print(f"No images found in '{final_path}'. Please check the contents.")
        return None, None, None

    if not web_images and not ai_images:
        print(
            f"The number '{number}' does not correspond to any valid images in 'web' or 'ai' folders."
        )
        return None, None, None

    if not web_images:
        print(f"No web images found with prefix '{number}' in '{web_path}'.")
        return None, None, None
    if not ai_images:
        print(f"No AI images found with prefix '{number}' in '{ai_path}'.")
        return None, None, None

    return final_images, web_images, ai_images, final_srcs, web_srcs, ai_srcs


def read_csv(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None


@st.cache_data
def show_simmilarity(group_id):
    images_data = load_images(group_id)
    if len(images_data) != 6:
        return None
    *_, web_srcs, ai_srcs = images_data

    similarities = st.session_state.get("similarities")

    web_df = similarities[similarities["Inspiration"].isin(web_srcs)].copy()
    ai_df = similarities[similarities["Inspiration"].isin(ai_srcs)].copy()

    web_df.loc[:, "Source"] = "Web"
    ai_df.loc[:, "Source"] = "AI"

    combined_df = pd.concat([web_df, ai_df])

    metrics = [
        "Color_Similarity",
        "ResNet_Similarity",
        "Dino_Similarity",
        "Contrast_Similarity",
    ]
    combined_df_melted = pd.melt(
        combined_df,
        id_vars=["Source"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Similarity",
    )

    custom_palette = {"AI": "#1E74C3", "Web": "#D85C56"}
    strip_palette = {"AI": "#0F4A8E", "Web": "#9A1E1A"}

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes((0, 0, 1, 1))

    sns.violinplot(
        x="Metric",
        y="Similarity",
        hue="Source",
        data=combined_df_melted,
        inner=None,
        linewidth=1.5,
        split=True,
        palette=custom_palette,
        alpha=0.75,
        edgecolor="black",
        ax=ax,
    )

    sns.stripplot(
        x="Metric",
        y="Similarity",
        hue="Source",
        data=combined_df_melted,
        palette=strip_palette,
        jitter=True,
        alpha=0.8,
        size=5,
        dodge=True,
        ax=ax,
    )

    ax.set_title(f"GROUP: {group_id}", fontsize=18, fontweight="bold")
    ax.set_xlabel("Metric", fontsize=14, fontweight="bold")
    ax.set_ylabel("Similarity", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title="Source")
    return fig

@st.cache_data
def compare_all_groups():
    ai_srcs_all = []
    web_srcs_all = []

    for i in range(8, 28):
        images_data = load_images(i)
        if len(images_data) != 6:
            continue
        *_, web_srcs, ai_srcs = images_data
        ai_srcs_all.extend(ai_srcs)
        web_srcs_all.extend(web_srcs)

    all_sims = st.session_state.get("similarities")
    web_df = all_sims[all_sims['Inspiration'].isin(web_srcs_all)].copy()
    ai_df = all_sims[all_sims['Inspiration'].isin(ai_srcs_all)].copy()
        
    web_df.loc[:, 'Source'] = 'Web'
    ai_df.loc[:, 'Source'] = 'AI'
        
    combined_df = pd.concat([web_df, ai_df])
    metrics = ['Color_Similarity', 'ResNet_Similarity', 'Dino_Similarity', 'Contrast_Similarity']
    combined_df_melted = pd.melt(combined_df, id_vars=['Source'], value_vars=metrics, 
                                var_name='Metric', value_name='Similarity')

    custom_palette = {'AI': '#1E74C3', 'Web': '#D85C56'}  

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes((0, 0, 1, 1))

    sns.violinplot(x="Metric", y="Similarity", hue="Source", data=combined_df_melted, 
                inner=None, linewidth=1.5, split=True, palette=custom_palette, alpha=0.75,
                edgecolor='black', ax=ax)  

    ax.set_title(f'ALL GROUPS', fontsize=18, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=14, fontweight='bold')
    ax.set_ylabel('Similarity', fontsize=14, fontweight='bold')

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Source')
    return fig


if "similarities" not in st.session_state:
    st.session_state["similarities"] = read_csv("csv/all_similarity.csv")

if not (current_index := st.session_state.get("index")):
    st.session_state["index"] = 0
    current_index = st.session_state["index"]

groupes = list(range(8, 26)) + [27]

per_group_tab, all_group_tab = st.tabs(["Per Group", "All Groups"])

with per_group_tab:
    st.pyplot(show_simmilarity(groupes[current_index]))

    back_button, next_button = st.columns(2)

    with back_button:
        if st.button("Back"):
            if current_index > 0:
                st.session_state["index"] = current_index - 1
            else:
                st.session_state["index"] = len(groupes) - 1
            st.rerun()

    with next_button:
        if st.button("Next"):
            if current_index < len(groupes) - 1:
                st.session_state["index"] = current_index + 1
            else:
                st.session_state["index"] = 0
            st.rerun()

with all_group_tab:
    st.pyplot(compare_all_groups())
