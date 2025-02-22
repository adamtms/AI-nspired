import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
import numpy as np
import streamlit as st
import os
import PIL
import sys

APP_PATH = os.path.dirname(os.path.abspath(sys.argv[0]))#get dirname of ./app.py

def imshow_on_axis(img, ax, title):
    img = img.astype(np.uint8)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

from typing import Literal

def load_image(path: str, type: Literal["ai", "final_submission", "web"]) -> np.ndarray:
    if type == "final_submission":
        group, file_name = path.split("_")
        path = os.path.join(APP_PATH, "src", "static", "final_submissions", group, file_name)
    elif type == "web":
        path = os.path.join(APP_PATH, "src", "static", "web", path)
    elif type == "ai":
        path = os.path.join(APP_PATH, "src", "static", "ai", path)
    img = cv2.imread(path)
    return img

def image_similarity(img_index):
    # Define the missing variables
    similarities = st.session_state.get("similarities")
    groups = similarities.groupby("Final_Submission")
    group_key = sorted(
        list(groups.groups.keys()),
        key=lambda img: [int(img.split("_")[0]), int(img.split("_")[1].split(".")[0])] #sort by group idx and image num
        )[img_index]
    image_similarities = groups.get_group(group_key)
    final_image_src = image_similarities["Final_Submission"].values[0]
    final_image = load_image(final_image_src, "final_submission")

    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(5, 6, figure=fig)

    ax_final_submission = fig.add_subplot(gs[0:2, 0:2])  
    imshow_on_axis(final_image, ax_final_submission, f"Final Submission: {final_image_src}")
    
    ax = fig.add_subplot(gs[2, 0])  # ax for 3,0
    ax.text(0.5, 0.5, 'TOP AI', fontsize=30, fontweight='bold', ha='center', va='center', color='#0F4A8E')
    ax.axis('off')
    
    
    ax = fig.add_subplot(gs[3, 0])  # ax for 3,0
    ax.text(0.5, 0.5, 'TOP WEB', fontsize=30, fontweight='bold', ha='center', va='center', color='#9A1E1A')
    ax.axis('off')

    metrics = ['Color_Similarity', 'ResNet_Similarity', 'Dino_Similarity', 'Contrast_Similarity']
    
    for index, metric in enumerate(metrics):
        web_inspirations = image_similarities[image_similarities["Source"] == "Web"]
        best_web = web_inspirations.loc[web_inspirations[metric].idxmax()]
        best_web_src = best_web["Inspiration"]
        best_web_score = best_web[metric]
        best_web_image = load_image(best_web_src, "web")

        ai_inspirations = image_similarities[image_similarities["Source"] == "AI"]
        best_ai = ai_inspirations.loc[ai_inspirations[metric].idxmax()]
        best_ai_src = best_ai["Inspiration"]
        best_ai_score = best_ai[metric]
        best_ai_image = load_image(best_ai_src, "ai")

        ax_web = fig.add_subplot(gs[3, index+2])
        imshow_on_axis(best_web_image, ax_web, f"Web: {best_web_src} - {best_web_score:.4f}")

        ax_ai = fig.add_subplot(gs[2, index+2])
        imshow_on_axis(best_ai_image, ax_ai, f"AI: {best_ai_src} - {best_ai_score:.4f}")

    combined_df_melted = pd.melt(image_similarities, id_vars=['Source'], value_vars=metrics, 
                                    var_name='Metric', value_name='Similarity')

    custom_palette = {'AI': '#1E74C3', 'Web': '#D85C56'}  
    strip_palette = {'AI': '#0F4A8E', 'Web': '#9A1E1A'}  

    ax_sin = fig.add_subplot(gs[0:2, 2:6])

    sns.violinplot(x="Metric", y="Similarity", hue="Source", data=combined_df_melted, 
                    inner=None, linewidth=1.5, split=True, palette=custom_palette, alpha=0.75,
                    edgecolor='black', ax=ax_sin)  


    sns.stripplot(x="Metric", y="Similarity", hue="Source", data=combined_df_melted, 
                    palette=strip_palette, jitter=True, alpha=0.8, size=5, dodge=True, ax=ax_sin)

    ax_sin.set_title("Distributions of Similarity Scores")  
    ax_sin.set_xlabel("Metric")
    ax_sin.set_ylabel("Similarity")

    return fig