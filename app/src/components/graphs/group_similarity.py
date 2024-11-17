import streamlit as st
import pandas as pd
from src.lib.data import load_images
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def group_similarity(group_id):

    similarities = st.session_state.get("similarities")
    filtered_df = similarities[similarities['Final_Submission'].str.startswith(f"{group_id}_")]

    metrics = ['Color_Similarity', 'ResNet_Similarity', 'Dino_Similarity', 'Contrast_Similarity']
    combined_df_melted = pd.melt(filtered_df, id_vars=['Source'], value_vars=metrics, 
                                 var_name='Metric', value_name='Similarity')
    
    custom_palette = {'AI': '#1E74C3', 'Web': '#D85C56'}  
    strip_palette = {'AI': '#0F4A8E', 'Web': '#9A1E1A'}  

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