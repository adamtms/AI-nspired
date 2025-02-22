import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def groups_similarity():

    all_sims = st.session_state.get("similarities")
    metrics = ['Color_Similarity', 'ResNet_Similarity', 'Dino_Similarity', 'Contrast_Similarity']
    combined_df_melted = pd.melt(all_sims, id_vars=['Source'], value_vars=metrics, 
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