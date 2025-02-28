import pandas as pd
from scipy.stats import pearsonr, ttest_ind
import matplotlib.pyplot as plt

expert_df = pd.read_csv("csv/expert_scores.csv")
sim_df = pd.read_csv("csv/all_similarities_with_srcs.csv")

score_cols = ["aesthetics", "value", "innovation", "surprise"]
similarity_cols = ["Color_Similarity", "ResNet_Similarity", "Dino_Similarity", "Contrast_Similarity"]

expert_df_clean = expert_df.dropna(subset=score_cols)
sim_df_clean = sim_df.dropna(subset=similarity_cols)

def extract_group_name(filename) -> str:
    return filename.split("_")[0]

sim_df_clean['Group'] = sim_df_clean['Final_Submission'].apply(extract_group_name)
sim_df_clean = sim_df_clean.astype({'Group': 'int64'})


def aggregate_data(
        expert_df: pd.DataFrame,
        sim_df: pd.DataFrame,
        expert_feature: str = 'aesthetics', 
        similarity_metric: str = 'Dino_Similarity', 
        expert_aggr_method: str = 'mean',
        sim_aggr_method: str = 'mean'
        ) -> pd.DataFrame:
    '''
    Aggregate expert scores per group.
    '''
    expert_group = expert_df.groupby("group").agg({expert_feature: expert_aggr_method}).reset_index()

    sim_group = sim_df.groupby(['Group', 'Source']).agg({similarity_metric: sim_aggr_method}).reset_index()
    
    sim_pivot = sim_group.pivot(index='Group', columns='Source', values=similarity_metric).reset_index()
    sim_pivot.columns.name = None  # Remove pivot index name

    merged_df = pd.merge(expert_group, sim_pivot, left_on="group", right_on="Group", how="inner")
    merged_df.drop(columns=["Group"], inplace=True)
    return merged_df

def correlation_analysis(
        df: pd.DataFrame, 
        expert_feature: str = 'aesthetics',
        similarity_metric: str = 'Dino_Similarity'
        ) -> None:
    '''
    Calculate the Pearson correlation between a given expert_feature scores 
    and a given similarity_metric for each Source.
    '''
    ai_valid = df.dropna(subset=["AI"])
    pearson_ai, p_value_ai = pearsonr(ai_valid[expert_feature], ai_valid["AI"])
    print(f"Correlation between {expert_feature} and AI {similarity_metric}:")
    print(f"Pearson Correlation: {pearson_ai:.3f}, p-value: {p_value_ai:.3f}")

    web_valid = df.dropna(subset=["Web"])
    pearson_web, p_value_web = pearsonr(web_valid[expert_feature], web_valid["Web"])
    print(f"\nCorrelation between {expert_feature} and WEB {similarity_metric}:")
    print(f"Pearson Correlation: {pearson_web:.3f}, p-value: {p_value_web:.3f}")

def stats_tests(df: pd.DataFrame, expert_feature: str = 'aesthetics') -> None:
    '''
    # Hypothesis example:
    # "Groups with higher AI-based Dino_Similarity scores have higher aesthetics scores than groups with higher WEB-based scores."
    # One way to test this: compare the aesthetics scores for groups where AI similarity > WEB similarity
    # versus groups where AI similarity is not greater than WEB similarity.
    '''
    df["Similarity_Diff"] = df["AI"] - df["Web"]

    # Define groups based on whether AI similarity is greater than WEB.
    group_ai_better = df[df["Similarity_Diff"] > 0][expert_feature]
    group_web_better = df[df["Similarity_Diff"] <= 0][expert_feature]

    # Perform an independent samples t-test (using Welch’s t-test which does not assume equal variances).
    t_stat, t_p_value = ttest_ind(group_ai_better, group_web_better, equal_var=False)
    print(f"\nT-test comparing {expert_feature} scores between groups with AI > Web and groups with AI <= Web:")
    print(f"T-statistic: {t_stat:.3f}, p-value: {t_p_value:.3f}")

def visualize(
        df: pd.DataFrame, 
        expert_feature: str = 'aesthetics', 
        similarity_metric: str = 'Dino_Similarity',
        sim_aggr_method: str = 'mean',
        exp_aggr_method: str = 'mean'
        ) -> None:
    '''
    Visualize the relationship between expert scores and similarity scores.
    '''
    plt.figure(figsize=(8, 6))

    for source, color in zip(["AI", "Web"], ['red', 'blue']):
        valid = df.dropna(subset=[source])
        plt.scatter(valid[source], valid[expert_feature], c=color, label=f'{source}-based')
    
    plt.xlabel(f"{sim_aggr_method} {similarity_metric}")
    plt.ylabel(f"{exp_aggr_method} {expert_feature} Score")
    plt.title(f"{expert_feature} vs. {similarity_metric}")
    plt.legend()
    plt.show()


for score_col in score_cols:
    for sim_col in similarity_cols:
        for exp_aggr_method in ['mean']:
            print(f"\n======\nExpert data aggregation method: {exp_aggr_method}")
            for sim_aggr_method in ['mean', 'median', 'max']:
                print(f"===\nSimilarity data aggregation method: {sim_aggr_method}")
                print(f"\nAnalyzing {score_col} and {sim_col}...")
                df = aggregate_data(expert_df_clean, 
                                    sim_df_clean, 
                                    expert_feature=score_col, 
                                    similarity_metric=sim_col, 
                                    expert_aggr_method=exp_aggr_method,
                                    sim_aggr_method=sim_aggr_method) 
                correlation_analysis(df, expert_feature=score_col, similarity_metric=sim_col)
                stats_tests(df, expert_feature=score_col)
                visualize(df, 
                          expert_feature=score_col, 
                          similarity_metric=sim_col,
                          sim_aggr_method=sim_aggr_method,
                          exp_aggr_method=exp_aggr_method)