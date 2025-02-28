# AI'nspired

---

For the streamlit app to work, you need to download the image files and put them inside **static**.

Change **Jupyter: Notebook File Root** to **${workspaceFolder}**, then every notebook should work.

---

## Expert scores analysis

Summary of the key findings based on the statistical tests. Note that significance is typically determined using a p-value threshold of 0.05. In these tests, a p-value below 0.05 is taken as evidence that the observed relationship or difference is unlikely to have occurred by chance.

### 1. Aesthetics

**Color_Similarity:**  
- **Mean/Median Aggregation:**  
  - **Correlations:** Both the AI (negative) and WEB (positive) correlations are not statistically significant.  
  - **T-test:** The t-test is significant (t = –2.610, p = 0.017), indicating that groups where AI Color_Similarity is higher than WEB tend to have lower aesthetics scores than groups where it is not.  
- **Max Aggregation:**  
  - The correlation for WEB is significant (r = 0.439, p = 0.025), suggesting that higher WEB-based similarity (when taking the maximum value) is associated with higher aesthetics scores.  
  - However, the t-test is not significant.

**ResNet_Similarity:**  
- **Mean Aggregation:**  
  - The AI correlation is significantly negative (r = –0.418, p = 0.034), meaning that higher AI ResNet_Similarity is associated with lower aesthetics ratings.  
  - The WEB correlation is not significant.  
  - The t-test does not show a significant difference between the groups.  
- **Max Aggregation:**  
  - The AI correlation trends negatively (r = –0.353, p = 0.077) and the t-test is borderline (t = –2.076, p = 0.051), suggesting a similar trend when using the maximum similarity values.

**Dino_Similarity:**  
- **Mean/Median Aggregation:**  
  - Neither the correlations nor the t-tests are statistically significant.  
- **Max Aggregation:**  
  - The WEB correlation trends moderately positive (r = 0.349, p = 0.081), and the t-test is significant (t = –2.944, p = 0.008).  
  - The negative t-statistic indicates that groups with higher AI Dino_Similarity (compared to WEB) tend to receive lower aesthetics scores.

**Contrast_Similarity:**  
- Across all aggregation methods, neither the correlations nor the t-tests are statistically significant.  
- **Conclusion for Aesthetics:**  
  Overall, when using mean or median aggregations, there is a tendency for groups with higher AI-based similarity (especially for Color and ResNet metrics) to score lower on aesthetics. The max aggregation for Dino_Similarity also reinforces that pattern, while WEB-based similarity sometimes shows a positive association with aesthetics.

---

### 2. Value

**Color_Similarity:**  
- **Max Aggregation:**  
  - The WEB correlation is significant (r = 0.410, p = 0.038), indicating that higher WEB-based Color_Similarity is associated with higher value scores.  
- Other aggregations and metrics do not show significant effects except for a non-significant trend with AI in some cases.

**ResNet_Similarity:**  
- **Mean Aggregation:**  
  - There is a significant negative correlation for AI (r = –0.395, p = 0.046), suggesting that groups with higher AI ResNet_Similarity tend to have lower value ratings.  
- Other tests (median or max) do not reach significance, though the max aggregation t-test is borderline (t = –1.900, p = 0.070).

**Dino_Similarity:**  
- **Max Aggregation:**  
  - The t-test is significant (t = –2.645, p = 0.017), indicating that groups with higher AI Dino_Similarity tend to have lower value scores, even though the direct correlations are not significant.

**Contrast_Similarity:**  
- No significant effects are observed for value.

**Conclusion for Value:**  
Some tests suggest that when final submissions are more similar to AI-generated inspirations (especially in ResNet and Dino metrics), the value scores tend to be lower. Conversely, a higher WEB-based similarity (in Color_Similarity with max aggregation) might be linked to higher value ratings.

---

### 3. Innovation

**Color_Similarity:**  
- **Mean/Median Aggregation:**  
  - The t-tests are significant (t = –2.824, p = 0.009), meaning that groups with higher AI-based Color_Similarity tend to have lower innovation scores.  
  - The AI correlation shows a trend toward a negative relationship (r ≈ –0.35, p just above 0.05) in mean aggregation.
- **Max Aggregation:**  
  - The WEB correlation is significant and positive (r = 0.401, p = 0.042), but the t-test is not significant.

**ResNet_Similarity:**  
- **Max Aggregation:**  
  - There is a significant negative correlation for AI (r = –0.495, p = 0.010), indicating that higher AI ResNet_Similarity is associated with lower innovation scores.  
  - The t-tests are not significant.

**Dino_Similarity:**  
- **Max Aggregation:**  
  - The t-test is significant (t = –3.050, p = 0.006), suggesting that groups with higher AI Dino_Similarity have lower innovation scores.

**Contrast_Similarity:**  
- None of the tests for innovation reach significance.

**Conclusion for Innovation:**  
There is consistent evidence—especially with mean/median aggregations for Color_Similarity and max for ResNet and Dino—that higher similarity to AI-generated inspirations is associated with lower innovation scores.

---

### 4. Surprise

**Color_Similarity:**  
- **Mean/Median Aggregation:**  
  - The AI correlation is significantly negative (r ≈ –0.425, p ≈ 0.030–0.020) and the t-tests are significant (t = –3.142, p = 0.004), indicating that groups with higher AI-based Color_Similarity tend to score lower on surprise.  
- **Max Aggregation:**  
  - The t-test is not significant.

**ResNet_Similarity:**  
- **Max Aggregation:**  
  - The AI correlation is significantly negative (r = –0.512, p = 0.007), showing a strong negative relationship between AI ResNet_Similarity and surprise scores.
  
**Dino_Similarity:**  
- **Max Aggregation:**  
  - The AI correlation is significantly negative (r = –0.403, p = 0.041) and the t-test is significant (t = –2.777, p = 0.012), again suggesting that higher AI-based similarity is associated with lower surprise scores.
  
**Contrast_Similarity:**  
- No significant effects are found for surprise.

**Conclusion for Surprise:**  
The tests indicate that higher similarity to AI inspirations—as measured by Color, ResNet, and Dino metrics (especially using mean/median or max aggregation)—is associated with lower surprise ratings.

---

### Overall Conclusions

- **General Pattern:**  
  Across several expert dimensions (aesthetics, innovation, and surprise), there is a recurring pattern: groups whose final submissions are more similar to AI-generated inspirations (i.e., when the similarity score for AI exceeds that for WEB) tend to receive lower expert scores.  
- **Metric-Specific Observations:**  
  - **Color_Similarity:**  
    The t-tests consistently indicate that groups with higher AI Color_Similarity score lower on aesthetics, innovation, and surprise.
  - **ResNet_Similarity and Dino_Similarity:**  
    Significant negative correlations and t-tests (especially with mean or max aggregation) reinforce the idea that higher AI-based similarity is associated with lower expert scores in several dimensions.
  - **Contrast_Similarity:**  
    There is little evidence of any relationship with expert scores.
- **WEB Similarity:**  
  In contrast, WEB-based similarity scores rarely show significant negative relationships—in a few cases, there’s even a positive association (for example, a significant positive correlation between WEB Color_Similarity and value when using max aggregation).

**Interpretation:**  
The data suggest that in this competition, final submissions that closely resemble AI-generated inspirations (as opposed to WEB-sourced ones) are generally rated lower by experts in terms of aesthetics, innovation, and surprise. This pattern is observed across several similarity metrics and aggregation methods, though it is not uniform across every test. These findings could imply that participants whose work leans too heavily on AI-inspired elements may be perceived as less original or compelling by experts.

---

This comprehensive analysis should guide further investigation into how the source of inspiration (AI vs. WEB) relates to expert evaluations of the final submissions.