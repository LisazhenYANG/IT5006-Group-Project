## 4.X Threshold Adjustment Analysis

In addition to evaluating the model using ROC-AUC, which measures ranking ability, it is also important to analyze the impact of the decision threshold on classification performance.

### 1. Motivation for Threshold Adjustment

By default, a threshold of 0.5 is used to convert predicted probabilities into binary labels. However, in the NIBRS dataset, this default threshold leads to extremely low recall, indicating that the model is overly conservative in predicting hotspot areas.

To address this issue, different threshold values were tested, including 0.5, 0.4, 0.3, and 0.2. The results show a clear trade-off between precision and recall.

### 2. Results of Threshold Adjustment

When using the default threshold of 0.5, the model achieves very low recall (approximately 0.056), meaning that most hotspot areas are missed. This suggests that the model only predicts hotspots when it is highly confident.

After lowering the threshold to 0.3, recall increases significantly to approximately 0.693, while precision remains at a reasonable level (around 0.341). The F1-score also improves substantially to 0.457, which is the best among the tested thresholds.

Further reducing the threshold to 0.2 increases recall even more (above 0.9), but at the cost of lower precision, indicating a higher number of false positives.

### 3. Interpretation

These results indicate that the model still retains useful ranking ability (as reflected by the ROC-AUC of approximately 0.62), but the default threshold of 0.5 is not suitable for this dataset.

By lowering the threshold, the model becomes less conservative and is able to identify a larger proportion of hotspot areas. This is particularly important for crime hotspot detection, where missing true hotspots may be more costly than generating some false positives.

The confusion matrix at threshold 0.3 further supports this observation, showing a substantial increase in correctly identified hotspot cases (true positives), although accompanied by an increase in false positives.

### 4. Final Decision

Based on the results, a threshold of 0.3 is selected as a more appropriate decision boundary for the NIBRS dataset, as it provides a better balance between precision and recall.

It is important to note that the ROC-AUC remains unchanged under different thresholds, as it is computed based on predicted probabilities rather than binary predictions. The threshold adjustment only affects classification-based metrics such as precision, recall, and F1-score.

### 5. Summary

Overall, threshold adjustment plays an important role in improving classification performance. While the model's ranking ability is limited by domain shift, selecting an appropriate threshold allows the model to better align with the task objective and improves its practical usefulness.