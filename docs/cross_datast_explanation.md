## 4.X Cross-dataset Explanation: Domain Shift Analysis

To evaluate the generalizability of the model, the Random Forest trained on the Chicago dataset was applied to the NIBRS dataset. A significant drop in performance was observed, with ROC-AUC decreasing from a high value on the Chicago dataset to approximately 0.62 on the NIBRS dataset. This indicates that the model does not generalize well across datasets.

This performance gap is mainly caused by a domain shift, where the data distribution and feature-label relationships differ between the two datasets. The issue can be analyzed from three aspects: feature distribution, feature importance, and their impact on model behavior.

### 1. Feature Distribution Shift

A direct comparison of feature means shows clear differences between the two datasets. For example, the average value of `theft_ratio` increases from approximately 0.22 in the Chicago dataset to 0.37 in the NIBRS dataset. At the same time, `institution_ratio`, which is almost zero in Chicago, rises to around 0.13 in NIBRS. In contrast, `other_ratio` decreases significantly from about 0.32 to 0.15.

These changes indicate that the overall composition of crime types and location categories differs substantially. In particular, the increase in institutional and commercial-related features suggests that the NIBRS dataset includes a broader range of environments (e.g., different cities or regions), while the Chicago dataset reflects a more consistent urban structure.

As a result, the feature space in the two datasets is not aligned. The model trained on Chicago is exposed to feature values and combinations that were rarely or never seen during training, which weakens its predictive ability.

### 2. Feature Importance Shift

To further understand the difference, two Random Forest models with identical hyperparameters were trained separately on the Chicago and NIBRS datasets, and their feature importance was compared.

In the Chicago dataset, crime-type features such as `theft_ratio` and `battery_ratio` dominate the model, indicating that hotspot prediction is mainly driven by the composition of crime types within each area. Location-related features play a relatively smaller role.

However, in the NIBRS dataset, location-related features such as `commercial_ratio` become the most important, and `institution_ratio` also gains noticeable importance. Meanwhile, some crime-related features, such as `assault_ratio`, increase in importance compared to Chicago.

This shift suggests that the relationship between features and the hotspot label is not consistent across datasets. In other words, the model learns dataset-specific patterns: what defines a "hotspot" in Chicago is not the same as in NIBRS.

### 3. Impact on Model Behavior

The combination of distribution shift and importance shift leads to a mismatch between the model and the new data. Specifically, the decision boundaries learned from the Chicago dataset are based on feature ranges and relationships that no longer hold in the NIBRS dataset.

As a result, the model's ability to correctly rank hotspot and non-hotspot areas is reduced. Although the ROC-AUC of approximately 0.62 indicates that the model still retains some ranking ability, the performance is much lower than in the original dataset.

This also explains why the model may still show some predictive signal, but fails to perform reliably when applied to a different data source.

### 4. Summary

Overall, the performance degradation is mainly caused by a domain shift problem. Differences in feature distributions and feature importance indicate that the underlying data patterns are not consistent across datasets. Therefore, a model trained on one dataset cannot be directly applied to another without adaptation.

This highlights the importance of considering data consistency and transferability when evaluating machine learning models in real-world scenarios.