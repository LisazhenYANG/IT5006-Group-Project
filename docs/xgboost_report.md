# XGBoost Model Report

## Model Implementation & Training

**Model Architecture**

We implemented an XGBoost binary classifier to predict whether a location is a crime hotspot. The model was trained on 539 samples with 35 engineered features, with a class ratio of 399 (non-hotspot) to 140 (hotspot). To handle class imbalance, `scale_pos_weight` was set to 2.85 (ratio of negative to positive samples).

**Training Process**

A Stratified 5-Fold Cross-Validation strategy was adopted throughout to ensure consistent class distribution across folds.

**Hyperparameter Tuning**

We used Optuna with Tree-structured Parzen Estimator (TPE) sampling — a Bayesian optimization approach — to search over 9 hyperparameters across 50 trials, optimizing for mean cross-validated AUC-ROC. The search space was:

| Parameter | Range |
|---|---|
| n_estimators | 100 – 600 |
| max_depth | 3 – 8 |
| learning_rate | 0.01 – 0.3 (log scale) |
| subsample | 0.5 – 1.0 |
| colsample_bytree | 0.5 – 1.0 |
| reg_alpha (L1) | 1e-3 – 10 (log scale) |
| reg_lambda (L2) | 1e-3 – 10 (log scale) |
| min_child_weight | 1 – 10 |
| gamma | 0.0 – 5.0 |

The best configuration was found at trial 42: `n_estimators=205, max_depth=3, learning_rate=0.071, subsample=0.780, colsample_bytree=0.714, reg_alpha=0.001, reg_lambda=0.703, min_child_weight=1, gamma=1.069`.

---

## Model Evaluation & Comparison

**Cross-Validation Results**

| Metric | Baseline (CV mean ± std) | Tuned (CV mean ± std) |
|---|---|---|
| Precision | 0.8806 ± 0.0378 | 0.8552 ± 0.0744 |
| Recall | 0.8786 ± 0.0535 | 0.9071 ± 0.0429 |
| F1-Score | 0.8782 ± 0.0323 | 0.8774 ± 0.0382 |
| AUC-ROC | 0.9718 ± 0.0135 | **0.9750 ± 0.0123** |

Tuning improved AUC-ROC from 0.9718 to 0.9750 and notably increased recall (0.8786 → 0.9071), which is desirable in crime prediction as missing a true hotspot is costlier than a false alarm.

**Test Set Performance**

| Metric | Score |
|---|---|
| Accuracy | 0.9221 |
| Precision | 0.8500 |
| Recall | 0.8500 |
| F1-Score | 0.8500 |
| AUC-ROC | **0.9851** |

**Spatial Accuracy Evaluation**

The test set was partitioned into four geographic quadrants using the median of `lat_mean` and `lon_mean`:

| Quadrant | N | Hotspot% | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|---|
| North-East | 12 | 50.0% | 1.0000 | 0.6667 | 0.8000 | 1.0000 |
| North-West | 27 | 22.2% | 0.7143 | 0.8333 | 0.7692 | 0.9762 |
| South-East | 27 | 25.9% | 0.8750 | 1.0000 | 0.9333 | 1.0000 |
| South-West | 11 | 9.1% | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The model performs consistently well across all quadrants (AUC-ROC ≥ 0.976). The North-West quadrant shows slightly lower F1 (0.769), likely due to its lower hotspot density (22.2%) making boundary cases harder to classify.

**Temporal Accuracy**

The feature set contains aggregated historical statistics without timestamps; temporal evaluation is therefore not applicable. This limitation is discussed in the Limitations section below.

**Model Robustness**

The low standard deviation across CV folds (AUC-ROC std = 0.0123) indicates stable performance and good generalisability to unseen data.

---

## Model Interpretation & Business Insights

**Feature Importance & Ablation**

The top 5 features by importance are `theft_ratio`, `commercial_ratio`, `deceptive_practice_ratio`, `district_2`, and `district_6`. Ablation results show that removing `district_2` caused the largest AUC-ROC drop (0.9851 → 0.9781), confirming that geographic district identity is a critical signal. Interestingly, removing `theft_ratio` marginally increased test AUC-ROC (0.9851 → 0.9868), suggesting it may introduce slight noise on the test set despite its high training importance.

**Key Findings from Feature Importance**

The top predictive features are crime-type ratios (e.g., `theft_ratio`, `commercial_ratio`, `deceptive_practice_ratio`) and district indicators (`district_2`, `district_6`), indicating that both the composition of past crime types and geographic district identity are key signals for hotspot formation.

**Actionable Insights for Law Enforcement**

1. **Prioritise patrol resources by crime type mix.** Locations with high theft and commercial crime ratios are among the strongest predictors of hotspots. Patrol schedules should weight these areas more heavily.

2. **Geographic concentration matters.** The spatial evaluation shows that model performance varies across quadrants, suggesting crime hotspots are not uniformly distributed. Resources should be directed toward the quadrants with the highest predicted hotspot density.

3. **Commercial context.** `commercial_ratio` is among the top features. Commercial zones tend to attract opportunistic crimes; dedicated surveillance or community policing programs in high-`commercial_ratio` areas may reduce hotspot risk.

4. **Early warning potential.** Because the model is trained on aggregated historical features rather than real-time events, it can serve as a strategic planning tool — informing monthly or quarterly resource allocation decisions rather than day-to-day dispatch.

**Limitations & Constraints**

1. **No temporal dimension.** The current feature set contains aggregated statistics without timestamps. The model cannot capture seasonal patterns, time-of-day effects, or evolving crime trends. Future work should incorporate temporal features (e.g., month, day-of-week, rolling windows).

2. **Geographic granularity.** Without official district labels, spatial evaluation relies on a simple lat/lon quadrant split. This may mask within-quadrant heterogeneity. Incorporating census tract or police district boundaries would improve spatial interpretability.

3. **Static hotspot definition.** The binary hotspot label is derived from historical data. In practice, crime patterns shift; the model requires periodic retraining to remain accurate.

4. **Class imbalance.** Hotspot locations are a minority class. Despite `scale_pos_weight` correction, the model may still under-detect emerging hotspots in areas with historically low crime counts.

5. **Causal vs. correlational.** High feature importance does not imply causality. Acting solely on model predictions without domain expertise risks reinforcing existing policing biases rather than addressing root causes of crime.
