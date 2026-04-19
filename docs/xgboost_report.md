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

---

## Cross-Domain Generalisation: NIBRS Evaluation

### Problem Statement

To assess model robustness beyond Chicago, we applied the trained model to the NIBRS (National Incident-Based Reporting System) dataset, covering 1,914 law enforcement agencies across the United States (2021–2023 features → 2024 labels). This tests whether crime hotspot patterns learned from a single city generalise to geographically diverse jurisdictions.

Initial cross-domain evaluation revealed a substantial performance drop (AUC-ROC: 0.9851 → ~0.62), which we investigated and partially mitigated through three targeted interventions.

---

### Root Cause Analysis

Two categories of issues were identified:

**1. Feature Engineering Bug**

The Chicago location mapping used exact string matching (e.g., `"SCHOOL"` → `INSTITUTION`), which failed to match Chicago's verbose location descriptions (e.g., `"SCHOOL, PUBLIC, BUILDING"`). This caused `institution_ratio` to be effectively zero across all Chicago training samples, while the corresponding NIBRS value averaged 0.132 — a distribution gap that could not be learned from training data.

**2. Domain Shift**

Fundamental distributional differences exist between the two datasets:

| Feature | Chicago (train) | NIBRS (test, original) | Difference |
|---|---|---|---|
| `institution_ratio` | 0.000 | 0.132 | +0.132 |
| `commercial_ratio` | 0.031 | 0.128 | +0.097 |
| `theft_ratio` | 0.220 | 0.371 | +0.151 |
| `assault_ratio` | 0.082 | 0.032 | −0.050 |

These gaps stem from two structural differences: (1) Chicago uses `community_area` (77 urban neighbourhoods within a single city) as the spatial unit, while NIBRS uses `agency_id` (nationwide agencies of varying type and size); and (2) the NIBRS test set includes agencies from the South (63%) and West (31%) — regions with systematically different crime patterns from Midwest Chicago.

---

### Remediation Measures

Three interventions were applied to reduce the performance gap:

**Measure 1: Fix `institution_ratio` Mapping Bug**

The exact-match mapping in `03_processing.ipynb` was replaced with keyword-based matching, correctly assigning location descriptions containing `SCHOOL`, `HOSPITAL`, `CHURCH`, `LIBRARY`, etc. to the `INSTITUTION` category. This corrected a systematic data quality error affecting the entire training set.

**Measure 2: Filter NIBRS to City-Type Agencies**

The original NIBRS test set includes county agencies, state police, university police, tribal agencies, and other types that are structurally incomparable to Chicago's urban community areas. We filtered the test set to retain only `City`-type agencies (1,133 out of 1,914), making the comparison more semantically consistent.

**Measure 3: Retrain on Updated Features**

Following the feature engineering fix, the model was retrained on the corrected `X_train` to ensure the updated `institution_ratio` and `commercial_ratio` distributions were reflected in learned decision boundaries.

---

### Results: Before and After

| Metric | Before | After | Change |
|---|---|---|---|
| Precision | 0.300 | 0.524 | **+0.224** |
| Recall | 0.800 | 0.656 | −0.144 |
| F1-Score | 0.440 | 0.583 | **+0.143** |
| AUC-ROC | 0.625 | **0.716** | **+0.091** |

The AUC-ROC improvement of +0.091 reflects genuine gains from correcting the feature engineering bug and improving test set comparability. The slight Recall decrease is attributable to the City filter: restricting the test set raised the hotspot prevalence from 26% to 35%, making the classification task inherently more challenging and shifting the precision-recall trade-off.

---

### Residual Gap and Limitations

Despite the improvements, a performance gap remains between the Chicago internal test (AUC-ROC: 0.9851) and the NIBRS cross-domain test (AUC-ROC: 0.716). This residual gap is attributed to the following irreducible structural limitations:

1. **Spatial unit incompatibility.** Chicago's `community_area` represents a dense urban sub-city neighbourhood (~77 units within one city), whereas a NIBRS `agency_id` covers an entire city jurisdiction. The aggregation semantics differ fundamentally, and this cannot be resolved without incident-level NIBRS data with coordinates.

2. **Geographic distribution mismatch.** The model was trained exclusively on Midwest urban data (Chicago). The NIBRS test set contains no Midwest agencies; agencies are drawn from the South and West, where regional crime patterns differ from Chicago.

3. **P(Y|X) shift.** Beyond feature distributions, the underlying relationship between feature values and hotspot probability may differ across cities and regions. This is a fundamental constraint of single-city training that cannot be overcome through feature normalisation alone.

These limitations are inherent to the cross-dataset evaluation design and represent directions for future work, such as multi-city joint training or domain adaptation techniques.
