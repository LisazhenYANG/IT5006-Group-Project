# Random Forest Model Report: Chicago Crime Hotspot Prediction

---

## 1. Problem Definition & Data Preparation

### 1.1 Problem Statement

The goal of this project is to predict whether a Chicago community area will be a **crime hotspot** in the following year. This is formulated as a **binary classification** task: given aggregated historical crime features for each of Chicago's 77 community areas, predict whether a community area will fall in the top 25% of all areas by crime count (hotspot = 1) or not (hotspot = 0).

The ability to identify hotspots one year in advance enables law enforcement to make proactive, data-driven resource allocation decisions — increasing patrol presence, deploying community policing programs, and prioritising investigative resources before crime escalates.

---

### 1.2 Dataset Overview

| Dataset | Source Period | Raw Records | Records After Preprocessing |
|---|---|---|---|
| Training | 2015–2024 | 2,519,642 | 2,424,554 |
| Test | 2025 | 236,552 | 231,996 |

The raw dataset contains 22 attributes per crime incident, including temporal information (date), crime classifications (primary_type, description), location attributes (latitude, longitude, district, ward, community_area, beat, location_description), and case outcomes (arrest, domestic).

---

### 1.3 Data Preprocessing & Cleaning

The preprocessing pipeline consists of four stages:

**Stage 1 — Column Reduction**
Nine redundant or non-predictive columns were dropped: `id`, `case_number`, `block`, `iucr`, `fbi_code`, `x_coordinate`, `y_coordinate`, `location`, `updated_on`. The retained columns are: `date`, `primary_type`, `location_description`, `arrest`, `beat`, `district`, `ward`, `community_area`, `latitude`, `longitude`.

**Stage 2 — Missing Value Handling**
- Rows missing `latitude`, `longitude`, or `district` were removed (42,362 rows removed from training data, 94 from test).
- Remaining missing `location_description` values (8,251 in training; 1,080 in test) were filled with `"UNKNOWN"`.
- Missing `community_area` values were imputed using the mode.

**Stage 3 — Deduplication and Type Conversion**
- Duplicate records sharing the same `date` and `beat` were removed (2,428,544 unique records retained).
- The `date` column was parsed into `datetime64` format to enable temporal feature extraction.
- `primary_type` and `location_description` were cast to `category` type for memory efficiency.

**Stage 4 — Geographic Boundary Validation**
Records with coordinates outside Chicago's geographic bounds (latitude: 41.64–42.02°N, longitude: −87.94 to −87.52°W) were removed, retaining 2,424,554 training records and 231,996 test records.

---

### 1.4 Feature Engineering Strategy

Raw incident-level records were transformed into **community-area level spatial features** capturing historical crime patterns. Each row in the final dataset represents one community area in one time window.

**Rolling Window Design**
A 3-year sliding window was used to generate training samples. For each target year from 2018 to 2024, features were derived from the preceding 3 years of crime data, and the label indicates whether the community area was a hotspot in the target year. This produces 7 windows × 77 community areas = **539 training samples**.

For the test set, features are derived from 2022–2024 data to predict hotspot status in 2025, producing **77 test samples**.

**Hotspot Label Definition**
A community area is labelled a hotspot (`y = 1`) if its annual crime count is at or above the 75th percentile across all 77 community areas in that year. This threshold produces a class ratio of approximately 26% hotspots vs. 74% non-hotspots.

**Engineered Features (36 total)**

| Feature Group | Features | Description |
|---|---|---|
| Crime volume | `crime_count_last3y` | Total crime incidents in the area over the 3-year window |
| Geographic center | `lat_mean`, `lon_mean` | Mean latitude and longitude of crimes in the area (MinMax scaled) |
| Location type ratios | `commercial_ratio`, `institution_ratio`, `other_ratio`, `public_ratio`, `residential_ratio` | Proportion of crimes by venue category |
| Crime type ratios | `theft_ratio`, `battery_ratio`, `criminal_damage_ratio`, `assault_ratio`, `deceptive_practice_ratio`, `other_crime_ratio` | Proportion of each major crime type within the area |
| District identity | `district_1` … `district_25` (22 columns) | One-hot encoded dominant police district of the area |

Location categories were derived by mapping 100+ raw `location_description` values into five groups: `PUBLIC_OUTDOOR`, `RESIDENTIAL`, `COMMERCIAL`, `INSTITUTION`, and `OTHER`. Crime type ratios cover the top 5 crime types by frequency (Theft, Battery, Criminal Damage, Assault, Deceptive Practice), with all remaining types aggregated into `other_crime_ratio`.

**Class Distribution (Training Set)**

| Class | Count | Proportion |
|---|---|---|
| Non-Hotspot (0) | 399 | 74.0% |
| Hotspot (1) | 140 | 26.0% |

The class imbalance ratio is approximately 2.85:1 (negative to positive).

---

## 2. Model Implementation & Training

### 2.1 Model Architecture

A **Random Forest Classifier** (scikit-learn `RandomForestClassifier`) was selected for this task. Random Forest is an ensemble method that constructs multiple decorrelated decision trees and aggregates their predictions via majority voting. Key design properties relevant to this problem:

- **Handles mixed feature types**: The feature set contains both continuous (ratios, coordinates) and binary (district indicators) features, which decision trees handle naturally without scaling (except for interpretability purposes).
- **Robust to class imbalance**: The `class_weight="balanced"` parameter automatically adjusts sample weights inversely proportional to class frequencies, correcting for the 2.85:1 imbalance.
- **Provides feature importance**: Impurity-based feature importances are a direct output of the model, supporting interpretability requirements.
- **Parallelisable**: `n_jobs=-1` enables full CPU utilisation, important given the two-stage grid search.

### 2.2 Training Process

Training proceeded in three stages:

**Stage 1 — Baseline Model**
An initial baseline was trained with `n_estimators=200`, `class_weight='balanced'`, and `random_state=42` to establish a performance reference before hyperparameter tuning.

**Stage 2 — Coarse Grid Search**
A `GridSearchCV` with Stratified 5-Fold Cross-Validation (`StratifiedKFold`, `n_splits=5`, `shuffle=True`, `random_state=42`) was applied over 54 parameter combinations. Multiple scoring metrics were tracked simultaneously (accuracy, precision, recall, F1, AUC-ROC), with `refit="roc_auc"` to select the best model by AUC-ROC.

| Hyperparameter | Search Range | Rationale |
|---|---|---|
| `n_estimators` | [100, 200, 300] | More trees → more stable predictions |
| `max_depth` | [None, 10, 20] | Controls overfitting via tree depth |
| `min_samples_leaf` | [1, 2, 4] | Smoother boundaries with larger values |
| `max_features` | ['sqrt', 'log2'] | Feature subsampling reduces inter-tree correlation |
| `class_weight` | ['balanced'] | Corrects for class imbalance |

Best coarse-search parameters: `{class_weight: balanced, max_depth: 10, max_features: sqrt, min_samples_leaf: 1, n_estimators: 300}`, Best CV AUC-ROC: **0.9701**

**Stage 3 — Fine Grid Search**
Building on the coarse result, a second grid search narrowed the search space around the best coarse-search region across 48 combinations:

| Hyperparameter | Fine Search Range |
|---|---|
| `n_estimators` | [250, 300, 350, 400] |
| `max_depth` | [8, 10, 12, 14] |
| `max_features` | ['sqrt'] |
| `min_samples_leaf` | [1, 2, 3] |
| `class_weight` | ['balanced'] |

### 2.3 Hyperparameter Tuning Results

**Top 10 Fine-Search Cross-Validation Results** (ranked by AUC-ROC):

| Rank | n_estimators | max_depth | min_samples_leaf | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|---|---|---|
| 1 | 350 | 10 | 1 | 0.9350 | 0.9024 | 0.8429 | 0.8699 | **0.9713** |
| 2 | 350 | 12 | 1 | 0.9406 | 0.9179 | 0.8500 | 0.8811 | 0.9711 |
| 3 | 400 | 10 | 1 | 0.9350 | 0.8981 | 0.8500 | 0.8710 | 0.9710 |
| 4 | 400 | 12 | 1 | 0.9387 | 0.9172 | 0.8429 | 0.8767 | 0.9707 |
| 5 | 250 | 10 | 1 | 0.9350 | 0.9040 | 0.8429 | 0.8699 | 0.9707 |
| 6 | 350 | 14 | 1 | 0.9424 | 0.9233 | 0.8500 | 0.8837 | 0.9704 |
| 7 | 300 | 8  | 1 | 0.9239 | 0.8681 | 0.8357 | 0.8502 | 0.9703 |
| 8 | 400 | 8  | 1 | 0.9258 | 0.8694 | 0.8429 | 0.8542 | 0.9702 |
| 9 | 300 | 10 | 1 | 0.9369 | 0.9107 | 0.8429 | 0.8732 | 0.9701 |
| 10 | 400 | 14 | 1 | 0.9406 | 0.9230 | 0.8429 | 0.8796 | 0.9701 |

**Optimal Hyperparameters (Final Model)**:

```
n_estimators    = 350
max_depth       = 10
max_features    = 'sqrt'
min_samples_leaf = 1
class_weight    = 'balanced'
random_state    = 42
n_jobs          = -1
```

Best Cross-Validation AUC-ROC: **0.9713**

The clustering of top results around `max_depth=10` and `min_samples_leaf=1` confirms that moderate tree depth provides the best bias-variance balance. Deeper trees (`max_depth=12, 14`) achieve higher accuracy and precision on the CV set but do not consistently improve AUC-ROC, suggesting the marginal gain in discriminative power does not justify the increased model complexity.

---

## 3. Model Evaluation & Comparison

### 3.1 Standard Classification Metrics

**Test Set Performance (Final Model, trained on full training set)**:

| Metric | Score |
|---|---|
| Accuracy | 0.9481 |
| Precision | **1.0000** |
| Recall | 0.8000 |
| F1-Score | 0.8889 |
| AUC-ROC | **0.9904** |

The model achieves perfect precision (1.0000), meaning every location it predicts as a hotspot is a true hotspot — no false positives. Recall of 0.8000 indicates that 80% of all true hotspots are correctly identified, with 20% missed. The AUC-ROC of 0.9904 reflects exceptional discriminative ability across all classification thresholds.

**Confusion Matrix (Test Set)**:

|  | Predicted: Non-Hotspot | Predicted: Hotspot |
|---|---|---|
| **Actual: Non-Hotspot** | 57 (TN) | 0 (FP) |
| **Actual: Hotspot** | 4 (FN) | 16 (TP) |

The model produces zero false positives — all 57 non-hotspot areas are correctly classified. The 4 false negatives represent community areas that were true hotspots but went undetected. This is consistent with the reported metrics: (57 + 16) / 77 = 0.9481 accuracy, and 16 / 20 = 0.8000 recall.

---

### 3.2 Cross-Validation Results

5-Fold Stratified Cross-Validation results (fine-tuned model, optimised configuration):

| Metric | CV Mean |
|---|---|
| Accuracy | 0.9350 |
| Precision | 0.9024 |
| Recall | 0.8429 |
| F1-Score | 0.8699 |
| AUC-ROC | **0.9713** |

The CV AUC-ROC of 0.9713 versus the test AUC-ROC of 0.9904 indicates strong generalisation, with the model performing even better on the held-out 2025 test data than on the training-era cross-validation folds. This suggests that the crime patterns in 2025 are well-represented by the 2015–2024 historical training features.

---

### 3.3 Spatial Accuracy Evaluation

The test set was partitioned into four geographic quadrants using the median values of `lat_mean` and `lon_mean` to evaluate spatial consistency of predictions:

| Quadrant | N | Hotspot% | Precision | Recall | F1 | Notes |
|---|---|---|---|---|---|---|
| North-East | — | — | — | — | — | Covered by XGBoost report; RF spatial breakdown not separately computed |
| Overall | 77 | 26.0% | 1.0000 | 0.8000 | 0.8889 | Zero false positives across all areas |

The Random Forest model's perfect precision across the full test set indicates that spatial prediction errors, when they occur, are exclusively false negatives (missed hotspots), never false alarms. This characteristic is particularly valuable in operational contexts where misallocating resources to non-hotspot areas carries a significant opportunity cost.

---

### 3.4 Temporal Accuracy

The feature set contains aggregated historical statistics computed over 3-year rolling windows. Temporal features (hour, weekday, month) are not included at the community-area level because the aggregation collapses individual event timestamps. The rolling window design does encode temporal evolution — each training sample represents a different year — but within-year temporal variation is not captured. This limitation is discussed further in Section 4.3.

---

### 3.5 Model Robustness Analysis — Feature Ablation

To validate the contribution of individual features, an ablation experiment was conducted by retraining the best model after removing `crime_count_last3y`:

| Configuration | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Full Model (with `crime_count_last3y`) | 0.9481 | 1.0000 | 0.8000 | 0.8889 | **0.9904** |
| Ablated Model (without `crime_count_last3y`) | 0.9221 | 0.8500 | 0.8500 | 0.8500 | 0.9833 |

Removing `crime_count_last3y` caused:
- Accuracy to drop from 0.9481 → 0.9221 (−2.6pp)
- Precision to drop from 1.0000 → 0.8500 (−15.0pp), introducing false positives for the first time
- AUC-ROC to drop from 0.9904 → 0.9833 (−0.71pp)

This confirms that raw crime volume (`crime_count_last3y`) is a critical feature. However, the ablated model still achieves AUC-ROC = 0.9833, demonstrating that the remaining 34 ratio-based and geographic features carry strong independent predictive signal.

Interestingly, the ablated model equalises precision and recall at 0.850 each, suggesting that `crime_count_last3y` primarily contributes to eliminating false positives — areas where crime composition ratios resemble hotspots but absolute crime volume is insufficient to warrant the label.

---

### 3.6 Comparison with Baseline

| Model Stage | AUC-ROC (CV) | Key Settings |
|---|---|---|
| Baseline | not evaluated | n_estimators=200, max_depth=None |
| After Coarse Search | 0.9701 | n_estimators=300, max_depth=10 |
| After Fine Search (Final) | **0.9713** | n_estimators=350, max_depth=10 |
| Final — Test Set | **0.9904** | Same as fine search best |

Note: The baseline model was trained without cross-validation; no CV AUC was computed for it. Comparisons are therefore made between the coarse-search best and the fine-search best.

Constraining `max_depth` to 10 was the most consistent finding across both search stages. The coarse search showed that `max_depth=None` (unbounded) was never selected as optimal, and the fine search confirmed that `max_depth=10` produced the highest CV AUC-ROC. The `sqrt` feature sampling was also consistently preferred over `log2`.

---

## 4. Model Interpretation & Business Insights

### 4.1 Feature Importance Analysis

Feature importances computed from the final model (mean decrease in impurity, aggregated across 350 trees):

| Rank | Feature | Importance |
|---|---|---|
| 1 | `theft_ratio` | 0.1185 |
| 2 | `battery_ratio` | 0.1094 |
| 3 | `lat_mean` | 0.0967 |
| 4 | `commercial_ratio` | 0.0906 |
| 5 | `deceptive_practice_ratio` | 0.0794 |
| 6 | `other_ratio` | 0.0784 |
| 7 | `criminal_damage_ratio` | 0.0617 |
| 8 | `other_crime_ratio` | 0.0599 |
| 9 | `residential_ratio` | 0.0583 |
| 10 | `public_ratio` | 0.0518 |
| 11 | `lon_mean` | 0.0508 |
| 12 | `assault_ratio` | 0.0291 |
| 13 | `district_2` | 0.0188 |
| 14 | `district_6` | 0.0172 |
| 15 | `district_9` | 0.0126 |

The top 4 features (`theft_ratio`, `battery_ratio`, `lat_mean`, `commercial_ratio`) together account for approximately **41.5% of total model importance**. The feature importance pattern reveals two dominant signal sources:

1. **Crime composition** (theft, battery, deceptive practice ratios): Areas where a disproportionate share of crimes are property crimes (theft) and interpersonal violence (battery) tend to be hotspots. These crime types are associated with opportunistic offending in densely populated or commercially active areas.

2. **Geographic position** (`lat_mean`, `lon_mean`): The spatial coordinates of an area carry significant predictive power, reflecting persistent geographic clustering of crime that is not fully captured by crime composition alone.

**Ablation Feature Importance (model without `crime_count_last3y`)**:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `battery_ratio` | 0.1240 |
| 2 | `theft_ratio` | 0.1169 |
| 3 | `commercial_ratio` | 0.1047 |
| 4 | `lat_mean` | 0.0899 |
| 5 | `other_ratio` | 0.0808 |

When `crime_count_last3y` is removed, `commercial_ratio` rises from rank 4 to rank 3 (importance: 0.0906 → 0.1047), confirming that commercial location density is a strong structural predictor of crime hotspot formation independent of raw volume.

---

### 4.2 Actionable Insights for Law Enforcement

**1. Target areas with high theft and battery composition.**
`theft_ratio` (11.9%) and `battery_ratio` (10.9%) are the two most important predictors. Community areas with higher proportions of theft and battery crimes are associated with elevated hotspot risk. Dedicated patrol scheduling and anti-theft programs in retail and public transit corridors should be prioritised in these areas.

**2. Monitor commercial zones as leading indicators.**
`commercial_ratio` ranks 4th in importance (9.1%). Commercial activity creates concentrated foot traffic and opportunity for property crime. Areas with increasing `commercial_ratio` over time should be tracked as early-warning signals for emerging hotspot formation, even before crime volume reaches the hotspot threshold.

**3. Use geographic coordinates for spatial resource allocation.**
`lat_mean` and `lon_mean` together contribute ~14.7% of model importance, reflecting persistent spatial inequality in crime distribution that is not fully explained by crime composition alone. The higher importance of `lat_mean` over `lon_mean` suggests that north–south position carries more predictive signal than east–west position, though the model does not explicitly identify which geographic zones are higher risk. Resource allocation should be informed by the model's community-area-level predictions rather than coordinate thresholds alone.

**4. Leverage zero-false-positive precision for confident deployment.**
The model's perfect precision (1.0000) on the test set means that its positive predictions can be acted upon with high confidence. Every community area flagged by the model as a hotspot for 2025 is genuinely a high-crime area. This makes the model appropriate for directing significant resource investments (e.g., permanent police sub-stations, community intervention programs) without risk of misallocating resources to false alarms.

**5. Supplement model predictions with recall-aware planning.**
With recall of 0.8000, the model misses approximately 1 in 5 true hotspots. Law enforcement should not treat areas not flagged by the model as definitively safe. Secondary monitoring protocols — such as monthly crime volume tracking against rolling baselines — should be applied to areas near the classification boundary to catch emerging hotspots before they reach critical levels.

**6. Use deceptive practice ratio as a fraud and financial crime proxy.**
`deceptive_practice_ratio` (7.9% importance) captures fraud, identity theft, and confidence crimes. Areas with elevated deceptive practice ratios may benefit from targeted consumer protection outreach, financial fraud task forces, and collaboration with banking institutions beyond traditional patrol responses.

---

### 4.3 Limitations and Constraints

**1. No within-year temporal granularity.**
The feature set aggregates crime incidents across 3 years without preserving temporal patterns such as hour-of-day, day-of-week, or seasonal variation. The model predicts annual hotspot status and cannot inform day-to-day or shift-level patrol scheduling. Incorporating temporal features (e.g., month-of-year crime volume trends) in future iterations would improve operational applicability.

**2. Small sample size at the community-area level.**
With only 77 community areas, the modelling dataset contains 539 training samples and 77 test samples. Despite strong performance, the small sample size limits statistical power for detecting subtle feature interactions and may cause instability in district-level importance estimates (e.g., the relative rankings of `district_2`, `district_6`, `district_9` are sensitive to small count changes).

**3. Static hotspot definition.**
The 75th percentile threshold for hotspot labelling is computed independently within each year. This means the absolute crime count required to be labelled a hotspot varies from year to year, potentially mislabelling areas that improved substantially but remained relatively high due to citywide trends.

**4. Impurity-based importance bias.**
Scikit-learn's impurity-based feature importance is known to overestimate the importance of high-cardinality continuous features (like `lat_mean`, `lon_mean`) relative to low-cardinality binary features (district indicators). Permutation importance or SHAP values would provide a more reliable importance ranking but were not computed in this analysis.

**5. Causal vs. correlational inference.**
High feature importance indicates predictive correlation, not causal influence. Acting on model predictions without domain expertise risks reinforcing historical policing patterns — for example, systematically flagging areas that have historically received higher patrol coverage (and therefore more recorded incidents) as hotspots, independent of underlying crime rates. Model predictions should always be interpreted alongside sociological and economic context.

**6. Temporal drift and model decay.**
The model is trained on 2015–2024 data and evaluated on 2025. Crime patterns evolve due to demographic shifts, economic conditions, and policy changes. The strong 2025 test performance (AUC-ROC = 0.9904) does not guarantee sustained accuracy beyond this period. Annual model retraining using the most recent 3-year window is recommended to maintain predictive validity.

---

## Appendix: Model Configuration Summary

| Parameter | Value |
|---|---|
| Model | `sklearn.ensemble.RandomForestClassifier` |
| n_estimators | 350 |
| max_depth | 10 |
| max_features | 'sqrt' |
| min_samples_leaf | 1 |
| class_weight | 'balanced' |
| random_state | 42 |
| n_jobs | -1 |
| CV Strategy | Stratified 5-Fold |
| Tuning Method | Two-stage GridSearchCV (coarse + fine) |
| Optimisation Metric | AUC-ROC |
| Training Samples | 539 |
| Test Samples | 77 |
| Features | 36 (including `crime_count_last3y`) |
| Training Period | 2015–2024 (rolling 3-year windows) |
| Prediction Target Year | 2025 |
