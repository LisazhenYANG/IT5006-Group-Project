# NIBRS Test Dataset Usage Guide

## 1. Overview

This folder contains the processed NIBRS dataset used to evaluate model generalization.

The dataset is constructed to match the feature structure of the Chicago training data, enabling direct testing of the trained model on a different dataset.

---

## 2. Files Description

### `X_test_nibrs.csv`

Feature dataset for model input.

* Each row represents one agency
* Features are aggregated from 2021–2023
* Includes:

  * crime type ratios (e.g., `theft_ratio`, `battery_ratio`, etc.)
  * location type ratios
  * `agency_id` (used only for merging)

---

### `y_test_nibrs.csv`

Ground truth labels.

* Each row represents one agency
* Includes:

  * `agency_id`
  * `label` (1 = hotspot, 0 = non-hotspot)
* Labels are defined based on 2024 crime data (top 25% threshold)

---

## 3. How to Use

### Step 1: Load data

```python
import pandas as pd

X = pd.read_csv("../data/nibrs/X_test_nibrs.csv")
y = pd.read_csv("../data/nibrs/y_test_nibrs.csv")
```

---

### Step 2: Merge datasets

```python
data = X.merge(y, on="agency_id", how="inner")
```

---

### Step 3: Prepare model input

```python
X_nibrs_model = data.drop(columns=["agency_id", "label"])
y_nibrs = data["label"]
```

---
### Step 4: Remove latitude/longitude and district one-hot columns in X_train/X_test
drop_cols = ["lat_mean", "lon_mean"] + [col for col in X_train.columns if col.startswith("district_")]

X_train_aligned = X_train.drop(columns=drop_cols, errors="ignore").copy()
X_test_aligned = X_test.drop(columns=drop_cols, errors="ignore").copy()
```

---

### Step 4: Align features with training data

```python
X_nibrs_model = X_nibrs_model.reindex(columns=X_train_aligned.columns, fill_value=0)
```

---

### Step 5: Make predictions

```python
y_pred = model.predict(X_nibrs_model)
y_prob = model.predict_proba(X_nibrs_model)[:, 1]
```

---

## 4. Important Notes

* `agency_id` is used only for merging and alignment, and must be removed before model prediction.
* Feature columns are aligned with the Chicago training dataset.
* Missing features are filled with 0.
* Only agencies with both historical data (2021–2023) and label data (2024) are included.

---

## 5. Purpose

This dataset is used to evaluate how well a model trained on Chicago data generalizes to a different dataset (NIBRS), simulating a real-world deployment scenario.
