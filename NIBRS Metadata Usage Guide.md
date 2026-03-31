# NIBRS Metadata Usage Guide

## 1. Overview

This file provides additional metadata for analysis purposes.

It is **not used for model training or prediction**, but is intended for post-hoc analysis and interpretation of results.

---

## 2. File Description

### `nibrs_meta.csv`

* Each row represents one agency
* Contains:

  * `agency_id`
  * `state_name`
  * `region_name`
  * `agency_type_name`

---

## 3. How to Use

### Step 1: Load metadata

```python
import pandas as pd

meta = pd.read_csv("nibrs_meta.csv")
```

---

### Step 2: Merge with prediction results

```python
analysis_df = data.merge(meta, on="agency_id", how="left")
```

---

## 4. Example Analysis

### By state

```python
analysis_df.groupby("state_name")["label"].mean()
```

### By agency type

```python
analysis_df.groupby("agency_type_name")["label"].mean()
```

---

## 5. Purpose

The metadata enables analysis from multiple perspectives:

* Geographic analysis (state / region)
* Institutional analysis (agency type)
* Model performance comparison across different contexts

---

## 6. Important Note

This file should **not be used as model input features**, but only for analysis and visualization.
