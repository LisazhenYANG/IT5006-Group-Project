# Crime Hotspot Prediction — IT5006 Group Project

A binary classification system that predicts whether a spatial unit will be a crime hotspot, trained on Chicago crime data (2015–2024) and evaluated on NIBRS federal data for cross-domain generalisability testing.

---

## Project Structure

```
IT5006-Group-Project/
├── data/
│   ├── raw/                    # Original Chicago crime CSVs (2015–2024, 2025)
│   ├── processed/              # Cleaned data output from notebook 02
│   ├── furtherprocessed/       # Intermediate processed CSVs output from notebook 03
│   ├── feature_engineering/    # Final feature matrices (X_train/test, y_train/test)
│   └── nibrs/                  # NIBRS federal dataset for cross-domain evaluation
├── deployment/                 # Streamlit prediction app (production entry point)
│   ├── app.py
│   ├── predict.py
│   ├── models/                 # Deployed model files (.pkl)
│   └── config/
│       └── feature_columns.json
├── notebooks/                  # Data pipeline and model training notebooks
├── src/
│   ├── dashboard/              # Interactive visualisation dashboard
│   ├── figures/                # Saved evaluation plots
│   └── models/                 # Trained model files (.pkl)
├── docs/                       # Model reports and analysis documentation
└── requirements.txt
```

---

## Data Pipeline

Run notebooks in order:

1. `notebooks/01_data_loader.ipynb` — load raw Chicago crime CSV
2. `notebooks/02_preprocessing.ipynb` — clean data, output to `data/processed/`
3. `notebooks/03_processing.ipynb` — spatial aggregation and feature mapping, output to `data/furtherprocessed/`
4. `notebooks/04_Spatial_feature_engineering.ipynb` — compute land-use and crime-type ratio features, output to `data/feature_engineering/`
5. Model training:
   - `notebooks/random_forest.ipynb` / `notebooks/random forest_nibrs.ipynb`
   - `notebooks/xgboost_modeling.ipynb`
   - `notebooks/logistic_regression.ipynb` / `notebooks/logistic_regression_nibrs.ipynb`

---

## Models

Three binary classifiers were trained on the same feature schema. The deployed models are the NIBRS variants (`*_nibrs.pkl`).

**Input features (11 total)**:
- Land-use ratios: `commercial_ratio`, `institution_ratio`, `other_ratio`, `public_ratio`, `residential_ratio`
- Crime-type ratios: `theft_ratio`, `battery_ratio`, `criminal_damage_ratio`, `assault_ratio`, `deceptive_practice_ratio`, `other_crime_ratio`

**Hotspot label**: a spatial unit is labelled hotspot (`1`) if its annual crime count is at or above the 75th percentile across all units in that year.

---

## Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Run the prediction MVP (from project root)
cd deployment && streamlit run app.py
```

The app supports: CSV upload → model selection (RF / XGBoost / LR) → prediction → optional evaluation if ground-truth `hotspot` column is present.

## Running the Interactive Dashboard

```bash
streamlit run src/dashboard/InteractiveDashboard.py
```

---

## Dependencies

See `requirements.txt`. Key packages: `scikit-learn`, `xgboost`, `streamlit`, `pandas`, `joblib`.
