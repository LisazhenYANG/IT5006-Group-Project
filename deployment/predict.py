from pathlib import Path
import json
import joblib
import pandas as pd


# Define important project paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "config"


# Store model file paths
MODEL_PATHS = {
    "rf": MODEL_DIR / "best_rf_nibrs.pkl",
    "xgb": MODEL_DIR / "xgboost_hotspot_nibrs.pkl",
    "lr": MODEL_DIR / "lr_nibrs.pkl",
}


def load_model(model_name: str):
    # Load a trained model based on the selected model name
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model name: {model_name}")

    model_path = MODEL_PATHS[model_name]

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)


def load_feature_columns() -> list:
    # Load the unified feature schema from JSON
    feature_file = CONFIG_DIR / "feature_columns.json"

    if not feature_file.exists():
        raise FileNotFoundError(f"Feature schema file not found: {feature_file}")

    with open(feature_file, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    if not isinstance(feature_columns, list):
        raise ValueError("feature_columns.json must contain a list of column names.")

    return feature_columns


def validate_input_dataframe(df: pd.DataFrame):
    # Check whether the input is a valid DataFrame
    if df is None:
        raise ValueError("Input DataFrame is None.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")


def align_features(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    # Create a copy to avoid modifying the original DataFrame
    aligned_df = df.copy()

    # Add missing columns with default value 0
    for col in feature_columns:
        if col not in aligned_df.columns:
            aligned_df[col] = 0

    # Keep only the required columns and enforce the correct order
    aligned_df = aligned_df[feature_columns]

    return aligned_df


def predict(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    # Validate input data
    validate_input_dataframe(df)

    # Load model and feature schema
    model = load_model(model_name)
    feature_columns = load_feature_columns()

    # Align input features to match training schema
    X = align_features(df, feature_columns)

    # Generate class predictions
    predictions = model.predict(X)

    # Build output DataFrame
    result_df = df.copy()
    result_df["prediction"] = predictions

    # Add prediction probabilities if supported
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
        result_df["probability"] = probabilities

    return result_df
