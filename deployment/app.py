import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

from predict import predict


# Configure page settings
st.set_page_config(
    page_title="Crime Hotspot Prediction MVP",
    layout="wide"
)


# Required feature columns for model input
REQUIRED_FEATURES = [
    "commercial_ratio",
    "institution_ratio",
    "other_ratio",
    "public_ratio",
    "residential_ratio",
    "theft_ratio",
    "battery_ratio",
    "criminal_damage_ratio",
    "assault_ratio",
    "deceptive_practice_ratio",
    "other_crime_ratio"
]


# Initialize session state
if "input_df" not in st.session_state:
    st.session_state.input_df = None

if "result_df" not in st.session_state:
    st.session_state.result_df = None

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "rf"


# App title
st.title("Crime Hotspot Prediction MVP")
st.markdown(
    "Upload a CSV file, choose a model, run prediction, and evaluate performance if the file contains a ground-truth label column."
)


# Sidebar controls
st.sidebar.header("Model Settings")

model_name = st.sidebar.selectbox(
    "Choose a model",
    options=["rf", "xgb", "lr"],
    index=["rf", "xgb", "lr"].index(st.session_state.selected_model)
)
st.session_state.selected_model = model_name

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

run_prediction = st.sidebar.button("Run Prediction")
clear_all = st.sidebar.button("Clear All")


# Clear state
if clear_all:
    st.session_state.input_df = None
    st.session_state.result_df = None
    st.rerun()


# Read uploaded CSV
if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)

        missing_cols = [col for col in REQUIRED_FEATURES if col not in input_df.columns]

        if missing_cols:
            st.sidebar.error(f"Missing required feature columns: {missing_cols}")
            st.session_state.input_df = None
            st.session_state.result_df = None
        else:
            st.session_state.input_df = input_df
            st.sidebar.success("CSV file uploaded successfully.")

    except Exception as e:
        st.sidebar.error(f"Failed to read CSV file: {e}")
        st.session_state.input_df = None
        st.session_state.result_df = None


# Run prediction
if run_prediction:
    if st.session_state.input_df is None:
        st.warning("Please upload a valid CSV file first.")
    else:
        try:
            result_df = predict(st.session_state.input_df, model_name)
            st.session_state.result_df = result_df
            st.success("Prediction completed successfully.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.session_state.result_df = None


# Two tabs only
tab1, tab2 = st.tabs(["Prediction", "Evaluation"])


with tab1:
    st.subheader("Prediction Results")

    if st.session_state.input_df is None:
        st.info("Please upload a CSV file to begin.")
    else:
        st.markdown("### Uploaded Data Preview")
        st.dataframe(st.session_state.input_df.head(), use_container_width=True)

        row_count, col_count = st.session_state.input_df.shape
        c1, c2 = st.columns(2)
        c1.metric("Rows", row_count)
        c2.metric("Columns", col_count)

        if st.session_state.result_df is not None:
            st.markdown("### Output Table")
            st.dataframe(st.session_state.result_df, use_container_width=True)

            if "prediction" in st.session_state.result_df.columns:
                prediction_counts = st.session_state.result_df["prediction"].value_counts().sort_index()

                c3, c4 = st.columns(2)
                c3.metric("Predicted Non-hotspot (0)", int(prediction_counts.get(0, 0)))
                c4.metric("Predicted Hotspot (1)", int(prediction_counts.get(1, 0)))

            csv_data = st.session_state.result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Prediction Results",
                data=csv_data,
                file_name=f"prediction_results_{model_name}.csv",
                mime="text/csv"
            )
        else:
            st.info("Click 'Run Prediction' to generate results.")


with tab2:
    st.subheader("Evaluation")

    if st.session_state.result_df is None:
        st.info("Please run prediction first.")
    else:
        if "hotspot" not in st.session_state.result_df.columns:
            st.warning("No ground-truth label column 'hotspot' was found in the uploaded file. Evaluation cannot be performed.")
        else:
            eval_df = st.session_state.result_df.copy()

            y_true = eval_df["hotspot"]
            y_pred = eval_df["prediction"]

            if "probability" not in eval_df.columns:
                st.warning("No probability column was generated, so ROC curve cannot be displayed.")
            else:
                y_proba = eval_df["probability"]

            st.markdown("### Evaluation Metrics")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
            c2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.4f}")
            c3.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.4f}")
            c4.metric("F1-score", f"{f1_score(y_true, y_pred, zero_division=0):.4f}")

            st.markdown("### Classification Report")
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).T
            st.dataframe(report_df, use_container_width=True)

            st.markdown("### Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)

            fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
            ax_cm.imshow(cm, cmap="Blues")
            ax_cm.set_xticks([0, 1])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(["Non-hotspot", "Hotspot"])
            ax_cm.set_yticklabels(["Non-hotspot", "Hotspot"])
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            ax_cm.set_title("Confusion Matrix")

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center")

            st.pyplot(fig_cm)

            if "probability" in eval_df.columns:
                st.markdown("### ROC Curve")
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                roc_auc = auc(fpr, tpr)

                fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
                ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
                ax_roc.plot([0, 1], [0, 1], linestyle="--")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.set_title("ROC Curve")
                ax_roc.legend(loc="lower right")

                st.pyplot(fig_roc)
