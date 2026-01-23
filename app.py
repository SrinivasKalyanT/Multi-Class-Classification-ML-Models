import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="ML Assignment 2 – Model Comparison", layout="wide")
st.title("📊 ML Assignment 2 – Model Evaluation App")
st.write(
    "Upload **test data only**, select a trained model, and view evaluation results."
)

# =========================
# MODEL LOADING
# =========================
MODEL_DIR = "model"
model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

available_models = {
    name: joblib.load(os.path.join(MODEL_DIR, file))
    for name, file in model_files.items()
    if os.path.exists(os.path.join(MODEL_DIR, file))
}

# =========================
# SIDEBAR – MODEL SELECTION
# =========================
st.sidebar.header("⚙️ Configuration")
selected_model_name = st.sidebar.selectbox(
    "Select Model", list(available_models.keys())
)
model = available_models[selected_model_name]

# =========================
# DATA UPLOAD
# =========================
st.subheader("📂 Upload Test Dataset")
st.write("Upload **X_test.csv** and **y_test.csv**")

X_file = st.file_uploader("Upload X_test.csv", type=["csv"], key="x")
y_file = st.file_uploader("Upload y_test.csv", type=["csv"], key="y")

if X_file and y_file:
    X_test = pd.read_csv(X_file)
    y_test = pd.read_csv(y_file).values.ravel() - 1

    # Ensure feature names match training
    X_test.columns = [f"f{i}" for i in range(X_test.shape[1])]

    st.success("✅ Test data uploaded successfully")

    # =========================
    # PREDICTION
    # =========================
    y_pred = model.predict(X_test)

    # =========================
    # METRICS
    # =========================
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    mcc = matthews_corrcoef(y_test, y_pred)

    st.subheader("📈 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("Precision (Macro)", f"{prec:.4f}")
    col3.metric("Recall (Macro)", f"{rec:.4f}")

    col4, col5 = st.columns(2)
    col4.metric("F1 Score (Macro)", f"{f1:.4f}")
    col5.metric("MCC", f"{mcc:.4f}")

    # =========================
    # CONFUSION MATRIX
    # =========================
    st.subheader("🔍 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # =========================
    # CLASSIFICATION REPORT
    # =========================
    st.subheader("📄 Classification Report")
    report = classification_report(y_test, y_pred)
    st.text(report)

else:
    st.info("⬅️ Please upload both X_test.csv and y_test.csv to continue")
