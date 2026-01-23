import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# ===============================
# CONFIG
# ===============================
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

X_TRAIN_PATH = "X_train.csv"
Y_TRAIN_PATH = "y_train.csv"
X_TEST_PATH = "X_test.csv"
Y_TEST_PATH = "y_test.csv"


# ===============================
# UTILITY FUNCTIONS
# ===============================
def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")
    return pd.read_csv(path)


def dataset_checks(X, y, name="Dataset"):
    print(f"\n🔍 Checking {name}...")

    # Shape check
    print(f"Shape X: {X.shape}, y: {y.shape}")

    # Missing values
    if X.isnull().sum().sum() > 0:
        print("⚠️ Missing values found in X — applying fillna(0)")
        X.fillna(0, inplace=True)

    if pd.isnull(y).sum() > 0:
        raise ValueError("❌ Missing labels found in y")

    # Label sanity
    unique_labels = np.unique(y)
    print(f"Unique labels: {unique_labels}")

    return X, y


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "F1": f1_score(y_test, y_pred, average="macro"),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }

    # AUC (only if predict_proba exists)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        metrics["AUC"] = roc_auc_score(y_test, y_proba, multi_class="ovr")
    else:
        metrics["AUC"] = np.nan

    return metrics


# ===============================
# LOAD DATA
# ===============================
print("📂 Loading datasets...")
X_train = load_csv(X_TRAIN_PATH)
y_train = load_csv(Y_TRAIN_PATH).values.ravel() - 1

X_test = load_csv(X_TEST_PATH)
y_test = load_csv(Y_TEST_PATH).values.ravel() - 1

# ===============================
# DATASET CHECKS
# ===============================
X_train, y_train = dataset_checks(X_train, y_train, "Training Data")
X_test, y_test = dataset_checks(X_test, y_test, "Test Data")

# Feature alignment
if X_train.shape[1] != X_test.shape[1]:
    raise ValueError("❌ Feature mismatch between train and test sets")

# Standardize feature names
X_train.columns = [f"f{i}" for i in range(X_train.shape[1])]
X_test.columns = X_train.columns

n_classes = len(np.unique(y_train))
print(f"\n✅ Number of classes: {n_classes}")


# ===============================
# MODELS
# ===============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        use_label_encoder=False,
        n_estimators=100,
        random_state=42,
    ),
}


# ===============================
# TRAIN + EVALUATE
# ===============================
results = []

for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    metrics["Model"] = name
    results.append(metrics)

    file_name = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, os.path.join(MODEL_DIR, file_name))
    print(f"✅ Model saved: {MODEL_DIR}/{file_name}")


# ===============================
# RESULTS SUMMARY
# ===============================
results_df = pd.DataFrame(results)[
    ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
]

print("\n📊 FINAL MODEL COMPARISON")
print(results_df)

results_df.to_csv("model_comparison_metrics.csv", index=False)
print("\n📁 Metrics saved to model_comparison_metrics.csv")
