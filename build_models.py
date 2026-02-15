# # ============================================================
# # Multi-Class Classification with Cross Validation
# # Dataset: UCI Human Activity Recognition
# # ============================================================
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
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
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
# LOAD DATA
# ===============================
def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


print("=" * 50)
print("Loading datasets...")
print("=" * 50)

X_train = load_csv(X_TRAIN_PATH)
y_train = load_csv(Y_TRAIN_PATH).values.ravel() - 1  # zero index

X_test = load_csv(X_TEST_PATH)
y_test = load_csv(Y_TEST_PATH).values.ravel() - 1

# Save feature names used during training
feature_names = X_train.columns.tolist()
joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

print("Feature names saved.")

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ===============================
# SPLIT TRAIN INTO TRAIN + VAL
# ===============================
X_train_part, X_val_part, y_train_part, y_val_part = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    stratify=y_train,
    random_state=42,
)

print("Train split:", X_train_part.shape)
print("Validation split:", X_val_part.shape)
print("=" * 50)

# ===============================
# CROSS VALIDATION SETUP
# ===============================
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro",
    "roc_auc": "roc_auc_ovr",
}


n_classes = len(np.unique(y_train))


# ===============================
# MODELS
# ===============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
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
# CROSS VALIDATION ON TRAIN_PART
# ===============================
results = []

print("\nPerforming 5-Fold CV on Training Split...")

for name, model in models.items():
    print(f"\nCross-validating {name}...")

    cv_results = cross_validate(
        model,
        X_train_part,
        y_train_part,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    metrics = {
        "Model": name,
        "CV_Accuracy": np.mean(cv_results["test_accuracy"]),
        "CV_Precision": np.mean(cv_results["test_precision"]),
        "CV_Recall": np.mean(cv_results["test_recall"]),
        "CV_F1": np.mean(cv_results["test_f1"]),
        "CV_AUC": np.mean(cv_results["test_roc_auc"]),
    }

    results.append(metrics)

results_df = pd.DataFrame(results)
print("\nCROSS VALIDATION RESULTS")
print("=" * 100)
print(results_df)
print("=" * 100)
results_df.to_csv("model_comparison_metrics.csv", index=False)

# ===============================
# TRAIN & SAVE ALL MODELS ON FULL TRAIN DATA
# ===============================
print("\n Training all models on full training data and saving them...")

trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name} on full data...")
    model.fit(X_train, y_train)

    file_name = name.lower().replace(" ", "_") + ".pkl"
    model_path = os.path.join(MODEL_DIR, file_name)

    joblib.dump(model, model_path)
    print(f"Saved: {model_path}")

    trained_models[name] = model


# ===============================
# SELECT BEST MODEL (Based on CV F1)
# ===============================
best_model_name = results_df.sort_values(by="CV_F1", ascending=False).iloc[0]["Model"]

print(f"\nBest Model from CV: {best_model_name}")

best_model = trained_models[best_model_name]

# Save best model separately
best_model_path = os.path.join(MODEL_DIR, "best_model.pkl")
joblib.dump(best_model, best_model_path)

print(f" Best model saved separately as: {best_model_path}")


# ===============================
# FINAL TEST EVALUATION
# ===============================
print("\n Final Test Evaluation")

y_pred = best_model.predict(X_test)

test_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, average="macro"),
    "Recall": recall_score(y_test, y_pred, average="macro"),
    "F1": f1_score(y_test, y_pred, average="macro"),
    "MCC": matthews_corrcoef(y_test, y_pred),
}

if hasattr(best_model, "predict_proba"):
    y_proba = best_model.predict_proba(X_test)
    test_metrics["AUC"] = roc_auc_score(y_test, y_proba, multi_class="ovr")
else:
    test_metrics["AUC"] = np.nan

print("=" * 100)
print("\nTEST SET METRICS")
print(test_metrics)

print("\n All models trained, saved, and evaluated successfully!")
print("=" * 100)
