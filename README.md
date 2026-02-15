# Human Activity Recognition – Multi-Class Classification

## 📋 Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset Description](#dataset-description)
- [Models Used](#models-used)
- [Setup Instructions](#setup-instructions)
- [Model Comparison Results](#model-comparison-results)
- [Final Test Performance](#final-test-performance)
- [Observations on Model Performance](#observations-on-model-performance)
- [Conclusion](#conclusion)

---

## Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to recognize human activities based on smartphone sensor data.

This is a **multi-class classification problem**, where the goal is to classify human activity into one of six categories using extracted time-domain and frequency-domain features.

**Evaluation Strategy:** Stratified 5-Fold Cross Validation is used for model comparison, and the best-performing model is evaluated on a separate test dataset.

---

## Dataset Description

The dataset used is the **UCI Human Activity Recognition Using Smartphones Dataset**.

### Dataset Details

- **Total Features:** 561
- **Total Classes:** 6
- **Activities:**
  1. Walking
  2. Walking Upstairs
  3. Walking Downstairs
  4. Sitting
  5. Standing
  6. Laying

### Feature Information

The features are pre-extracted from:
- Accelerometer signals
- Gyroscope signals
- Time-domain signals
- Frequency-domain signals

Each observation corresponds to a fixed-width sliding window of sensor data.

---

## Models Used

The following six machine learning models were implemented:

1. **Logistic Regression**
2. **Decision Tree**
3. **K-Nearest Neighbors (kNN)**
4. **Naive Bayes (Gaussian)**
5. **Random Forest (Ensemble)**
6. **XGBoost (Ensemble)**

### Evaluation Metrics

- Accuracy
- AUC (One-vs-Rest)
- Precision (Macro)
- Recall (Macro)
- F1 Score (Macro)
- MCC (Matthews Correlation Coefficient)

---

## 🛠 Setup Instructions

### 1️⃣ Create Virtual Environment

#### On Linux / macOS
```bash
python -m venv myenv
```

#### On Windows
```bash
python -m venv myenv
```

### 2️⃣ Activate Virtual Environment

#### On Linux / macOS
```bash
source myenv/bin/activate
```

#### On Windows
```bash
myenv\Scripts\activate
```

### 3️⃣ Install Required Dependencies

After activating the virtual environment, install the required packages:
```bash
pip install -r requirements.txt
```

---

## 📊 Model Comparison Results (Cross Validation)

| ML Model Name        | Accuracy | AUC    | Precision | Recall | F1     |
|----------------------|----------|--------|-----------|--------|--------|
| Logistic Regression  | 0.9854   | 0.9994 | 0.9863    | 0.9862 | 0.9862 |
| Decision Tree        | 0.9301   | 0.9575 | 0.9296    | 0.9290 | 0.9290 |
| KNN                  | 0.9623   | 0.9963 | 0.9654    | 0.9642 | 0.9645 |
| Naive Bayes          | 0.7507   | 0.9568 | 0.7762    | 0.7557 | 0.7509 |
| Random Forest        | 0.9798   | 0.9995 | 0.9802    | 0.9803 | 0.9801 |
| **XGBoost**          | **0.9891** | **0.9998** | **0.9894** | **0.9894** | **0.9894** |

---

## 🎯 Final Test Performance (Best Model: XGBoost)

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 0.9382 |
| Precision | 0.9398 |
| Recall    | 0.9365 |
| F1 Score  | 0.9374 |
| MCC       | 0.9261 |
| AUC       | 0.9970 |

---

## 🔎 Observations on Model Performance

| ML Model Name          | Observation                                                                                                                                                      |
|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression    | Performed extremely well, indicating that the dataset is nearly linearly separable in high-dimensional feature space. Demonstrates strong generalization ability. |
| Decision Tree          | Lower performance compared to ensemble methods due to high variance and overfitting tendencies.                                                                  |
| kNN                    | Strong performance due to well-separated clusters in feature space. Computationally expensive for large datasets.                                                |
| Naive Bayes            | Lowest performance because the conditional independence assumption between features does not hold for this dataset.                                              |
| Random Forest          | High accuracy and stability. Reduced variance compared to single Decision Tree through bagging.                                                                  |
| XGBoost                | Best performing model in cross-validation. Boosting effectively reduces both bias and variance.                                                                  |

---

## 🏆 Conclusion

Among all models, **XGBoost** achieved the highest cross-validation performance and strong test performance with an accuracy of **93.82%** on the test set.

**Key Findings:**
- Ensemble methods (Random Forest and XGBoost) significantly outperformed individual models
- XGBoost's gradient boosting technique proved most effective for this multi-class classification task
- Stratified 5-Fold Cross Validation ensured balanced class distribution across folds and reliable performance estimation
- The high performance of Logistic Regression suggests the dataset features are well-engineered and nearly linearly separable

This project demonstrates the effectiveness of ensemble learning techniques, particularly gradient boosting, in achieving superior classification performance for human activity recognition tasks.