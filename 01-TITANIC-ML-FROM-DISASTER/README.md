# Kaggle Titanic: Machine Learning from Disaster

This repository contains my solution for Kaggle's classic **Titanic: Machine Learning from Disaster** competition. The goal is to predict which passengers survived the Titanic shipwreck using passenger data (such as name, age, gender, socio-economic class, etc.).

## 📊 Project Overview
* **Objective:** Predict binary classification (`Survived`: 0 or 1).
* **Model Used:** Random Forest Classifier (`Scikit-Learn`)
* **Validation Strategy:** 5-Fold Cross-Validation & local train/test split.
* **Target Metric:** Accuracy Score

---

## 🛠️ Feature Engineering & Data Pipeline

Real-world data is messy, and a significant portion of this project involved processing features to ensure the Random Forest algorithm could interpret them efficiently:

1. **Handling Missing Values:** * Dropped rows with missing `Fare` data in the training set.
   * Imputed missing values in the `Age` and `Fare` columns using their respective dataset **medians** to avoid data distortion.
2. **Categorical Encoding:** * Converted the string-based `Sex` column into a binary numerical format (`male: 0`, `female: 1`).
3. **Feature Creation:** * Engineered a new feature, `Family_Size`, by combining `SibSp` (siblings/spouses) and `Parch` (parents/children) plus the passenger themselves (`+ 1`) to capture traveling group dynamics.

---

## 🔬 Hyperparameter Tuning & Evaluation

To find the optimal constraints for the Random Forest and prevent **overfitting** (which naturally occurs when trees grow too deep on specific numerical columns like `Age` and `Fare`), I implemented an automated hyperparameter search.

* **Grid Search Optimization (`GridSearchCV`):** Tested combinations across `n_estimators` (number of trees) and `max_depth` (tree height) utilizing 5-fold cross-validation (`cv=5`) to ensure evaluation stability across the entire training subset.
* **Final Model Parameters:** `n_estimators=100`, `max_depth=10`, `random_state=1`.

### Results
* **Local Validation Accuracy:** **80.72%** 🚀

---

## 📁 Repository Structure

```text
01-TITANIC-ML-FROM-DISASTER/
│
├── titanic_data/
│   ├── train.csv              # Training data provided by Kaggle
│   └── test.csv               # Unlabeled evaluation data from Kaggle
│
├── model.py                   # Central Python script containing execution pipeline
├── kaggle_submission.csv      # Final exported predictions file for leaderboard upload
└── README.md                  # Project documentation