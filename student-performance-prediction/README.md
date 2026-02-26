# Comparative Analysis of Regression and Classification Models for Predicting Student Academic Performance (UCI)

## Overview
This project predicts a student's **final grade (G3)** using demographic, behavioral, and school-related features from the **UCI Student Performance** dataset (`student-mat.csv`).  
I compare two approaches:
- **Regression**: predict the exact final grade (0–20)
- **Classification**: predict performance category (**A / B / C**)

---

## Problem Statement
### What are we predicting and why?
- **Target:** `G3` (final grade)
- **Why it matters:** Early performance estimation can help educators identify students who may need support and understand which factors are associated with stronger outcomes.

### Two modeling tasks
1. **Regression task:** Predict the exact final grade (`G3`) as a number.
2. **Classification task:** Predict grade category:
   - **A:** `G3 >= 15`
   - **B:** `10 <= G3 < 15`
   - **C:** `G3 < 10`

---

## Dataset
- **Source:** UCI Student Performance Dataset  
  https://archive.ics.uci.edu/ml/datasets/Student+Performance
- **File used:** `student-mat.csv` (Math course), semicolon-separated (`sep=';'`)

### Dataset summary
- **Rows:** 395  
- **Columns:** 33 original features + grade columns (`G1`, `G2`, `G3`)
- **Target:** `G3` (final grade)

---

## Project Structure
- `notebooks/01_eda.ipynb` — distributions, outliers (Z-score), correlations
- `notebooks/02_regression_models.ipynb` — Linear Regression, Random Forest, XGBoost (RMSE/MAE/R²)
- `notebooks/03_classification_models.ipynb` — Logistic Regression, Random Forest, XGBoost (weighted/macro F1 + confusion matrices)
- `data/raw/student-mat.csv` — dataset location used by the notebooks
- `requirements.txt` — dependencies

---

## Exploratory Data Analysis (EDA)
EDA was performed to understand distributions, detect outliers, and examine correlations with `G3`.

### What I checked
- **Distributions** of numeric variables (histograms)
- **Outliers** using boxplots and Z-score method
- **Correlation matrix** to inspect relationships with `G3`

### Key insights
- `failures` and `absences` show **negative relationships** with `G3`.
- `studytime` tends to have a **positive relationship** with `G3` (not perfectly, but noticeable).
- `G1` and `G2` are **highly correlated** with `G3`, so I excluded them for a more realistic prediction setting.

---

## Preprocessing
### Why drop `G1` and `G2`?
`G1` and `G2` are prior grades. Including them can inflate results because the model can “cheat” by using earlier exam scores.  
This project focuses on more realistic predictors like study time, family background, absences, and lifestyle factors.

### Steps
- Dropped `G1` and `G2`
- One-hot encoded categorical features (`pd.get_dummies(drop_first=True)`)
- Train/test split (80/20, `random_state=42`)
- Evaluation uses appropriate metrics for both tasks

---

## Regression Models (Predict Exact Grade)
### Models used
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

### Metrics
- **RMSE** (root mean squared error)
- **MAE** (mean absolute error)
- **R²** (explained variance)

### Results (from my run)
| Model | RMSE | MAE | R² |
|------|-----:|----:|---:|
| Linear Regression | 4.21 | 3.44 | 0.136 |
| Random Forest Regressor | **3.90** | 3.11 | **0.257** |
| XGBoost Regressor | 3.91 | **3.09** | 0.256 |

**Interpretation:**  
Tree-based models (Random Forest and XGBoost) performed better than Linear Regression
