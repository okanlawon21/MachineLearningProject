# Scikit-learn Hands-on Practice – January 4, 2026

Welcome!  
This README summarizes today’s hands-on machine learning practice, covering regression, classification, troubleshooting, and model visualization with Python’s scikit-learn library.  
Each exercise was designed to strengthen my understanding of classic ML workflows and prepare me for moving into deep learning with frameworks like PyTorch.

---

## 📝 Project Overview

Today’s project focused on practical, step-by-step implementation of regression and classification tasks using real-world datasets. The work involved building data pipelines, optimizing models, troubleshooting data issues, and visualizing results to build a solid base for future machine learning projects.

---

## 🚩 Objectives

- Implement and evaluate common ML algorithms for regression and classification.
- Develop skills with pipelines, preprocessing, dimensionality reduction, and hyperparameter tuning (GridSearchCV).
- Learn troubleshooting techniques for data loading and cache issues.
- Use visualizations to interpret confusion matrices, predictions, and errors.
- Assess personal readiness to transition from scikit-learn to PyTorch.

---

## 📁 Datasets Used

- **Iris data** (Regression - Classification)
- **California Housing** (Regression)
- **OpenML House Prices** (Regression)
- **Scikit-learn Digits** (Classification - handwritten digits)

---

## 🗂️ Work Done

### 1. **Regression with Random Forest**
- Created a pipeline with `StandardScaler`, `PCA`, and `RandomForestRegressor`.
- Applied grid search (`GridSearchCV`) to tune model hyperparameters.
- Evaluated predictions using RMSE and R² score.
- Visualized actual vs. predicted outcomes and distribution of errors.

### 2. **Dataset Troubleshooting**
- Identified and addressed dataset cache corruption issues (especially with California Housing).
- Used Python scripting to delete problematic cache files and ensure fresh downloads for smooth data loading.

### 3. **Simple Classification – Digits Dataset**
- Trained a `LogisticRegression` model to classify handwritten digits.
- Measured accuracy to validate model performance.

### 4. **Enhanced Digits Classification**
- Plotted confusion matrix using heatmap for deeper insight into classification results.
- Displayed sample digit images with predicted and true labels for interpretability.

---

## 📊 Results

- Developed robust pipelines for both regression and classification tasks.
- Achieved reliable prediction performance (RMSE, R², accuracy) on each dataset.
- Resolved data loading issues, ensuring reproducible experiments.
- Gained insightful visualizations to understand errors and model strengths/weaknesses.

---

## 📝 Summary

These scikit-learn exercises helped reinforce fundamental skills in preprocessing, modeling, evaluation, and troubleshooting.  
With this foundation, I am well-prepared to start learning PyTorch for deep learning projects, such as custom neural networks and image recognition.

