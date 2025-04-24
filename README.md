# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using supervised learning on an imbalanced dataset.

## Objective
Build a predictive model to classify transactions as fraudulent or legitimate based on patterns in transaction data. The main challenge is handling severe class imbalance.

## Dataset
- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- Features are numerical and anonymized (PCA transformed).
- Imbalance: ~0.17% of transactions are fraudulent.

## Features
- `V1` to `V28`: Principal components obtained via PCA.
- `Time`, `Amount`: Raw features (preprocessing needed).
- `Class`: Target variable (`0` = legitimate, `1` = fraud)

## 🛠Tools & Techniques
- Data preprocessing (scaling, undersampling/oversampling with SMOTE)
- Models: Logistic Regression, Random Forest, XGBoost
- Evaluation: Precision, Recall, F1-Score, ROC-AUC
- Handling Imbalance: SMOTE, stratified splitting

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook notebooks/fraud_detection.ipynb
