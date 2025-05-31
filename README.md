# Parkinson’s Disease Detection Using XGBoost

This project implements a machine learning model to detect Parkinson’s Disease using vocal features. The model is built with **XGBoost**, a powerful gradient boosting algorithm, leveraging Python libraries such as `pandas`, `scikit-learn`, and `xgboost`.

---

## Dataset

- The dataset used is the **Parkinson’s Disease dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Parkinsons).
- It contains biomedical voice measurements from 195 people, including 147 with Parkinson’s and 48 healthy individuals.
- Features represent various vocal attributes; the target label `status` indicates disease presence (1) or absence (0).

---

## Project Workflow

1. **Data Loading:** The dataset is loaded and preprocessed.
2. **Feature Scaling:** Features are scaled using `StandardScaler` for better model performance.
3. **Train-Test Split:** The dataset is split into training and testing sets.
4. **Model Building:** An `XGBClassifier` is trained on the training data.
5. **Evaluation:** The model's accuracy is calculated on the test data.
6. **Optional:** Additional evaluation metrics, feature importance, and hyperparameter tuning can be applied.

---

## Why XGBoost?

- **High Accuracy:** XGBoost is known for superior predictive performance on tabular data.
- **Efficient:** Fast training with optimized memory usage.
- **Regularization:** Built-in L1 and L2 regularization to prevent overfitting.
- **Feature Importance:** Helps interpret which features influence predictions.
- **Robust:** Handles complex patterns and imbalanced datasets well.

---

## Requirements

- Python 3.6+
- pandas
- numpy
- scikit-learn
- xgboost

Install required packages with:

```bash
pip install pandas numpy scikit-learn xgboost
