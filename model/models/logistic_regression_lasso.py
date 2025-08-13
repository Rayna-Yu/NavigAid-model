import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    log_loss
)
from sklearn.calibration import calibration_curve
import numpy as np

# Load dataset
df = pd.read_csv('model/datasets/final_csv/continuous_night_data.csv')
X = df.drop(columns=['label']).astype('float32')
y = df['label'].astype('int')

# Define models
models = {
    "Logistic Regression (No penalty)": LogisticRegression(penalty=None, solver='lbfgs', class_weight='balanced', max_iter=1000),
    "Logistic Regression (L1 Lasso)": LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', max_iter=1000),
    "Logistic Regression (L2 Ridge)": LogisticRegression(penalty='l2', solver='lbfgs', class_weight='balanced', max_iter=1000),
}

# Stratified CV
skf = StratifiedKFold(n_splits=5)
results = {}

for name, model in models.items():
    all_y_true, all_y_pred, all_y_probs = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_probs.extend(y_prob)

    results[name] = {
        'y_true': np.array(all_y_true),
        'y_pred': np.array(all_y_pred),
        'y_prob': np.array(all_y_probs)
    }

    print(f"\n{name} Classification Report:")
    print(classification_report(all_y_true, all_y_pred))
    logloss = log_loss(all_y_true, all_y_probs)
    print(f"{name} Log Loss: {logloss:.4f}")

# ROC Curve comparison
plt.figure(figsize=(8, 6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid()
plt.show()

# Precision-Recall Curve comparison
plt.figure(figsize=(8, 6))
for name, res in results.items():
    precision, recall, _ = precision_recall_curve(res['y_true'], res['y_prob'])
    avg_precision = average_precision_score(res['y_true'], res['y_prob'])
    plt.plot(recall, precision, label=f'{name} (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve Comparison')
plt.legend()
plt.grid()
plt.show()

# Calibration Curve comparison
plt.figure(figsize=(8, 6))
for name, res in results.items():
    prob_true, prob_pred = calibration_curve(res['y_true'], res['y_prob'], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label=name)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curve Comparison')
plt.legend()
plt.grid()
plt.show()

# Coefficient comparison
coef_df = pd.DataFrame(index=X.columns)

for name, model in models.items():
    model.fit(X, y)
    coef_df[name] = model.coef_[0]

print("\nCoefficient Comparison Table:\n")
print(coef_df)

# Plot coefficient comparison
plt.figure(figsize=(12, 6))
coef_df.plot(kind='bar', figsize=(12, 6))
plt.title("Logistic Regression Coefficients: No Penalty vs L1 vs L2")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Numerical difference checks
print("\nDifference between L1 and L2 coefficients:")
print((coef_df["Logistic Regression (L1 Lasso)"] - coef_df["Logistic Regression (L2 Ridge)"]).round(6))

print("\nDifference between No Penalty and L1 coefficients:")
print((coef_df["Logistic Regression (No penalty)"] - coef_df["Logistic Regression (L1 Lasso)"]).round(6))
