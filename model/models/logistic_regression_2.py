import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    log_loss
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Load data
df = pd.read_csv('model/datasets/final_csv/flags_and_crash_data.csv')

X = df.drop(columns=['label']).astype('float32')
y = df['label'].astype('int')

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
model = make_pipeline(poly, LogisticRegression(max_iter=1000, class_weight='balanced'))

scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
print("Average CV AUC with polynomial features:", scores.mean())

# cross-validate
cv_accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy')
cv_auc = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
print(f"CV accuracy scores: {cv_accuracy}")
print(f"Average CV accuracy: {cv_accuracy.mean():.4f}")
print(f"Average CV AUC: {cv_auc.mean():.4f}")

# stratified CV
skf = StratifiedKFold(n_splits=5)
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

# report
print("\nClassification report (CV combined):")
print(classification_report(all_y_true, all_y_pred))

# confustion matrix
cm = confusion_matrix(all_y_true, all_y_pred)
plt.figure()
plt.title("Confusion Matrix (CV combined)")
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# roc curve
fpr, tpr, _ = roc_curve(all_y_true, all_y_probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (CV combined)')
plt.legend()
plt.grid()
plt.show()

# precision-recall curve
precision, recall, _ = precision_recall_curve(all_y_true, all_y_probs)
avg_precision = average_precision_score(all_y_true, all_y_probs)
plt.figure()
plt.plot(recall, precision, label=f'AP = {avg_precision:.2f}', color='purple')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall Curve (CV combined)')
plt.legend()
plt.grid()
plt.show()

# log loss
logloss = log_loss(all_y_true, all_y_probs)
print(f"Log Loss (CV combined): {logloss:.4f}")

# Fit the pipeline on all data (PolynomialFeatures + LogisticRegression)
model_full = make_pipeline(
    PolynomialFeatures(degree=2, interaction_only=False, include_bias=False),
    LogisticRegression(max_iter=1000, class_weight='balanced')
)
model_full.fit(X, y)

# Extract polynomial feature names
poly_features = model_full.named_steps['polynomialfeatures'].get_feature_names_out(input_features=X.columns)

# Extract logistic regression coefficients
coefs = model_full.named_steps['logisticregression'].coef_[0]
intercept = model_full.named_steps['logisticregression'].intercept_[0]

print("\nFeature coefficients (weights) with polynomial features:")
for feat, coef in zip(poly_features, coefs):
    print(f"{feat}: {coef:.4f}")
print(f"Intercept (bias): {intercept:.4f}")

# Predicted probabilities on full data with polynomial pipeline
y_prob_full = model_full.predict_proba(X)[:, 1]

# Calibration curve
prob_true, prob_pred = calibration_curve(y, y_prob_full, n_bins=10)

plt.figure()
plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Polynomial Logistic Regression')
plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curve')
plt.legend()
plt.grid()
plt.show()
