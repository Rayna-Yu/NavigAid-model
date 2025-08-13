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

# Load dataset
df = pd.read_csv('model/datasets/final_csv/continuous_night_data.csv')
X = df.drop(columns=['label']).astype('float32')
y = df['label'].astype('int')

model = LogisticRegression(class_weight='balanced', max_iter=1000)

# Simple CV accuracy and AUC
cv_accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy')
cv_auc = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
print(f"CV accuracy scores: {cv_accuracy}")
print(f"Average CV accuracy: {cv_accuracy.mean():.4f}")
print(f"Average CV AUC: {cv_auc.mean():.4f}")

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

print("\nClassification report (CV combined):")
print(classification_report(all_y_true, all_y_pred))

cm = confusion_matrix(all_y_true, all_y_pred)
plt.figure()
plt.title("Confusion Matrix (CV combined)")
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve for all CV predictions
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

# Precision-Recall Curve for all CV predictions
precision, recall, _ = precision_recall_curve(all_y_true, all_y_probs)
avg_precision = average_precision_score(all_y_true, all_y_probs)
plt.figure()
plt.plot(recall, precision, label=f'AP = {avg_precision:.2f}', color='purple')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (CV combined)')
plt.legend()
plt.grid()
plt.show()

# Log loss on combined predictions
logloss = log_loss(all_y_true, all_y_probs)
print(f"Log Loss (CV combined): {logloss:.4f}")

#Fit model on entire dataset to get final coefficients
model_full = LogisticRegression(max_iter=1000)
model_full.fit(X, y)

# Print feature coefficients
print("\nFeature coefficients (weights):")
for name, coef in zip(X.columns, model_full.coef_[0]):
    print(f"{name}: {coef:.4f}")
print(f"Intercept (bias): {model_full.intercept_[0]:.4f}")

# Get predicted probabilities on full data
y_prob_full = model_full.predict_proba(X)[:, 1]

prob_true, prob_pred = calibration_curve(y, y_prob_full, n_bins=10)

plt.figure()
plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Logistic Regression')
plt.plot([0,1], [0,1], linestyle='--', color='gray', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration Curve')
plt.legend()
plt.grid()
plt.show()

