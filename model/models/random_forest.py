import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import joblib

# Load Data
df = pd.read_csv('model/datasets/final_csv/continuous_night_data.csv')
X = df.drop(columns=['label'])
y = df['label']

X_trainval, X_holdout, y_trainval, y_holdout = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train Random Forest

# rf = RandomForestClassifier(bootstrap=False, n_estimators=100, min_samples_leaf=1, min_samples_split=2, random_state=42)
rf = RandomForestClassifier(n_estimators=300, random_state=42)

acc_scores = cross_val_score(rf, X_trainval, y_trainval, cv=cv, scoring='accuracy')
auc_scores = cross_val_score(rf, X_trainval, y_trainval, cv=cv, scoring='roc_auc')
print(f"Average CV accuracy: {acc_scores.mean():.4f}")
print(f"Average CV AUC: {auc_scores.mean():.4f}")

rf.fit(X_trainval, y_trainval)

# Calibration
calibrated_rf = CalibratedClassifierCV(rf, method='isotonic', cv=5)
calibrated_rf.fit(X_trainval, y_trainval)

prob_uncal = rf.predict_proba(X_holdout)[:, 1]
prob_cal = calibrated_rf.predict_proba(X_holdout)[:, 1]

# Calibration curve
frac_pos_uncal, mean_pred_uncal = calibration_curve(y_holdout, prob_uncal, n_bins=10)
frac_pos_cal, mean_pred_cal = calibration_curve(y_holdout, prob_cal, n_bins=10)

plt.plot(mean_pred_uncal, frac_pos_uncal, 'o-', label="Uncalibrated")
plt.plot(mean_pred_cal, frac_pos_cal, 'o-', label="Calibrated")
plt.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")
plt.xlabel("Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curve")
plt.legend()
plt.show()

# Save calibrated model
joblib.dump(calibrated_rf, "final/random_forest_night_calibrated.pkl")

# Evaluate on Holdout (Post-Calibration)
y_pred = calibrated_rf.predict(X_holdout)
y_prob = calibrated_rf.predict_proba(X_holdout)[:, 1]

holdout_acc = (y_pred == y_holdout).mean()
print(f"Holdout Accuracy: {holdout_acc:.4f}")
print(classification_report(y_pred, y_holdout))

# Sanity check
y_shuffled = shuffle(y_trainval, random_state=42)
rf_shuffled = RandomForestClassifier(n_estimators=300, random_state=42)
rf_shuffled.fit(X_trainval, y_shuffled)
y_pred_shuffled = rf_shuffled.predict(X_holdout)
acc_shuffled = (y_pred_shuffled == y_holdout).mean()
print(f"Holdout Accuracy with shuffled labels: {acc_shuffled:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_holdout, y_pred)
ConfusionMatrixDisplay(cm).plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Holdout Set)')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_holdout, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Holdout Set)")
plt.legend()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_holdout, y_prob)
avg_prec = average_precision_score(y_holdout, y_prob)
plt.plot(recall, precision, label=f"Avg Precision = {avg_prec:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Holdout Set)")
plt.legend()
plt.show()

# Probability Distribution
sns.histplot(y_prob[y_holdout == 0], color="blue", label="No crash", kde=True, bins=12, stat="density", alpha=0.5)
sns.histplot(y_prob[y_holdout == 1], color="red", label="Crash", kde=True, bins=12, stat="density", alpha=0.5)
plt.xlabel("Predicted Probability (Crash)")
plt.title("Predicted Probability Distribution")
plt.legend()
plt.show()

# Feature Importance (from base estimator)
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.barh(importances['feature'], importances['importance'], color='lightgreen')
plt.gca().invert_yaxis()
plt.xlabel('Importance (Gini)')
plt.title('Random Forest Feature Importance')
plt.show()

# Permutation Importance
perm = permutation_importance(rf, X_trainval, y_trainval, n_repeats=10, random_state=42)
perm_df = pd.DataFrame({'feature': X.columns, 'importance': perm.importances_mean}).sort_values('importance', ascending=False)

plt.barh(perm_df['feature'], perm_df['importance'], color='coral')
plt.gca().invert_yaxis()
plt.xlabel('Permutation Importance')
plt.title('Permutation Feature Importance')
plt.show()

# SHAP Analysis
X_sample = X_trainval.sample(400, random_state=42)
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sample)
plot_height = max(6, 0.4 * X_sample.shape[1])
shap.summary_plot(shap_values, X_sample, plot_size=(plot_height * 0.75, plot_height))

# # SHAP Dependency Plots
top_features = importances['feature'].head(8)
# shap_vals_for_plot = shap_values[1]
# for feat in top_features:
#     shap.dependence_plot(feat, shap_vals_for_plot, X_sample)

# Partial Dependence Plots
for feat in top_features:
    PartialDependenceDisplay.from_estimator(rf, X_trainval, [feat], kind='average')
    plt.show()

# Learning Curve
train_sizes, train_scores, valid_scores = learning_curve(
    rf, X_trainval, y_trainval, cv=cv, scoring='accuracy', n_jobs=-1
)
plt.plot(train_sizes, train_scores.mean(axis=1), label="Train")
plt.plot(train_sizes, valid_scores.mean(axis=1), label="Validation")
plt.xlabel("Training Samples")
plt.ylabel("Accuracy")
plt.ylim(0.5, 1)
plt.title("Learning Curve")
plt.legend()
plt.show()

# Cumulative Gain & Lift
sorted_idx = np.argsort(y_prob)[::-1]
y_sorted = y_holdout.iloc[sorted_idx]
cum_pos_found = np.cumsum(y_sorted) / sum(y_sorted)
perc_samples = np.arange(1, len(y_sorted)+1) / len(y_sorted)
lift = cum_pos_found / perc_samples

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(perc_samples, cum_pos_found, label="Model", color="royalblue", linewidth=2)
axes[0].plot(perc_samples, perc_samples, 'k--', label="Random")
axes[0].set_xlabel("Fraction of Samples")
axes[0].set_ylabel("Fraction of Positives Found")
axes[0].set_title("Cumulative Gain Curve")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(perc_samples, lift, label="Model", color="darkorange", linewidth=2)
axes[1].plot([0, 1], [1, 1], 'k--', label="Random")
axes[1].set_xlabel("Fraction of Samples")
axes[1].set_ylabel("Lift")
axes[1].set_title("Lift Curve")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
