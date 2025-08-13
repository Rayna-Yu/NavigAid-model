import pandas as pd
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


# Load dataset
df = pd.read_csv('model/datasets/final_csv/continuous_night_data.csv')
X = df.drop(columns=['label'])
y = df['label']

# Hold out 20% of data for final validation (test)
X_trainval, X_holdout, y_trainval, y_holdout = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# param_dist = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
#     'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
#     'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
#     'max_features': ['sqrt', 'log2', None]
# }

# rf = RandomForestClassifier(random_state=42)

# search = RandomizedSearchCV(
#     rf, param_distributions=param_dist,
#     n_iter=30, cv=5, scoring='roc_auc', random_state=42,
#     n_jobs=-1, verbose=1
# )

# search.fit(X, y)
# print("Best params:", search.best_params_)
# print("Best CV AUC:", search.best_score_)

rf = RandomForestClassifier(
    n_estimators=300,
    # min_samples_split=2,
    # min_samples_leaf=5,
    # max_depth=12,
    random_state=42,
)

# Cross-validation setup on training data only
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation accuracy and AUC on training data
accuracies = cross_val_score(rf, X_trainval, y_trainval, cv=cv, scoring='accuracy')
aucs = cross_val_score(rf, X_trainval, y_trainval, cv=cv, scoring='roc_auc')
print(f"Average CV accuracy (train set): {accuracies.mean():.4f}")
print(f"Average CV AUC (train set): {aucs.mean():.4f}")

# Fit on full training data (train+val)
rf.fit(X_trainval, y_trainval)

train_accuracies = []
val_accuracies = []

for train_idx, val_idx in cv.split(X_trainval, y_trainval):
    X_train, X_val = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
    y_train, y_val = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]
    
    rf.fit(X_train, y_train)
    
    train_acc = rf.score(X_train, y_train)
    val_acc = rf.score(X_val, y_val)
    
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

print(f"Mean training accuracy (CV folds): {np.mean(train_accuracies):.4f}")
print(f"Mean validation accuracy (CV folds): {np.mean(val_accuracies):.4f}")

plt.figure()
plt.plot(train_accuracies, label='Training accuracy', marker='o')
plt.plot(val_accuracies, label='Validation accuracy', marker='o')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy per Fold (train set)')
plt.legend()
plt.show()

# Evaluate final model on holdout set
holdout_acc = rf.score(X_holdout, y_holdout)
y_holdout_pred = rf.predict(X_holdout)
cm = confusion_matrix(y_holdout, y_holdout_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Holdout Set)')
plt.show()

y_holdout_prob = rf.predict_proba(X_holdout)[:, 1]
fpr, tpr, _ = roc_curve(y_holdout, y_holdout_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Holdout Set)')
plt.legend(loc="lower right")
plt.show()

precision, recall, _ = precision_recall_curve(y_holdout, y_holdout_prob)
avg_precision = average_precision_score(y_holdout, y_holdout_prob)

plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label=f'Avg Precision = {avg_precision:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Holdout Set)')
plt.legend(loc='upper right')
plt.show()

trainval_acc = rf.score(X_trainval, y_trainval)
print(f"Training accuracy on full train+val set: {trainval_acc:.4f}")
print(f"Holdout set accuracy: {holdout_acc:.4f}")

# Confusion Matrix
y_pred = rf.predict(X)
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# ROC curve
y_prob = rf.predict_proba(X)[:, 1]
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Precision-recall curve
precision, recall, _ = precision_recall_curve(y, y_prob)
avg_precision = average_precision_score(y, y_prob)

plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label=f'Avg Precision = {avg_precision:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='upper right')
plt.show()

# Feature importance
importances = rf.feature_importances_
feat_imp_df = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values('importance', ascending=False)

plt.figure(figsize=(10,6))
plt.barh(feat_imp_df['feature'], feat_imp_df['importance'], color='lightgreen')
plt.xlabel('Feature Importance (Gini)')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()
plt.show()

# SHAP values
X_sample = X.sample(400, random_state=42)
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sample)

shap_values_class1 = shap_values[1] if isinstance(shap_values, list) else shap_values
shap_values_class1 = np.array(shap_values_class1)
shap_means = np.abs(shap_values_class1).mean(axis=0)
if shap_means.ndim == 2 and shap_means.shape[1] == 2:
    shap_means = shap_means[:, 1]

importance_df = pd.DataFrame({
    'feature': X.columns,
    'mean_abs_shap': shap_means
}).sort_values('mean_abs_shap', ascending=False)

print("\nTop features from SHAP:")
print(importance_df.head(10))

plt.figure(figsize=(10, 6))
plt.barh(
    importance_df['feature'], 
    importance_df['mean_abs_shap'], 
    color='skyblue'
)
plt.xlabel('Mean Absolute SHAP Value')
plt.title('Feature Importance from SHAP Values')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Waterfall plot
pred_probs = rf.predict_proba(X_sample)[:, 1]

# Highest predicted probability sample
sample_idx_1 = np.argmax(pred_probs)
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[sample_idx_1, :, 1], 
        base_values=explainer.expected_value[1],
        data=X_sample.iloc[sample_idx_1],
        feature_names=X_sample.columns
    ),
    max_display=len(X_sample.columns)
)
plt.show()

# Mid probability (closest to 0.5)
sample_idx_mid = np.argmin(np.abs(pred_probs - 0.5))
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[sample_idx_mid, :, 1], 
        base_values=explainer.expected_value[1],
        data=X_sample.iloc[sample_idx_mid],
        feature_names=X_sample.columns
    ),
    max_display=len(X_sample.columns)
)
plt.show()

# Lowest predicted probability sample
sample_idx_0 = np.argmin(pred_probs)
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[sample_idx_0, :, 1], 
        base_values=explainer.expected_value[1],
        data=X_sample.iloc[sample_idx_0],
        feature_names=X_sample.columns
    ),
    max_display=len(X_sample.columns)
)
plt.show()

# Dependency plots
for feature in importance_df['feature'].head(5):
    shap.dependence_plot(
        feature,
        shap_values[:, :, 1],
        X_sample,
        show=True
    )
    plt.tight_layout()
    plt.show()

# Interaction values
top_features = importance_df['feature'].head(8).tolist()
X_top = X_sample[top_features]
shap_interaction_values = explainer.shap_interaction_values(X_top)

if isinstance(shap_interaction_values, list):
    interaction_values_class1 = shap_interaction_values[1]
else:
    interaction_values_class1 = shap_interaction_values

# Fix class dim if necessary
if interaction_values_class1.ndim == 4 and interaction_values_class1.shape[-1] == 2:
    interaction_values_class1 = interaction_values_class1[:, :, :, 1]

interaction_importance = np.abs(interaction_values_class1).mean(axis=0)

interaction_df = pd.DataFrame(
    interaction_importance,
    index=top_features,
    columns=top_features
)

top_pairs = (
    interaction_df.where(~np.eye(interaction_df.shape[0], dtype=bool))
    .stack()
    .sort_values(ascending=False)
)

print("\nTop feature interactions:")
print(top_pairs.head(10))

# SHAP interaction summary plot
if isinstance(shap_interaction_values, list):
    shap_interaction_to_plot = shap_interaction_values[1]
else:
    shap_interaction_to_plot = shap_interaction_values

if shap_interaction_to_plot.ndim == 4 and shap_interaction_to_plot.shape[-1] == 2:
    shap_interaction_to_plot = shap_interaction_to_plot[:, :, :, 1]

shap.summary_plot(
    shap_interaction_to_plot,
    X_top.values,
    plot_type="interaction",
    feature_names=X_top.columns.tolist(),
    show=True
)
plt.show()

# Learning Curve
train_sizes, train_scores, valid_scores = learning_curve(
    rf, X, y, cv=cv, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
)

plt.figure(figsize=(8,5))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='r', label='Training accuracy')
plt.plot(train_sizes, valid_scores.mean(axis=1), 'o-', color='g', label='Validation accuracy')
plt.xlabel('Training set size')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.show()

# Permutation importance
perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42)

perm_df = pd.DataFrame({'feature': X.columns, 'importance': perm_importance.importances_mean}).sort_values('importance', ascending=False)

plt.figure(figsize=(10,6))
plt.barh(perm_df['feature'], perm_df['importance'], color='coral')
plt.xlabel('Permutation Importance')
plt.title('Permutation Feature Importance')
plt.gca().invert_yaxis()
plt.show()

# Calibrate using isotonic regression
calibrated_rf = CalibratedClassifierCV(estimator=rf, method='isotonic', cv=5)
calibrated_rf.fit(X_train, y_train)

# Compare calibration curves
prob_rf = rf.predict_proba(X)[:, 1]
prob_cal_rf = calibrated_rf.predict_proba(X)[:, 1]

fraction_pos_rf, mean_pred_rf = calibration_curve(y, prob_rf, n_bins=10)
fraction_pos_cal, mean_pred_cal = calibration_curve(y, prob_cal_rf, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(mean_pred_rf, fraction_pos_rf, "o-", label="RF (uncalibrated)")
plt.plot(mean_pred_cal, fraction_pos_cal, "o-", label="RF (calibrated)")
plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration Curve")
plt.legend()
plt.show()
