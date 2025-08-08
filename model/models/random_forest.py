import pandas as pd
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('model/datasets/final_csv/flags_and_crash_data.csv')
X = df.drop(columns=['label'])
y = df['label']

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
aucs = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')

print(f"Average CV accuracy: {accuracies.mean():.4f}")
print(f"Average CV AUC: {aucs.mean():.4f}")

# Fit on all data
rf.fit(X, y)

# SHAP
X_sample = X.sample(400, random_state=42)  # smaller sample for speed
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

print("\nTop features from regular SHAP:")
print(importance_df.head(10))

# Find top interactions
top_features = importance_df['feature'].head(8).tolist()
X_top = X_sample[top_features]

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

# Run on top features
shap_interaction_values = explainer.shap_interaction_values(X_top)

if isinstance(shap_interaction_values, list):
    interaction_values_class1 = shap_interaction_values[1]  # select positive class
else:
    interaction_values_class1 = shap_interaction_values


# If the last dimension corresponds to classes, select positive class index 1
if interaction_values_class1.ndim == 4 and interaction_values_class1.shape[-1] == 2:
    interaction_values_class1 = interaction_values_class1[:, :, :, 1]

# Mean absolute interaction strengths over samples (axis=0)
interaction_importance = np.abs(interaction_values_class1).mean(axis=0)

interaction_df = pd.DataFrame(
    interaction_importance,
    index=top_features,
    columns=top_features
)

# sort top feature interaction pairs (exclude diagonal)
top_pairs = (
    interaction_df.where(~np.eye(interaction_df.shape[0], dtype=bool))
    .stack()
    .sort_values(ascending=False)
)

print("\nTop feature interactions:")
print(top_pairs.head(10))

# If shap_interaction_values is a list, use positive class interactions
if isinstance(shap_interaction_values, list):
    shap_interaction_to_plot = shap_interaction_values[1]
else:
    shap_interaction_to_plot = shap_interaction_values


# Fix the class dimension for interaction values
if shap_interaction_to_plot.ndim == 4 and shap_interaction_to_plot.shape[-1] == 2:
    shap_interaction_to_plot = shap_interaction_to_plot[:, :, :, 1]

shap.summary_plot(
    shap_interaction_to_plot,
    X_top.values,
    plot_type="interaction",
    feature_names=X_top.columns.tolist(),
    show=True
)


