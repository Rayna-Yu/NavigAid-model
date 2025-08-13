from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import shap
import pandas as pd

# Load dataset
df = pd.read_csv('model/datasets/final_csv/night_flags_and_crash_data.csv')
X = df.drop(columns=['label']).astype('float32')
y = df['label'].astype('int')

# Split your data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train HistGradientBoostingClassifier
hgb_model = HistGradientBoostingClassifier(random_state=42)
hgb_model.fit(X_train, y_train)

# Predictions
y_pred = hgb_model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# SHAP analysis
explainer = shap.Explainer(hgb_model, X_train)
shap_values = explainer(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test)
