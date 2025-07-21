# -----------------------------
# Train XGBoost Model & Save Artifacts
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv(r"D:\VSCODE\Datanalaysis\AIML IBM\Dataset\Employee_Salary_Dataset.csv")

# -----------------------------
# Feature Engineering
# -----------------------------
df['Salary_per_Year'] = df['Salary'] / (df['Experience_Years'] + 1)
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

# -----------------------------
# Feature Selection
# -----------------------------
features = ['Experience_Years', 'Age', 'Gender', 'Salary_per_Year']
X = df[features]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Train XGBoost Model
# -----------------------------
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# -----------------------------
# Evaluate Model
# -----------------------------
y_pred = xgb_model.predict(X_test_scaled)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# -----------------------------
# Save Model and Scaler
# -----------------------------
xgb_model.save_model("xgb_model.json")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and scaler saved successfully!")