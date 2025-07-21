# -----------------------------
# Employee Salary Prediction – Streamlit App (Cloud/Local Compatible)
# -----------------------------
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load Model & Scaler
# -----------------------------
model = xgb.XGBRegressor()
model.load_model("xgb_model.json")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")
st.title("💼 Employee Salary Prediction")
st.markdown("Predict salary based on age, experience, and gender using a trained AI model.")

# -----------------------------
# Option to Upload CSV (Mandatory in Cloud)
# -----------------------------
uploaded_file = st.file_uploader("Upload Employee Salary CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write(df.head())
else:
    st.warning("⚠️ Please upload a CSV file to proceed.")
    st.stop()

# -----------------------------
# Input Fields
# -----------------------------
exp = st.slider("Years of Experience", 0, 40, 5)
age = st.slider("Age", 18, 65, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
gender_encoded = 1 if gender == "Male" else 0

# -----------------------------
# Prediction
# -----------------------------
input_df = pd.DataFrame([[exp, age, gender_encoded, 0]],
                        columns=['Experience_Years', 'Age', 'Gender', 'Salary_per_Year'])
scaled = scaler.transform(input_df)
salary_pred = model.predict(scaled)[0]
st.success(f"💰 Predicted Salary: ₹{salary_pred:,.2f}")

# -----------------------------
# SHAP Explainability
# -----------------------------
if st.button("Show Explainability"):
    st.subheader("Feature Contribution (SHAP)")
    explainer = shap.Explainer(model)
    shap_values = explainer(scaled)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(bbox_inches='tight')

# -----------------------------
# END OF APP
# -----------------------------
