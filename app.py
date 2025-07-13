import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load('crop_model.pkl')

st.set_page_config(page_title="🌱 Crop Recommendation", layout="centered")
st.title("🌾 Smart Krishi Assistant – Crop Recommendation")
st.header("🧪 Enter Soil and Environmental Parameters")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, step=1)
    P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, step=1)
    K = st.number_input("Potassium (K)", min_value=0, max_value=200, step=1)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1, format="%.2f")

with col2:
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, step=0.1, format="%.2f")
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1, format="%.2f")
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, step=0.1, format="%.2f")

# Predict
if st.button("🔍 Recommend Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    predicted_crop = model.predict(input_data)[0]
    st.success(f"✅ Recommended Crop: **{predicted_crop}**")
