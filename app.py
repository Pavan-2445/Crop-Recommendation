import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="🌱 Crop Recommendation", layout="centered")
st.title("🌾 Smart Krishi Assistant – Crop Recommendation")
st.header("🧪 Enter Soil and Environmental Parameters")

# --- Inputs ---
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

# --- Load Model ---
try:
    model_obj = joblib.load("crop_recommendation_model (1).pkl")
    # Support both dictionary format and direct model
    if isinstance(model_obj, dict):
        model = model_obj["model"]
        encoder = model_obj.get("encoder", None)
    else:
        model = model_obj
        encoder = None
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# --- Predict ---
if st.button("🔍 Recommend Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    predicted_crop = model.predict(input_data)[0]

    # Decode label if encoder exists
    if encoder:
        predicted_crop = encoder.inverse_transform([predicted_crop])[0]

    st.success(f"✅ Recommended Crop: **{predicted_crop}**")
