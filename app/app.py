import streamlit as st
import numpy as np
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
SAVED_WEIGHTS = RESULTS / "saved_weights"
theta = np.load(SAVED_WEIGHTS / "theta_normal_B.npy")
mean_B = np.load(SAVED_WEIGHTS / "mean_B.npy")
std_B = np.load(SAVED_WEIGHTS / "std_B.npy")
st.set_page_config(
    page_title="Solar Power Prediction",
    page_icon="☀️"
)
st.title("Solar Power Prediction")
st.write(
    "Predict plant AC power using public weather data."
)
hour = st.number_input(
    "Hour of Day",
    min_value=0,
    max_value=23,
    value=12,
    step=1
)
sw_radiation = st.number_input(
    "Shortwave Radiation (W/m²)",
    min_value=0.0,
    value=500.0,
    step=10.0
)
temp_2m = st.number_input(
    "Temperature 2m (°C)",
    value=25.0,
    step=1.0
)
cloud_cover = st.number_input(
    "Cloud Cover (%)",
    min_value=0.0,
    max_value=100.0,
    value=20.0,
    step=1.0
)
if st.button("Predict AC Power"):
    sin_hour = np.sin(2 * np.pi * hour / 24)
    cos_hour = np.cos(2 * np.pi * hour / 24)
    features = np.array([
        sw_radiation,
        temp_2m,
        cloud_cover,
        sin_hour,
        cos_hour
    ])
    scaled_features = (features - mean_B) / std_B
    X = np.concatenate([[1], scaled_features])
    prediction = X @ theta
    prediction = max(prediction, 0)
    # Display prediction
    st.success(
        f"Predicted AC Power: {prediction:.2f} kW"
    )
