import os
import joblib
import requests
import pandas as pd
import datetime
import numpy as np
import hopsworks
import streamlit as st
st.set_page_config(page_title="Karachi AQI Forecast", page_icon="🌎", layout="wide")
# Hopsworks API Key (Replace with your own)
HOPSWORKS_API_KEY = "EP5aAsjdPutNPjHf.qXzpVQ2wrS8dHURwxxJMggYsRgWHpy42SN2CqvSB5xdHGOdqZoezwioQU9tqj4Cc"

# OpenWeather API Key (Replace with your own)
OPENWEATHER_API_KEY = "05f1a56c472a46f3fb42f0a3775cc7e7"

# Karachi Coordinates
LAT, LON = 24.8607, 67.0011
st.markdown("""
    <style>
        .main {background-color: #f4f4f4; color: #000000;}
        .stApp {background-color: #ffffff; padding: 20px;}
    </style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
        padding: 20px;
    }
    .title-container {
        text-align: center;
        margin-bottom: 20px;
    }
    .main-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .weather-box {
        background: rgba(0, 0, 0, 0.5);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
    }
    .st-emotion-cache-h4xjwg{
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);   }
    .weather-box h3 {
        color: #ffcc00;
        font-size: 24px;
    }
    .prediction-table-container {
        margin: 20px auto;
        width: 80%;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #1e3c72, #2a5298);
        border-radius: 10px;
        padding: 20px;
        color: white;
    }
    .footer {
        text-align: center;
        color: #d1d1d1;
        margin-top: 20px;
    }
    .st-emotion-cache-6qob1r{
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        }
</style>
""", unsafe_allow_html=True)
# Connect to Hopsworks
try:
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    fs = project.get_feature_store()
except Exception as e:
    print(f"❌ Failed to connect: {str(e)}")

# Function to Load Model from Hopsworks
def load_model_from_hopsworks(model_name="linear_regression_model"):
    """Fetch model from Hopsworks Model Registry."""
    try:
        mr = project.get_model_registry()
        model_version = 16  # Change if needed
        formatted_model_name = model_name.lower().replace(" ", "_")

        try:
            model = mr.get_model(formatted_model_name, version=model_version)
        except:
            st.warning(f"⚠️ Version {model_version} not found. Fetching latest version.")
            model = mr.get_model(formatted_model_name)  # Fetch latest version
        
        model_dir = model.download()
        model_file = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
        if not model_file:
            st.error(f"❌ No .pkl model file found in {model_dir}.")
            return None
        
        model_path = os.path.join(model_dir, model_file[0])
        return joblib.load(model_path)

    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Function to Fetch AQI Forecast from OpenWeather
def fetch_aqi_forecast():
    """Fetch AQI forecast data from OpenWeather API."""
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error("⚠️ Failed to fetch AQI data. Check OpenWeather API Key.")
        return None

# Function to Handle Missing Values & Outliers
def clean_data(df):
    """Fill NaN values with mean and remove outliers using IQR."""
    
    # Fill NaN values with column means
    df.fillna(df.mean(), inplace=True)

    # Remove outliers using IQR
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df.clip(lower=lower_bound, upper=upper_bound, axis=1)
    return df

# Function to Process Forecast Data
def process_forecast_data(data):
    """Extract relevant features and clean data."""
    if not data:
        return None

    forecast_list = data["list"][:3]  # Get next 3 days' data
    processed_data = []
    
    previous_aqi = None
    rolling_pm10 = []
    rolling_pm25 = []

    for i, entry in enumerate(forecast_list):
        components = entry["components"]
        date = datetime.datetime.utcnow() + datetime.timedelta(days=i+1)

        aqi = entry.get("main", {}).get("aqi", np.nan)  # Handle missing AQI

        # Compute AQI Change Rate
        aqi_change_rate = 0
        if previous_aqi is not None and previous_aqi > 0:
            aqi_change_rate = (aqi - previous_aqi) / previous_aqi
        previous_aqi = aqi

        # Get pollutant values
        pm10 = components.get("pm10", np.nan)
        pm25 = components.get("pm2_5", np.nan)
        no2 = components.get("no2", np.nan)
        o3 = components.get("o3", np.nan)

        # Rolling statistics
        rolling_pm10.append(pm10)
        rolling_pm25.append(pm25)
        
        rolling_pm10_mean = pd.Series(rolling_pm10).rolling(window=3, min_periods=1).mean().iloc[-1]
        rolling_pm10_std = pd.Series(rolling_pm10).rolling(window=3, min_periods=1).std().iloc[-1]
        rolling_pm25_mean = pd.Series(rolling_pm25).rolling(window=3, min_periods=1).mean().iloc[-1]
        rolling_pm25_std = pd.Series(rolling_pm25).rolling(window=3, min_periods=1).std().iloc[-1]

        features = [
            pm25, pm10, no2, o3, date.hour, date.day, date.month,
            aqi_change_rate, date.weekday(), 1 if date.weekday() in [5, 6] else 0,
            rolling_pm25_mean, rolling_pm10_mean, rolling_pm25_std, rolling_pm10_std
        ]
        
        processed_data.append(features)

    df = pd.DataFrame(processed_data, columns=[
        "pm2_5", "pm10", "no2", "o3", "hour", "day", "month",
        "aqi_change_rate", "day_of_week", "is_weekend",
        "rolling_pm25_mean", "rolling_pm10_mean", "rolling_pm25_std", "rolling_pm10_std"
    ])

    return clean_data(df).values  # Clean data & return array

# Function to Predict AQI Change Rate
def predict_aqi(model, input_data):
    """Predict AQI Change Rate using the trained model.""" 
    expected_features = model.feature_names_in_  # Ensure correct order
    df = pd.DataFrame(input_data, columns=expected_features)  

    predictions = model.predict(df)
    return predictions

# Function to Interpret AQI Change
def interpret_aqi_change(change_rate):
    """Interpret the AQI change rate."""
    if change_rate >= 5:
        return "🔴 Significant Deterioration"
    elif change_rate >= 4:
        return "🟠 Slight Increase"
    elif change_rate >= 3:
        return "🟡 Stable"
    elif change_rate >= 2:
        return "🟢 Slight Improvement"
    else:
        return "💚 Major Improvement"

# Streamlit UI
st.title("🌎 Karachi AQI Forecast")

# Load Model
model = load_model_from_hopsworks()

if model:
    print(f"✅ Model loaded successfully!")

    # Fetch AQI Forecast
    st.subheader("📡 Fetching AQI Forecast for Karachi...")
    forecast_data = fetch_aqi_forecast()

    if forecast_data:
        processed_data = process_forecast_data(forecast_data)
        if processed_data is not None:
            predictions = predict_aqi(model, processed_data)

            # Display Predictions
            st.subheader("🔮 Predicted AQI for Next 3 Days:")
            for i, pred in enumerate(predictions):
                date = (datetime.datetime.utcnow() + datetime.timedelta(days=i+1)).strftime('%Y-%m-%d')
                interpretation = interpret_aqi_change(pred)
                st.write(f"📅 {date}: Change Rate = {pred:.0f} ({interpretation})")
        else:
            st.error("⚠️ Error processing forecast data.")
    else:
        st.error("⚠️ Failed to fetch AQI data.")
else:
    st.error("⚠️ Model could not be loaded.")

hazardous_detected = any(predictions>3)
if hazardous_detected:
    with st.sidebar:
        st.markdown("""
        <div style="background-color: white;color:black; border: 2px solid #ff4d4d; border-radius: 10px; padding: 15px; text-align: center;">
            <h3 style="color: #ff4d4d;">🚨 Hazardous AQI!</h3>
            <p>AQI is hazardous in the next 3 days.<br>
            - Dont go outside.<br>
            - Keep Hydrated.<br>
        </div>
        """, unsafe_allow_html=True)
else:
    with st.sidebar:
        st.markdown("""
        <div style="background-color: white;color:black; border: 2px solid #4dff4d; border-radius: 10px; padding: 15px; text-align: center;">
            <h3 style="color: #4dff4d;">✅ AQI is Safe!</h3>
            <p>No hazardous AQI detected in the forecast.</p>
        </div>
        """, unsafe_allow_html=True)
