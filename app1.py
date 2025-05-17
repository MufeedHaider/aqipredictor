import os
import joblib
import requests
import pandas as pd
import numpy as np
import hopsworks
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="Karachi AQI Forecast", 
    page_icon="🌎", 
    layout="wide"
)

# API Keys
HOPSWORKS_API_KEY = "EP5aAsjdPutNPjHf.qXzpVQ2wrS8dHURwxxJMggYsRgWHpy42SN2CqvSB5xdHGOdqZoezwioQU9tqj4Cc"
OPENWEATHER_API_KEY = "05f1a56c472a46f3fb42f0a3775cc7e7"
LAT, LON = 24.8607, 67.0011

# Define AQI categories
AQI_CATEGORIES = {
    "Good": {"range": (0, 50), "color": "#00e400"},
    "Moderate": {"range": (51, 100), "color": "#ffff00"},
    "Unhealthy for Sensitive Groups": {"range": (101, 150), "color": "#ff7e00"},
    "Unhealthy": {"range": (151, 200), "color": "#ff0000"},
    "Very Unhealthy": {"range": (201, 300), "color": "#99004c"},
    "Hazardous": {"range": (301, 500), "color": "#7e0023"}
}

# Simple CSS for styling
st.markdown("""
<style>
    .stApp {
        background: #121212;
        color: #E5E7EB;
    }
    h1, h2, h3 {
        color: #F0F9FF;
    }
    .card {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-box {
        background: rgba(17, 24, 39, 0.7);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #94A3B8;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Connect to Hopsworks
@st.cache_resource
def connect_to_hopsworks():
    try:
        project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
        fs = project.get_feature_store()
        return project, fs
    except Exception as e:
        st.error(f"Failed to connect to Hopsworks: {str(e)}")
        return None, None

# Load Model from Hopsworks
@st.cache_resource
def load_model_from_hopsworks(model_name="linear_regression_model"):
    try:
        project, _ = connect_to_hopsworks()
        if not project:
            return None
            
        mr = project.get_model_registry()
        model_version = 16
        formatted_model_name = model_name.lower().replace(" ", "_")

        try:
            model = mr.get_model(formatted_model_name, version=model_version)
        except:
            model = mr.get_model(formatted_model_name)
        
        model_dir = model.download()
        model_file = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
        if not model_file:
            return None
        
        model_path = os.path.join(model_dir, model_file[0])
        return joblib.load(model_path)

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Fetch Weather Data
@st.cache_data(ttl=3600)
def fetch_current_weather():
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&units=metric&appid={OPENWEATHER_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch weather data: {response.status_code}")
        return None

# Fetch AQI Forecast
@st.cache_data(ttl=3600)
def fetch_aqi_forecast():
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch AQI data: {response.status_code}")
        return None

# Process Forecast Data
def process_forecast_data(data):
    if not data:
        return None

    forecast_list = data.get("list", [])
    if len(forecast_list) < 3:
        return None

    forecast_list = forecast_list[:3]
    processed_data = []
    
    previous_aqi = None
    rolling_pm10 = []
    rolling_pm25 = []

    for i, entry in enumerate(forecast_list):
        components = entry.get("components", {})
        date = datetime.utcnow() + timedelta(days=i+1)

        aqi = entry.get("main", {}).get("aqi", np.nan)

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
        
        rolling_pm10_mean = pd.Series(rolling_pm10).rolling(window=min(3, len(rolling_pm10)), min_periods=1).mean().iloc[-1]
        rolling_pm10_std = pd.Series(rolling_pm10).rolling(window=min(3, len(rolling_pm10)), min_periods=1).std().iloc[-1]
        rolling_pm25_mean = pd.Series(rolling_pm25).rolling(window=min(3, len(rolling_pm25)), min_periods=1).mean().iloc[-1]
        rolling_pm25_std = pd.Series(rolling_pm25).rolling(window=min(3, len(rolling_pm25)), min_periods=1).std().iloc[-1]

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

    # Fill missing values
    df.fillna(df.mean(), inplace=True)
    return df

# Predict AQI
def predict_aqi(model, input_data):
    try:
        required_features = model.feature_names_in_
        
        missing_features = set(required_features) - set(input_data.columns)
        if missing_features:
            for feature in missing_features:
                input_data[feature] = 0
                
        input_data = input_data[required_features]
        predictions = model.predict(input_data)
        return predictions
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# Create AQI Chart
def create_aqi_chart(predictions, dates):
    df = pd.DataFrame({
        'Date': dates,
        'AQI Change Rate': predictions
    })
    
    # Create color scale
    colors = []
    for pred in predictions:
        if pred >= 5:
            colors.append("#7e0023")  # Hazardous
        elif pred >= 4:
            colors.append("#ff0000")  # Unhealthy
        elif pred >= 3:
            colors.append("#ff7e00")  # Moderate
        elif pred >= 2:
            colors.append("#ffff00")  # Good
        else:
            colors.append("#00e400")  # Very Good
    
    # Create bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['AQI Change Rate'],
        marker_color=colors,
        text=df['AQI Change Rate'].round(1),
        textposition='auto',
    ))
    
    # Customize layout
    fig.update_layout(
        title="AQI Change Rate Forecast",
        xaxis_title="Date",
        yaxis_title="Change Rate",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#e5e7eb"),
        height=400
    )
    
    return fig

# Create pollutant chart
def create_pollutant_chart(input_data):
    if input_data is None:
        return None
        
    pollutants = input_data[["pm2_5", "pm10", "no2", "o3"]].iloc[0].to_dict()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=list(pollutants.keys()),
        y=list(pollutants.values()),
        marker_color=['#38BDF8', '#818CF8', '#A78BFA', '#F472B6'],
        text=[f"{val:.2f}" for val in pollutants.values()],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Current Pollutant Levels",
        xaxis_title="Pollutant",
        yaxis_title="Value (μg/m³)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#e5e7eb"),
        height=350
    )
    
    return fig

# Main application
def main():
    # App header
    st.markdown("<h1>🌎 Karachi Air Quality Forecast</h1>", unsafe_allow_html=True)
    
    # Load data and model
    with st.spinner("Loading data..."):
        model = load_model_from_hopsworks()
        current_weather = fetch_current_weather()
        forecast_data = fetch_aqi_forecast()
        
        if not model or not forecast_data:
            st.error("Failed to load required data")
            st.stop()
    
    # Display weather information
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📍 Karachi, Pakistan")
        
        if current_weather:
            temp = current_weather.get("main", {}).get("temp", "N/A")
            humidity = current_weather.get("main", {}).get("humidity", "N/A")
            weather_desc = current_weather.get("weather", [{}])[0].get("description", "N/A").capitalize()
            
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Temperature</div>
                <div class="metric-value">{temp}°C</div>
                <div>{weather_desc}</div>
            </div>
            <div class="metric-box">
                <div class="metric-title">Humidity</div>
                <div class="metric-value">{humidity}%</div>
                <div>Last updated: {datetime.now().strftime('%H:%M')}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Weather data unavailable")
            
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Process and display AQI forecast
    with col2:
        processed_data = process_forecast_data(forecast_data)
        
        if processed_data is not None:
            predictions = predict_aqi(model, processed_data)
            
            if predictions is not None:
                forecast_dates = [(datetime.utcnow() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(len(predictions))]
                
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### 📈 AQI Forecast")
                
                # Display AQI chart
                aqi_chart = create_aqi_chart(predictions, forecast_dates)
                st.plotly_chart(aqi_chart, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error("Error making predictions")
        else:
            st.error("Error processing forecast data")
    
    # Display pollutant levels
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🔬 Pollutant Levels")
    
    if processed_data is not None:
        pollutant_chart = create_pollutant_chart(processed_data)
        if pollutant_chart:
            st.plotly_chart(pollutant_chart, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Simple footer
    st.markdown("<p style='text-align:center;color:#94A3B8;'>Karachi AQI Forecast | Data from OpenWeather API</p>", unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()

