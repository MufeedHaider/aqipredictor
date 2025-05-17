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













# import os
# import joblib
# import requests
# import pandas as pd
# import datetime
# import numpy as np
# import hopsworks
# import streamlit as st
# import plotly.express as px
# import plotly.graph_objects as go
# from datetime import datetime, timedelta

# # Configure page
# st.set_page_config(
#     page_title="Karachi AQI Forecast", 
#     page_icon="🌎", 
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Hopsworks API Key (Replace with your own)
# HOPSWORKS_API_KEY = "EP5aAsjdPutNPjHf.qXzpVQ2wrS8dHURwxxJMggYsRgWHpy42SN2CqvSB5xdHGOdqZoezwioQU9tqj4Cc"

# # OpenWeather API Key (Replace with your own)
# OPENWEATHER_API_KEY = "05f1a56c472a46f3fb42f0a3775cc7e7"

# # Karachi Coordinates
# LAT, LON = 24.8607, 67.0011

# # Define AQI categories and colors
# AQI_CATEGORIES = {
#     "Good": {"range": (0, 50), "color": "#00e400", "description": "Air quality is satisfactory, and air pollution poses little or no risk."},
#     "Moderate": {"range": (51, 100), "color": "#ffff00", "description": "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."},
#     "Unhealthy for Sensitive Groups": {"range": (101, 150), "color": "#ff7e00", "description": "Members of sensitive groups may experience health effects. The general public is less likely to be affected."},
#     "Unhealthy": {"range": (151, 200), "color": "#ff0000", "description": "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."},
#     "Very Unhealthy": {"range": (201, 300), "color": "#99004c", "description": "Health alert: The risk of health effects is increased for everyone."},
#     "Hazardous": {"range": (301, 500), "color": "#7e0023", "description": "Health warning of emergency conditions: everyone is more likely to be affected."}
# }

# # Enhanced styling with improved CSS
# st.markdown("""
# <style>
#     /* Main app styling */
#     .stApp {
#         background: linear-gradient(135deg, #121212, #1E3A8A);
#         color: #E5E7EB;
#         font-family: 'Segoe UI', 'Roboto', sans-serif;
#     }
    
#     /* Header styling */
#     h1, h2, h3 {
#         color: #F0F9FF;
#         font-weight: 600;
#         margin-bottom: 1rem;
#     }
    
#     h1 {
#         background: linear-gradient(90deg, #38BDF8, #818CF8);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         font-size: 2.5rem;
#         letter-spacing: -0.025em;
#         text-align: center;
#         padding-bottom: 0.5rem;
#         border-bottom: 2px solid rgba(255, 255, 255, 0.1);
#         margin-bottom: 2rem;
#     }
    
#     /* Card styling */
#     .card {
#         background: rgba(30, 41, 59, 0.7);
#         border-radius: 12px;
#         padding: 1.5rem;
#         margin-bottom: 1.5rem;
#         box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
#         border: 1px solid rgba(255, 255, 255, 0.1);
#         backdrop-filter: blur(10px);
#     }
    
#     /* Forecast card */
#     .forecast-card {
#         text-align: center;
#         padding: 1rem;
#         border-radius: 8px;
#         transition: transform 0.3s ease;
#     }
    
#     .forecast-card:hover {
#         transform: translateY(-5px);
#     }
    
#     /* Status indicators */
#     .status-good {
#         color: #00e400;
#     }
    
#     .status-moderate {
#         color: #ffff00;
#     }
    
#     .status-sensitive {
#         color: #ff7e00;
#     }
    
#     .status-unhealthy {
#         color: #ff0000;
#     }
    
#     .status-very-unhealthy {
#         color: #99004c;
#     }
    
#     .status-hazardous {
#         color: #7e0023;
#     }
    
#     /* Alert box styling */
#     .alert-box {
#         padding: 1rem;
#         border-radius: 8px;
#         margin-bottom: 1rem;
#         font-weight: 500;
#     }
    
#     .alert-success {
#         background-color: rgba(0, 200, 83, 0.2);
#         border-left: 4px solid #00C853;
#         color: #E0F2F1;
#     }
    
#     .alert-warning {
#         background-color: rgba(255, 152, 0, 0.2);
#         border-left: 4px solid #FF9800;
#         color: #FFF3E0;
#     }
    
#     .alert-danger {
#         background-color: rgba(244, 67, 54, 0.2);
#         border-left: 4px solid #F44336;
#         color: #FFEBEE;
#     }
    
#     /* Data metric styles */
#     .metric-container {
#         display: flex;
#         flex-wrap: wrap;
#         gap: 1rem;
#         justify-content: space-between;
#         margin-bottom: 1rem;
#     }
    
#     .metric-box {
#         background: rgba(17, 24, 39, 0.7);
#         border-radius: 8px;
#         padding: 1rem;
#         min-width: 150px;
#         flex: 1;
#         text-align: center;
#         border: 1px solid rgba(255, 255, 255, 0.05);
#     }
    
#     .metric-title {
#         font-size: 0.9rem;
#         color: #94A3B8;
#         margin-bottom: 0.5rem;
#     }
    
#     .metric-value {
#         font-size: 1.8rem;
#         font-weight: 600;
#         margin-bottom: 0.5rem;
#     }
    
#     /* Footer styling */
#     .footer {
#         margin-top: 3rem;
#         text-align: center;
#         padding: 1rem;
#         color: #94A3B8;
#         font-size: 0.8rem;
#         border-top: 1px solid rgba(255, 255, 255, 0.1);
#     }
    
#     /* Make sure Streamlit-specific elements match our theme */
#     .stButton button {
#         background-color: #3B82F6;
#         color: white;
#         border-radius: 6px;
#         padding: 0.5rem 1rem;
#         font-weight: 500;
#         border: none;
#         transition: all 0.3s ease;
#     }
    
#     .stButton button:hover {
#         background-color: #2563EB;
#         box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
#     }
    
#     .stTextInput input, .stSelectbox, .stMultiselect {
#         background-color: rgba(17, 24, 39, 0.7);
#         border: 1px solid rgba(255, 255, 255, 0.1);
#         border-radius: 6px;
#         color: white;
#     }
    
#     /* Tabs styling */
#     .stTabs [data-baseweb="tab-list"] {
#         gap: 8px;
#     }
    
#     .stTabs [data-baseweb="tab"] {
#         background-color: rgba(30, 41, 59, 0.7);
#         border-radius: 6px 6px 0 0;
#         padding: 0.5rem 1rem;
#         border: 1px solid rgba(255, 255, 255, 0.1);
#         border-bottom: none;
#     }
    
#     .stTabs [aria-selected="true"] {
#         background-color: rgba(59, 130, 246, 0.2);
#         border-bottom: 2px solid #3B82F6;
#     }
    
#     /* Loading spinner */
#     .stSpinner > div {
#         border-color: #3B82F6 #3B82F6 transparent !important;
#     }
    
#     /* Sidebar styling */
#     [data-testid="stSidebar"] {
#         background-color: rgba(17, 24, 39, 0.95);
#         border-right: 1px solid rgba(255, 255, 255, 0.05);
#         padding: 2rem 1rem;
#     }
    
#     [data-testid="stSidebarNav"] {
#         background-color: transparent;
#     }
    
#     [data-testid="stSidebarNav"] li {
#         margin-bottom: 0.5rem;
#     }
    
#     /* Chart background */
#     .js-plotly-plot {
#         background-color: rgba(17, 24, 39, 0.5) !important;
#         border-radius: 8px;
#         padding: 1rem;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Connect to Hopsworks
# @st.cache_resource
# def connect_to_hopsworks():
#     try:
#         project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
#         fs = project.get_feature_store()
#         return project, fs
#     except Exception as e:
#         st.error(f"❌ Failed to connect to Hopsworks: {str(e)}")
#         return None, None

# # Function to Load Model from Hopsworks
# @st.cache_resource
# def load_model_from_hopsworks(model_name="linear_regression_model"):
#     """Fetch model from Hopsworks Model Registry."""
#     try:
#         project, _ = connect_to_hopsworks()
#         if not project:
#             return None
            
#         mr = project.get_model_registry()
#         model_version = 16  # Change if needed
#         formatted_model_name = model_name.lower().replace(" ", "_")

#         try:
#             model = mr.get_model(formatted_model_name, version=model_version)
#         except:
#             st.warning(f"⚠️ Version {model_version} not found. Fetching latest version.")
#             model = mr.get_model(formatted_model_name)  # Fetch latest version
        
#         model_dir = model.download()
#         model_file = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
#         if not model_file:
#             st.error(f"❌ No .pkl model file found in {model_dir}.")
#             return None
        
#         model_path = os.path.join(model_dir, model_file[0])
#         return joblib.load(model_path)

#     except Exception as e:
#         st.error(f"❌ Error loading model: {str(e)}")
#         return None

# # Function to Fetch Current Weather from OpenWeather
# @st.cache_data(ttl=3600)  # Cache for 1 hour
# def fetch_current_weather():
#     """Fetch current weather data from OpenWeather API."""
#     url = f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&units=metric&appid={OPENWEATHER_API_KEY}"
#     response = requests.get(url)
    
#     if response.status_code == 200:
#         return response.json()
#     else:
#         st.error(f"⚠️ Failed to fetch weather data. Status code: {response.status_code}")
#         return None

# # Function to Fetch AQI Forecast from OpenWeather
# @st.cache_data(ttl=3600)  # Cache for 1 hour
# def fetch_aqi_forecast():
#     """Fetch AQI forecast data from OpenWeather API."""
#     url = f"http://api.openweathermap.org/data/2.5/air_pollution/forecast?lat={LAT}&lon={LON}&appid={OPENWEATHER_API_KEY}"
#     response = requests.get(url)
    
#     if response.status_code == 200:
#         return response.json()
#     else:
#         st.error(f"⚠️ Failed to fetch AQI data. Status code: {response.status_code}")
#         return None

# # Function to categorize AQI values
# def categorize_aqi(aqi_value):
#     """Return the category for a given AQI value."""
#     for category, info in AQI_CATEGORIES.items():
#         if info["range"][0] <= aqi_value <= info["range"][1]:
#             return category, info["color"], info["description"]
#     return "Unknown", "#808080", "No data available"

# # Function to Handle Missing Values & Outliers
# def clean_data(df):
#     """Fill NaN values with mean and remove outliers using IQR."""
    
#     # Fill NaN values with column means
#     df.fillna(df.mean(), inplace=True)

#     # Remove outliers using IQR
#     Q1 = df.quantile(0.25)
#     Q3 = df.quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR

#     df = df.clip(lower=lower_bound, upper=upper_bound, axis=1)
#     return df

# # Function to Process Forecast Data
# def process_forecast_data(data):
#     """Extract relevant features and clean data."""
#     if not data:
#         return None

#     # Make sure we have enough entries in the list
#     forecast_list = data.get("list", [])
#     if len(forecast_list) < 3:
#         st.error(f"⚠️ Not enough forecast data points. Only {len(forecast_list)} available.")
#         return None

#     forecast_list = forecast_list[:3]  # Get next 3 days' data
#     processed_data = []
    
#     previous_aqi = None
#     rolling_pm10 = []
#     rolling_pm25 = []

#     for i, entry in enumerate(forecast_list):
#         components = entry.get("components", {})
#         date = datetime.utcnow() + timedelta(days=i+1)

#         aqi = entry.get("main", {}).get("aqi", np.nan)  # Handle missing AQI

#         # Compute AQI Change Rate
#         aqi_change_rate = 0
#         if previous_aqi is not None and previous_aqi > 0:
#             aqi_change_rate = (aqi - previous_aqi) / previous_aqi
#         previous_aqi = aqi

#         # Get pollutant values
#         pm10 = components.get("pm10", np.nan)
#         pm25 = components.get("pm2_5", np.nan)
#         no2 = components.get("no2", np.nan)
#         o3 = components.get("o3", np.nan)

#         # Rolling statistics
#         rolling_pm10.append(pm10)
#         rolling_pm25.append(pm25)
        
#         rolling_pm10_mean = pd.Series(rolling_pm10).rolling(window=min(3, len(rolling_pm10)), min_periods=1).mean().iloc[-1]
#         rolling_pm10_std = pd.Series(rolling_pm10).rolling(window=min(3, len(rolling_pm10)), min_periods=1).std().iloc[-1]
#         rolling_pm25_mean = pd.Series(rolling_pm25).rolling(window=min(3, len(rolling_pm25)), min_periods=1).mean().iloc[-1]
#         rolling_pm25_std = pd.Series(rolling_pm25).rolling(window=min(3, len(rolling_pm25)), min_periods=1).std().iloc[-1]

#         features = [
#             pm25, pm10, no2, o3, date.hour, date.day, date.month,
#             aqi_change_rate, date.weekday(), 1 if date.weekday() in [5, 6] else 0,
#             rolling_pm25_mean, rolling_pm10_mean, rolling_pm25_std, rolling_pm10_std
#         ]
        
#         processed_data.append(features)

#     # Create DataFrame with expected column names
#     df = pd.DataFrame(processed_data, columns=[
#         "pm2_5", "pm10", "no2", "o3", "hour", "day", "month",
#         "aqi_change_rate", "day_of_week", "is_weekend",
#         "rolling_pm25_mean", "rolling_pm10_mean", "rolling_pm25_std", "rolling_pm10_std"
#     ])

#     return clean_data(df)  # Clean data & return DataFrame

# # Function to Predict AQI Change Rate
# def predict_aqi(model, input_data):
#     """Predict AQI Change Rate using the trained model."""
#     try:
#         # Make sure we have the correct feature names
#         required_features = model.feature_names_in_
        
#         # Check if all required features are present
#         missing_features = set(required_features) - set(input_data.columns)
#         if missing_features:
#             st.error(f"⚠️ Missing features: {missing_features}")
#             # Add missing features with default values
#             for feature in missing_features:
#                 input_data[feature] = 0
                
#         # Make sure columns are in the correct order
#         input_data = input_data[required_features]
        
#         # Make predictions
#         predictions = model.predict(input_data)
#         return predictions
#     except Exception as e:
#         st.error(f"⚠️ Prediction error: {str(e)}")
#         return None

# # Function to Interpret AQI Change
# def interpret_aqi_change(change_rate):
#     """Interpret the AQI change rate."""
#     if change_rate >= 5:
#         return {"status": "🔴 Significant Deterioration", "description": "Air quality is expected to worsen significantly.", "class": "status-hazardous"}
#     elif change_rate >= 4:
#         return {"status": "🟠 Slight Increase", "description": "Expect a small increase in pollution levels.", "class": "status-unhealthy"}
#     elif change_rate >= 3:
#         return {"status": "🟡 Stable", "description": "Air quality is expected to remain relatively stable.", "class": "status-moderate"}
#     elif change_rate >= 2:
#         return {"status": "🟢 Slight Improvement", "description": "A small improvement in air quality is expected.", "class": "status-good"}
#     else:
#         return {"status": "💚 Major Improvement", "description": "Air quality should improve significantly.", "class": "status-good"}

# # Create AQI visualization
# def create_aqi_chart(predictions, dates):
#     """Create an AQI prediction visualization."""
#     # Create a dataframe for the chart
#     df = pd.DataFrame({
#         'Date': dates,
#         'AQI Change Rate': predictions
#     })
    
#     # Create color scale
#     colors = []
#     for pred in predictions:
#         if pred >= 5:
#             colors.append("#7e0023")  # Hazardous
#         elif pred >= 4:
#             colors.append("#ff0000")  # Unhealthy
#         elif pred >= 3:
#             colors.append("#ff7e00")  # Moderate
#         elif pred >= 2:
#             colors.append("#ffff00")  # Good
#         else:
#             colors.append("#00e400")  # Very Good
    
#     # Create bar chart
#     fig = go.Figure()
#     fig.add_trace(go.Bar(
#         x=df['Date'],
#         y=df['AQI Change Rate'],
#         marker_color=colors,
#         text=df['AQI Change Rate'].round(1),
#         textposition='auto',
#     ))
    
#     # Customize layout
#     fig.update_layout(
#         title="AQI Change Rate Forecast",
#         xaxis_title="Date",
#         yaxis_title="Change Rate",
#         plot_bgcolor='rgba(0,0,0,0)',
#         paper_bgcolor='rgba(0,0,0,0)',
#         font=dict(color="#e5e7eb"),
#         height=400,
#         margin=dict(l=40, r=40, t=50, b=40),
#     )
    
#     # Add threshold lines
#     fig.add_shape(
#         type="line",
#         x0=df['Date'].min(),
#         y0=5,
#         x1=df['Date'].max(),
#         y1=5,
#         line=dict(color="#7e0023", width=2, dash="dash"),
#     )
    
#     fig.add_shape(
#         type="line",
#         x0=df['Date'].min(),
#         y0=4,
#         x1=df['Date'].max(),
#         y1=4,
#         line=dict(color="#ff0000", width=2, dash="dash"),
#     )
    
#     fig.add_shape(
#         type="line",
#         x0=df['Date'].min(),
#         y0=3,
#         x1=df['Date'].max(),
#         y1=3,
#         line=dict(color="#ff7e00", width=2, dash="dash"),
#     )
    
#     fig.add_shape(
#         type="line",
#         x0=df['Date'].min(),
#         y0=2,
#         x1=df['Date'].max(),
#         y1=2,
#         line=dict(color="#ffff00", width=2, dash="dash"),
#     )
    
#     return fig

# # Create pollutant comparison chart
# def create_pollutant_chart(input_data):
#     """Create a comparison chart of pollutants."""
#     if input_data is None:
#         return None
        
#     # Extract pollutant data
#     pollutants = input_data[["pm2_5", "pm10", "no2", "o3"]].iloc[0].to_dict()
    
#     # Create a bar chart
#     fig = go.Figure()
    
#     # Add bars for each pollutant
#     fig.add_trace(go.Bar(
#         x=list(pollutants.keys()),
#         y=list(pollutants.values()),
#         marker_color=['#38BDF8', '#818CF8', '#A78BFA', '#F472B6'],
#         text=[f"{val:.2f}" for val in pollutants.values()],
#         textposition='auto',
#     ))
    
#     # Customize layout
#     fig.update_layout(
#         title="Current Pollutant Levels",
#         xaxis_title="Pollutant",
#         yaxis_title="Value (μg/m³)",
#         plot_bgcolor='rgba(0,0,0,0)',
#         paper_bgcolor='rgba(0,0,0,0)',
#         font=dict(color="#e5e7eb"),
#         height=350,
#         margin=dict(l=40, r=40, t=50, b=40),
#     )
    
#     return fig

# # Display health recommendations based on AQI
# def display_health_recommendations(predictions):
#     """Display health recommendations based on predicted AQI."""
#     worst_aqi = max(predictions)
    
#     st.markdown("<div class='card'>", unsafe_allow_html=True)
#     st.markdown("### 🏥 Health Recommendations")
    
#     if worst_aqi >= 5:
#         st.markdown("""
#         <div class="alert-box alert-danger">
#             <strong>⚠️ HAZARDOUS AIR QUALITY EXPECTED</strong>
#             <ul>
#                 <li>Stay indoors with windows and doors closed</li>
#                 <li>Use air purifiers if available</li>
#                 <li>Avoid all outdoor physical activities</li>
#                 <li>Wear N95/KN95 masks if you must go outside</li>
#                 <li>Keep hydrated and monitor symptoms</li>
#                 <li>Seek medical attention if experiencing difficulty breathing</li>
#             </ul>
#         </div>
#         """, unsafe_allow_html=True)
#     elif worst_aqi >= 4:
#         st.markdown("""
#         <div class="alert-box alert-warning">
#             <strong>⚠️ UNHEALTHY AIR QUALITY EXPECTED</strong>
#             <ul>
#                 <li>Limit outdoor activities, especially for sensitive groups</li>
#                 <li>Keep windows closed during peak pollution hours</li>
#                 <li>Consider using air purifiers indoors</li>
#                 <li>Wear masks when outdoors for extended periods</li>
#                 <li>Stay hydrated</li>
#             </ul>
#         </div>
#         """, unsafe_allow_html=True)
#     elif worst_aqi >= 3:
#         st.markdown("""
#         <div class="alert-box alert-warning">
#             <strong>⚠️ MODERATE AIR QUALITY EXPECTED</strong>
#             <ul>
#                 <li>Unusually sensitive people should consider reducing prolonged outdoor activities</li>
#                 <li>Keep windows closed during peak traffic hours</li>
#                 <li>Consider air purifiers for vulnerable individuals</li>
#                 <li>Stay hydrated</li>
#             </ul>
#         </div>
#         """, unsafe_allow_html=True)
#     else:
#         st.markdown("""
#         <div class="alert-box alert-success">
#             <strong>✅ GOOD AIR QUALITY EXPECTED</strong>
#             <ul>
#                 <li>Air quality is satisfactory and poses little or no risk</li>
#                 <li>Enjoy outdoor activities</li>
#                 <li>Keep monitoring air quality if you have respiratory issues</li>
#             </ul>
#         </div>
#         """, unsafe_allow_html=True)
    
#     st.markdown("</div>", unsafe_allow_html=True)

# # Main application
# def main():
#     # App header
#     st.markdown("<h1>🌎 Karachi Air Quality Forecast</h1>", unsafe_allow_html=True)
    
#     # Display loading spinner while fetching data
#     with st.spinner("Loading data and model..."):
#         # Load model
#         model = load_model_from_hopsworks()

#         if not model:
#             st.error("⚠️ Failed to load the prediction model. Please check your API keys and connections.")
#             st.stop()
            
#         # Fetch current weather
#         current_weather = fetch_current_weather()
        
#         # Fetch AQI forecast
#         forecast_data = fetch_aqi_forecast()
        
#         if not forecast_data:
#             st.error("⚠️ Failed to fetch AQI forecast data. Please check your API keys and connections.")
#             st.stop()
    
#     # Create tabs for different sections
#     tab1, tab2, tab3 = st.tabs(["📊 Forecast", "🔍 Analysis", "ℹ️ About"])
    
#     with tab1:
#         # Current weather and AQI overview
#         col1, col2 = st.columns([1, 2])
        
#         with col1:
#             st.markdown("<div class='card'>", unsafe_allow_html=True)
#             st.markdown("### 📍 Karachi, Pakistan")
            
#             if current_weather:
#                 # Extract weather data
#                 temp = current_weather.get("main", {}).get("temp", "N/A")
#                 feels_like = current_weather.get("main", {}).get("feels_like", "N/A")
#                 humidity = current_weather.get("main", {}).get("humidity", "N/A")
#                 weather_desc = current_weather.get("weather", [{}])[0].get("description", "N/A").capitalize()
#                 wind_speed = current_weather.get("wind", {}).get("speed", "N/A")
#                 icon_code = current_weather.get("weather", [{}])[0].get("icon", "01d")
                
#                 # Weather icon mapping to emojis
#                 weather_icons = {
#                     "01": "☀️", "02": "⛅", "03": "☁️", "04": "☁️",
#                     "09": "🌧️", "10": "🌦️", "11": "⛈️", "13": "❄️", "50": "🌫️"
#                 }
#                 icon_prefix = icon_code[:2]
#                 weather_emoji = weather_icons.get(icon_prefix, "🌡️")
                
#                 # Display current weather
#                 st.markdown(f"""
#                 <div class="metric-container">
#                     <div class="metric-box">
#                         <div class="metric-title">Temperature</div>
#                         <div class="metric-value">{temp}°C</div>
#                         <div>{weather_emoji} {weather_desc}</div>
#                     </div>
#                     <div class="metric-box">
#                         <div class="metric-title">Humidity</div>
#                         <div class="metric-value">{humidity}%</div>
#                         <div>💧 Feels like {feels_like}°C</div>
#                     </div>
#                 </div>
#                 <div class="metric-container">
#                     <div class="metric-box">
#                         <div class="metric-title">Wind Speed</div>
#                         <div class="metric-value">{wind_speed} m/s</div>
#                         <div>💨 {datetime.now().strftime('%H:%M')}</div>
#                     </div>
#                     <div class="metric-box">
#                         <div class="metric-title">Last Updated</div>
#                         <div class="metric-value">{datetime.now().strftime('%d %b')}</div>
#                         <div>🕒 {datetime.now().strftime('%H:%M')}</div>
#                     </div>
#                 </div>
#                 """, unsafe_allow_html=True)
#             else:
#                 st.warning("⚠️ Weather data unavailable")
                
#             st.markdown("</div>", unsafe_allow_html=True)
        
#         with col2:
#             # Process forecast data
#             processed_data = process_forecast_data(forecast_data)
            
#             if processed_data is not None:
#                 # Make predictions
#                 predictions = predict_aqi(model, processed_data)
                
#                 if predictions is not None:
#                     # Create forecast dates
#                     forecast_dates = [(datetime.utcnow() + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(len(predictions))]
                    
#                     # Display chart
#                     st.markdown("<div class='card'>", unsafe_allow_html=True)
#                     st.markdown("### 📈 AQI Forecast")
                    
#                     # Display AQI chart
#                     aqi_chart = create_aqi_chart(predictions, forecast_dates)
#                     st.plotly_chart(aqi_chart, use_container_width=True)
                    
#                     # Display forecast cards
#                     st.markdown("<div style='display: flex; justify-content: space-between; gap: 10px;'>", unsafe_allow_html=True)
                    
#                     for i, (pred, date) in enumerate(zip(predictions, forecast_dates)):
#                         interpretation = interpret_aqi_change(pred)
                        
#                         # Set background color based on status
#                         if pred >= 5:
#                             bg_color = "rgba(126, 0, 35, 0.3)"
#                         elif pred >= 4:
#                             bg_color = "rgba(255, 0, 0, 0.3)"
#                         elif pred >= 3:
#                             bg_color = "rgba(255, 126, 0, 0.3)"
#                         elif pred >= 2:
#                             bg_color = "rgba(255, 255, 0, 0.3)"
#                         else:
#                             bg_color = "rgba(0, 228, 0, 0.3)"
                        
#                         st.markdown(f"""
#                         <div class="forecast-card" style="background-color: {bg_color}; flex: 1;">
#                             <h4 style="margin: 0;">{date}</h4>
#                             <div style="font-size: 1.8rem; font-weight: 600; margin: 10px 0;">{pred:.1f}</div>
#                             <div class="{interpretation['class']}" style="font-weight: 500;">{interpretation['status']}</div>
#                             <div style="font-size: 0.8rem; margin-top: 5px;">{interpretation['description']}</div>
#                         </div>
#                         """, unsafe_allow_html=True)
                    
#                     st.markdown("</div>", unsafe_allow_html=True)
#                     st.markdown("</div>", unsafe_allow_html=True)
                    
#                     # Display health recommendations
#                     display_health_recommendations(predictions)
                    
#                     # Display pollutant comparison
#                     st.markdown("<div class='card'>", unsafe_allow_html=True)
#                     st.markdown("### 🔬 Pollutant Analysis")
                    
#                     # Show pollutant chart
#                     pollutant_chart = create_pollutant_chart(processed_data)
#                     if pollutant_chart:
#                         st.plotly_chart(pollutant_chart, use_container_width=True)
                    
#                     # Pollutant descriptions
#                     st.markdown("""
#                     <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
#                         <div class="metric-box">
#                             <div class="metric-title">PM2.5</div>
#                             <div style="font-size: 0.9rem;">Fine particulate matter that can penetrate deep into the lungs. Sources include vehicle emissions, industrial activities, and combustion.</div>
#                         </div>
#                         <div class="metric-box">
#                             <div class="metric-title">PM10</div>
#                             <div style="font-size: 0.9rem;">Larger particulate matter that can cause respiratory issues. Common sources include dust, pollen, and construction activities.</div>
#                         </div>
#                         <div class="metric-box">
#                             <div class="metric-title">NO2</div>
#                             <div style="font-size: 0.9rem;">Nitrogen dioxide from vehicle emissions and power plants. Can cause respiratory inflammation and reduced lung function.</div>
#                         </div>
#                         <div class="metric-box">
#                             <div class="metric-title">O3</div>
#                             <div style="font-size: 0.9rem;">Ground-level ozone formed by chemical reactions between pollutants. Can trigger asthma attacks and cause lung damage.</div>
#                         </div>
#                     </div>
#                     """, unsafe_allow_html=True)
#                 else:
#                     st.error("⚠️ Error making predictions. Please check the model and input data.")
#             else:
#                 st.error("⚠️ Error processing forecast data.")
    
#     with tab2:
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.markdown("### 🔍 AQI Analysis and Trends")
        
#         # Add some analysis content
#         st.markdown("""
#         This section provides deeper insights into Karachi's air quality patterns and trends.
#         """)
        
#         # Historical Data Visualization (placeholder)
#         if processed_data is not None:
#             # Create a sample trend chart (this would ideally use actual historical data)
#             dates = pd.date_range(end=datetime.now(), periods=14).tolist()
#             hist_values = np.random.normal(3.5, 1.0, 14).clip(1, 6).tolist()
            
#             # Create historical trend chart
#             hist_df = pd.DataFrame({
#                 'Date': dates,
#                 'AQI Change Rate': hist_values
#             })
            
#             fig = px.line(hist_df, x='Date', y='AQI Change Rate', 
#                         title='AQI Change Rate Trend (Past 14 Days)',
#                         markers=True)
            
#             fig.update_layout(
#                 plot_bgcolor='rgba(0,0,0,0)',
#                 paper_bgcolor='rgba(0,0,0,0)',
#                 font=dict(color="#e5e7eb"),
#                 height=350,
#                 margin=dict(l=40, r=40, t=50, b=40),
#             )
            
#             # Add reference lines
#             for level, color in [(5, "#7e0023"), (4, "#ff0000"), (3, "#ff7e00"), (2, "#ffff00")]:
#                 fig.add_shape(
#                     type="line",
#                     x0=hist_df['Date'].min(),
#                     y0=level,
#                     x1=hist_df['Date'].max(),
#                     y1=level,
#                     line=dict(color=color, width=1, dash="dash"),
#                 )
            
#             st.plotly_chart(fig, use_container_width=True)
            
#             # Add insights
#             st.markdown("""
#             #### Key Insights:
#             - Karachi's air quality fluctuates significantly with seasonal changes
#             - Industrial emissions and traffic are major contributors to air pollution
#             - Winter months typically see worse air quality due to temperature inversions
#             - Weekends show marginal improvement in air quality compared to weekdays
#             """)
            
#             # Contributing factors
#             st.markdown("""
#             #### Major Contributing Factors:
#             1. **Industrial Emissions**: Karachi hosts numerous industries that release various pollutants
#             2. **Vehicle Emissions**: The large number of vehicles and traffic congestion
#             3. **Construction Activities**: Ongoing construction projects generate particulate matter
#             4. **Weather Conditions**: Temperature inversions can trap pollutants near the ground
#             5. **Geographic Factors**: Coastal location affects pollutant dispersion
#             """)
#         else:
#             st.warning("⚠️ Analysis data unavailable")
            
#         st.markdown("</div>", unsafe_allow_html=True)
        
#         # Add comparison with other cities
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.markdown("### 🌐 Karachi vs Other Cities")
        
#         # Create sample comparison data
#         cities = ["Karachi", "Lahore", "Delhi", "Mumbai", "Beijing"]
#         aqi_values = [3.8, 4.9, 5.2, 3.2, 3.6]  # Example values
        
#         # Create comparison chart
#         city_df = pd.DataFrame({
#             'City': cities,
#             'AQI Change Rate': aqi_values
#         })
        
#         city_colors = []
#         for val in aqi_values:
#             if val >= 5:
#                 city_colors.append("#7e0023")
#             elif val >= 4:
#                 city_colors.append("#ff0000")
#             elif val >= 3:
#                 city_colors.append("#ff7e00")
#             elif val >= 2:
#                 city_colors.append("#ffff00")
#             else:
#                 city_colors.append("#00e400")
        
#         city_fig = go.Figure()
#         city_fig.add_trace(go.Bar(
#             x=city_df['City'],
#             y=city_df['AQI Change Rate'],
#             marker_color=city_colors,
#             text=city_df['AQI Change Rate'].round(1),
#             textposition='auto',
#         ))
        
#         city_fig.update_layout(
#             title="AQI Comparison with Major Cities",
#             xaxis_title="City",
#             yaxis_title="AQI Change Rate",
#             plot_bgcolor='rgba(0,0,0,0)',
#             paper_bgcolor='rgba(0,0,0,0)',
#             font=dict(color="#e5e7eb"),
#             height=350,
#             margin=dict(l=40, r=40, t=50, b=40),
#         )
        
#         st.plotly_chart(city_fig, use_container_width=True)
        
#         st.markdown("""
#         #### Comparative Analysis:
#         - Karachi's air quality is generally better than Delhi and Lahore
#         - However, it lags behind many European and North American cities
#         - Seasonal patterns affect all South Asian cities similarly
#         - Karachi's coastal location provides some natural ventilation advantages
#         """)
        
#         st.markdown("</div>", unsafe_allow_html=True)
        
#     with tab3:
#         st.markdown("<div class='card'>", unsafe_allow_html=True)
#         st.markdown("### ℹ️ About This Application")
        
#         st.markdown("""
#         This application provides air quality forecasts for Karachi, Pakistan using machine learning. 
#         The model was trained on historical air quality data and weather patterns to predict AQI change rates.
        
#         #### Data Sources:
#         - Weather and AQI data from OpenWeather API
#         - Model hosted on Hopsworks Feature Store
        
#         #### AQI Change Rate Interpretation:
#         - **5+**: Significant deterioration in air quality
#         - **4-5**: Slight increase in pollution levels
#         - **3-4**: Stable air quality conditions
#         - **2-3**: Slight improvement in air quality
#         - **<2**: Major improvement in air quality
        
#         #### Features Used for Prediction:
#         - PM2.5 and PM10 concentrations
#         - NO2 and O3 levels
#         - Temporal features (hour, day, month, weekday)
#         - Rolling statistics of pollutants
        
#         #### How to Use This App:
#         1. The main dashboard shows current conditions and forecasts
#         2. The Analysis tab provides historical trends and comparisons
#         3. Health recommendations are based on predicted AQI levels
        
#         #### Limitations:
#         - Predictions are estimates and may vary due to unforeseen events
#         - The model is updated periodically with new training data
#         - Local variations in air quality may occur within the city
#         """)
        
#         st.markdown("</div>", unsafe_allow_html=True)
    
#     # Sidebar with health information and settings
#     with st.sidebar:
#         st.markdown("### ⚕️ Health Impact")
        
#         # Add health impact information
#         st.markdown("""
#         <div style="background-color: rgba(17, 24, 39, 0.7); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
#             <h4 style="margin-top: 0; color: #f0f9ff;">Who's Most at Risk?</h4>
#             <ul style="padding-left: 20px;">
#                 <li>Children and elderly</li>
#                 <li>People with asthma or COPD</li>
#                 <li>Pregnant women</li>
#                 <li>People with heart conditions</li>
#                 <li>Outdoor workers</li>
#             </ul>
#         </div>
#         """, unsafe_allow_html=True)
        
#         # Add protection measures
#         st.markdown("""
#         <div style="background-color: rgba(17, 24, 39, 0.7); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
#             <h4 style="margin-top: 0; color: #f0f9ff;">Protection Measures</h4>
#             <ul style="padding-left: 20px;">
#                 <li>Use N95/KN95 masks outdoors</li>
#                 <li>Install air purifiers at home</li>
#                 <li>Keep windows closed during peak hours</li>
#                 <li>Stay hydrated</li>
#                 <li>Monitor symptoms</li>
#                 <li>Limit outdoor exercise when AQI is high</li>
#             </ul>
#         </div>
#         """, unsafe_allow_html=True)
        
#         # Add settings section
#         st.markdown("### ⚙️ Settings")
        
#         # Add a refresh button
#         if st.button("🔄 Refresh Data"):
#             st.cache_data.clear()
#             st.experimental_rerun()
        
#         # Add a theme selector (placeholder)
#         theme = st.selectbox("🎨 Theme", ["Dark Blue (Default)", "Dark Green", "Dark Purple"])
        
#         # Add notification settings (placeholder)
#         notify = st.checkbox("🔔 Enable Notifications", value=False)
        
#         if notify:
#             st.text_input("📧 Email for Alerts")
#             alert_threshold = st.slider("Alert Threshold", 1, 6, 4)
#             st.markdown(f"You'll be notified when AQI exceeds {alert_threshold}")
    
#     # Footer
#     st.markdown("""
#     <div class="footer">
#         <p>© 2025 Karachi AQI Forecast | Data from OpenWeather API | Model hosted on Hopsworks</p>
#         <p>For questions or feedback, please contact: support@karachiaqi.example.com</p>
#     </div>
#     """, unsafe_allow_html=True)

# # Run the application
# if __name__ == "__main__":
#     main()
