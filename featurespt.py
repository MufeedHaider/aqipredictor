import requests
import datetime
import time
import os
import pandas as pd
import numpy as np

import hopsworks

# OpenWeather API Key
API_KEY = '05f1a56c472a46f3fb42f0a3775cc7e7'
# Hopsworks API Key
HOPS_API_KEY = 'EP5aAsjdPutNPjHf.qXzpVQ2wrS8dHURwxxJMggYsRgWHpy42SN2CqvSB5xdHGOdqZoezwioQU9tqj4Cc'

# OpenWeather API URL
URL = "http://api.openweathermap.org/data/2.5/air_pollution/history?lat=24.8607&lon=67.0011&start={start}&end={end}&appid={API}"

def fetch_data(api_key, start, end):
    """
    Fetch data from the OpenWeather API.
    """
    url = URL.format(start=start, end=end, API=api_key)
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('list', [])
    else:
        print(f"Error {response.status_code}: {response.text}")
        return []

def process_data(raw_data):
    """
    Process raw API data into a pandas DataFrame.
    """
    processed_data = []
    for entry in raw_data:
        timestamp = datetime.datetime.utcfromtimestamp(entry["dt"])
        processed_data.append({
            "timestamp": timestamp,
            "aqi": entry["main"]["aqi"],
            "pm25": entry["components"]["pm2_5"],
            "pm10": entry["components"]["pm10"],
            "no2": entry["components"]["no2"],
            "o3": entry["components"]["o3"]
        })
    return pd.DataFrame(processed_data)

def remove_outliers_iqr(data, numeric_cols):
    """
    Remove outliers using the IQR method.
    """
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    return data

def clean_data(data):
    """
    Clean and enhance the DataFrame by adding features and handling missing data.
    """
    data.fillna(data.mean(), inplace=True)

    # Remove outliers using IQR method
    numeric_cols = ['pm25', 'pm10', 'no2', 'o3', 'aqi']
    data = remove_outliers_iqr(data, numeric_cols)

    # Add time-based features
    data["hour"] = data["timestamp"].dt.hour
    data["day"] = data["timestamp"].dt.day
    data["month"] = data["timestamp"].dt.month
    data["day_week"] = data["timestamp"].dt.dayofweek
    data["is_weekend"] = data["day_week"].isin([5, 6]).astype(int)

    # Add rolling statistics
    rolling_window = 24  # 24-hour rolling window
    data["rolling_pm25_mean"] = data["pm25"].rolling(window=rolling_window).mean()
    data["rolling_pm10_mean"] = data["pm10"].rolling(window=rolling_window).mean()
    data["rolling_pm25_std"] = data["pm25"].rolling(window=rolling_window).std()
    data["rolling_pm10_std"] = data["pm10"].rolling(window=rolling_window).std()

    # Calculate AQI change rate
    data["aqi_change_rate"] = data["aqi"].diff()

    # Remove NaN rows after rolling and diff
    data.dropna(inplace=True)

    # Add a unique ID for Hopsworks
    data.reset_index(drop=True, inplace=True)
    data["id"] = data.index + 1

    return data



def store_data_to_hopsworks(data):
    """
    Store cleaned data into Hopsworks Feature Store.
    """
    try:
        # Log in to Hopsworks
        project = hopsworks.login(api_key_value=HOPS_API_KEY)
        fs = project.get_feature_store()

        # Feature group details
        feature_group_name = "air_quality_features"
        version = 3
        description = "Cleaned air quality data and derived features"
        primary_key = ["id"]

        # Delete existing feature group if it exists
        try:
            feature_group = fs.get_feature_group(name=feature_group_name, version=version)
            feature_group.delete()
            print(f"Existing feature group '{feature_group_name}' deleted.")
        except Exception as e:
            print(f"No existing feature group to delete: {e}")

        # Create new feature group and insert data
        feature_group = fs.create_feature_group(
            name=feature_group_name,
            version=version,
            description=description,
            primary_key=primary_key,
            online_enabled=False
        )
        feature_group.insert(data)
        print(f"Data successfully stored in Hopsworks Feature Store as '{feature_group_name}'.")
    except Exception as e:
        print(f"Error storing data to Hopsworks: {e}")

def fetch_and_process_data(api_key, days=365):
    """
    Fetch and process data for the last 'days' days.
    """
    end_date = datetime.datetime.utcnow()
    start_date = end_date - datetime.timedelta(days=days)

    all_data = []
    while start_date < end_date:
        start_unix = int(start_date.timestamp())
        end_unix = int((start_date + datetime.timedelta(days=1)).timestamp())
        print(f"Fetching data from {start_date} to {start_date + datetime.timedelta(days=1)}")
        raw_data = fetch_data(api_key, start_unix, end_unix)
        if raw_data:
            all_data.extend(raw_data)
        start_date += datetime.timedelta(days=1)
        time.sleep(1)  # Prevent API rate limits

    if not all_data:
        print("No data fetched from the API.")
        return

    # Process and clean data
    raw_df = process_data(all_data)
    clean_df = clean_data(raw_df)
    # Store cleaned data in Hopsworks
    store_data_to_hopsworks(clean_df)

if __name__ == "__main__":
    fetch_and_process_data(API_KEY, days=365)
