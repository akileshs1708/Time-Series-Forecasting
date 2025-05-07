import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from datetime import date
from src.api_handler import fetch_time_series_data
from src.preprocessing import preprocess_data
from src.model import train_arima_model
from src.forecast import forecast_future
from src.evaluate import evaluate_model
from src.utils import train_test_split_series

st.set_page_config(page_title="Time-Series Forecasting", layout="wide")
st.title("Time-Series Forecasting (Open-Meteo REST API)")

# Default API and UI inputs
api_url = "https://archive-api.open-meteo.com/v1/archive"
latitude = st.number_input("Latitude (default = INDIA)", value=20.5937)  # India's lat
longitude = st.number_input("Longitude (default = INDIA)", value=78.9629)  # India's lon
timezone = "Asia/Kolkata"

start_date = st.date_input("Start Date", value=date(2023, 1, 1))
end_date = st.date_input("End Date", value=date(2023, 4, 1))
forecast_days = st.slider("Forecast Days", min_value=1, max_value=30, value=7)

if st.button("Fetch and Forecast"):
    with st.spinner("Fetching data..."):
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max",
            "timezone": timezone
        }
        df = fetch_time_series_data(api_url, params=params)
        st.success("Data fetched successfully!")

        st.subheader("Raw Data")
        st.write(df.head())

        cleaned = preprocess_data(df)
        st.subheader("Cleaned Time Series")
        st.line_chart(cleaned)

        # Split
        train, test = train_test_split_series(cleaned, test_size=0.2)
        st.write("Training Data:", train.shape)
        st.write("Testing Data:", test.shape)

        model = train_arima_model(train)
        forecast, conf_int = forecast_future(model, train, len(test))

        # Plot actual vs forecast
        st.subheader("Forecast vs Actual")
        result = pd.DataFrame({
            "Actual": test,
            "Forecast": forecast
        })
        st.line_chart(result)

        mae, rmse, mape = evaluate_model(model, test, forecast)
        st.metric("MAE", f"{mae:.2f}")
        st.metric("RMSE", f"{rmse:.2f}")
        st.metric("MAPE", f"{mape:.2f}%")
