import pandas as pd

def forecast_future(model, original_series, days):
    forecast = model.get_forecast(steps=days)
    forecast_df = pd.DataFrame({
        'Forecast': forecast.predicted_mean,
        'Lower': forecast.conf_int().iloc[:, 0],
        'Upper': forecast.conf_int().iloc[:, 1],
    })
    forecast_df.index = pd.date_range(start=original_series.index[-1] + pd.Timedelta(days=1), periods=days)
    return forecast_df['Forecast'], forecast.conf_int()
