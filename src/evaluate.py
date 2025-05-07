from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_model(model, actual, forecast):
    mae = mean_absolute_error(actual, forecast)
    mse = mean_squared_error(actual, forecast)
    rmse = mse ** 0.5
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    return mae, rmse, mape
