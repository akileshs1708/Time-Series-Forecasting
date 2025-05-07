import requests
import pandas as pd


def fetch_time_series_data(url: str, params: dict, headers: dict = None) -> pd.DataFrame:
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()

    # Modify based on actual API structure
    df = pd.DataFrame({
        'timestamp': data['daily']['time'],
        'value': data['daily']['temperature_2m_max']
    })
    return df
