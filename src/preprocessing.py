import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.Series:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.sort_index()
    df = df.dropna()
    df = df[~df.index.duplicated(keep='first')]
    df = df.asfreq('D')
    return df['value']
