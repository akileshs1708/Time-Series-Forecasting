def train_test_split_series(series, test_size=0.2):
    split_point = int(len(series) * (1 - test_size))
    train = series[:split_point]
    test = series[split_point:]
    return train, test
