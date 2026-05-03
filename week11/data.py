import numpy as np
from sklearn.preprocessing import MinMaxScaler


def generate_time_series(n_points=1200, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, 8 * np.pi, n_points)
    trend = 0.03 * t
    seasonality = np.sin(t) + 0.5 * np.sin(2 * t) + 0.25 * np.sin(4 * t)
    noise = np.random.normal(0, 0.15, n_points)
    return (trend + seasonality + noise).astype(np.float32)


def create_sequences(series, window_size=30):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_data(window_size=30, n_points=1200, train_ratio=0.70, val_ratio=0.15):
    series = generate_time_series(n_points)

    n_train = int(n_points * train_ratio)
    n_val = int(n_points * val_ratio)

    train_series = series[:n_train]
    val_series = series[n_train:n_train + n_val]
    test_series = series[n_train + n_val:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_series.reshape(-1, 1)).flatten()
    val_scaled = scaler.transform(val_series.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test_series.reshape(-1, 1)).flatten()

    X_train, y_train = create_sequences(train_scaled, window_size)
    X_val, y_val = create_sequences(val_scaled, window_size)
    X_test, y_test = create_sequences(test_scaled, window_size)

    X_train = X_train.reshape(-1, window_size, 1)
    X_val = X_val.reshape(-1, window_size, 1)
    X_test = X_test.reshape(-1, window_size, 1)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, series, n_train, n_val
