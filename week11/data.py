import numpy as np
from sklearn.preprocessing import MinMaxScaler


def generate_time_series(n_points=1200, seed=42):
    # Fijamos la semilla para que los resultados sean reproducibles en cada ejecución
    np.random.seed(seed)

    # Creamos un eje de tiempo con n_points valores entre 0 y 8π
    t = np.linspace(0, 8 * np.pi, n_points)

    # Componente de tendencia: la serie crece lentamente con el tiempo
    trend = 0.03 * t

    # Componente estacional: superposición de tres frecuencias (imita ciclos reales)
    seasonality = np.sin(t) + 0.5 * np.sin(2 * t) + 0.25 * np.sin(4 * t)

    # Ruido gaussiano: simula la variabilidad aleatoria de datos del mundo real
    noise = np.random.normal(0, 0.15, n_points)

    return (trend + seasonality + noise).astype(np.float32)


def create_sequences(series, window_size=30):
    """
    Convierte la serie en pares (entrada, salida) usando ventana deslizante.
    Cada entrada X es una ventana de `window_size` pasos consecutivos.
    La salida y es el valor inmediatamente siguiente a esa ventana.
    """
    X, y = [], []
    for i in range(len(series) - window_size):
        # Tomamos window_size valores como contexto histórico
        X.append(series[i:i + window_size])
        # El modelo debe predecir el valor siguiente
        y.append(series[i + window_size])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_data(window_size=30, n_points=1200, train_ratio=0.70, val_ratio=0.15):
    series = generate_time_series(n_points)

    # Calculamos los tamaños de cada partición
    n_train = int(n_points * train_ratio)
    n_val   = int(n_points * val_ratio)
    # El resto (15%) queda como conjunto de prueba

    train_series = series[:n_train]
    val_series   = series[n_train:n_train + n_val]
    test_series  = series[n_train + n_val:]

    # Escalamos entre 0 y 1. El scaler se ajusta SOLO con datos de entrenamiento
    # para evitar que información de validación o prueba "filtre" hacia el modelo
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_series.reshape(-1, 1)).flatten()
    val_scaled   = scaler.transform(val_series.reshape(-1, 1)).flatten()
    test_scaled  = scaler.transform(test_series.reshape(-1, 1)).flatten()

    # Construimos las secuencias de ventana deslizante para cada partición
    X_train, y_train = create_sequences(train_scaled, window_size)
    X_val,   y_val   = create_sequences(val_scaled,   window_size)
    X_test,  y_test  = create_sequences(test_scaled,  window_size)

    # Las RNN esperan entrada con forma (muestras, pasos_de_tiempo, características)
    # Añadimos la dimensión de características = 1 (una sola variable)
    X_train = X_train.reshape(-1, window_size, 1)
    X_val   = X_val.reshape(-1, window_size, 1)
    X_test  = X_test.reshape(-1, window_size, 1)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, series, n_train, n_val
