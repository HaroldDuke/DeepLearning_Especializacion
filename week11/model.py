import tensorflow as tf
from tensorflow.keras import layers


def build_simple_rnn(window_size=30):
    """
    Red recurrente básica (SimpleRNN).
    Aprende dependencias a corto plazo, pero puede perder información
    en secuencias largas por el problema del gradiente que se desvanece.
    """
    model = tf.keras.Sequential([
        # Primera capa recurrente: 64 unidades, devuelve secuencia completa
        # para que la segunda capa recurrente tenga contexto paso a paso
        layers.SimpleRNN(64, activation='tanh', return_sequences=True,
                         input_shape=(window_size, 1)),

        # Segunda capa recurrente: 32 unidades, devuelve solo el último estado
        layers.SimpleRNN(32, activation='tanh'),

        # Capa densa intermedia para aprender combinaciones no lineales del estado
        layers.Dense(16, activation='relu'),

        # Salida: un único valor numérico (predicción del siguiente paso)
        layers.Dense(1)
    ], name='SimpleRNN')

    # MSE como pérdida porque es una regresión; MAE como métrica legible
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_lstm(window_size=30):
    """
    Red LSTM (Long Short-Term Memory).
    Usa puertas de entrada, olvido y salida para decidir qué información
    conservar o descartar, lo que le permite capturar dependencias largas.
    """
    model = tf.keras.Sequential([
        # Primera capa LSTM: return_sequences=True pasa la secuencia completa
        # a la segunda capa, preservando el contexto temporal en cada paso
        layers.LSTM(64, return_sequences=True, input_shape=(window_size, 1)),

        # Segunda capa LSTM: resume toda la secuencia en un único vector de estado
        layers.LSTM(32),

        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ], name='LSTM')

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_gru(window_size=30):
    """
    Red GRU (Gated Recurrent Unit).
    Versión simplificada del LSTM con solo dos puertas (reset y update).
    Alcanza rendimiento similar al LSTM con menos parámetros y mayor velocidad.
    """
    model = tf.keras.Sequential([
        # Primera capa GRU: igual que LSTM, return_sequences alimenta la siguiente capa
        layers.GRU(64, return_sequences=True, input_shape=(window_size, 1)),

        # Segunda capa GRU: produce el vector de estado final de la secuencia
        layers.GRU(32),

        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ], name='GRU')

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
