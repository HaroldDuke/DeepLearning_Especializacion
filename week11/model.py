import tensorflow as tf
from tensorflow.keras import layers


def build_simple_rnn(window_size=30):
    model = tf.keras.Sequential([
        layers.SimpleRNN(64, activation='tanh', return_sequences=True,
                         input_shape=(window_size, 1)),
        layers.SimpleRNN(32, activation='tanh'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ], name='SimpleRNN')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_lstm(window_size=30):
    model = tf.keras.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=(window_size, 1)),
        layers.LSTM(32),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ], name='LSTM')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_gru(window_size=30):
    model = tf.keras.Sequential([
        layers.GRU(64, return_sequences=True, input_shape=(window_size, 1)),
        layers.GRU(32),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ], name='GRU')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
