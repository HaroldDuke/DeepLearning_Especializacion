import numpy as np
import tensorflow as tf

from data import prepare_data
from model import build_simple_rnn, build_lstm, build_gru
from train import train_model
from evaluate import (
    compute_metrics,
    plot_full_series,
    plot_predictions,
    plot_training_history,
    print_comparison,
)

WINDOW_SIZE = 30
N_POINTS = 1200
EPOCHS = 50
BATCH_SIZE = 32
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

np.random.seed(42)
tf.random.set_seed(42)


def main():
    print("=== Semana 11: RNN para Predicción de Series de Tiempo ===\n")

    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, series, n_train, n_val = (
        prepare_data(WINDOW_SIZE, N_POINTS, TRAIN_RATIO, VAL_RATIO)
    )

    print(f"Entrenamiento : {X_train.shape}")
    print(f"Validación    : {X_val.shape}")
    print(f"Prueba        : {X_test.shape}\n")

    plot_full_series(series, n_train, n_val, save_path='plots/serie_completa.png')

    architectures = {
        'SimpleRNN': build_simple_rnn(WINDOW_SIZE),
        'LSTM':      build_lstm(WINDOW_SIZE),
        'GRU':       build_gru(WINDOW_SIZE),
    }

    histories = {}
    results = {}

    for name, model in architectures.items():
        print(f"\n{'='*50}")
        print(f"  Entrenando {name}")
        print(f"{'='*50}")
        model.summary()

        history = train_model(
            model, X_train, y_train, X_val, y_val,
            epochs=EPOCHS, batch_size=BATCH_SIZE
        )
        histories[name] = history

        y_pred = model.predict(X_test, verbose=0)
        metrics = compute_metrics(y_test, y_pred)
        results[name] = metrics

        print(f"\nMétricas de prueba — {name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")

        plot_predictions(
            y_test, y_pred.flatten(),
            title=f'{name} — Predicción vs Real (conjunto de prueba)',
            save_path=f'plots/prediccion_{name.lower()}.png'
        )

    plot_training_history(histories, save_path='plots/curvas_entrenamiento.png')
    print("\n=== Comparación Final de Modelos ===")
    print_comparison(results)


if __name__ == '__main__':
    main()
