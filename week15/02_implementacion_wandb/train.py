"""Entrenamiento del clasificador MNIST con integración Weights & Biases.

Ejecución:
    python train.py                 # corre con hiperparámetros por defecto
    python train.py --epochs 5      # corre con argumentos
    wandb agent <SWEEP_ID>          # corre como parte de un sweep
"""
import argparse
import os

import numpy as np
import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger

from model import build_cnn


PROJECT_NAME = "unicundi-deeplearning-w15"
ENTITY = None  # Cambiar a "nombre-del-equipo" si crearon un equipo en W&B


def load_data(batch_size: int):
    """Carga MNIST, normaliza a [0, 1] y devuelve tres tf.data.Dataset."""
    (x_full, y_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_full = x_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_full = np.expand_dims(x_full, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    y_full = tf.keras.utils.to_categorical(y_full, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Partición 80/20 sobre el set de 60K para train/val
    n_train = int(0.8 * len(x_full))
    x_train, x_val = x_full[:n_train], x_full[n_train:]
    y_train, y_val = y_full[:n_train], y_full[n_train:]

    train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .shuffle(10_000, seed=42)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices((x_val, y_val))
              .batch(batch_size)
              .prefetch(tf.data.AUTOTUNE))
    test_ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
               .batch(batch_size)
               .prefetch(tf.data.AUTOTUNE))

    return train_ds, val_ds, test_ds, (x_test, y_test)


def get_optimizer(name: str, learning_rate: float):
    """Devuelve el optimizador según el nombre."""
    name = name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    raise ValueError(f"Optimizer desconocido: {name}")


def train(config=None):
    """Función principal de entrenamiento. Compatible con sweeps de W&B."""
    # Inicializa el run; cuando se usa con `wandb agent`, config viene del sweep
    run = wandb.init(project=PROJECT_NAME, entity=ENTITY, config=config)
    config = wandb.config

    # 1. Datos
    train_ds, val_ds, test_ds, (x_test_arr, y_test_arr) = load_data(config.batch_size)

    # 2. Modelo
    model = build_cnn(
        num_filters=config.num_filters,
        dense_units=config.dense_units,
        dropout=config.dropout,
    )
    optimizer = get_optimizer(config.optimizer, config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # 3. Entrenamiento con callback de W&B (loggea métricas de cada época automáticamente)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=[WandbMetricsLogger(log_freq="epoch")],
        verbose=1,
    )

    # 4. Evaluación final en el test set (datos nunca vistos durante entrenamiento)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})
    print(f"\nTest loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

    # 5. Loggeo de predicciones de ejemplo como tabla visual
    sample_preds = model.predict(x_test_arr[:20], verbose=0)
    sample_labels = np.argmax(sample_preds, axis=1)
    sample_true = np.argmax(y_test_arr[:20], axis=1)
    table = wandb.Table(columns=["imagen", "real", "predicción", "correcto"])
    for i in range(20):
        table.add_data(
            wandb.Image(x_test_arr[i]),
            int(sample_true[i]),
            int(sample_labels[i]),
            bool(sample_true[i] == sample_labels[i]),
        )
    wandb.log({"predicciones_de_muestra": table})

    # 6. Guardado del modelo como artifact (versionado en W&B)
    os.makedirs("models", exist_ok=True)
    model_path = "models/mnist_classifier.keras"
    model.save(model_path)
    artifact = wandb.Artifact("mnist_classifier", type="model",
                              description=f"CNN entrenada con {config.optimizer} lr={config.learning_rate}")
    artifact.add_file(model_path)
    run.log_artifact(artifact)

    wandb.finish()
    return history, test_acc


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena un CNN para MNIST con tracking en W&B")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "sgd", "rmsprop"])
    parser.add_argument("--num_filters", type=int, default=32)
    parser.add_argument("--dense_units", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    return vars(parser.parse_args())


if __name__ == "__main__":
    config = parse_args()
    train(config=config)
