"""Definición del modelo CNN para clasificación de dígitos MNIST.

Arquitectura simple pero efectiva:
- 2 bloques convolucionales (Conv2D + MaxPool)
- 1 capa densa intermedia con Dropout
- Capa de salida con 10 neuronas (softmax) para los 10 dígitos
"""
import tensorflow as tf
from tensorflow.keras import layers, Sequential


def build_cnn(num_filters: int = 32,
              dense_units: int = 128,
              dropout: float = 0.3) -> tf.keras.Model:
    """Construye una CNN simple para MNIST.

    Args:
        num_filters: filtros base de las capas convolucionales.
        dense_units: neuronas de la capa densa intermedia.
        dropout: tasa de dropout para regularización.

    Returns:
        Modelo Keras compilable.
    """
    model = Sequential([
        layers.Input(shape=(28, 28, 1)),

        # Bloque convolucional 1: extrae bordes y trazos básicos
        layers.Conv2D(num_filters, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=2),

        # Bloque convolucional 2: combina bordes en partes de dígitos
        layers.Conv2D(num_filters * 2, kernel_size=3, activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=2),

        # Cabezal de clasificación
        layers.Flatten(),
        layers.Dropout(dropout),
        layers.Dense(dense_units, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ], name="mnist_cnn")

    return model
