import numpy as np
import tensorflow as tf


def load_mnist(batch_size: int = 128, buffer_size: int = 60_000):
    """Carga MNIST y lo prepara como tf.data.Dataset listo para entrenar la GAN.

    - Convierte imágenes a float32 con forma (28, 28, 1).
    - Escala los píxeles al rango [-1, 1] porque el generador termina en tanh
      y el discriminador debe ver el mismo rango en reales y falsos.
    - Mezcla y agrupa en lotes para entrenamiento por mini-batches.
    """
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32")
    x_train = (x_train - 127.5) / 127.5  # [0, 255] -> [-1, 1]
    x_train = np.expand_dims(x_train, axis=-1)  # (N, 28, 28) -> (N, 28, 28, 1)

    dataset = (
        tf.data.Dataset.from_tensor_slices(x_train)
        .shuffle(buffer_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset, x_train.shape
