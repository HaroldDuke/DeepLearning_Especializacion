import tensorflow as tf
from tensorflow.keras import layers, Sequential


def build_generator(noise_dim: int = 100) -> tf.keras.Model:
    """Generador: ruido (noise_dim,) -> imagen (28, 28, 1) en rango [-1, 1].

    Arranca con una proyección densa que se reformatea a (7, 7, 256) y luego
    expande espacialmente con Conv2DTranspose hasta llegar a 28x28.
    BatchNorm estabiliza el entrenamiento y LeakyReLU evita gradientes muertos.
    La capa final usa tanh para producir píxeles en [-1, 1] (mismo rango que
    las imágenes reales preprocesadas en data.py).
    """
    model = Sequential(name="generator")

    # Proyección densa del vector de ruido a un volumen 7x7x256
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((7, 7, 256)))

    # 7x7 -> 7x7 (sin upsampling, solo refina features)
    model.add(layers.Conv2DTranspose(128, kernel_size=5, strides=1,
                                     padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # 7x7 -> 14x14
    model.add(layers.Conv2DTranspose(64, kernel_size=5, strides=2,
                                     padding="same", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    # 14x14 -> 28x28, salida con tanh
    model.add(layers.Conv2DTranspose(1, kernel_size=5, strides=2,
                                     padding="same", use_bias=False,
                                     activation="tanh"))

    return model


def build_discriminator() -> tf.keras.Model:
    """Discriminador: imagen (28, 28, 1) -> logit (real vs falso).

    CNN clásica que reduce la resolución con strides=2 y aumenta canales.
    Devuelve un logit (sin sigmoid) porque la pérdida se calcula con
    `from_logits=True`, que es numéricamente más estable.
    Dropout ayuda a que el discriminador no domine demasiado rápido al generador.
    """
    model = Sequential(name="discriminator")

    # 28x28 -> 14x14
    model.add(layers.Conv2D(64, kernel_size=5, strides=2, padding="same",
                            input_shape=(28, 28, 1)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    # 14x14 -> 7x7
    model.add(layers.Conv2D(128, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))  # logit, sin sigmoid

    return model
