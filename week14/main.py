import numpy as np
import tensorflow as tf

from data import load_mnist
from model import build_generator, build_discriminator
from train import train
from evaluate import plot_loss_curves, plot_real_vs_fake


# Hiperparámetros globales del experimento
NOISE_DIM  = 100   # dimensión del vector de ruido de entrada al generador
BATCH_SIZE = 128   # muestras por paso de entrenamiento
EPOCHS     = 30    # 30 épocas son suficientes para ver dígitos reconocibles en MNIST

# Semillas para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)


def main():
    print("=== Semana 14: GAN para generación de dígitos MNIST ===\n")

    # 1. Cargamos MNIST escalado a [-1, 1] (rango de la salida tanh del generador)
    dataset, shape = load_mnist(batch_size=BATCH_SIZE)
    print(f"Dataset: {shape[0]} imágenes de {shape[1]}x{shape[2]}\n")

    # 2. Construimos las dos redes que competirán durante el entrenamiento
    generator = build_generator(noise_dim=NOISE_DIM)
    discriminator = build_discriminator()

    print("Arquitectura del GENERADOR:")
    generator.summary()
    print("\nArquitectura del DISCRIMINADOR:")
    discriminator.summary()

    # 3. Entrenamiento adversarial: alterna G y D, guarda imágenes por época
    history = train(
        dataset=dataset,
        generator=generator,
        discriminator=discriminator,
        epochs=EPOCHS,
        noise_dim=NOISE_DIM,
        batch_size=BATCH_SIZE,
    )

    # 4. Visualizaciones finales: curvas de pérdida + reales vs falsas
    plot_loss_curves(history, save_path="plots/curvas_perdida.png")

    # Tomamos un lote real cualquiera para la comparación visual
    real_batch = next(iter(dataset))
    plot_real_vs_fake(real_batch, generator, NOISE_DIM,
                      save_path="plots/reales_vs_falsas.png")

    print("\nEntrenamiento finalizado. Resultados guardados en plots/ y models/")


if __name__ == "__main__":
    main()
