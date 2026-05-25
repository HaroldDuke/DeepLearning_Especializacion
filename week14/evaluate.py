import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def save_image_grid(images, epoch: int, save_path: str, n_cols: int = 4):
    """Guarda una grilla de imágenes generadas para una época dada.

    Las imágenes vienen en rango [-1, 1] (salida tanh), las llevamos a [0, 1]
    para visualizarlas en escala de grises.
    """
    images = (images + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    n = images.shape[0]
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.8, n_rows * 1.8))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols, i % n_cols]
        ax.axis("off")
        if i < n:
            ax.imshow(tf.squeeze(images[i]).numpy(), cmap="gray")

    fig.suptitle(f"Imágenes generadas — Época {epoch}", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(history: dict, save_path: str):
    """Curvas de pérdida del generador y el discriminador por época.

    Es la herramienta principal para diagnosticar el equilibrio adversarial:
    si la pérdida del discriminador colapsa a 0, está dominando al generador;
    si la del generador explota, el generador no está aprendiendo.
    """
    epochs = range(1, len(history["gen_loss"]) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["gen_loss"], label="Generador", linewidth=2)
    plt.plot(epochs, history["disc_loss"], label="Discriminador", linewidth=2)
    plt.xlabel("Época")
    plt.ylabel("Pérdida (BCE)")
    plt.title("Curvas de pérdida adversarial")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_real_vs_fake(real_images, generator, noise_dim: int, save_path: str,
                     n_samples: int = 8):
    """Comparación lado a lado de imágenes reales del dataset contra falsas
    generadas, útil para evaluar cualitativamente la calidad final."""
    noise = tf.random.normal([n_samples, noise_dim])
    fake = generator(noise, training=False)
    fake = (fake + 1.0) / 2.0
    real = (real_images[:n_samples] + 1.0) / 2.0

    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 1.4, 3.2))

    for i in range(n_samples):
        axes[0, i].imshow(tf.squeeze(real[i]).numpy(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(tf.squeeze(fake[i]).numpy(), cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_title("Reales", loc="left", fontsize=11)
    axes[1, 0].set_title("Generadas", loc="left", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
