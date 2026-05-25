import os
import time
import numpy as np
import tensorflow as tf

from evaluate import save_image_grid


# Pérdida única para ambos modelos: BCE sobre logits (estable numéricamente)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_logits, fake_logits):
    """El discriminador quiere etiquetar las reales como 1 y las falsas como 0."""
    real_loss = cross_entropy(tf.ones_like(real_logits), real_logits)
    fake_loss = cross_entropy(tf.zeros_like(fake_logits), fake_logits)
    return real_loss + fake_loss


def generator_loss(fake_logits):
    """El generador quiere engañar al discriminador: que las falsas parezcan reales (=1)."""
    return cross_entropy(tf.ones_like(fake_logits), fake_logits)


@tf.function
def train_step(real_images, generator, discriminator,
               gen_optimizer, disc_optimizer, noise_dim, batch_size):
    """Un paso de entrenamiento adversarial:

    1. El generador produce un lote de imágenes falsas a partir de ruido.
    2. El discriminador evalúa reales y falsas.
    3. Se calculan pérdidas para ambos y se actualizan sus pesos por separado.
    """
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated = generator(noise, training=True)

        real_logits = discriminator(real_images, training=True)
        fake_logits = discriminator(generated, training=True)

        gen_loss = generator_loss(fake_logits)
        disc_loss = discriminator_loss(real_logits, fake_logits)

    # Gradientes y actualización independiente de cada red
    grads_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    grads_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(grads_gen, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(grads_disc, discriminator.trainable_variables))

    return gen_loss, disc_loss


def train(dataset, generator, discriminator,
          epochs: int, noise_dim: int, batch_size: int,
          plots_dir: str = "plots", models_dir: str = "models"):
    """Loop principal de entrenamiento de la GAN.

    Guarda en cada época una grilla de imágenes generadas con un ruido fijo
    (seed) para poder observar la evolución visual del generador.
    Retorna el historial de pérdidas para graficarlo después.
    """
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

    # Semilla fija de ruido para comparar la misma "muestra" en cada época
    seed = tf.random.normal([16, noise_dim])

    history = {"gen_loss": [], "disc_loss": []}

    for epoch in range(1, epochs + 1):
        start = time.time()
        gen_losses, disc_losses = [], []

        for real_batch in dataset:
            g_loss, d_loss = train_step(
                real_batch, generator, discriminator,
                gen_optimizer, disc_optimizer,
                noise_dim, batch_size,
            )
            gen_losses.append(float(g_loss))
            disc_losses.append(float(d_loss))

        # Promedio por época para el historial
        avg_gen = float(np.mean(gen_losses))
        avg_disc = float(np.mean(disc_losses))
        history["gen_loss"].append(avg_gen)
        history["disc_loss"].append(avg_disc)

        # Guardamos imágenes para inspección visual
        samples = generator(seed, training=False)
        save_image_grid(
            samples, epoch,
            save_path=os.path.join(plots_dir, f"epoca_{epoch:03d}.png"),
        )

        elapsed = time.time() - start
        print(f"Época {epoch:3d}/{epochs} | "
              f"gen_loss: {avg_gen:.4f} | disc_loss: {avg_disc:.4f} | "
              f"tiempo: {elapsed:.1f}s")

    # Guardamos los pesos del generador entrenado para reproducir resultados
    generator.save(os.path.join(models_dir, "generator.keras"))
    discriminator.save(os.path.join(models_dir, "discriminator.keras"))

    return history
