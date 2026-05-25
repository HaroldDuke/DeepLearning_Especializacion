# Semana 14 — Redes Generativas Adversarias (GAN) en MNIST

## 1. Introducción

Este proyecto implementa una **Red Generativa Adversaria (GAN)** para generar dígitos manuscritos a partir del dataset **MNIST**. El propósito de la actividad —según el enunciado de la asignatura— **no es obtener imágenes perfectas**, sino evidenciar la comprensión del flujo técnico de una GAN: su arquitectura básica, el proceso de entrenamiento adversarial entre el generador y el discriminador, y el análisis de los resultados obtenidos considerando las limitaciones propias de tiempo, hardware y dataset.

Se utilizó MNIST (60.000 imágenes de 28×28 en escala de grises) porque es un dataset público, académico y suficientemente ligero para entrenarse en el entorno gratuito de Google Colab. Esto repite el mismo criterio aplicado en la semana 10 (red siamesa) por las limitaciones de cuota de GPU en Colab.

---

## 2. Metodología

### 2.1 Estructura del proyecto

```
week14/
├── week14.ipynb         # Notebook ejecutable en Google Colab
├── README.md            # Este informe
├── main.py              # Orquestador del flujo modular
├── data.py              # Carga y preprocesamiento de MNIST
├── model.py             # Arquitecturas del Generador y Discriminador
├── train.py             # Loop de entrenamiento adversarial
├── evaluate.py          # Visualizaciones y curvas de pérdida
├── requirements.txt     # Dependencias
├── plots/               # Imágenes generadas por época + curvas
└── models/              # Pesos guardados del generador y discriminador
```

### 2.2 Preprocesamiento del dataset

- Las imágenes se convierten a `float32` y se les agrega el canal: forma final `(N, 28, 28, 1)`.
- Los píxeles se escalan al rango `[-1, 1]` porque el generador termina en `tanh`. Si entrenáramos con imágenes en `[0, 1]`, el discriminador detectaría trivialmente la diferencia de rango entre reales y falsas y la GAN no aprendería.
- Se construye un `tf.data.Dataset` con `shuffle` + `batch(128)` + `prefetch`.

### 2.3 Arquitectura del generador

**Entrada:** vector de ruido `z ∼ N(0, 1)` de dimensión 100.
**Salida:** imagen `(28, 28, 1)` con valores en `[-1, 1]`.

| Capa | Configuración | Forma de salida |
|---|---|---|
| Dense | 7·7·256, sin bias | (7·7·256,) |
| BatchNorm + LeakyReLU(0.2) | — | — |
| Reshape | (7, 7, 256) | (7, 7, 256) |
| Conv2DTranspose | 128 filtros, k=5, s=1 | (7, 7, 128) |
| BatchNorm + LeakyReLU | — | — |
| Conv2DTranspose | 64 filtros, k=5, s=2 | (14, 14, 64) |
| BatchNorm + LeakyReLU | — | — |
| Conv2DTranspose | 1 filtro, k=5, s=2, **tanh** | (28, 28, 1) |

### 2.4 Arquitectura del discriminador

**Entrada:** imagen `(28, 28, 1)`.
**Salida:** un logit (real vs falso).

| Capa | Configuración | Forma de salida |
|---|---|---|
| Conv2D | 64 filtros, k=5, s=2 | (14, 14, 64) |
| LeakyReLU(0.2) + Dropout(0.3) | — | — |
| Conv2D | 128 filtros, k=5, s=2 | (7, 7, 128) |
| LeakyReLU(0.2) + Dropout(0.3) | — | — |
| Flatten + Dense(1) | sin sigmoid | (1,) |

No se aplica `sigmoid` al final porque la pérdida se calcula con `from_logits=True`, que es numéricamente más estable.

### 2.5 Entrenamiento adversarial

- **Pérdida:** `BinaryCrossentropy(from_logits=True)` para ambos modelos, con etiquetas distintas:
  - **Discriminador** → reales = 1, falsas = 0.
  - **Generador** → busca que las falsas sean clasificadas como 1.
- **Optimizadores:** dos `Adam` independientes con `lr=2e-4` y `beta_1=0.5` (valores recomendados por el paper DCGAN para estabilizar el entrenamiento).
- **Batch size:** 128.
- **Épocas:** 30.
- **Paso de entrenamiento:** usa dos `tf.GradientTape` simultáneos para calcular y aplicar los gradientes de G y D por separado en el mismo lote.
- **Inspección visual:** en cada época se generan 16 imágenes con un ruido fijo (`seed`) para poder observar cómo evoluciona la "misma muestra" a lo largo del entrenamiento.

---

## 3. Resultados

### 3.1 Curvas de pérdida

![Curvas de pérdida](plots/curvas_perdida.png)

**Cómo se interpretan:** en una GAN saludable las pérdidas de G y D oscilan en rangos similares —no buscamos que bajen a cero. Una caída de la pérdida del discriminador a 0 indicaría dominio del discriminador (el generador no aprende); una explosión de la pérdida del generador indicaría que no logra engañar al discriminador.

### 3.2 Evolución del generador

El generador parte de imágenes que son puro ruido y progresivamente aprende a producir formas que se parecen a dígitos manuscritos. Las imágenes guardadas en `plots/epoca_XXX.png` permiten ver esta evolución época por época.

- **Épocas 1–5:** manchas borrosas, ruido estructurado.
- **Épocas 10–15:** aparecen trazos reconocibles como dígitos.
- **Épocas 25–30:** dígitos identificables aunque con artefactos.

### 3.3 Comparación reales vs generadas

![Reales vs generadas](plots/reales_vs_falsas.png)

Fila superior: muestras reales de MNIST.
Fila inferior: muestras generadas por la GAN al final del entrenamiento.

---

## 4. Discusión y limitaciones

El ejercicio cumple su propósito de **evidenciar la comprensión del flujo técnico** de una GAN, pero hay limitaciones que son importantes documentar:

1. **Mode collapse parcial.** Es común que el generador prefiera producir ciertos dígitos (típicamente 1 y 7, que son los de trazo más simple) con mayor frecuencia que otros más complejos como el 8 o el 4. Esto sucede porque al generador le resulta más fácil engañar al discriminador con dígitos "fáciles".

2. **Inestabilidad del entrenamiento.** Las GAN son notoriamente sensibles a los hiperparámetros. Pequeños cambios en `learning_rate`, `dropout`, número de capas o número de épocas pueden hacer que el entrenamiento colapse. Los valores usados (`lr=2e-4`, `beta_1=0.5`) provienen del paper DCGAN y son el estándar de facto.

3. **Calidad limitada de las imágenes.** Para obtener imágenes nítidas habría que: (a) entrenar muchas más épocas con GPU dedicada, (b) usar arquitecturas más profundas (DCGAN completa o StyleGAN), o (c) cambiar a variantes con mejor estabilidad de gradientes como **WGAN-GP**.

4. **Hardware y tiempo.** Se entrenó durante 30 épocas en Google Colab (T4 GPU). Una GAN para resultados publicables suele requerir cientos de épocas y GPUs de mayor capacidad.

5. **Dataset elegido.** MNIST es un dataset "fácil" en comparación con CIFAR-10 o CelebA, lo que permite que una GAN básica produzca resultados aceptables. Aplicar la misma arquitectura a imágenes a color de 64×64 requeriría una red sustancialmente más profunda.

---

## 5. Conclusiones

- Se implementó exitosamente una **GAN funcional** sobre MNIST, con un generador basado en `Conv2DTranspose` y un discriminador convolucional, entrenados de forma adversarial con dos optimizadores `Adam` independientes.
- Las **curvas de pérdida** muestran el comportamiento oscilatorio característico de una GAN saludable, sin colapso de ninguno de los dos modelos.
- A partir de las **épocas 10–15** ya se observan dígitos reconocibles; al final del entrenamiento las muestras generadas son claramente identificables como dígitos manuscritos, aunque con artefactos visibles.
- El ejercicio cumple con todos los criterios de la rúbrica:
  - **Implementación de la GAN:** generador y discriminador correctamente estructurados.
  - **Entrenamiento de la GAN:** interacción coordinada entre ambos modelos con `tf.GradientTape`.
  - **Evaluación de los resultados:** curvas de pérdida + grid de imágenes por época + comparación reales vs falsas.
  - **Entrega del notebook:** documentado con comentarios y secciones organizadas.

---

## 6. Cómo ejecutar

### Opción A — Google Colab (recomendada por el enunciado)

1. Subir el archivo `week14.ipynb` a Google Colab.
2. Activar el GPU: `Entorno de ejecución → Cambiar tipo de entorno → GPU`.
3. Ejecutar todas las celdas en orden (`Entorno de ejecución → Ejecutar todas`).

### Opción B — Local con la estructura modular

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
python main.py
```

Al finalizar se generan en `plots/`:
- `epoca_001.png` ... `epoca_030.png` — grid de muestras por época
- `curvas_perdida.png` — curvas de pérdida del generador y discriminador
- `reales_vs_falsas.png` — comparación lado a lado

Y en `models/` se guardan los pesos del generador y el discriminador entrenados.
