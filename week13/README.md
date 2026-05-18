# 🧠 Autoencoder Denoising con MNIST

**Actividad práctica guiada — Aprendizaje No Supervisado y Redes Generativas**

---

## 📋 Descripción general

Esta actividad implementa un **Denoising Autoencoder Convolucional** utilizando el dataset MNIST en Google Colab. El objetivo central es comprender cómo una red neuronal puede aprender representaciones comprimidas de los datos y reconstruir imágenes afectadas por ruido gaussiano, fortaleciendo conceptos de aprendizaje no supervisado y arquitecturas encoder–decoder.

El modelo recibe imágenes de dígitos escritos a mano con ruido artificialmente añadido y aprende, sin etiquetas, a recuperar la imagen limpia original.

---

## 🎯 Objetivos de aprendizaje

- Comprender la arquitectura **Encoder–Decoder** y el concepto de espacio latente (cuello de botella).
- Implementar un flujo completo de preprocesamiento: normalización y adición de ruido gaussiano.
- Entrenar un modelo con buenas prácticas: callbacks, hiperparámetros justificados y monitoreo de convergencia.
- Evaluar la calidad de reconstrucción con métricas cuantitativas (**MSE**, **PSNR**, **SSIM**).
- Documentar resultados y presentar conclusiones técnicas a partir del comportamiento observado.

---

## 📁 Estructura del proyecto

```
📦 autoencoder-denoising-mnist/
├── README.md                              ← Este informe
└── autoencoder_denoising_mnist.ipynb      ← Notebook principal (Google Colab)
```

### Organización del notebook

| # | Celda | Tipo | Criterio |
|---|-------|------|----------|
| 0 | Instalación de dependencias | Código | — |
| 1 | Verificación del entorno | Código | CA-5 |
| 2 | Carga y normalización MNIST | Código | CA-1 |
| 3 | Adición de ruido gaussiano | Código | CA-1 |
| 4 | Visualización: originales vs ruidosas | Código | CA-1, CA-4 |
| 5 | Descripción de la arquitectura | Markdown | CA-2, CA-5 |
| 6 | Implementación del Autoencoder | Código | CA-2 |
| 7 | Entrenamiento del modelo | Código | CA-3 |
| 8 | Curvas de pérdida (convergencia) | Código | CA-3, CA-4 |
| 9 | Métricas cuantitativas: MSE, PSNR, SSIM | Código | CA-4 |
| 10 | Visualización comparativa final | Código | CA-4 |
| 11 | Análisis por clase de dígito | Código | CA-4 |
| 12 | Conclusiones técnicas | Markdown | CA-3, CA-4, CA-5 |
| 13 | Guardado del modelo | Código | CA-5 |

---

## 🏗️ Arquitectura del modelo

El modelo sigue un diseño simétrico **Encoder → Espacio Latente → Decoder**:

```
Entrada (28 × 28 × 1)
│
├─── ENCODER ──────────────────────────────────────────────
│    Conv2D(32, 3×3, relu)   →  (28 × 28 × 32)
│    MaxPooling2D(2×2)        →  (14 × 14 × 32)
│    Conv2D(16, 3×3, relu)   →  (14 × 14 × 16)
│    MaxPooling2D(2×2)        →  ( 7 ×  7 × 16)  ← ESPACIO LATENTE
│
├─── DECODER ──────────────────────────────────────────────
│    Conv2D(16, 3×3, relu)   →  ( 7 ×  7 × 16)
│    UpSampling2D(2×2)        →  (14 × 14 × 16)
│    Conv2D(32, 3×3, relu)   →  (14 × 14 × 32)
│    UpSampling2D(2×2)        →  (28 × 28 × 32)
│    Conv2D( 1, 3×3, sigmoid) →  (28 × 28 ×  1)  ← IMAGEN RECONSTRUIDA
│
└─── Parámetros totales: ~33 000
```

**Factor de compresión:** 28×28 = 784 → 7×7×16 = 784 dimensiones latentes (compresión 1:1 en cantidad pero con representación aprendida y robusta al ruido).

**Elecciones de diseño:**

- `padding="same"` en todas las capas convolucionales para conservar dimensiones espaciales.
- `relu` en capas internas para evitar el problema del gradiente desvaneciente.
- `sigmoid` en la capa de salida para mantener valores en [0, 1], consistente con la normalización de entrada.
- `binary_crossentropy` como función de pérdida, apropiada para reconstrucción de valores continuos en [0, 1].

---

## ⚙️ Hiperparámetros

| Hiperparámetro | Valor | Justificación |
|----------------|-------|---------------|
| `NOISE_FACTOR` | 0.4 | Ruido perceptiblemente alto pero con información recuperable |
| `EPOCHS` | 30 | Suficiente para convergencia; EarlyStopping detiene antes si aplica |
| `BATCH_SIZE` | 128 | Balance entre velocidad y estabilidad del gradiente |
| `OPTIMIZER` | Adam (lr=1e-3) | Adaptativo; buen rendimiento general en visión por computadora |
| `LOSS` | Binary Crossentropy | Apropiada para salidas en [0, 1] |
| EarlyStopping `patience` | 5 | Evita sobreentrenamiento; restaura mejores pesos |
| ReduceLROnPlateau `factor` | 0.5 | Reduce LR a la mitad ante estancamiento |

---

## 📊 Métricas de evaluación

Se utilizan tres métricas complementarias para evaluar la calidad de reconstrucción:

### MSE — Mean Squared Error
Error promedio por píxel entre la imagen original y la reconstruida. Valores menores indican mejor reconstrucción. Escala: [0, 1].

```
MSE = (1/N) × Σ (original_i − reconstruida_i)²
```

### PSNR — Peak Signal-to-Noise Ratio
Métrica estándar en compresión y restauración de imágenes. Expresada en decibeles (dB).

```
PSNR = 20 × log₁₀(MAX / √MSE)
```

| Rango PSNR | Interpretación |
|------------|----------------|
| < 20 dB | Calidad baja |
| 20 – 30 dB | Calidad aceptable |
| > 30 dB | Buena calidad |
| > 40 dB | Excelente calidad |

### SSIM — Structural Similarity Index
Evalúa la similitud estructural entre imágenes considerando luminancia, contraste y estructura. Rango [0, 1]; valores cercanos a 1 indican alta similitud.

---

## 🔄 Flujo de la actividad

```
1. Preprocesamiento
   MNIST (raw) → Normalización [0,1] → + Ruido gaussiano (σ=0.4)
          ↓
2. Entrenamiento
   Entrada: imagen ruidosa  →  [Encoder]  →  Espacio Latente
                                                     ↓
   Objetivo: imagen limpia  ←  [Decoder]  ←  Representación comprimida
          ↓
3. Evaluación
   Predicción sobre test set → Cálculo MSE / PSNR / SSIM
   Visualización: Original | Ruidosa | Reconstruida
          ↓
4. Conclusiones
   Análisis de convergencia + interpretación de métricas + propuestas de mejora
```

---

## 🛠️ Tecnologías utilizadas

| Tecnología | Versión | Uso |
|------------|---------|-----|
| Python | 3.10+ | Lenguaje principal |
| TensorFlow / Keras | 2.x | Construcción y entrenamiento del modelo |
| NumPy | 1.23+ | Operaciones matriciales y métricas |
| Matplotlib | 3.5+ | Visualizaciones |
| Google Colab | — | Entorno de ejecución con GPU |

---

## ▶️ Cómo ejecutar

### En Google Colab (recomendado)

1. Abrir [colab.research.google.com](https://colab.research.google.com)
2. `Archivo` → `Subir cuaderno` → seleccionar `autoencoder_denoising_mnist.ipynb`
3. Activar GPU: `Entorno de ejecución` → `Cambiar tipo de entorno de ejecución` → `GPU`
4. Ejecutar todas las celdas en orden: `Entorno de ejecución` → `Ejecutar todo`

### En VS Code / entorno local

1. Abrir el archivo `.ipynb` con la extensión Jupyter de VS Code
2. Ejecutar primero la **Celda 0** (instalación de dependencias)
3. Continuar con las celdas restantes en orden

> **Nota:** El tiempo de entrenamiento es de aproximadamente 2–3 minutos con GPU (Colab) y 10–15 minutos sin GPU (CPU local).

---

## ✅ Criterios de aceptación

| Criterio | Descripción | Celdas |
|----------|-------------|--------|
| **CA-1** | Dataset MNIST cargado correctamente; normalización y adición de ruido adecuadas; código eficiente y documentado | 2, 3, 4 |
| **CA-2** | Autoencoder implementado con fase de codificación y decodificación bien estructuradas; buenas prácticas de programación | 5, 6 |
| **CA-3** | Modelo entrenado con hiperparámetros adecuados y buena convergencia; resultados documentados | 7, 8 |
| **CA-4** | Evaluación con métricas claras (MSE, PSNR, SSIM); visualización comparativa efectiva entre imágenes ruidosas y restauradas | 9, 10, 11 |
| **CA-5** | Código documentado con comentarios claros en cada paso; organización lógica y fácil de seguir | Todos |

---

## 📚 Conceptos clave

**Autoencoder:** Red neuronal que aprende a comprimir datos en una representación de menor dimensión (encoder) y luego reconstruirlos (decoder). La restricción del cuello de botella obliga a la red a capturar solo la información más relevante.

**Denoising Autoencoder:** Variante que recibe datos corrompidos como entrada y aprende a producir la versión limpia. Esta tarea más difícil produce representaciones latentes más robustas y generalizables.

**Espacio latente:** Representación interna comprimida aprendida por el encoder. Captura las características estadísticas más importantes de los datos sin necesidad de etiquetas.

**Aprendizaje no supervisado:** El modelo aprende la estructura de los datos a partir de la señal de reconstrucción únicamente, sin usar etiquetas de clase (los dígitos 0–9 nunca son provistos al modelo durante el entrenamiento).

---

## 💡 Extensiones sugeridas

- **Variational Autoencoder (VAE):** Agrega una distribución probabilística al espacio latente, permitiendo generar nuevas imágenes interpolando entre puntos del espacio latente.
- **Convolutional Autoencoder más profundo:** Añadir un tercer bloque para capturar características de mayor nivel de abstracción.
- **Batch Normalization:** Insertar capas `BatchNormalization` después de cada `Conv2D` para acelerar la convergencia y mejorar la estabilidad.
- **Diferentes tipos de ruido:** Probar ruido salt-and-pepper, ruido de Poisson o máscaras aleatorias (inpainting) para evaluar la robustez del modelo.
- **Visualización del espacio latente:** Extraer el encoder como submodelo y proyectar las representaciones latentes en 2D con t-SNE para observar la organización por dígito.

---
