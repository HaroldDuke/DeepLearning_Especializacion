# Informe Final — Semana 15
## Trabajo Cooperativo sobre Data Journey, Acceso y Manipulación de Datos, Monitoreo y Logging, y Model Serving con Weights & Biases

**Especialización en Deep Learning — Universidad de Cundinamarca**
**Curso:** 601539 / DEEP LEARNING - CONCEPTOS / FU / CAD2202023205 / EIAIPA2026

### Equipo

- **Laura Amado** — Investigadora teórica
- **Harold Duque** — ML Engineer
- **Miguel Ángel Córdoba** — Analista de experimentos
- **Jensul Villalba** — Tech Writer

---

## 1. Resumen ejecutivo

Este proyecto evidencia la aplicación práctica de los conceptos fundamentales de **MLOps** sobre un caso de estudio concreto: el entrenamiento de un clasificador CNN sobre el dataset MNIST. La entrega cubre los cuatro pilares que enmarca el enunciado de la actividad:

1. **Data Journey** — recorrido completo del dato desde su fuente hasta el modelo.
2. **Acceso y manipulación de datos** — pipelines reproducibles con `tf.data`.
3. **Monitoreo y logging del entrenamiento** — registro completo de cada experimento en **Weights & Biases**.
4. **Model Serving** — versionado del modelo entrenado como artifact, sentando las bases para servirlo en producción.

El propósito no fue obtener el mejor clasificador posible, sino **evidenciar el ciclo de vida completo** alrededor del modelo y la **trazabilidad** que MLOps aporta al desarrollo de Deep Learning.

---

## 2. Marco conceptual

Los conceptos clave que sustentan este proyecto están desarrollados en detalle en los siguientes documentos:

| Documento | Concepto | Páginas |
|---|---|---|
| `01_investigacion/data_journey.md` | Data Journey: etapas y aplicación práctica | ~3 |
| `01_investigacion/acceso_datos.md` | Acceso y manipulación de datos | ~3 |
| `01_investigacion/monitoreo_logging.md` | Monitoreo y logging del entrenamiento | ~3 |
| `01_investigacion/model_serving.md` | Model Serving: alternativas y consideraciones | ~3 |

**Idea central:** un modelo no termina cuando se entrena. Existe todo un ciclo —antes (datos) y después (despliegue, monitoreo)— que determina si el modelo genera valor real.

---

## 3. Caso de estudio

**Dataset:** MNIST (60.000 imágenes de entrenamiento + 10.000 de prueba, dígitos manuscritos de 28×28 en escala de grises).

**Modelo:** CNN simple con la siguiente arquitectura:

```
Conv2D(num_filters, 3x3) → MaxPool(2x2)
Conv2D(num_filters*2, 3x3) → MaxPool(2x2)
Flatten → Dropout → Dense(dense_units, ReLU) → Dense(10, Softmax)
```

**Pérdida:** Categorical Crossentropy
**Métrica:** Accuracy
**Optimizador:** explorado vía sweep (`adam`, `sgd`, `rmsprop`)

**Razón de la elección:** MNIST con CNN es el "Hello World" del Deep Learning. Su simplicidad permite entrenar runs rápidamente (2-3 min por run), lo que habilita **múltiples experimentos comparables** — el verdadero foco de esta entrega.

---

## 4. Implementación del Data Journey

El recorrido del dato en este proyecto se materializa así:

```
┌──────────────┐     ┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│   Fuente     │ ──> │   Ingesta    │ ──> │ Manipulación   │ ──> │   Modelo     │
│  Keras API   │     │ load_data()  │     │ norm + reshape │     │  tf.data.DS  │
└──────────────┘     └──────────────┘     └────────────────┘     └──────────────┘
```

| Etapa | Implementación |
|---|---|
| **Fuente** | Dataset MNIST distribuido con Keras |
| **Ingesta** | `tf.keras.datasets.mnist.load_data()` |
| **Manipulación** | (1) Cast a `float32`, (2) Normalización a `[0, 1]`, (3) Agregar canal `(N, 28, 28, 1)`, (4) One-hot encoding de etiquetas |
| **Particionamiento** | 80% train / 20% val sobre el set de 60K + 10K test reservados |
| **Carga eficiente** | `tf.data.Dataset` con `shuffle(seed=42)`, `batch`, `prefetch(AUTOTUNE)` |
| **Reproducibilidad** | Semillas fijas en `numpy` y `tensorflow` |

---

## 5. Implementación en Weights & Biases

### 5.1 Configuración del proyecto

- **Proyecto:** `unicundi-deeplearning-w15`
- **Plataforma:** wandb.ai (plan gratuito)
- **Link público:** ver `link_wandb_publico.md`

### 5.2 Qué se loggea en cada run

| Categoría | Datos loggeados |
|---|---|
| **Hiperparámetros** | `learning_rate`, `batch_size`, `optimizer`, `num_filters`, `dense_units`, `dropout`, `epochs`, `architecture` |
| **Métricas por época** | `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy` (vía `WandbMetricsLogger`) |
| **Métricas finales** | `test_loss`, `test_accuracy` |
| **Visualizaciones** | Matriz de confusión (PNG) |
| **Tablas** | `predicciones_de_muestra`: 20 imágenes con label real y predicción |
| **Artifacts** | `mnist_classifier`: modelo Keras versionado con metadata |

### 5.3 Experimentos ejecutados

Se ejecutaron al menos 3 runs base + un sweep adicional para evidenciar la exploración sistemática de hiperparámetros. **Ver capturas en `03_capturas_wandb/`.**

---

## 6. Análisis de resultados

> 📝 **Esta sección la rellena Miguel Ángel** después de correr los runs y tomar las capturas.

### 6.1 Comparación de runs

[Pegar aquí el análisis basado en `01_runs_comparison.png`]

Hallazgos clave:
- El optimizador **[XYZ]** con `learning_rate = [VALOR]` obtuvo la mejor `val_accuracy` ([%]).
- **[Observación sobre SGD vs Adam]**.
- **[Observación sobre el efecto del batch_size]**.

### 6.2 Curvas de entrenamiento

[Pegar aquí el análisis basado en `02_loss_curves.png`]

- **Convergencia**: los modelos con Adam convergen en ~3-5 épocas; SGD requiere más épocas.
- **Sobreajuste**: la diferencia entre `train_accuracy` y `val_accuracy` es de **[X puntos]**, lo que sugiere **[buena generalización / sobreajuste leve]**.

### 6.3 Importancia de hiperparámetros

[Pegar aquí el análisis basado en `03_hyperparameters_parallel.png`]

W&B nos permite ver con coordenadas paralelas qué hiperparámetros impactan más la métrica objetivo. En nuestro caso, el factor más influyente fue **[learning_rate / optimizer]**.

### 6.4 Versionado del modelo

[Pegar aquí el análisis basado en `04_artifacts.png`]

Cada run produjo una versión del artifact `mnist_classifier`. Esto evidencia el concepto de **versionado** del Model Serving: cualquier versión puede recuperarse e implementarse en producción de forma reproducible.

---

## 7. Aproximación a Model Serving

Aunque la actividad no exige implementar serving funcional, exploramos el concepto:

### 7.1 Modelo listo para servir

El modelo se guarda en formato `.keras` (Keras nativo) y se sube como **artifact versionado** a W&B. Desde ahí puede descargarse en cualquier entorno con:

```python
import wandb
api = wandb.Api()
artifact = api.artifact('USUARIO/unicundi-deeplearning-w15/mnist_classifier:v0')
artifact.download()
```

### 7.2 Alternativas de despliegue evaluadas

| Opción | Caso de uso ideal | Esfuerzo |
|---|---|---|
| **FastAPI** local | Demo, MVP | Bajo |
| **TensorFlow Serving** | Producción escalable con TF | Medio |
| **TF Lite** | App móvil offline | Medio |
| **AWS SageMaker** | Producción gestionada en la nube | Medio-alto |
| **HuggingFace Spaces** | Demo público compartible | Bajo |

Para una hipotética puesta en producción de este clasificador, recomendaríamos **FastAPI + Docker** como punto de partida (simple, controlable, sin vendor lock-in), migrando a **TF Serving** si el volumen lo amerita.

---

## 8. Colaboración en el equipo

La organización del trabajo está descrita en detalle en `04_colaboracion/roles_y_tareas.md`. En resumen:

- **División clara** de roles según fortalezas: investigación, implementación, análisis, documentación.
- **Repositorio compartido** en GitHub con commits diferenciados por autor.
- **Dependencias secuenciales** respetadas: la investigación precedió a la implementación; el análisis dependió de los runs; la documentación final integró todo.
- **Revisión cruzada** entre miembros para asegurar calidad.

---

## 9. Conclusiones

1. **MLOps no es opcional.** Aplicar Data Journey, monitoreo y versionado convierte un experimento en un proceso reproducible y trazable.
2. **Weights & Biases es una herramienta poderosa y accesible.** Con pocas líneas de código se obtiene visibilidad total sobre los experimentos, comparación entre runs y versionado de modelos.
3. **El modelo no termina cuando se entrena.** El ciclo continúa con serving, monitoreo en producción y feedback al siguiente ciclo de entrenamiento.
4. **El trabajo en equipo se beneficia de buenas prácticas MLOps.** La trazabilidad de W&B permitió que cada miembro avanzara en paralelo sin pisarse el trabajo.
5. **El experimento más simple bien instrumentado** vale más que un modelo complejo sin trazabilidad.

---

## 10. Referencias

- Sambasivan et al. (2021). *"Everyone wants to do the model work, not the data work"*. Google Research.
- Sculley et al. (2015). *"Hidden Technical Debt in Machine Learning Systems"*. NeurIPS.
- Huyen, C. (2022). *Designing Machine Learning Systems*. O'Reilly.
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.).
- Weights & Biases Documentation — https://docs.wandb.ai
- TensorFlow Data Pipeline Guide — https://www.tensorflow.org/guide/data

---

## Anexos

- `01_investigacion/` — Documentos teóricos
- `02_implementacion_wandb/` — Código fuente
- `03_capturas_wandb/` — Capturas del dashboard
- `04_colaboracion/` — Documentación del trabajo en equipo
- `link_wandb_publico.md` — Link al dashboard público
