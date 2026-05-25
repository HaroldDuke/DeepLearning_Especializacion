# Semana 15 — MLOps básico con Weights & Biases

**Especialización en Deep Learning — Universidad de Cundinamarca**

Trabajo cooperativo orientado a comprender el ciclo de vida de los datos y del modelo dentro de un flujo básico de aprendizaje profundo, aplicando conceptos de **Data Journey**, **Acceso y Manipulación de Datos**, **Monitoreo y Logging del Entrenamiento** y una aproximación inicial al **Model Serving** mediante **Weights & Biases**.

## Equipo

| Nombre | Rol |
|---|---|
| Laura Amado | Investigadora teórica |
| Harold Duque | ML Engineer |
| Miguel Ángel Córdoba | Analista de experimentos |
| Jensul Villalba | Tech Writer |

> Detalle completo de roles, tareas y metodología en [`04_colaboracion/roles_y_tareas.md`](04_colaboracion/roles_y_tareas.md).

---

## Estructura del proyecto

```
week15/
│
├── README.md                          ← Este archivo (resumen ejecutivo)
├── requirements.txt                   ← Dependencias
│
├── 01_investigacion/                  ← Marco conceptual (Criterio 1)
│   ├── data_journey.md
│   ├── acceso_datos.md
│   ├── monitoreo_logging.md
│   └── model_serving.md
│
├── 02_implementacion_wandb/           ← Código fuente (Criterios 2 y 3)
│   ├── model.py                       ← Definición del CNN
│   ├── train.py                       ← Entrenamiento con W&B
│   ├── sweep.yaml                     ← Búsqueda de hiperparámetros
│   └── week15.ipynb                   ← Notebook integrador (Colab)
│
├── 03_capturas_wandb/                 ← Evidencia del dashboard (Criterio 3)
│   ├── INSTRUCCIONES.md
│   ├── 01_runs_comparison.png
│   ├── 02_loss_curves.png
│   ├── 03_hyperparameters_parallel.png
│   ├── 04_artifacts.png
│   └── 05_predictions_table.png
│
├── 04_colaboracion/                   ← Trabajo en equipo (Criterio 4)
│   └── roles_y_tareas.md
│
└── 05_presentacion/                   ← Documentación final (Criterio 5)
    ├── informe_final.md
    └── link_wandb_publico.md
```

## Mapeo con la rúbrica de calificación

| # | Criterio | Cubierto en | Puntos |
|---|---|---|---|
| 1 | Investigación y comprensión de conceptos clave | [`01_investigacion/`](01_investigacion/) | 1.0 |
| 2 | Implementación práctica en Weights & Biases | [`02_implementacion_wandb/`](02_implementacion_wandb/) | 1.0 |
| 3 | Monitoreo y logging del modelo | [`03_capturas_wandb/`](03_capturas_wandb/) + W&B dashboard | 1.0 |
| 4 | Colaboración en el trabajo en equipo | [`04_colaboracion/roles_y_tareas.md`](04_colaboracion/roles_y_tareas.md) + commits en GitHub | 1.0 |
| 5 | Documentación y presentación del proyecto | [`05_presentacion/informe_final.md`](05_presentacion/informe_final.md) | 1.0 |
| | **Total** | | **5.0** |

---

## Cómo ejecutar

### Opción A — Google Colab (recomendada)

1. Abrir [`02_implementacion_wandb/week15.ipynb`](02_implementacion_wandb/week15.ipynb) en Google Colab.
2. Activar el GPU: `Entorno de ejecución → Cambiar tipo de entorno → GPU`.
3. Crear cuenta en https://wandb.ai/site (gratis) si no la tienen.
4. Ejecutar todas las celdas (`Entorno de ejecución → Ejecutar todas`).
5. Cuando pregunte por la API key, péguenla desde https://wandb.ai/authorize.

El notebook entrena el modelo, loggea todo en W&B y guarda el modelo como artifact.

### Opción B — Local

```bash
# Desde la carpeta week15/
pip install -r requirements.txt
wandb login
python 02_implementacion_wandb/train.py
```

### Opción C — Sweep de hiperparámetros (avanzado)

```bash
cd 02_implementacion_wandb
wandb sweep sweep.yaml          # devuelve un SWEEP_ID
wandb agent <SWEEP_ID> --count 6
```

Esto ejecuta 6 runs con combinaciones de hiperparámetros distintas, todos comparables en el dashboard.

---

## Resumen del flujo implementado

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA JOURNEY                                │
│  MNIST ─> load_data ─> normalize ─> reshape ─> tf.data.Dataset      │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ENTRENAMIENTO                                 │
│   ┌────────┐    ┌──────────┐    ┌─────────────────────────────┐    │
│   │  CNN   │ ─> │ Adam/SGD │ ─> │ Logs por época en W&B       │    │
│   └────────┘    └──────────┘    │  - train_loss               │    │
│                                 │  - train_accuracy           │    │
│                                 │  - val_loss                 │    │
│                                 │  - val_accuracy             │    │
│                                 └─────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     EVALUACIÓN + ARTIFACTS                          │
│   Test set ─> métricas finales ─> tabla de predicciones ─>          │
│   Modelo guardado como `mnist_classifier:vN` artifact en W&B        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                MODEL SERVING (CONCEPTO)                             │
│   Modelo versionado listo para servir vía FastAPI / TF Serving /    │
│   SageMaker / etc. (no implementado en esta entrega, solo discutido)│
└─────────────────────────────────────────────────────────────────────┘
```

---

## Link público del proyecto en W&B

Ver [`05_presentacion/link_wandb_publico.md`](05_presentacion/link_wandb_publico.md) para el link compartible al dashboard.

---

## Tecnologías

- **TensorFlow / Keras** 2.10+ — framework de Deep Learning
- **Weights & Biases** — plataforma de experiment tracking
- **NumPy / Matplotlib** — manipulación numérica y visualización
- **Python** 3.10+
- **Google Colab** — entorno de ejecución

---

## Referencias clave

- Huyen, C. (2022). *Designing Machine Learning Systems*. O'Reilly.
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.).
- Sculley et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS.
- Documentación de W&B — https://docs.wandb.ai
---

## 📈 Resultados de Ejecución

Durante la ejecución se registraron dos runs principales en **Weights & Biases**:

### 🔹 Run `proud-yogurt-1`
- **Mejor epoch:** 10  
- **Best Val Accuracy:** 0.9941  
- **Test Accuracy:** 0.9958  
- **Parámetros entrenables:** 1,701,578  
- **Duración por epoch:** ~174.45 s  
- **Dataset:** Train 51,000 | Val 9,000 | Test 10,000  

📌 Dashboard: [Ver run en W&B](https://wandb.ai/mcordobafigueroa-universidad-de-cundinamarca/ciclo-vida-deep-learning/runs/lvmr8rez)

---

### 🔹 Run `happy-lion-2`
- **Mejor epoch:** 10  
- **Best Val Accuracy:** 0.9941  
- **Test Accuracy:** 0.9958  
- **Parámetros entrenables:** 1,701,578  
- **Duración por epoch:** ~178.80 s  
- **Latencia de inferencia (serving):** ~2.18 ms (p50)  
- **Dataset:** Train 51,000 | Val 9,000 | Test 10,000  

📌 Dashboard: [Ver run en W&B](https://wandb.ai/mcordobafigueroa-universidad-de-cundinamarca/ciclo-vida-deep-learning/runs/ompqswiq)

---

## 📝 Conclusión Final

El pipeline logró un desempeño sobresaliente en MNIST, alcanzando **más del 99.5% de accuracy en test** y exportando el modelo en formatos **TorchScript** y **ONNX**.  
Los resultados fueron trazados y versionados en W&B, incluyendo métricas, artefactos y simulaciones de latencia de inferencia, lo que demuestra un flujo completo de **ciclo de vida de datos y modelo en Deep Learning**.
<img width="1825" height="930" alt="huu" src="https://github.com/user-attachments/assets/8f9ca51f-6b77-407b-9bd9-7148568ad604" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/17604c56-6c0e-44aa-b159-bf751420f9e0" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/17d042e1-6808-4e0a-a6b3-838c87719ce2" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/7d3c613e-7767-4b27-aa4f-0fdbdfe498f1" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/5210ae6c-c52b-4d97-8b8e-4c3e205c147d" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/03022ff3-8b35-4edf-a1bc-f52e1ade5f1c" />





