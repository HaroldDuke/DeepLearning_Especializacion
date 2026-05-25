# Monitoreo y Logging del Entrenamiento

## Definición

El **monitoreo y logging** del entrenamiento es el proceso de **registrar, observar y analizar** todo lo que ocurre durante el entrenamiento de un modelo de Machine Learning: métricas, hiperparámetros, recursos del sistema, gradientes, predicciones intermedias, artefactos generados, etc.

Sin un sistema de monitoreo y logging adecuado, entrenar un modelo es como **conducir un carro con los ojos vendados**: no sabes si está mejorando, si está sobreajustando, si se está colapsando o por qué un cambio funcionó mejor que otro.

## ¿Por qué es crítico en ML?

### Problema 1: el experimento se vuelve invisible
En desarrollo tradicional ejecutamos código y vemos el resultado. En ML el "resultado" del entrenamiento son **decenas de métricas que evolucionan a lo largo del tiempo**. Sin loggearlas, se pierden.

### Problema 2: comparar experimentos a mano es imposible
Un proyecto típico genera decenas o cientos de runs con distintos hiperparámetros. Sin una herramienta de tracking es imposible recordar "qué configuración dio la mejor accuracy hace dos semanas".

### Problema 3: reproducibilidad
Si no registras qué hiperparámetros, qué datos y qué código produjeron un resultado, **no podrás reproducirlo**. Esto es inaceptable en proyectos serios.

### Problema 4: detección temprana de problemas
- ¿La loss explotó en la época 5? → ajustar `learning_rate`.
- ¿La accuracy de validación dejó de subir? → activar early stopping.
- ¿El uso de GPU es bajo? → optimizar el data pipeline.

## ¿Qué se loggea?

### A. Hiperparámetros (al inicio del run)
Configuración usada para entrenar:
- `learning_rate`, `batch_size`, `optimizer`, `num_epochs`
- Arquitectura: número de capas, neuronas, dropout
- Semilla aleatoria
- Versión del dataset

### B. Métricas (durante el entrenamiento)
Valores que evolucionan época a época o paso a paso:
- **Loss** de entrenamiento y validación
- **Métricas de calidad**: accuracy, precision, recall, F1, AUC
- **Métricas de regresión**: MSE, RMSE, MAE, R²
- **Métricas específicas**: BLEU, perplexity, FID, etc.

### C. Métricas del sistema
- Uso de GPU/CPU/RAM
- Throughput (samples/seg)
- Tiempo por época

### D. Artefactos
Archivos generados durante el entrenamiento:
- **Modelos** (pesos guardados)
- **Curvas y gráficas** (PNG)
- **Predicciones** sobre el set de validación
- **Matrices de confusión**, ROC, etc.

### E. Visualizaciones complejas
- Imágenes generadas (en GANs)
- Mapas de atención (en NLP/Vision)
- Embeddings (proyecciones 2D/3D)

## Herramientas de monitoreo y logging

| Herramienta | Tipo | Pros | Contras |
|---|---|---|---|
| **TensorBoard** | Local, open source | Integrado con TF/PyTorch, gratis | Compartir runs es difícil |
| **Weights & Biases (W&B)** | SaaS | Dashboard web, sweeps, artifacts, gratis para uso individual | Requiere cuenta y API key |
| **MLflow** | Open source (self-hosted) | Flexible, multi-framework, incluye registry | Setup más complejo |
| **Neptune.ai** | SaaS | UI muy buena, equipos | Plan gratuito limitado |
| **Comet ML** | SaaS | Similar a W&B, fuerte en experimentación | Menos popular |

## ¿Por qué elegimos Weights & Biases?

1. **Hosting en la nube gratuito** — no requiere infraestructura.
2. **Integración trivial** — `pip install wandb` + 3 líneas de código.
3. **Dashboard interactivo** — comparar runs, hacer queries.
4. **Artifacts** — versionado de datasets y modelos.
5. **Sweeps** — búsqueda automática de hiperparámetros.
6. **Link público compartible** — útil para entregas académicas y demos.
7. **Standard de la industria** — usado por OpenAI, NVIDIA, Toyota.

## Cómo se integra W&B en el código (ejemplo de nuestro proyecto)

```python
import wandb

# 1. Inicializar el run con hiperparámetros
wandb.init(
    project="unicundi-deeplearning-w15",
    config={
        "learning_rate": 0.001,
        "batch_size": 64,
        "optimizer": "adam",
        "num_epochs": 10,
    },
)
config = wandb.config

# 2. Durante el entrenamiento, loggear métricas
for epoch in range(config.num_epochs):
    train_loss, train_acc = train_one_epoch(...)
    val_loss, val_acc = evaluate(...)

    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    })

# 3. Al final, subir el modelo como artifact
artifact = wandb.Artifact("mnist_classifier", type="model")
artifact.add_file("model.keras")
wandb.log_artifact(artifact)

# 4. Cerrar el run
wandb.finish()
```

## Sweeps: la herramienta más potente de W&B

Un **sweep** es una búsqueda automática de hiperparámetros. En lugar de probar combinaciones a mano:

```yaml
# sweep.yaml
method: bayes  # 'grid', 'random' o 'bayes'
metric:
  name: val_accuracy
  goal: maximize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.01
  batch_size:
    values: [32, 64, 128]
  optimizer:
    values: ['adam', 'sgd']
```

```bash
wandb sweep sweep.yaml          # crea el sweep, devuelve un ID
wandb agent <SWEEP_ID>          # ejecuta runs en paralelo
```

W&B prueba combinaciones automáticamente y muestra cuál maximiza tu métrica objetivo.

## Buenas prácticas

1. **Loggear todo desde el principio**, no después de que algo falle.
2. **Nombres consistentes** de métricas (`train_loss` siempre, no `tl` o `loss_train`).
3. **Tags** para agrupar experimentos relacionados.
4. **Notas** en cada run explicando qué se probó.
5. **Guardar el modelo como artifact** al final de cada run exitoso.
6. **Comparar 2-3 runs** lado a lado antes de tomar decisiones.

## Conclusión

El monitoreo y logging convierten al entrenamiento de ML de un proceso opaco en un experimento **trazable, reproducible y comparable**. Es la práctica más rentable de adoptar: el costo es bajo (algunas líneas de código) y el beneficio es masivo (visibilidad total sobre lo que estás haciendo).

## Referencias

- Weights & Biases Docs — https://docs.wandb.ai
- Géron, A. — *Hands-On Machine Learning* (Cap. sobre entrenamiento y validación)
- Sculley et al. — *"Hidden Technical Debt in Machine Learning Systems"* (NeurIPS 2015)
