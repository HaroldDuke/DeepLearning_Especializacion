# Link público del proyecto en Weights & Biases

## Proyecto

**Nombre del proyecto:** `unicundi-deeplearning-w15`

**Link público:**
```
https://wandb.ai/<USUARIO-O-EQUIPO>/unicundi-deeplearning-w15
```

> ⚠️ **PENDIENTE:** Reemplazar `<USUARIO-O-EQUIPO>` por el usuario real de W&B una vez creado el proyecto.

## Cómo hacer público el proyecto

1. Ir al proyecto en W&B.
2. Click en **Settings** (esquina superior derecha del proyecto).
3. En **Privacy** seleccionar **"Open"** o **"Public"**.
4. Confirmar.

## Runs ejecutados

Listar al menos 3 runs con sus hiperparámetros y métricas finales. **Ejemplo (rellenar con valores reales):**

| Run name | Optimizer | LR | Batch size | Val accuracy | Test accuracy | Ejecutado por |
|---|---|---|---|---|---|---|
| `cnn-adam-lr0.001-bs64` | adam | 0.001 | 64 | 0.992 | 0.991 | Harold |
| `cnn-adam-lr0.0001-bs64` | adam | 0.0001 | 64 | 0.985 | 0.983 | Harold |
| `cnn-sgd-lr0.001-bs128` | sgd | 0.001 | 128 | 0.973 | 0.971 | Miguel Ángel |
| `cnn-rmsprop-lr0.005-bs32` | rmsprop | 0.005 | 32 | 0.989 | 0.988 | Miguel Ángel |

## Artifacts generados

- **`mnist_classifier:v0`** — Modelo del run base
- **`mnist_classifier:v1`** — Modelo del run con lr más bajo
- **`mnist_classifier:v2`** — Modelo del run con SGD

> Cada versión incluye en su metadata los hiperparámetros con que fue entrenada, evidenciando el concepto de **versionado de modelos** discutido en `model_serving.md`.
