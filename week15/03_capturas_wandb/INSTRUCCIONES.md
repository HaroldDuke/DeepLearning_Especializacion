# Instrucciones para tomar capturas del dashboard de W&B

> **Responsable:** Miguel Ángel Córdoba
> **Tiempo estimado:** 15-20 minutos
> **Pre-requisito:** Harold debe haber corrido el notebook al menos 3 veces con hiperparámetros distintos.

## Capturas necesarias (en orden)

Guarden cada captura en esta carpeta con el nombre exacto indicado, en formato PNG.

### 1. `01_runs_comparison.png` — Tabla comparativa de runs

**Cómo obtenerla:**
1. Vayan al proyecto en W&B: `https://wandb.ai/<usuario>/unicundi-deeplearning-w15`
2. En la vista principal verán la **tabla de runs**.
3. Asegúrense de que se vean al menos las columnas: `Name`, `val_accuracy`, `val_loss`, `test_accuracy`, `learning_rate`, `batch_size`, `optimizer`.
4. **Captura completa** de la tabla.

### 2. `02_loss_curves.png` — Curvas de loss y accuracy

**Cómo obtenerla:**
1. En la vista de runs, **seleccionen las 3-4 corridas** (checkbox a la izquierda).
2. Clicken **"Add panel"** o vayan a la pestaña **"Workspace"** que ya muestra los paneles por defecto.
3. Verán paneles con `train_loss`, `val_loss`, `train_accuracy`, `val_accuracy`.
4. **Captura de los 4 paneles** mostrando las múltiples runs superpuestas.

### 3. `03_hyperparameters_parallel.png` — Coordenadas paralelas

**Cómo obtenerla:**
1. En el workspace del proyecto, busquen el panel **"Parallel coordinates"** (W&B lo crea automáticamente si hay varios runs).
2. Si no existe: **Add panel → Parallel coordinates**.
3. Configuren los ejes con los hiperparámetros (`learning_rate`, `batch_size`, `optimizer`) y la métrica final (`val_accuracy`).
4. **Captura** del panel mostrando cómo cada run "trazó" una línea distinta.

### 4. `04_artifacts.png` — Modelos versionados

**Cómo obtenerla:**
1. En el proyecto, vayan a la pestaña **"Artifacts"** (menú lateral izquierdo).
2. Verán el artifact `mnist_classifier` con sus versiones (v0, v1, v2...).
3. **Captura** mostrando las versiones del modelo.

### 5. `05_predictions_table.png` — Tabla interactiva de predicciones

**Cómo obtenerla:**
1. Entren al detalle de uno de los runs (cliquen en su nombre).
2. Vayan a la pestaña **"Logs"** o **"Workspace"**.
3. Encontrarán la tabla `predicciones_de_muestra` con miniaturas de los dígitos, etiqueta real y predicción.
4. **Captura** de la tabla mostrando algunas filas.

### 6. (Bonus) `06_sweep_results.png` — Resultados del sweep

Si lograron correr el `wandb sweep` + `wandb agent`:
1. Vayan a la pestaña **"Sweeps"** del proyecto.
2. Verán el sweep con todas las runs ejecutadas.
3. **Captura** del panel de resultados (típicamente muestra una grilla de paneles y la importancia de cada hiperparámetro).

---

## Tips para mejores capturas

- Usen **modo claro** o **modo oscuro** consistente en todas (queda más prolijo).
- Capturen la **ventana completa** del panel, no solo una porción.
- Si la pantalla es muy pequeña, hagan zoom out en el navegador para que entre todo.
- En macOS: `Cmd + Shift + 4` para selección rectangular.
- En Windows: `Win + Shift + S` para herramienta de recorte.

---

## Plantilla del análisis (a incluir en el informe final)

Después de tomar las capturas, escriban un párrafo de análisis para cada una. Plantilla:

```markdown
### Análisis: Comparación de runs

A partir de la tabla comparativa observamos que la configuración con
[OPTIMIZADOR] y learning_rate [VALOR] obtuvo la mejor val_accuracy
([VALOR]). El run con SGD requirió un learning rate más alto para
converger a niveles comparables a Adam, lo cual es consistente con
la teoría: SGD necesita pasos más grandes porque no adapta el lr.

### Análisis: Curvas de loss

Las curvas muestran [convergencia / sobreajuste / inestabilidad].
Notamos que [observación]. La diferencia entre train_accuracy y
val_accuracy es de [X puntos], lo que sugiere [generalización
adecuada / sobreajuste leve].
```

Una vez tomadas las capturas y escrito el análisis, avisarle a Jensul para que lo integre al informe final.
