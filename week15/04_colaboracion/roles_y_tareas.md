# Roles y División del Trabajo en Equipo

## Equipo

Este proyecto fue desarrollado de manera colaborativa por el siguiente equipo de creadores de oportunidades de la **Especialización en Deep Learning** de la **Universidad de Cundinamarca**:

| Nombre | Rol | Responsabilidad principal |
|---|---|---|
| **Laura Amado** | Investigadora teórica | Investigación y redacción de los 4 documentos conceptuales |
| **Harold Duque** | ML Engineer | Implementación del modelo y entrenamiento con W&B |
| **Miguel Ángel Córdoba** | Analista de experimentos | Sweep de hiperparámetros, capturas y análisis del dashboard |
| **Jensul Villalba** | Tech Writer | Documentación, informe final y presentación |

---

## Detalle de tareas por miembro

### 👩‍🎓 Laura Amado — Investigadora teórica

**Entregables:**
- `01_investigacion/data_journey.md` — Definición, etapas y aplicación práctica del Data Journey.
- `01_investigacion/acceso_datos.md` — Componentes, herramientas y buenas prácticas de acceso y manipulación de datos.
- `01_investigacion/monitoreo_logging.md` — Por qué loggear, qué loggear y herramientas (W&B, MLflow, TensorBoard).
- `01_investigacion/model_serving.md` — Tipos de serving, herramientas y aspectos críticos en producción.

**Aporte clave:** establecer el marco conceptual que sustenta toda la implementación práctica del equipo.

---

### 👨‍💻 Harold Duque — ML Engineer

**Entregables:**
- `02_implementacion_wandb/model.py` — Definición de la arquitectura CNN.
- `02_implementacion_wandb/train.py` — Script de entrenamiento con integración de W&B (logging de métricas, artifacts, predicciones).
- `02_implementacion_wandb/week15.ipynb` — Notebook integrador ejecutable en Google Colab.
- Ejecución de los 3+ runs iniciales en W&B.

**Aporte clave:** materializar los conceptos teóricos en código funcional, integrando todas las prácticas de MLOps en el flujo de entrenamiento.

---

### 🔬 Miguel Ángel Córdoba — Analista de experimentos

**Entregables:**
- `02_implementacion_wandb/sweep.yaml` — Configuración del sweep de hiperparámetros (revisión y ajuste).
- Ejecución del sweep con `wandb agent`.
- `03_capturas_wandb/` — Capturas del dashboard de W&B:
  - `01_runs_comparison.png` — Tabla comparativa de runs.
  - `02_loss_curves.png` — Curvas de loss y accuracy.
  - `03_hyperparameters_parallel.png` — Coordenadas paralelas de hiperparámetros.
  - `04_artifacts.png` — Modelos versionados como artifacts.
  - `05_predictions_table.png` — Tabla interactiva de predicciones.
- Análisis escrito de los resultados del sweep en `05_presentacion/informe_final.md`.

**Aporte clave:** convertir los runs en evidencia visual y conclusiones cuantitativas que muestran cuáles hiperparámetros funcionaron mejor.

---

### ✍️ Jensul Villalba — Tech Writer

**Entregables:**
- `README.md` — Documento principal con resumen ejecutivo, estructura y guía de ejecución.
- `05_presentacion/informe_final.md` — Informe completo integrando teoría, implementación y resultados.
- `05_presentacion/link_wandb_publico.md` — Link público del dashboard de W&B para el profe.
- `04_colaboracion/roles_y_tareas.md` — Este documento.
- Revisión de estilo y consistencia de todos los documentos.

**Aporte clave:** asegurar que la entrega sea **comprensible, profesional y completa**, dando coherencia narrativa a los aportes de todos.

---

## Metodología de trabajo

1. **Repositorio compartido en GitHub** — Cada miembro hace commits con su autoría para evidenciar contribución individual.
2. **Comunicación asíncrona** — Coordinación por mensajería del equipo.
3. **Revisión cruzada** — Cada documento es revisado por al menos otro miembro antes de considerarse final.
4. **Dependencias claras** — La investigación teórica (Laura) precede a la implementación (Harold); el análisis (Miguel Ángel) depende de los runs ejecutados; la documentación final (Jensul) integra los aportes anteriores.

## Cronograma

| Fecha | Hito | Responsable |
|---|---|---|
| Día 1 | Investigación teórica (4 documentos) | Laura |
| Día 2 | Implementación del modelo + primeros 3 runs | Harold |
| Día 3 | Sweep + capturas + análisis | Miguel Ángel |
| Día 4 | Informe final + README + revisión global | Jensul + todos |
| Día 5 | Entrega final | Todos |

## Evidencia de colaboración

- **Commits diferenciados** en el repositorio compartido (autoría por miembro).
- **Pull Requests** con revisión cruzada entre miembros.
- Este documento como **mapa explícito de contribuciones**.
- **Link público del proyecto en W&B** mostrando los runs con autoría diferenciada (cada run lleva el nombre del miembro que lo ejecutó).
