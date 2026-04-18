# Semana 6 – Bias/Variance, Overfitting y Regularización con scikit-learn

## Objetivo

Profundizar en el trade-off bias/variance y el fenómeno de sobreajuste (overfitting), aplicando métodos de regularización para mejorar la capacidad de generalización de una red neuronal. La evidencia se construye comparando un modelo base (sin regularización) frente a un modelo regularizado, observando el comportamiento del entrenamiento y la evaluación para sustentar conclusiones técnicas.

---

## Herramientas y Dataset

| Elemento | Detalle |
|---|---|
| Lenguaje | Python 3 |
| Librería principal | scikit-learn (`MLPClassifier`) |
| Dataset | Sintético — `make_classification` |
| Muestras | 600 (450 entrenamiento / 150 prueba) |
| Features | 30 (10 informativas, 10 redundantes) |
| Ruido en etiquetas | 10 % (`flip_y=0.10`) — induce overfitting |

El dataset fue diseñado intencionalmente con ruido y pocas muestras para hacer visible el fenómeno de sobreajuste.

---

## Arquitectura de la Red Neuronal

Ambos modelos comparten la misma arquitectura para que la comparación sea justa:

```
Capa entrada (30) → Dense(512) → Dense(256) → Dense(128) → Dense(64) → Salida(2)
Activación: ReLU · Optimizador: Adam
```

---

## Modelos Comparados

### Modelo Base — Sin regularización (Alta Varianza)
- `alpha = 1e-9` (regularización L2 prácticamente nula)
- Sin Early Stopping
- Tiende a memorizar los datos de entrenamiento

### Modelo Regularizado — L2 + Early Stopping (Menor Varianza)
- `alpha = 0.15` (penalización L2 moderada)
- Early Stopping activado (`n_iter_no_change=20`)
- Aprende patrones más generales

---

## Resultados

| Métrica | Base (sin reg.) | Regularizado |
|---|---|---|
| Accuracy entrenamiento | **1.0000** | 0.9556 |
| Accuracy prueba (test) | 0.9200 | 0.8867 |
| **Brecha (overfitting gap)** | **0.0800** | 0.0689 |
| Épocas entrenadas | 47 | 32 |

El modelo base alcanza accuracy perfecto en entrenamiento (100 %), señal clara de **sobreajuste**: memoriza los datos en lugar de generalizar. La regularización reduce la brecha entre entrenamiento y prueba y detiene el entrenamiento antes de sobreajustar.

---

## Visualizaciones (`bias_variance_sklearn.png`)

La figura generada contiene 6 gráficas:

1. **Curva de pérdida (Loss)** — evolución del error por época en ambos modelos.
2. **Accuracy comparativo** — barras de entrenamiento vs prueba para cada modelo.
3. **Brecha de generalización** — diferencia Train − Test Accuracy por modelo.
4. **Curva de aprendizaje – Modelo Base** — la brecha train/validación persiste aunque se agreguen más datos → diagnóstico de **alta varianza**.
5. **Curva de aprendizaje – Modelo Regularizado** — brecha más estrecha y curvas convergentes.
6. **Curva de validación (α)** — muestra las tres zonas del trade-off Bias/Variance según el valor de regularización L2:
   - α pequeño → **Overfitting** (alta varianza)
   - α óptimo → **Punto de equilibrio**
   - α grande → **Underfitting** (alto sesgo)

---

## Técnicas de Regularización Aplicadas

**Regularización L2 (Weight Decay)**
Penaliza los pesos grandes en la función de costo, obligando a la red a aprender representaciones más simples y distribuidas. Reduce la varianza sin aumentar excesivamente el sesgo.

**Early Stopping**
Detiene el entrenamiento cuando la pérdida en validación deja de mejorar durante `n_iter_no_change` épocas consecutivas. Previene que el modelo continúe ajustándose al ruido del conjunto de entrenamiento.

---

## Análisis Bias-Variance

| Diagnóstico | Señal en las gráficas | Solución |
|---|---|---|
| Alta Varianza (Overfitting) | Curva train alta, validación baja; brecha grande | Regularización L2, Dropout, más datos |
| Alto Sesgo (Underfitting) | Ambas curvas bajas y juntas | Más capacidad, menos regularización |
| Equilibrio | Brecha pequeña, validación razonablemente alta | α óptimo, arquitectura adecuada |

---

## Ejecución

```bash
python3 bias_variance_sklearn.py
```

Requiere: `numpy`, `matplotlib`, `scikit-learn`

```bash
pip install numpy matplotlib scikit-learn
```

---

## Conclusiones

1. El modelo base alcanza accuracy perfecto en entrenamiento pero generaliza peor en datos nuevos, evidenciando overfitting por **alta varianza**.
2. La regularización L2 reduce los pesos excesivos y, combinada con Early Stopping, disminuye la brecha de generalización.
3. La **curva de validación** es la herramienta más directa para diagnosticar el trade-off Bias/Variance: permite encontrar el valor óptimo de α antes de entrenar el modelo final.
4. Existe un equilibrio entre sesgo y varianza — regularizar demasiado produce underfitting (alto sesgo), mientras que regularizar poco produce overfitting (alta varianza).
