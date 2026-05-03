# Semana 11 — Redes Neuronales Recurrentes para Series de Tiempo

## Objetivo

Implementar y comparar tres arquitecturas recurrentes (SimpleRNN, LSTM y GRU) para predecir valores futuros en una serie de tiempo sintética. El propósito es entender cómo los modelos recurrentes capturan dependencias temporales: usan información del pasado para anticipar el futuro.

---

## Estructura del proyecto

```
week11/
├── main.py          # Punto de entrada — orquesta todo el flujo
├── data.py          # Generación de datos y construcción de secuencias
├── model.py         # Definición de arquitecturas (SimpleRNN, LSTM, GRU)
├── train.py         # Lógica de entrenamiento con callbacks
├── evaluate.py      # Métricas y visualizaciones
├── requirements.txt # Dependencias
├── models/          # Pesos guardados del mejor modelo por arquitectura
└── plots/           # Gráficas generadas automáticamente
```

---

## Datos

Se genera una serie de tiempo sintética de **1200 pasos** que combina:

- **Tendencia lineal** creciente
- **Estacionalidad** con tres frecuencias superpuestas (sen, 2×, 4×)
- **Ruido gaussiano** para simular condiciones reales

Este tipo de señal es representativa de series del mundo real como demanda, tráfico o consumo energético.

La serie se divide en:

| Partición     | Proporción | Uso                              |
|---------------|-----------|----------------------------------|
| Entrenamiento | 70 %      | Ajuste de pesos                  |
| Validación    | 15 %      | Seguimiento durante entrenamiento|
| Prueba        | 15 %      | Evaluación final no vista        |

> El escalado (MinMaxScaler) se ajusta **solo sobre el conjunto de entrenamiento** para evitar fuga de información hacia validación y prueba.

---

## Construcción de secuencias

El problema se formula como predicción de un paso adelante con ventana deslizante:

```
Entrada:  [t-29, t-28, ..., t-1, t]  →  Salida: t+1
```

Con `window_size = 30`, cada muestra es una secuencia de 30 valores pasados con forma `(30, 1)`.

---

## Arquitecturas

Las tres arquitecturas comparten la misma estructura de capas; solo cambia el tipo de celda recurrente.

### SimpleRNN
Celda recurrente básica. Aprende dependencias a corto plazo. Puede sufrir desvanecimiento del gradiente en secuencias largas.

```
SimpleRNN(64) → SimpleRNN(32) → Dense(16) → Dense(1)
```

### LSTM (Long Short-Term Memory)
Introduce puertas de entrada, olvido y salida para controlar qué información conservar o descartar. Maneja dependencias a largo plazo con mayor efectividad.

```
LSTM(64) → LSTM(32) → Dense(16) → Dense(1)
```

### GRU (Gated Recurrent Unit)
Simplificación del LSTM con dos puertas (reset y update). Rendimiento comparable al LSTM con menos parámetros y mayor velocidad de entrenamiento.

```
GRU(64) → GRU(32) → Dense(16) → Dense(1)
```

Todos los modelos usan:
- **Pérdida:** MSE (Mean Squared Error)
- **Optimizador:** Adam
- **Métrica de seguimiento:** MAE

---

## Entrenamiento

Cada modelo se entrena hasta 50 épocas con los siguientes callbacks:

| Callback              | Configuración                              |
|-----------------------|--------------------------------------------|
| EarlyStopping         | paciencia = 5 épocas sobre `val_loss`      |
| ModelCheckpoint       | guarda el mejor peso en `models/`          |
| ReduceLROnPlateau     | reduce LR × 0.5 si no mejora en 3 épocas  |

---

## Métricas de evaluación

| Métrica | Descripción                                              |
|---------|----------------------------------------------------------|
| MSE     | Error cuadrático medio — penaliza errores grandes        |
| RMSE    | Raíz del MSE — en la misma unidad que los datos          |
| MAE     | Error absoluto medio — más robusto ante outliers         |
| R²      | Coeficiente de determinación — 1.0 es predicción perfecta|

---

## Cómo ejecutar

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
python main.py
```

Al finalizar se generan en `plots/`:

**Serie completa con partición de datos**
![Serie completa](plots/serie_completa.png)

**Predicción vs Real por modelo**
![SimpleRNN](plots/prediccion_simplernn.png)
![LSTM](plots/prediccion_lstm.png)
![GRU](plots/prediccion_gru.png)

**Curvas de entrenamiento comparadas**
![Curvas de entrenamiento](plots/curvas_entrenamiento.png)

Y en consola se imprime la tabla comparativa final:

```
Modelo        RMSE        MAE         R²
--------------------------------------------
SimpleRNN   0.012345   0.009876    0.987654
LSTM        0.008123   0.006543    0.993210
GRU         0.008456   0.006789    0.992870
```

---

## Conclusiones esperadas

- **LSTM y GRU** superan a **SimpleRNN** en RMSE y R² gracias a sus mecanismos de memoria controlada.
- Las tres arquitecturas demuestran que las RNN capturan efectivamente las dependencias temporales de la serie.
- GRU alcanza resultados similares a LSTM con menos parámetros, lo que lo hace más eficiente en series de tiempo convencionales.
- El uso correcto de callbacks (EarlyStopping + ReduceLROnPlateau) evita sobreajuste sin requerir ajuste manual de épocas.
