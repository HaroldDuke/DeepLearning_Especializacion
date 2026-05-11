# README del proyecto: Transformer con atención para series de tiempo

## OBJETIVO
Este proyecto implementa un modelo de aprendizaje profundo basado en un mecanismo de atención y un Transformer simplificado para predecir valores futuros de una serie de tiempo mensual. Además, el código busca comparar este enfoque con un modelo LSTM y analizar el desempeño obtenido en términos de predicción e interpretabilidad.

## ESTRUCTURA DEL PROYECTO
El código sigue un flujo ordenado de trabajo que inicia con la importación de librerías, continúa con la carga y preparación de datos, luego define la arquitectura del modelo, realiza el entrenamiento, evalúa resultados y finalmente incorpora visualizaciones y una comparación con LSTM .

## Proyecto
─ Importación de librerías

─ Carga y preprocesamiento de datos

─ Construcción de secuencias

─ Implementación del mecanismo de atención

─ Definición del Transformer simplificado

─ Entrenamiento del modelo

─ Evaluación con métricas

─ Visualización de pesos de atención

─ Comparación con LSTM

## DATOS
El código utiliza una serie de tiempo univariada con datos mensuales de pasajeros de aerolínea entre 1949 y 1960, con un total de 144 observaciones. Esta serie presenta tendencia creciente y estacionalidad anual, por lo que resulta adecuada para probar modelos capaces de capturar dependencias temporales.

Características de los datos:

- Serie temporal univariada.
- Frecuencia mensual, construida con `pd.date_range(..., freq='MS')`.
- Total de registros: 144.
- Variable objetivo: número de pasajeros por periodo.
- Escalamiento previo al entrenamiento con `MinMaxScaler` en el rango `[0, 1]`.

## CONSTRUCCIÓN DE SECUENCIA
El código transforma la serie de tiempo en un problema supervisado mediante una ventana deslizante. Cada muestra toma 12 valores consecutivos como entrada y usa el valor siguiente como salida objetivo.

La función `crear_secuencias(datos, seq_len)` genera dos estructuras: `X`, que contiene las secuencias de entrada, y `y`, que almacena el siguiente valor a predecir. Después, se aplica una división 80/20 respetando el orden temporal y se convierten los datos a tensores PyTorch con forma `(batch, seq_len, features)`.

```text
Entrada (X): [t1, t2, t3, ..., t12]
Salida  (y): t13
```

## ARQUITECTURA
La solución principal del código es un **Transformer simplificado** implementado en PyTorch. Primero se define una clase `Atencion` que construye el mecanismo de atención escalada por producto punto usando proyecciones lineales para `Q`, `K` y `V`, y luego aplica `softmax` para obtener los pesos de atención.

Después se define la clase `TransformerSimple`, cuya arquitectura sigue la secuencia `Embedding → Atención → Add & Norm → Feed Forward → Pooling → Predicción`. El modelo proyecta la entrada a un espacio `d_model=32`, aplica atención, incorpora conexiones residuales con normalización, utiliza una red feed-forward y resume la secuencia con un promedio temporal para producir una salida escalar.

Componentes principales:

- Capa de embedding: `nn.Linear(1, d_model)`.
- Atención escalada por producto punto implementada manualmente.
- Normalización con `LayerNorm`.
- Red feed-forward con activación `ReLU`.
- Capa final `nn.Linear(d_model, 1)` para regresión.

Además, el código incluye un modelo `LSTMModelo` como línea base, compuesto por una capa `nn.LSTM` y una capa lineal de salida, para comparar el desempeño del Transformer con una arquitectura recurrente clásica.

## ENTRENAMIENTO
El modelo se entrena usando la función de pérdida `MSELoss`, el optimizador `Adam` con tasa de aprendizaje `1e-3` y un total de 80 épocas [file:11]. Durante cada época se ejecuta una fase de entrenamiento sobre el conjunto de entrenamiento y una fase de validación sobre el conjunto de prueba, guardando el historial de pérdidas en ambas fases.

El código usa `batch_size=16` y mezcla el conjunto de entrenamiento con `shuffle=True` dentro del `DataLoader`. También imprime el avance cada 20 épocas y genera una curva de pérdida para visualizar la evolución del aprendizaje.

Parámetros clave:

- Optimizador: Adam.
- Learning rate: `1e-3`.
- Función de pérdida: `MSELoss.
- Épocas: 80.
- Batch size:.

## METRICAS DE EVALUACIÓN
La evaluación se realiza sobre el conjunto de prueba después de desnormalizar las predicciones al rango original. El código calcula MSE, RMSE, MAE, MAPE y \(R^2\), lo que permite medir el error desde diferentes perspectivas.

Resultados reportados por el código:

- MSE: 4979.93.
- RMSE: 70.57.
- MAE: 56.19.
- MAPE: 12.60%.
- \(R^2\): 0.2160.

También se genera una gráfica de predicción vs. valores reales y una visualización de pesos de atención para analizar qué posiciones temporales resultan más importantes en la decisión del modelo.

## CONCLUSIONES
El código presenta una implementación clara y didáctica de un Transformer simplificado aplicado a predicción de series temporales. Su valor no está solo en la predicción, sino también en la posibilidad de interpretar el comportamiento del modelo a través de los pesos de atención y de contrastarlo con una LSTM.

Como resultado, el proyecto funciona como una base sólida para seguir experimentando con modelos secuenciales, mejorar el desempeño predictivo, ajustar hiperparámetros o extender el enfoque hacia problemas de series de tiempo más complejos [file:11].
