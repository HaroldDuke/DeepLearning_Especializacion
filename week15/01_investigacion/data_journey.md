# Data Journey

## Definición

El **Data Journey** (o "viaje del dato") es el ciclo completo que recorre un dato desde que se genera en su fuente original hasta que es consumido por un modelo de Machine Learning para inferencia o entrenamiento. Es uno de los pilares fundamentales de MLOps porque la calidad y trazabilidad de los datos determinan directamente la calidad del modelo resultante.

A diferencia del software tradicional, en Machine Learning **el dato es código**: una versión distinta de los datos produce un modelo distinto. Por eso entender, documentar y versionar el Data Journey es esencial.

## Etapas del Data Journey

```
[Fuente] → [Ingesta] → [Almacenamiento] → [Limpieza] → [Transformación] → [Particionamiento] → [Modelo]
```

### 1. Fuente (Data Source)
El origen del dato. Puede ser:
- **Bases de datos relacionales** (PostgreSQL, MySQL, SQL Server)
- **Data Lakes** (S3, Google Cloud Storage, Azure Data Lake)
- **APIs externas** (REST, GraphQL, gRPC)
- **Streaming** (Kafka, Kinesis, Pub/Sub)
- **Sensores IoT** o dispositivos físicos
- **Archivos planos** (CSV, JSON, Parquet)
- **Datasets públicos académicos** (MNIST, CIFAR, ImageNet)

### 2. Ingesta (Ingestion)
Es el proceso de **traer el dato** desde la fuente al entorno de procesamiento. Puede ser:
- **Batch**: por lotes en intervalos definidos (ej. cada noche).
- **Streaming**: en tiempo real, evento por evento.
- **Micro-batch**: lotes pequeños frecuentes.

### 3. Almacenamiento (Storage)
Una vez ingerido, el dato se persiste en un formato eficiente para el procesamiento posterior: Parquet, ORC, Feather, HDF5, TFRecord o bases especializadas como **Feature Stores** (Feast, Tecton).

### 4. Limpieza (Cleaning)
Proceso de eliminar imperfecciones del dato:
- Valores nulos o faltantes (`NaN`)
- Duplicados
- Outliers (valores atípicos)
- Inconsistencias de formato (mayúsculas/minúsculas, fechas, encoding)
- Errores tipográficos

### 5. Transformación (Transformation / Feature Engineering)
Convertir el dato crudo en **features** que el modelo pueda consumir:
- **Normalización / Estandarización** (escalar a [0,1] o media 0 desviación 1)
- **Encoding** (One-Hot, Label, Target, Embeddings)
- **Creación de variables derivadas** (ratios, agregados temporales, lags)
- **Reducción de dimensionalidad** (PCA, t-SNE)

### 6. Particionamiento (Splitting)
Dividir el dataset en subconjuntos:
- **Train** (60–80%) — para ajustar los pesos del modelo
- **Validation** (10–20%) — para ajustar hiperparámetros y detener el entrenamiento
- **Test** (10–20%) — para evaluación final imparcial

Es crítico evitar **data leakage**: que información del test "se filtre" al training.

### 7. Consumo por el modelo
El dato finalmente alimenta al modelo a través de pipelines optimizados como `tf.data.Dataset` (TensorFlow), `DataLoader` (PyTorch) o `Dataset` (HuggingFace).

## Aplicación práctica en este proyecto (MNIST)

| Etapa | Implementación en nuestro proyecto |
|---|---|
| **Fuente** | Dataset MNIST público distribuido con Keras (`tf.keras.datasets.mnist`) |
| **Ingesta** | `load_data()` descarga 60.000 imágenes de entrenamiento + 10.000 de prueba |
| **Almacenamiento** | En memoria como arrays NumPy (`uint8`, 28×28) |
| **Limpieza** | No requerida — MNIST viene curado |
| **Transformación** | (1) Conversión a `float32`, (2) Normalización a `[0, 1]`, (3) Agregar canal: `(N, 28, 28) → (N, 28, 28, 1)` |
| **Particionamiento** | 80% train / 20% val a partir del set de 60K; 10K reservados para test |
| **Consumo** | `tf.data.Dataset` con `batch`, `shuffle` y `prefetch` |

## Buenas prácticas

1. **Documentar el linaje del dato** (data lineage): de dónde viene cada feature.
2. **Versionar los datos**, no solo el código. Herramientas: DVC, LakeFS, W&B Artifacts.
3. **Crear Data Cards** que describan datasets como las "fichas técnicas" de un producto.
4. **Validar la calidad del dato** automáticamente (Great Expectations, Pandera, TFDV).
5. **Reproducibilidad**: fijar semillas aleatorias en las particiones.

## Conclusión

El Data Journey transforma el dato de un activo "crudo y desordenado" en una entrada estructurada, confiable y trazable que alimenta un modelo. Sin un Data Journey bien diseñado, el resto del pipeline de ML se construye sobre cimientos frágiles: por más sofisticado que sea el modelo, **"garbage in, garbage out"**.

## Referencias

- Google Cloud — *Practitioners Guide to MLOps* (2021)
- Sambasivan et al. — *"Everyone wants to do the model work, not the data work"* (Google Research, 2021)
- TensorFlow Data Validation (TFDV) — https://www.tensorflow.org/tfx/data_validation
