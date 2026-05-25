# Acceso y Manipulación de Datos

## Definición

El **acceso y manipulación de datos** se refiere al conjunto de prácticas, herramientas y patrones que permiten **traer datos desde su fuente, transformarlos y prepararlos** para ser consumidos por un modelo de Machine Learning de forma **eficiente, reproducible y escalable**.

Si el Data Journey describe el "qué" recorre el dato, el acceso y manipulación describe el "cómo" lo hacemos en la práctica.

## Componentes principales

### 1. Acceso a los datos (Data Access)

Cómo nos conectamos a la fuente del dato:

| Tipo de fuente | Herramienta típica |
|---|---|
| Base de datos SQL | `sqlalchemy`, `psycopg2`, `pyodbc` |
| Archivos planos | `pandas`, `polars`, `pyarrow` |
| APIs REST | `requests`, `httpx` |
| Almacenamiento en la nube | `boto3` (S3), `google-cloud-storage`, `azure-storage-blob` |
| Datasets académicos | `tf.keras.datasets`, `torchvision.datasets`, `huggingface.datasets` |
| Streaming | `kafka-python`, `confluent-kafka` |

### 2. Carga eficiente (Efficient Loading)

Para que el entrenamiento no se vea limitado por I/O (entrada/salida), se usan **data pipelines** optimizados:

- **TensorFlow**: `tf.data.Dataset` con operaciones como `map`, `batch`, `shuffle`, `cache`, `prefetch`
- **PyTorch**: `torch.utils.data.Dataset` + `DataLoader` con `num_workers`
- **HuggingFace**: `datasets.Dataset` con streaming y caching automático

Patrones clave para no perder tiempo cargando datos:
- **Prefetching**: cargar el siguiente batch mientras el modelo procesa el actual
- **Paralelismo**: múltiples workers cargando en paralelo
- **Caching**: mantener en memoria/disco datos ya procesados
- **Mixed precision**: usar `float16` cuando sea posible

### 3. Manipulación (Data Manipulation)

Transformar el dato crudo en features listas para el modelo:

**Operaciones comunes:**
- **Selección y filtrado** (`df[df['col'] > x]`)
- **Agregación** (`groupby`, `pivot`, `resample`)
- **Joins** entre tablas
- **Limpieza** (manejo de nulos, duplicados, outliers)
- **Encoding categórico** (One-Hot, Label, Target Encoding)
- **Escalado** (`MinMaxScaler`, `StandardScaler`)
- **Aumento de datos** (Data Augmentation) — rotaciones, flips, crops en imágenes

**Herramientas:**
- **pandas / polars** — manipulación tabular
- **NumPy** — operaciones numéricas vectorizadas
- **scikit-learn** — preprocesadores reutilizables
- **albumentations / tf.image** — augmentation de imágenes

### 4. Validación de calidad del dato

Antes de entrenar, hay que verificar que el dato cumple expectativas:

| Validación | Ejemplo |
|---|---|
| Rango de valores | Píxeles entre [0, 1] tras normalizar |
| Distribución de clases | ¿Está balanceado? |
| Forma esperada | Imágenes de exactamente 28×28×1 |
| Ausencia de nulos | Ningún `NaN` en las features |
| Tipos de datos | `float32` y no `float64` |

**Herramientas:**
- **Great Expectations** — validaciones declarativas
- **Pandera** — schemas con type hints
- **TensorFlow Data Validation (TFDV)** — específico para ML

### 5. Reproducibilidad

Que el mismo código + los mismos datos produzcan el mismo resultado:
- **Semillas aleatorias fijas** (`np.random.seed`, `tf.random.set_seed`)
- **Versionado de datasets** (DVC, W&B Artifacts, LakeFS)
- **Documentación de transformaciones** (data cards, model cards)

## Aplicación práctica en este proyecto

```python
# 1. Acceso: dataset distribuido con Keras
(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Manipulación: normalización + agregar canal
x_train_full = x_train_full.astype('float32') / 255.0
x_train_full = np.expand_dims(x_train_full, axis=-1)
y_train_full = tf.keras.utils.to_categorical(y_train_full, 10)

# 3. Particionamiento: 80% train / 20% val
n_train = int(0.8 * len(x_train_full))
x_train, x_val = x_train_full[:n_train], x_train_full[n_train:]
y_train, y_val = y_train_full[:n_train], y_train_full[n_train:]

# 4. Carga eficiente: tf.data con shuffle + prefetch
train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(10_000)
            .batch(64)
            .prefetch(tf.data.AUTOTUNE))
```

## Buenas prácticas

1. **Separar acceso de manipulación**: un módulo carga el dato, otro lo transforma.
2. **Pipelines determinísticos** con semillas fijas.
3. **Documentar cada transformación** y por qué se hace.
4. **No tocar el test set** durante el desarrollo (sólo evaluación final).
5. **Validar la calidad del dato** antes de cada entrenamiento.

## Diferencia entre acceso y manipulación

| Aspecto | Acceso | Manipulación |
|---|---|---|
| **Qué hace** | Trae el dato | Transforma el dato |
| **Cuándo** | Una vez por sesión | Cada vez que se usa el dato |
| **Herramienta típica** | `boto3`, `requests` | `pandas`, `tf.data` |
| **Costo** | I/O (red, disco) | CPU/GPU |

## Conclusión

Un buen sistema de acceso y manipulación de datos hace que el resto del pipeline de ML sea **fluido, reproducible y rápido**. Es la "tubería" que conecta los datos crudos con el modelo. Si está mal diseñado, el cuello de botella deja de ser el modelo y pasa a ser el flujo de datos.

## Referencias

- TensorFlow tf.data guide — https://www.tensorflow.org/guide/data
- PyTorch DataLoader docs — https://pytorch.org/docs/stable/data.html
- Géron, A. — *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed., 2022)
