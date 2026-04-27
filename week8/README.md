# 🧠 Clasificación de Imágenes con CNN y Transfer Learning (CIFAR-10)

## 📌 Descripción del Proyecto

Este proyecto tiene como objetivo implementar y comparar dos enfoques de aprendizaje profundo para la clasificación de imágenes utilizando el dataset **CIFAR-10**:

1. Una **Red Neuronal Convolucional (CNN)** construida desde cero.
2. Un modelo basado en **Transfer Learning** utilizando **MobileNetV2**.

Se busca analizar el rendimiento de ambos modelos y entender cómo el uso de modelos preentrenados puede mejorar significativamente la precisión.

---

## 📊 Dataset: CIFAR-10

El dataset CIFAR-10 contiene:

* 60,000 imágenes a color (32x32 píxeles)
* 10 clases:

  * Avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco y camión
* División:

  * 50,000 imágenes de entrenamiento
  * 10,000 imágenes de prueba

---

## ⚙️ Tecnologías Utilizadas

* Python
* TensorFlow / Keras
* Matplotlib

---

## 🚀 Implementación

### 1. Carga y Preprocesamiento

* Se carga el dataset CIFAR-10 desde Keras.
* Se normalizan los valores de píxeles en el rango `[0,1]`.

```python
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```

---

### 2. Modelo CNN (Desde Cero)

#### Arquitectura:

* Capas convolucionales con activación ReLU
* Capas de pooling para reducción espacial
* Capa densa con dropout para evitar overfitting
* Capa final con Softmax (10 clases)

```python
cnn_model = models.Sequential([
    layers.Input(shape=(32,32,3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

#### Compilación:

```python
cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

### 3. Entrenamiento CNN

* Épocas: 10
* Resultado final:

  * **Precisión entrenamiento:** ~67%
  * **Precisión validación:** ~68%

---

### 4. Transfer Learning con MobileNetV2

#### Ajustes realizados:

* Redimensionamiento de imágenes a 96x96
* Uso de pesos preentrenados en ImageNet
* Congelación del modelo base

```python
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(96,96,3)
)
base_model.trainable = False
```

#### Capas añadidas:

```python
tl_model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

---

### 5. Entrenamiento Transfer Learning

* Épocas: 10
* Resultado final:

  * **Precisión entrenamiento:** ~80%
  * **Precisión validación:** ~81.5%

---

## 📈 Resultados y Análisis

### 🔹 CNN desde cero

* Aprendizaje progresivo y estable
* Limitaciones en precisión debido a:

  * Arquitectura relativamente simple
  * Dataset complejo (CIFAR-10)
* Presenta ligera tendencia al overfitting controlado con dropout

### 🔹 Transfer Learning (MobileNetV2)

* Mejora significativa en precisión (+13%)
* Entrenamiento más eficiente en términos de calidad del modelo
* Aprovecha características previamente aprendidas en ImageNet
* Generaliza mejor sobre los datos de prueba

---

## 📊 Comparación

| Modelo            | Precisión Entrenamiento | Precisión Validación |
| ----------------- | ----------------------- | -------------------- |
| CNN desde cero    | ~67%                    | ~68%                 |
| Transfer Learning | ~80%                    | ~81.5%               |

---

## 📉 Visualización

Se grafican:

* Precisión (accuracy)
* Pérdida (loss)

Para analizar el comportamiento del entrenamiento y validación.

---

## 🧠 Conclusiones

* El modelo CNN desde cero funciona correctamente, pero su rendimiento es limitado.
* El uso de **Transfer Learning** mejora considerablemente los resultados sin necesidad de entrenar desde cero.
* MobileNetV2 demuestra ser una excelente opción para tareas de clasificación con datasets pequeños o medianos.
* La reutilización de modelos preentrenados reduce el tiempo de desarrollo y mejora la precisión.

---

## 🔧 Posibles Mejoras

* Fine-tuning del modelo base (descongelar capas finales)
* Aumentar número de épocas
* Data augmentation
* Probar otras arquitecturas (ResNet, EfficientNet)

---

## ▶️ Ejecución

1. Instalar dependencias:

```bash
pip install tensorflow matplotlib
```

2. Ejecutar el script

