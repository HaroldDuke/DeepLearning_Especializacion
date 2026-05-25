# Model Serving

## Definición

El **Model Serving** (servido de modelos) es el proceso de **poner un modelo de Machine Learning entrenado a disposición de usuarios o sistemas** para que puedan hacer predicciones en tiempo real o por lotes. Es el último eslabón del ciclo de vida del modelo: aquí el modelo deja de ser un experimento y empieza a generar valor real.

Entrenar un modelo es solo la mitad del trabajo. Sin un sistema de serving, el modelo es un archivo `.h5` que nadie usa.

## ¿Por qué es importante?

1. **Valor real**: un modelo guardado en disco no genera ningún beneficio para el usuario final.
2. **Integración**: las aplicaciones (apps móviles, sitios web, sistemas internos) necesitan consumir predicciones.
3. **Escalabilidad**: servir requiere arquitectura distinta a entrenar — baja latencia, alto throughput, alta disponibilidad.
4. **Monitoreo en producción**: hay que saber si el modelo sigue funcionando bien con datos reales (concept drift, data drift).

## Tipos de Model Serving

### 1. Batch Serving (predicciones por lotes)

El modelo procesa un gran volumen de datos de una sola vez en un horario programado (ej. todas las noches).

**Ejemplos:**
- Predecir el churn de todos los clientes una vez al mes.
- Generar recomendaciones diarias para todos los usuarios.
- Calificar todos los créditos pendientes.

**Tecnologías:** Apache Spark, AWS Batch, cron + scripts Python, Airflow.

### 2. Online / Real-Time Serving (predicciones bajo demanda)

El modelo responde a peticiones individuales en tiempo real, normalmente vía una API HTTP.

**Ejemplos:**
- Detección de fraude al momento de una transacción.
- Clasificación de imagen en una app móvil.
- Asistente conversacional (ChatGPT).

**Requisitos:** latencia baja (<100ms), alta disponibilidad (99.9%+).

### 3. Streaming Serving

El modelo consume eventos de un stream (Kafka, Kinesis) y genera predicciones de forma continua.

**Ejemplos:**
- Monitoreo de tweets en tiempo real.
- Detección de anomalías en sensores IoT.

### 4. Edge Serving

El modelo se ejecuta directamente **en el dispositivo del usuario** (móvil, IoT, navegador) sin necesidad de servidor.

**Ejemplos:**
- Reconocimiento facial offline en un iPhone.
- Modelos en navegador con TensorFlow.js.
- Visión por computadora en drones.

**Tecnologías:** TensorFlow Lite, ONNX Runtime, Core ML, TensorFlow.js.

## Alternativas concretas para servir un modelo

| Herramienta | Tipo | Pros | Contras |
|---|---|---|---|
| **FastAPI** | Framework Python | Simple, async, documentación automática (OpenAPI) | Requiere construir todo a mano |
| **Flask** | Framework Python | Maduro, ecosistema grande | Síncrono, menos performante |
| **TensorFlow Serving** | Servidor especializado de Google | Optimizado para modelos TF, versionado integrado | Solo TensorFlow |
| **TorchServe** | Servidor de PyTorch | Optimizado para PyTorch, multi-modelo | Solo PyTorch |
| **NVIDIA Triton** | Servidor universal | Multi-framework (TF, PyTorch, ONNX), GPU optimizado | Más complejo de configurar |
| **AWS SageMaker** | Plataforma gestionada (cloud) | Auto-scaling, versionado, A/B testing | Costo, vendor lock-in |
| **Google Vertex AI** | Plataforma gestionada (cloud) | Integración con GCP | Costo, vendor lock-in |
| **Streamlit / Gradio** | Apps interactivas | Demo en minutos, UI incluida | No apto para producción seria |
| **HuggingFace Spaces** | Hosting gratuito | Comparte demos al instante | Pensado para demos, no producción |

## Ejemplo conceptual con FastAPI

Aunque en esta entrega no es obligatorio implementarlo, así se vería un endpoint básico que sirviera nuestro clasificador MNIST:

```python
from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="MNIST Classifier API")
model = tf.keras.models.load_model("mnist_classifier.keras")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Recibe una imagen y devuelve el dígito predicho."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28))
    array = np.array(image).astype("float32") / 255.0
    array = array.reshape(1, 28, 28, 1)

    probabilities = model.predict(array)[0]
    predicted_digit = int(np.argmax(probabilities))

    return {
        "digit": predicted_digit,
        "confidence": float(probabilities[predicted_digit]),
        "all_probabilities": probabilities.tolist(),
    }
```

Se levantaría con `uvicorn app:app --reload` y se llamaría con:

```bash
curl -X POST -F "file=@digit.png" http://localhost:8000/predict
```

## Aspectos críticos al servir un modelo

### 1. Formato del modelo
- **Keras nativo** (`.keras`, `.h5`) — flexible pero atado a TF
- **SavedModel** (`saved_model.pb`) — formato estándar de TF
- **ONNX** — formato abierto, portable entre frameworks
- **TorchScript** — para modelos PyTorch
- **TensorFlow Lite** — para edge/móvil

### 2. Versionado
Cada cambio del modelo debe quedar trazado: `v1.0`, `v1.1`, ... Idealmente con A/B testing antes de promover una nueva versión.

### 3. Monitoreo en producción
- **Latencia** de las predicciones
- **Throughput** (peticiones/segundo)
- **Data drift** (cambian las features de entrada)
- **Concept drift** (cambia la relación features → target)
- **Tasa de errores** del modelo

### 4. Seguridad
- Autenticación (API keys, OAuth)
- Rate limiting
- Validación de entrada (evitar inputs maliciosos)
- Encriptación TLS

### 5. Costo computacional
Servir grandes modelos puede ser caro: cuantización, distillation, batching son técnicas para reducir costos.

## Conclusión

Model Serving es lo que convierte un modelo entrenado en un **producto funcional**. Existen múltiples opciones según el caso de uso: desde una API simple con FastAPI hasta plataformas gestionadas como SageMaker o Vertex AI.

En el ciclo MLOps, el serving cierra el círculo: el modelo se entrena, se loggea, se versiona, se sirve, se monitorea — y los datos del serving alimentan el siguiente ciclo de entrenamiento. **El modelo no termina cuando se entrena; el modelo empieza cuando se sirve.**

## Referencias

- Sato, D. — *"Continuous Delivery for Machine Learning"* (Martin Fowler, 2019)
- Huyen, C. — *Designing Machine Learning Systems* (O'Reilly, 2022)
- TensorFlow Serving docs — https://www.tensorflow.org/tfx/serving
- FastAPI docs — https://fastapi.tiangolo.com
