Semana 4 – Actividad 4: Aplicación de Métodos de Regularización en una Red Neuronal
# Regularización en Redes Neuronales

## Objetivo

El objetivo de esta actividad fue aplicar técnicas de regularización en una red neuronal con el fin de reducir el sobreajuste (overfitting) y mejorar la capacidad de generalización del modelo. Para ello, se realizó una comparación entre un modelo sin regularización y otro que incorpora técnicas como Dropout y regularización L2, evaluando su desempeño mediante métricas como loss y accuracy.

---

## Metodología

La actividad se desarrolló utilizando un entorno de notebooks en Google Colab y la librería TensorFlow/Keras. El proceso consistió en las siguientes etapas:

1. Carga de datos: Se utilizó el dataset MNIST, compuesto por imágenes de dígitos escritos a mano.
2. Preprocesamiento: Los datos fueron normalizados para mejorar el entrenamiento del modelo.
3. Construcción del modelo sin regularización:
   - Red neuronal con capas densas.
   - Función de activación ReLU.
   - Capa de salida con Softmax.
4. Construcción del modelo con regularización:
   - Se agregó regularización L2 en las capas densas.
   - Se implementó Dropout para desactivar neuronas aleatoriamente durante el entrenamiento.
5. Entrenamiento de ambos modelos:
   - Se entrenaron durante 10 épocas.
   - Se utilizó un conjunto de validación para evaluar el rendimiento.
6. Evaluación y comparación:
   - Se analizaron métricas como pérdida (loss) y precisión (accuracy).
   - Se visualizaron gráficas de entrenamiento y validación.
   - Se compararon resultados numéricos finales.

---

##  Resultados

###  Modelo sin regularización

- Train Loss: 0.0155  
- Train Accuracy: 0.9951  
- Validation Loss: 0.0829  
- Validation Accuracy: 0.9777  
- Diferencia Loss: 0.0675  
- Diferencia Accuracy: 0.0174  

###  Modelo con regularización

- Train Loss: 0.2886  
- Train Accuracy: 0.9412  
- Validation Loss: 0.2032  
- Validation Accuracy: 0.9695  
- Diferencia Loss: 0.0854  
- Diferencia Accuracy: 0.0283  

---

## Análisis de Resultados

A partir de los resultados obtenidos, se observa que el modelo sin regularización presenta un mejor desempeño en términos de precisión tanto en entrenamiento como en validación. Esto indica que el modelo logra ajustarse muy bien a los datos disponibles.

Sin embargo, este comportamiento también puede ser indicativo de un leve sobreajuste, ya que el modelo alcanza una precisión muy alta en entrenamiento y una diferencia notable en la pérdida frente a los datos de validación. Esto sugiere que el modelo está aprendiendo patrones muy específicos del conjunto de entrenamiento.

Por otro lado, el modelo con regularización muestra un desempeño más conservador. La inclusión de técnicas como Dropout y L2 incrementa el error en entrenamiento, lo cual es esperado, ya que se limita la capacidad del modelo para memorizar los datos. En validación, aunque el rendimiento es ligeramente inferior, se mantiene relativamente estable.

La diferencia entre métricas de entrenamiento y validación en ambos modelos permite analizar el nivel de generalización. Aunque en este caso el modelo sin regularización no presenta un sobreajuste extremo, sí se evidencia una tendencia mayor en comparación con el modelo regularizado.

---

## Conclusiones

El desarrollo de esta actividad permitió evidenciar de manera práctica el impacto de las técnicas de regularización en el entrenamiento de redes neuronales, particularmente en la reducción del sobreajuste (overfitting) y la mejora de la capacidad de generalización del modelo.

En primer lugar, el modelo sin regularización alcanzó un desempeño superior en los datos de entrenamiento, con una precisión de 99.51% y un valor de pérdida bastante bajo. Sin embargo, al evaluar su rendimiento en datos de validación, se observó una disminución en la precisión (97.77%) y un incremento en la pérdida, lo que indica que el modelo tiende a ajustarse demasiado a los datos de entrenamiento, perdiendo capacidad de generalizar correctamente ante nuevos datos.

Por otro lado, el modelo con regularización presentó un comportamiento más equilibrado. Aunque su precisión en entrenamiento fue menor (94.12%) y su pérdida más alta, esto es esperado debido a la aplicación de técnicas como Dropout y regularización L2, las cuales limitan la complejidad del modelo. En los datos de validación, el modelo alcanzó una precisión de 96.95%, mostrando un rendimiento ligeramente inferior al modelo sin regularización, pero manteniendo un comportamiento más controlado.

Es importante destacar que, en este caso específico, el modelo sin regularización obtuvo mejores resultados en términos de precisión final. Sin embargo, esto no implica necesariamente que sea un mejor modelo en todos los contextos, ya que su mayor diferencia entre entrenamiento y validación evidencia una tendencia al sobreajuste. En contraste, el modelo regularizado, aunque menos preciso, es potencialmente más robusto frente a datos no vistos.

En conclusión, la regularización no siempre mejora directamente las métricas de desempeño, pero sí contribuye a construir modelos más estables y confiables. La elección entre un modelo con o sin regularización dependerá del equilibrio deseado entre precisión y capacidad de generalización, así como del contexto en el que será aplicado el modelo.

