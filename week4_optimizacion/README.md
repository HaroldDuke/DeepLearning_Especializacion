Semana 4 – Actividad 3: Aplicación de Técnicas de Optimización en una Red Neuronal en Google Colab
# Experimento de Optimización en Redes Neuronales (Compuerta AND)

##  Descripción General
Este proyecto implementa una red neuronal simple en **Google Colab** para aprender la lógica de la compuerta **AND**.  
El objetivo principal es **aplicar técnicas de optimización** y comparar cómo distintos **optimizadores** y **tasas de aprendizaje** afectan el proceso de entrenamiento.

##  Flujo del Notebook
1. **Importación de librerías**  
   - `numpy` para manejar los datos.  
   - `tensorflow/keras` para construir y entrenar la red neuronal.  
   - `matplotlib` para graficar la evolución del entrenamiento.  

2. **Definición del dataset**  
   - Entradas: todas las combinaciones binarias de dos variables (0,0), (0,1), (1,0), (1,1).  
   - Salida esperada: compuerta AND → `[0,0,0,1]`.  

3. **Construcción del modelo**  
   - Capa oculta con 4 neuronas y activación ReLU.  
   - Capa de salida con 1 neurona y activación sigmoide.  
   - Este diseño permite aprender relaciones lógicas simples.  

4. **Configuración de optimizadores**  
   - `SGD` con tasas de aprendizaje 0.1 y 0.01.  
   - `Adam` con tasas de aprendizaje 0.01 y 0.001.  

5. **Entrenamiento**  
   - Se entrena el modelo por 200 épocas con cada configuración.  
   - Se guarda el historial de métricas (*loss* y *accuracy*).  

6. **Visualización y resultados**  
   - Se grafica la evolución del **loss** (error) durante el entrenamiento.  
   - Se imprime en consola el **loss final** y el **accuracy final** para cada configuración.  

## 📊 Resultados

| Configuración   | Accuracy | Error (Loss) |
| --------------- | -------- | ------------ |
| SGD (lr=0.1)    | 1.00     | 0.2561       |
| SGD (lr=0.01)   | 0.75     | 0.5346       |
| Adam (lr=0.01)  | 1.00     | 0.0841       |
| Adam (lr=0.001) | 0.75     | 0.5794       |


**Análisis

- El **loss** muestra cómo el error disminuye con el tiempo, evidenciando la convergencia del modelo.  
- El **accuracy** final indica qué tan bien aprendió la red la tabla de verdad de la compuerta AND.  
- Configuraciones con **Adam** suelen ser más estables y rápidas en alcanzar precisión perfecta.  
- Configuraciones con **SGD** dependen fuertemente de la tasa de aprendizaje:  
  - Con `lr=0.1` puede converger rápido.  
  - Con `lr=0.01` converge más lento y puede quedarse en valores intermedios.  
- Adam (lr=0.01) obtuvo el mejor desempeño, con menor error y convergencia más rápida.
- SGD (lr=0.1) logró aprender correctamente, pero con mayor error final.
- Configuraciones con lr bajo (0.01 y 0.001) mostraron aprendizaje incompleto (accuracy = 0.75).
- El loss permitió observar mejor la calidad del entrenamiento que el accuracy.

---
** Justificación

Se seleccionaron SGD y Adam porque:
SGD permite observar el impacto directo de la tasa de aprendizaje
Adam ajusta automáticamente los parámetros, mejorando estabilidad
Adam es más adecuado en este caso porque:
Converge más rápido
Reduce mejor el error
Requiere menos ajuste manual

---
** Impacto de la Optimización

Mejora significativa en la convergencia del modelo

Reducción del error final (loss)

Diferencias claras entre configuraciones

Evidencia de que la tasa de aprendizaje es crítica

---
** Discusión

Los resultados obtenidos muestran que el rendimiento del modelo depende significativamente tanto del optimizador como de la tasa de aprendizaje. Aunque todas las configuraciones permiten cierto nivel de aprendizaje, solo algunas logran una convergencia completa.

Se observa que Adam con una tasa de aprendizaje de 0.01 logra el mejor equilibrio entre velocidad y precisión, alcanzando un error mínimo y una clasificación perfecta. Esto se debe a su capacidad de ajustar dinámicamente las actualizaciones de los pesos, lo que mejora la estabilidad del entrenamiento.

Por otro lado, SGD presenta mayor sensibilidad a la tasa de aprendizaje. Con un valor alto (0.1), el modelo logra converger, pero con mayor error residual, lo que sugiere que las actualizaciones pueden ser menos precisas. En contraste, con una tasa más baja (0.01), el aprendizaje es insuficiente, lo que evidencia un problema de convergencia lenta.

Además, se evidencia que el accuracy no es suficiente para evaluar completamente el modelo, ya que configuraciones con el mismo accuracy presentan diferencias importantes en el loss. Esto resalta la importancia de analizar múltiples métricas.

En conjunto, estos resultados confirman que la elección del optimizador y sus hiperparámetros es un factor crítico en el entrenamiento de redes neuronales, incluso en problemas simples.

---
** Conclusiones

- La red neuronal es capaz de aprender la lógica de la compuerta AND con distintas configuraciones de optimización.  
- El **optimizador Adam** mostró mejor desempeño en términos de estabilidad y rapidez de convergencia.  
- El **SGD** requiere un ajuste cuidadoso de la tasa de aprendizaje para evitar que el modelo se quede en un error alto o oscile demasiado.  
- Graficar el **loss** es más informativo que graficar el accuracy en datasets pequeños, ya que el accuracy toma valores discretos y puede verse “digital”.  
- Este experimento demuestra la importancia de los hiperparámetros en el entrenamiento de redes neuronales, incluso en problemas simples.  


