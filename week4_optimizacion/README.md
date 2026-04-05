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
- El **loss** muestra cómo el error disminuye con el tiempo, evidenciando la convergencia del modelo.  
- El **accuracy** final indica qué tan bien aprendió la red la tabla de verdad de la compuerta AND.  
- Configuraciones con **Adam** suelen ser más estables y rápidas en alcanzar precisión perfecta.  
- Configuraciones con **SGD** dependen fuertemente de la tasa de aprendizaje:  
  - Con `lr=0.1` puede converger rápido.  
  - Con `lr=0.01` converge más lento y puede quedarse en valores intermedios.  

##  Conclusiones
- La red neuronal es capaz de aprender la lógica de la compuerta AND con distintas configuraciones de optimización.  
- El **optimizador Adam** mostró mejor desempeño en términos de estabilidad y rapidez de convergencia.  
- El **SGD** requiere un ajuste cuidadoso de la tasa de aprendizaje para evitar que el modelo se quede en un error alto o oscile demasiado.  
- Graficar el **loss** es más informativo que graficar el accuracy en datasets pequeños, ya que el accuracy toma valores discretos y puede verse “digital”.  
- Este experimento demuestra la importancia de los hiperparámetros en el entrenamiento de redes neuronales, incluso en problemas simples.  
