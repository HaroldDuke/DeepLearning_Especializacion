#OPTIMIZACIÓN DE HIPERPARÁETROS EN REDES NEURONALES🧠

##OBJETIVO🎯::: 
El objetivo de la actividad fue identificar como los cambios de configuración del entrenamiento de una red neuronal puede afectar sus resultados. Se comparo la tasa de aprendizaje para observar como cambia la forma en la que el modelo aprende, que tan rápido mejora y qué tan estable se comporta durante el proceso.

##METODOLOGÍA: Para desarrollar la actividad se utilizó una red neuronal sencilla en Google Colab con ayuda de TensorFlow y Keras.

Se trabajó con el conjunto de datos Fashion MNIST, que contiene imágenes de prendas de vestir como zapatos, camisetas y bolsos.
Se realizaron dos pruebas con el mismo modelo, manteniendo iguales todos los demás valores, y únicamente cambiando la tasa de aprendizaje:

Configuración 1: learning rate = 0.01 Configuración 2: learning rate = 0.001
En ambos casos se utilizaron:
10 épocas batch size de 32 mismo optimizador (Adam)

Después del entrenamiento se revisaron los resultados por medio de gráficas de loss y accuracy para comparar el comportamiento del modelo.

##RESULTADOS📊: 
En la primera configuración, el modelo aprendió más rápido durante las primeras épocas, ya que el error disminuyó rápidamente.
En la segunda configuración, el aprendizaje fue un poco más lento, pero se observó un comportamiento más estable y constante.
Al final, la segunda configuración presentó mejores resultados generales, ya que el modelo logró una mejor precisión y menos variaciones en el proceso.

##JUSTIFICACIÓN DE LAS TÉCNICAS UTILIZADAS: 
Se eligió cambiar la tasa de aprendizaje porque es uno de los factores más importantes durante el entrenamiento de una red neuronal.
Este valor define qué tan grandes son los pasos que da el modelo mientras aprende:
Si el valor es alto, aprende más rápido. Si el valor es bajo, aprende de forma más controlada.
La idea de comparar ambas configuraciones fue entender cuál de las dos permite obtener mejores resultados.

##ANÁLISIS DE RESULTADOS🔍:
Con base en las gráficas, se pudo observar que cuando la tasa de aprendizaje fue más alta, el modelo avanzó rápido al inicio, pero presentó algunos cambios bruscos.
Por otro lado, cuando la tasa fue menor, el proceso fue más suave y ordenado.
Esto permitió evidenciar que no siempre aprender rápido significa aprender mejor, ya que un aprendizaje más controlado puede llevar a resultados más precisos.

##IMPACTO:
Este tipo de pruebas permite entender cómo pequeños cambios en la configuración pueden mejorar significativamente el rendimiento del modelo.
En situaciones reales, esto es importante porque ayuda a construir modelos más confiables y eficientes para resolver problemas como clasificación de imágenes, predicción de datos o automatización de tareas.

##DISCUSIÓN💬: Durante la actividad fue posible notar que la configuración del entrenamiento influye directamente en el comportamiento de la red neuronal.
Aunque una tasa de aprendizaje alta permite ver resultados rápidos, también puede hacer que el modelo no aprenda de la mejor manera.
En cambio, una tasa más baja requiere más paciencia, pero ofrece resultados más estables.
Por esto, es importante probar diferentes configuraciones antes de decidir cuál usar.

##CONCLUSIONES:

La tasa de aprendizaje influye mucho en el desempeño del modelo. Un valor alto hace que aprenda rápido, pero puede ser menos estable. Un valor más bajo mejora la estabilidad y los resultados finales. Comparar configuraciones ayuda a entender mejor el comportamiento de la red neuronal. La optimización de hiperparámetros es clave para obtener mejores resultados.


 # 🧠 Optimización de Hiperparámetros en Redes Neuronales                                                                                                                              
                                                                                                                                                                                        
  ---                                                                                                                                                                                   
                                                                                                                                                                                        
  ## 🎯 Objetivo  
                                                                                                                                                                                        
  Identificar cómo los cambios en la configuración del entrenamiento de una red neuronal                                                                                                
  afectan sus resultados. Se compararon dos tasas de aprendizaje para observar cómo
  varían la velocidad de aprendizaje, la estabilidad y el rendimiento final del modelo.                                                                                                 
                                                                                                                                                                                        
  ---                                                                                                                                                                                   
                                                                                                                                                                                        
  ## 🧪 Metodología

  Se implementó una red neuronal en **Google Colab** usando **TensorFlow** y **Keras**,                                                                                                 
  entrenada sobre el dataset **Fashion MNIST** (imágenes de prendas de vestir: zapatos,
  camisetas, bolsos, etc.).                                                                                                                                                             
                  
  Se realizaron **dos experimentos** con el mismo modelo y arquitectura, variando                                                                                                       
  únicamente la tasa de aprendizaje:
                                                                                                                                                                                        
  | Parámetro       | Configuración 1 | Configuración 2 |
  |-----------------|:--------------:|:--------------:|                                                                                                                                 
  | Learning rate   | `0.01`         | `0.001`        |
  | Épocas          | 10             | 10             |                                                                                                                                 
  | Batch size      | 32             | 32             |
  | Optimizador     | Adam           | Adam           |                                                                                                                                 
                  
  Los resultados se evaluaron mediante gráficas de **loss** y **accuracy** por época.                                                                                                   
                  
  ---                                                                                                                                                                                   
                  
  ## 📊 Resultados

  | Métrica              | Config. 1 (`lr=0.01`)        | Config. 2 (`lr=0.001`)         |                                                                                              
  |----------------------|------------------------------|-------------------------------|
  | Velocidad inicial    | Alta — error bajó rápido     | Moderada — descenso gradual   |                                                                                               
  | Estabilidad          | Variaciones bruscas          | Comportamiento suave          |                                                                                               
  | Precisión final      | Menor                        | **Mayor**                     |                                                                                               
  | Resultado general    | Aprendizaje rápido inestable | **Mejor rendimiento global**  |                                                                                               
                                                                                                                                                                                        
  ---                                                                                                                                                                                   
                                                                                                                                                                                        
  ## 🔍 Análisis de Resultados

  Las gráficas mostraron que:

  - Con `lr=0.01`, el modelo avanzó rápido en las primeras épocas pero presentó                                                                                                         
    oscilaciones visibles en la curva de pérdida.
  - Con `lr=0.001`, el proceso fue más suave y constante, convergiendo a mejores                                                                                                        
    valores de precisión.                                                                                                                                                               
                                                                                                                                                                                        
  > **Conclusión clave:** aprender rápido no siempre significa aprender mejor.                                                                                                          
  > Un aprendizaje más controlado puede producir resultados más precisos y confiables.
                                                                                                                                                                                        
  ---             

  ## ⚙️ Justificación de la Técnica                                                                                                                                                     
   
  La **tasa de aprendizaje** es uno de los hiperparámetros más influyentes en el                                                                                                        
  entrenamiento de una red neuronal. Determina el tamaño de los pasos que da el
  optimizador al ajustar los pesos del modelo:                                                                                                                                          
                                                                                                                                                                                        
  - **Valor alto →** convergencia rápida, pero riesgo de inestabilidad o divergencia.                                                                                                   
  - **Valor bajo →** convergencia más lenta, pero mayor estabilidad y precisión final.                                                                                                  
                                                                                                                                                                                        
  Comparar ambas configuraciones permite entender empíricamente cuál produce mejores                                                                                                    
  resultados para el problema dado.                                                                                                                                                     
                                                                                                                                                                                        
  ---             

  ## 💡 Impacto

  Este tipo de experimentos tiene aplicación directa en proyectos reales de Machine                                                                                                     
  Learning, ya que pequeños cambios en la configuración pueden mejorar significativamente
  el rendimiento del modelo en tareas como:                                                                                                                                             
                                                                                                                                                                                        
  - Clasificación de imágenes
  - Predicción de datos                                                                                                                                                                 
  - Automatización de tareas

  Entender cómo ajustar hiperparámetros permite construir modelos más **confiables**                                                                                                    
  y **eficientes**.
                                                                                                                                                                                        
  ---             

  ## 💬 Discusión

  La actividad evidenció que la configuración del entrenamiento influye directamente en                                                                                                 
  el comportamiento de la red neuronal. Una tasa de aprendizaje alta ofrece resultados
  visibles rápidamente, pero compromete la estabilidad. Una tasa más baja requiere                                                                                                      
  más épocas para converger, pero ofrece resultados más sólidos.
                                                                                                                                                                                        
  Por esta razón, es recomendable **probar múltiples configuraciones** antes de                                                                                                         
  seleccionar la definitiva, especialmente en proyectos donde la precisión es crítica.                                                                                                  
                                                                                                                                                                                        
  ---             
                                                                                                                                                                                        
  ## ✅ Conclusiones

  1. La tasa de aprendizaje es un hiperparámetro crítico que afecta directamente el                                                                                                     
     desempeño del modelo.
  2. Un valor alto (`0.01`) acelera el aprendizaje inicial pero introduce inestabilidad.                                                                                                
  3. Un valor bajo (`0.001`) produce un entrenamiento más estable y mejores resultados                                                                                                  
     finales.                                                                                                                                                                           
  4. Comparar configuraciones permite comprender mejor el comportamiento de la red.                                                                                                     
  5. La optimización de hiperparámetros es esencial para maximizar el rendimiento de                                                                                                    
     modelos de Machine Learning.
                                                                                                                                                                                        
  ---             
                                                                                                                                                                                        
  *Dataset: [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) —                                                                                                         
  Framework: TensorFlow/Keras — Entorno: Google Colab*
                                                                                                                                                                                        
  Los cambios principales:
  - Tabla comparativa de configuraciones y resultados, más fácil de leer que el texto plano
  - Jerarquía visual clara con headers ## y separadores ---                                                                                                                             
  - Énfasis con negrita en los puntos clave                
  - Corrección del typo: HIPERPARÁETROS → HIPERPARÁMETROS                                                                                                                               
  - Cita destacada para la conclusión principal del análisis
  - Footer con referencias al dataset y herramientas       


