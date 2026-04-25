# Informe Técnico: Convolución, Padding y Stride en Imágenes

## Introducción
La práctica tuvo como finalidad comprender de manera práctica cómo funciona la operación de convolución aplicada a imágenes digitales. Se buscó implementar la operación de forma manual en Python, sin depender de librerías de alto nivel, y aplicar filtros sobre una imagen para observar cómo cambian los resultados cuando se modifican los parámetros de padding y stride. El objetivo fue evidenciar cómo estas decisiones afectan el tamaño de salida y la información que se conserva en los mapas de características.

## Metodología
Se programó una función de convolución que recibe una imagen en escala de grises y un kernel definido por el usuario. La función incluye:

- **Padding**: añade un marco de ceros alrededor de la imagen, permitiendo que el kernel se aplique también en los bordes.
- **Stride**: define el paso con el que se mueve el kernel sobre la imagen, afectando directamente la resolución del resultado.
- **Cálculo de dimensiones**: se determinan las dimensiones de salida según el tamaño de la imagen, el kernel, el padding y el stride.

Para la práctica se utilizó un kernel de detección de bordes, ya que este tipo de filtro permite visualizar con claridad los efectos de la convolución. Se aplicó la función en tres escenarios distintos:

- Padding = 0, Stride = 1  
- Padding = 1, Stride = 1  
- Padding = 0, Stride = 2  

## Resultados

### Caso 1 (Padding = 0, Stride = 1)
La imagen resultante fue más pequeña que la original. Se observó pérdida de información en los bordes, ya que el kernel no pudo aplicarse en las zonas extremas. El mapa de características muestra bordes internos, pero incompletos.

### Caso 2 (Padding = 1, Stride = 1)
El tamaño de salida se mantuvo igual al de la imagen original. El padding permitió aplicar el kernel en toda la imagen, preservando los bordes. El resultado muestra un mapa de características más completo y fiel.

### Caso 3 (Padding = 0, Stride = 2)
La salida se redujo notablemente en tamaño. Al avanzar con pasos de dos píxeles, se perdió detalle fino y se obtuvo una representación más compacta. La estructura general de la imagen se conserva, pero con menor resolución.

## Análisis
La práctica permitió comprobar que:

- La convolución es una herramienta poderosa para extraer características relevantes de las imágenes, como bordes y texturas.
- El padding es necesario cuando se quiere evitar la pérdida de información en los bordes y mantener las dimensiones originales. Sin padding, los mapas de características tienden a ser más pequeños y menos completos.
- El stride controla la resolución del mapa de salida. Un stride mayor reduce el tamaño y simplifica la información, lo que puede ser útil para disminuir el costo computacional, pero implica sacrificar detalle.

Estos hallazgos son directamente aplicables al diseño de redes convolucionales en visión por computador, donde la elección de padding y stride determina el balance entre precisión y eficiencia.

## Impacto y Aprendizaje
Más allá de cumplir con la implementación técnica, la práctica permitió:

- Entender de manera tangible cómo las decisiones de diseño afectan la información que se conserva en un modelo.
- Valorar la importancia de la reproducibilidad: el código y los resultados pueden ser replicados fácilmente en Google Colab y compartidos en GitHub.
- Reconocer que incluso operaciones aparentemente simples, como la convolución, tienen implicaciones profundas en el rendimiento y la precisión de sistemas de visión por computador.
- Desarrollar habilidades de programación más sólidas, al implementar manualmente procesos que normalmente se delegan a librerías.
- Generar un criterio propio sobre cuándo conviene usar padding o stride, dependiendo del objetivo: preservar detalle, reducir dimensiones o ganar eficiencia.

## Conclusión
La práctica permitió comprender de manera clara y fundamentada cómo funcionan la convolución, el padding y el stride. Se obtuvo evidencia visual y técnica que demuestra el impacto de estos parámetros en el procesamiento de imágenes. El ejercicio no solo cumplió con los objetivos planteados, sino que aportó un aprendizaje significativo sobre el balance entre precisión y eficiencia en el diseño de modelos de visión por computador.
