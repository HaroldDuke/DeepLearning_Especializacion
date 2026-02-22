# Week1
# DeepLearning_Especializacion
Conceptualización sobre deep learning

# Grafico del ejercicio:
<img width="663" height="487" alt="image" src="https://github.com/user-attachments/assets/3f0cc6b6-14aa-444f-8fa3-6492815dce17" />

## 1) ¿Qué es el bias (b)?
-Es como el "nivel de exigencia" de la neurona. Si el bias es alto, la neurona se activa fácil. Si es muy negativo, se pone difícil de activar, como alguien que necesita mucho para convencerse.
## 2) ¿Qué significa "se parece a AND"?
-Básicamente es una neurona se comporta igualito a la compuerta lógica AND: solo da 1 cuando las dos entradas son 1. Si alguna es 0, el resultado es 0. Es la típica de "solo si los dos dicen sí, pasa".
## 3) ¿Por qué con b=-0.5 me salen más '1'?
-Porque al subir el bias (hacerlo menos negativo), le baja la vara a la neurona. Ahora no necesita tanta "evidencia" para activarse, entonces se prende más fácil y salen más unos.
## 4) ¿Qué pasa si b es muy negativo (ej. -2.0)?
-La neurona se vuelve súper estricta, tipo profe exigente. Necesita que ambas entradas estén en 1 y aún así apenas le alcanza. Básicamente sube tanto el umbral que casi nada la activa.
## 5) ¿Qué es z?
-Es el cálculo intermedio, lo que pasa "por dentro" antes de decidir si la neurona se activa o no. Se calcula como z = x1*w1 + x2*w2 + b. Si z sale mayor o igual a 0, la neurona dice "1" (se activa). Si z es negativo, dice "0" (no pasa nada).
## 6) ¿Dónde respondo las preguntas?
-Se usar google colab pero, use lo que tenia mas a la mano en este caso directamente en visual studio code
## 7) ¿Qué debo entregar?
-El notebook .ipynb metido en tu repo de GitHub, en este caso paso el archivo main que genere con comentarios.
## 8) Me dio diferente al compañero, ¿está mal?
-No necesariamente. Revisá que no hayas tocado los pesos w1 y w2. Si solo cambiaste el bias b, hice varias pruebas con diferentes pesos, esto se evidencia en los resultados de la consola.
## 9) ¿Esto ya es entrenar un modelo?
-Todavía no. claro, esto simplemente por el momento es pasar datos a un perceptron basico, basicamente una pequeñisima neurona validadora

![DeepLearning](https://github.com/user-attachments/assets/e8d1c274-45de-4309-989c-e3e42e300f75)

