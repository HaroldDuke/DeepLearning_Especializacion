import numpy as np
#Función de activación, o validación basicamente es un if else que valida si el valor es mayor a 0 o no
def step(z):
    return 1 if z > 0 else 0

#Función de la neurona
def neurona(x1, x2, w1, w2, b):
    z = x1*w1 + x2*w2 + b
    print(f"z: {z}")
    y = step(z)
    return z, y

#Asignación de Valores de entrada
w1, w2, b = 1, 0.5, -0.5
#Lista de casos de entrada para el perceptron, basicamente es una lista de tuplas que contiene los valores 
# de entrada para el perceptron
casos = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1)
]
#Ciclo for para recorrer los casos
for x1, x2 in casos:
    z, y = neurona(x1, x2, w1, w2, b)
    print(f"entrada: ({x1}, {x2}) z={z:.1f} -> salida: {y}")
