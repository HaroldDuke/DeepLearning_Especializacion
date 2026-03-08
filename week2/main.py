import pandas as pd

# 1. Definición de la lógica
def neurona_perceptron(x1, x2, w1, w2, b):
    # Cálculo del puntaje z
    z = (x1 * w1) + (x2 * w2) + b
    # Regla de activación (Clasificación 0/1)
    y = 1 if z >= 0 else 0
    return z, y

# 2. Configuración de parámetros (Pesos y Sesgo)
w1, w2, b = 1.0, 0.5, -0.7  # Puedes ajustar b para ver cambios

# 3. Conjunto de pruebas (Casos controlados)
entradas = [(0,0), (0,1), (1,0), (1,1)]
resultados = []

for x1, x2 in entradas:
    z, y = neurona_perceptron(x1, x2, w1, w2, b)
    resultados.append({"X1": x1, "X2": x2, "Puntaje Z": z, "Clasificación (Salida)": y})

# 4. Mostrar resultados visibles
df = pd.DataFrame(resultados)
display(df)