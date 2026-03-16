import numpy as np

np.random.seed(42)

# ============================================================
# FUNCIONES DE ACTIVACIÓN Y SUS DERIVADAS
# ============================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_deriv(a):
    return (a > 0).astype(float)

def tanh_act(z):
    return np.tanh(z)

def tanh_deriv(a):
    return 1 - a ** 2

ACTIVACIONES = {
    "sigmoid": (sigmoid, sigmoid_deriv),
    "relu":    (relu,    relu_deriv),
    "tanh":    (tanh_act, tanh_deriv),
}

# ============================================================
# FUNCIÓN DE PÉRDIDA: Error Cuadrático Medio (MSE)
# ============================================================

def mse_loss(y_real, y_pred):
    return np.mean((y_real - y_pred) ** 2)

# ============================================================
# RED NEURONAL: 2 entradas → capa oculta → 1 salida
# ============================================================

class RedNeuronal:
    """
    Arquitectura: 2 neuronas de entrada, N neuronas ocultas, 1 neurona de salida.
    Usa la activación elegida en la capa oculta y sigmoid en la salida.
    """

    def __init__(self, neuronas_ocultas=4, activacion="sigmoid", lr=0.5):
        self.lr = lr
        self.act_nombre = activacion
        self.act, self.act_d = ACTIVACIONES[activacion]

        self.W1 = np.random.randn(2, neuronas_ocultas) * 0.5
        self.b1 = np.zeros((1, neuronas_ocultas))
        self.W2 = np.random.randn(neuronas_ocultas, 1) * 0.5
        self.b2 = np.zeros((1, 1))

    # ----- Forward pass -----
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.act(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    # ----- Backward pass (Backpropagation) -----
    def backward(self, X, y, salida):
        m = X.shape[0]

        # Gradiente de la capa de salida
        dL_da2 = -(y - salida)                       # derivada de MSE
        da2_dz2 = sigmoid_deriv(salida)               # derivada de sigmoid
        delta2 = dL_da2 * da2_dz2                     # delta salida

        dW2 = (self.a1.T @ delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m

        # Gradiente de la capa oculta
        delta1 = (delta2 @ self.W2.T) * self.act_d(self.a1)

        dW1 = (X.T @ delta1) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m

        # Actualización de pesos y sesgos (Gradient Descent)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    # ----- Entrenamiento completo -----
    def entrenar(self, X, y, epocas=5000, imprimir_cada=1000):
        historial_loss = []
        for epoca in range(1, epocas + 1):
            salida = self.forward(X)
            loss = mse_loss(y, salida)
            historial_loss.append(loss)
            self.backward(X, y, salida)

            if epoca % imprimir_cada == 0 or epoca == 1:
                print(f"  Época {epoca:>5d}  |  Loss (MSE): {loss:.6f}")

        return historial_loss

    def predecir(self, X, umbral=0.5):
        salida = self.forward(X)
        return (salida >= umbral).astype(int)


# ============================================================
# DATASET: Problema XOR (no linealmente separable)
# ============================================================

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# ============================================================
# ENTRENAMIENTO Y COMPARACIÓN DE FUNCIONES DE ACTIVACIÓN
# ============================================================

EPOCAS = 10000
LR = 1.0
NEURONAS = 4

resultados = {}

print("=" * 60)
print("  RED NEURONAL DESDE CERO — BACKPROPAGATION")
print("  Problema: XOR  |  Arquitectura: 2 → 4 → 1")
print("=" * 60)

for nombre_act in ["sigmoid", "relu", "tanh"]:
    print(f"\n{'─' * 60}")
    print(f"  Activación: {nombre_act.upper()}")
    print(f"{'─' * 60}")

    np.random.seed(42)
    red = RedNeuronal(neuronas_ocultas=NEURONAS, activacion=nombre_act, lr=LR)
    historial = red.entrenar(X, y, epocas=EPOCAS, imprimir_cada=2000)

    predicciones = red.predecir(X)
    salidas_raw = red.forward(X)

    print(f"\n  Resultados finales ({nombre_act}):")
    print(f"  {'Entrada':<12} {'Esperado':<10} {'Predicho':<10} {'Salida raw':<12}")
    for i in range(len(X)):
        print(f"  {str(X[i]):<12} {y[i][0]:<10} {predicciones[i][0]:<10} {salidas_raw[i][0]:<12.4f}")

    aciertos = np.sum(predicciones == y)
    precision = aciertos / len(y) * 100
    print(f"\n  Precisión: {precision:.0f}% ({aciertos}/{len(y)})")
    print(f"  Loss final: {historial[-1]:.6f}")

    resultados[nombre_act] = {
        "historial": historial,
        "precision": precision,
        "loss_final": historial[-1],
    }

# ============================================================
# TABLA COMPARATIVA DE RESULTADOS
# ============================================================

print(f"\n{'=' * 60}")
print("  TABLA COMPARATIVA — FUNCIONES DE ACTIVACIÓN")
print(f"{'=' * 60}")
print(f"  {'Activación':<12} {'Loss final':<14} {'Precisión':<12} {'Convergió':<10}")
print(f"  {'─' * 46}")
for nombre, datos in resultados.items():
    convergio = "Sí" if datos["precision"] == 100 else "No"
    print(f"  {nombre:<12} {datos['loss_final']:<14.6f} {datos['precision']:.0f}%{'':8s} {convergio:<10}")

# ============================================================
# INSPECCIÓN DE PESOS — ÚLTIMA RED ENTRENADA (tanh)
# ============================================================

print(f"\n{'=' * 60}")
print("  INSPECCIÓN DE PARÁMETROS (última red: tanh)")
print(f"{'=' * 60}")
print(f"\n  Pesos capa oculta (W1):\n{red.W1}")
print(f"\n  Sesgos capa oculta (b1):\n{red.b1}")
print(f"\n  Pesos capa salida (W2):\n{red.W2}")
print(f"\n  Sesgos capa salida (b2):\n{red.b2}")

# ============================================================
# GRÁFICO DE PÉRDIDA DURANTE EL ENTRENAMIENTO
# ============================================================

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for nombre, datos in resultados.items():
        ax.plot(datos["historial"], label=f"{nombre} (loss final={datos['loss_final']:.4f})")

    ax.set_title("Evolución de la pérdida (MSE) durante el entrenamiento")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    import os
    ruta_grafico = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loss_comparison.png")
    fig.savefig(ruta_grafico, dpi=150)
    print(f"\n  Gráfico guardado en: {ruta_grafico}")
except ImportError:
    print("\n  [INFO] matplotlib no disponible. Instálalo con: pip install matplotlib")

# ============================================================
# SEGUIMIENTO PASO A PASO (una época con sigmoid)
# ============================================================

print(f"\n{'=' * 60}")
print("  DEMOSTRACIÓN PASO A PASO — 1 Época (Sigmoid)")
print(f"{'=' * 60}")

np.random.seed(0)
demo = RedNeuronal(neuronas_ocultas=2, activacion="sigmoid", lr=0.5)

print("\n  --- Pesos ANTES del entrenamiento ---")
print(f"  W1 = {demo.W1.flatten()}")
print(f"  b1 = {demo.b1.flatten()}")
print(f"  W2 = {demo.W2.flatten()}")
print(f"  b2 = {demo.b2.flatten()}")

salida = demo.forward(X)
loss_antes = mse_loss(y, salida)
print(f"\n  Forward pass → Salidas: {salida.flatten().round(4)}")
print(f"  Loss (MSE): {loss_antes:.6f}")

demo.backward(X, y, salida)
print("\n  --- Pesos DESPUÉS de 1 actualización (backprop) ---")
print(f"  W1 = {demo.W1.flatten().round(6)}")
print(f"  b1 = {demo.b1.flatten().round(6)}")
print(f"  W2 = {demo.W2.flatten().round(6)}")
print(f"  b2 = {demo.b2.flatten().round(6)}")

salida2 = demo.forward(X)
loss_despues = mse_loss(y, salida2)
print(f"\n  Forward pass (post-update) → Salidas: {salida2.flatten().round(4)}")
print(f"  Loss (MSE): {loss_despues:.6f}")
print(f"  Reducción del error: {((loss_antes - loss_despues) / loss_antes * 100):.2f}%")

print(f"\n{'=' * 60}")
print("  FIN DE LA EJECUCIÓN")
print(f"{'=' * 60}")
