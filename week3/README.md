# Week 3 — Backpropagation y Funciones de Activación

## Objetivo

Implementar y validar el proceso de aprendizaje de una red neuronal mediante **backpropagation** y el uso de **funciones de activación**, evidenciando cómo la red ajusta sus parámetros (pesos y sesgos) para reducir el error durante el entrenamiento.

## Problema: XOR

Se eligió el problema **XOR** porque no es linealmente separable, lo que obliga a la red a usar una capa oculta para aprender la relación entre entradas y salidas:

| X1 | X2 | Salida esperada |
|----|----|-----------------|
| 0  | 0  | 0               |
| 0  | 1  | 1               |
| 1  | 0  | 1               |
| 1  | 1  | 0               |

Un perceptrón simple (como el de la semana 1) **no puede resolver XOR**. Se necesita al menos una capa oculta con backpropagation.

## Arquitectura de la Red

```
Entrada (2) → Capa oculta (4 neuronas) → Salida (1)
```

- **Capa oculta**: usa la función de activación configurable (sigmoid, ReLU o tanh).
- **Capa de salida**: siempre usa sigmoid para producir una probabilidad entre 0 y 1.
- **Función de pérdida**: Error Cuadrático Medio (MSE).

## Conceptos Clave Implementados

### 1. Forward Pass (Propagación hacia adelante)
Se calcula la salida de cada capa en secuencia:
- $z_1 = X \cdot W_1 + b_1$ → $a_1 = \text{activación}(z_1)$
- $z_2 = a_1 \cdot W_2 + b_2$ → $a_2 = \text{sigmoid}(z_2)$

### 2. Función de Pérdida (Loss)
Se utiliza el **MSE** para medir qué tan lejos está la predicción del valor real:

$$\text{MSE} = \frac{1}{m} \sum (y - \hat{y})^2$$

### 3. Backpropagation (Propagación del error hacia atrás)
Se calculan los gradientes de la pérdida respecto a cada peso y sesgo, propagando el error desde la salida hacia la entrada usando la **regla de la cadena**:

- Se calcula el **delta** de la capa de salida.
- Se propaga el error a la capa oculta.
- Se actualizan los parámetros con **Gradient Descent**: $W = W - \alpha \cdot \nabla W$

### 4. Funciones de Activación

| Función | Fórmula | Rango | Características |
|---------|---------|-------|-----------------|
| **Sigmoid** | $\frac{1}{1+e^{-z}}$ | (0, 1) | Suave, pero puede sufrir de gradientes que se desvanecen |
| **ReLU** | $\max(0, z)$ | [0, ∞) | Rápida, pero neuronas pueden "morir" (salida siempre 0) |
| **Tanh** | $\tanh(z)$ | (-1, 1) | Centrada en cero, mejor convergencia que sigmoid |

## Cómo Ejecutar

```bash
cd week3
python main.py
```

**Dependencias**: solo `numpy`. Opcionalmente `matplotlib` para generar el gráfico de pérdida.

```bash
pip install numpy matplotlib
```

## Qué Muestra la Ejecución

1. **Entrenamiento con 3 activaciones** (sigmoid, ReLU, tanh) durante 10,000 épocas, mostrando la evolución del loss.
2. **Predicciones finales** vs valores esperados para cada función de activación.
3. **Tabla comparativa** de loss final, precisión y convergencia.
4. **Inspección de parámetros** (pesos y sesgos) de la red entrenada.
5. **Demostración paso a paso** de una época completa: pesos antes → forward → loss → backprop → pesos después → reducción del error.
6. **Gráfico** de la evolución de la pérdida comparando las 3 funciones de activación (guardado como `loss_comparison.png`).

## Conclusiones

1. **Backpropagation funciona**: en cada época, los pesos se actualizan en dirección contraria al gradiente del error, reduciendo progresivamente la pérdida. La demostración paso a paso muestra cómo incluso una sola actualización ya reduce el error.

2. **La función de activación importa**: no todas las funciones convergen igual de rápido ni con la misma estabilidad.
   - **Tanh** tiende a converger más rápido por estar centrada en cero.
   - **Sigmoid** converge pero más lento (gradientes pequeños lejos del origen).
   - **ReLU** puede tener problemas con neuronas muertas en problemas pequeños como XOR.

3. **Sin capa oculta no hay solución**: el XOR demuestra que se necesitan capas intermedias para aprender fronteras de decisión no lineales, justificando la existencia del deep learning.

4. **El learning rate influye directamente**: un valor muy bajo hace el entrenamiento lento; uno muy alto puede impedir la convergencia. El valor de 1.0 fue elegido por funcionar bien para este problema pequeño.
