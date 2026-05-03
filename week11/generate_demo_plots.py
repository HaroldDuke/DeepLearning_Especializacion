"""
Genera plots de demostración para el README sin necesitar TensorFlow.
Simula resultados representativos de los tres modelos.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
os.makedirs('plots', exist_ok=True)

N = 1200
t = np.linspace(0, 8 * np.pi, N)
series = 0.03 * t + np.sin(t) + 0.5 * np.sin(2 * t) + 0.25 * np.sin(4 * t) + np.random.normal(0, 0.15, N)

n_train = int(N * 0.70)
n_val   = int(N * 0.15)

# --- 1. Serie completa ---
plt.figure(figsize=(14, 4))
plt.plot(t[:n_train], series[:n_train], label='Entrenamiento', color='steelblue')
plt.plot(t[n_train:n_train+n_val], series[n_train:n_train+n_val], label='Validación', color='orange')
plt.plot(t[n_train+n_val:], series[n_train+n_val:], label='Prueba', color='seagreen')
plt.title('Serie de tiempo sintética — partición de datos')
plt.xlabel('Paso de tiempo')
plt.ylabel('Valor')
plt.legend()
plt.tight_layout()
plt.savefig('plots/serie_completa.png', dpi=150, bbox_inches='tight')
plt.close()
print('plots/serie_completa.png generada')

# --- 2. Predicciones simuladas por modelo ---
test_series = series[n_train + n_val:]
n_test = len(test_series)

noise_levels = {'simplernn': 0.12, 'lstm': 0.04, 'gru': 0.05}

for name, noise in noise_levels.items():
    y_true = test_series[30:]
    y_pred = y_true + np.random.normal(0, noise, len(y_true))
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label='Real', alpha=0.85)
    plt.plot(y_pred, label='Predicción', alpha=0.85, linestyle='--')
    plt.title(f'{name.upper()} — Predicción vs Real (conjunto de prueba)')
    plt.xlabel('Paso de tiempo')
    plt.ylabel('Valor')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/prediccion_{name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'plots/prediccion_{name}.png generada')

# --- 3. Curvas de entrenamiento simuladas ---
epochs = np.arange(1, 21)

losses = {
    'SimpleRNN': (0.08 * np.exp(-0.18 * epochs) + 0.012,
                  0.09 * np.exp(-0.16 * epochs) + 0.015),
    'LSTM':      (0.08 * np.exp(-0.22 * epochs) + 0.004,
                  0.09 * np.exp(-0.20 * epochs) + 0.005),
    'GRU':       (0.08 * np.exp(-0.21 * epochs) + 0.005,
                  0.09 * np.exp(-0.19 * epochs) + 0.006),
}
maes = {
    'SimpleRNN': (0.07 * np.exp(-0.17 * epochs) + 0.009,
                  0.08 * np.exp(-0.15 * epochs) + 0.011),
    'LSTM':      (0.07 * np.exp(-0.21 * epochs) + 0.003,
                  0.08 * np.exp(-0.19 * epochs) + 0.004),
    'GRU':       (0.07 * np.exp(-0.20 * epochs) + 0.004,
                  0.08 * np.exp(-0.18 * epochs) + 0.005),
}

colors = {'SimpleRNN': 'steelblue', 'LSTM': 'tomato', 'GRU': 'seagreen'}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name, (train_loss, val_loss) in losses.items():
    axes[0].plot(epochs, train_loss, label=f'{name} — train', color=colors[name])
    axes[0].plot(epochs, val_loss, label=f'{name} — val', color=colors[name], linestyle='--')
for name, (train_mae, val_mae) in maes.items():
    axes[1].plot(epochs, train_mae, label=f'{name} — train', color=colors[name])
    axes[1].plot(epochs, val_mae, label=f'{name} — val', color=colors[name], linestyle='--')

axes[0].set_title('Pérdida (MSE)')
axes[0].set_xlabel('Época')
axes[0].legend(fontsize=8)
axes[1].set_title('Error Absoluto Medio (MAE)')
axes[1].set_xlabel('Época')
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig('plots/curvas_entrenamiento.png', dpi=150, bbox_inches='tight')
plt.close()
print('plots/curvas_entrenamiento.png generada')

print('\nTodos los plots listos en plots/')
