import os
import numpy as np
import matplotlib.pyplot as plt


def compute_metrics(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}


def _save_and_show(path):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_full_series(series, n_train, n_val, save_path=None):
    t = np.arange(len(series))
    plt.figure(figsize=(14, 4))
    plt.plot(t[:n_train], series[:n_train], label='Entrenamiento', color='steelblue')
    plt.plot(t[n_train:n_train + n_val], series[n_train:n_train + n_val],
             label='Validación', color='orange')
    plt.plot(t[n_train + n_val:], series[n_train + n_val:],
             label='Prueba', color='seagreen')
    plt.title('Serie de tiempo sintética — partición de datos')
    plt.xlabel('Paso de tiempo')
    plt.ylabel('Valor')
    plt.legend()
    plt.tight_layout()
    _save_and_show(save_path)


def plot_predictions(y_true, y_pred, title='Predicciones vs Real', save_path=None):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label='Real', alpha=0.85)
    plt.plot(y_pred, label='Predicción', alpha=0.85, linestyle='--')
    plt.title(title)
    plt.xlabel('Paso de tiempo')
    plt.ylabel('Valor (normalizado)')
    plt.legend()
    plt.tight_layout()
    _save_and_show(save_path)


def plot_training_history(histories, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, history in histories.items():
        axes[0].plot(history.history['loss'], label=f'{name} — train')
        axes[0].plot(history.history['val_loss'], label=f'{name} — val', linestyle='--')
        axes[1].plot(history.history['mae'], label=f'{name} — train')
        axes[1].plot(history.history['val_mae'], label=f'{name} — val', linestyle='--')

    for ax, title in zip(axes, ['Pérdida (MSE)', 'Error Absoluto Medio (MAE)']):
        ax.set_title(title)
        ax.set_xlabel('Época')
        ax.legend()

    plt.tight_layout()
    _save_and_show(save_path)


def print_comparison(results):
    print(f"\n{'Modelo':<12} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
    print('-' * 44)
    for name, m in results.items():
        print(f"{name:<12} {m['RMSE']:>10.6f} {m['MAE']:>10.6f} {m['R2']:>10.6f}")
    best = min(results, key=lambda k: results[k]['RMSE'])
    print(f"\nMejor modelo (menor RMSE): {best}")
