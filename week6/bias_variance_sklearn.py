"""
Actividad 4 — Bias/Variance, Overfitting y Regularización
Dataset sintético + sklearn MLPClassifier
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ============================================================
# 1. DATASET SINTÉTICO
#    - Pocas muestras + ruido en etiquetas → facilita overfitting
# ============================================================
X, y = make_classification(
    n_samples=600,
    n_features=30,
    n_informative=10,
    n_redundant=10,
    flip_y=0.10,          # 10 % de ruido en etiquetas
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"Entrenamiento : {X_train.shape}")
print(f"Prueba        : {X_test.shape}")

# ============================================================
# 2. ARQUITECTURA COMPARTIDA
# ============================================================
CAPAS = (512, 256, 128, 64)

# ============================================================
# 3. MODELO BASE  — sin regularización (alta varianza)
# ============================================================
modelo_base = MLPClassifier(
    hidden_layer_sizes=CAPAS,
    activation="relu",
    alpha=1e-9,            # regularización L2 casi nula
    max_iter=400,
    early_stopping=False,
    random_state=42,
    verbose=False,
)
modelo_base.fit(X_train, y_train)

acc_train_base = accuracy_score(y_train, modelo_base.predict(X_train))
acc_test_base  = accuracy_score(y_test,  modelo_base.predict(X_test))

# ============================================================
# 4. MODELO REGULARIZADO — L2 + Early Stopping (menor varianza)
# ============================================================
modelo_reg = MLPClassifier(
    hidden_layer_sizes=CAPAS,
    activation="relu",
    alpha=0.15,            # penalización L2 moderada
    max_iter=400,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=42,
    verbose=False,
)
modelo_reg.fit(X_train, y_train)

acc_train_reg = accuracy_score(y_train, modelo_reg.predict(X_train))
acc_test_reg  = accuracy_score(y_test,  modelo_reg.predict(X_test))

# ============================================================
# 5. RESUMEN EN CONSOLA
# ============================================================
print("\n" + "=" * 60)
print(f"{'Métrica':<35} {'Base':>10} {'Regularizado':>13}")
print("-" * 60)
print(f"{'Accuracy entrenamiento':<35} {acc_train_base:>10.4f} {acc_train_reg:>13.4f}")
print(f"{'Accuracy prueba (test)':<35} {acc_test_base:>10.4f} {acc_test_reg:>13.4f}")
print(f"{'Brecha (overfitting gap)':<35} {acc_train_base - acc_test_base:>10.4f} {acc_train_reg - acc_test_reg:>13.4f}")
print(f"{'Épocas entrenadas':<35} {modelo_base.n_iter_:>10} {modelo_reg.n_iter_:>13}")
print("=" * 60)

print("\n--- Reporte modelo base ---")
print(classification_report(y_test, modelo_base.predict(X_test)))
print("--- Reporte modelo regularizado ---")
print(classification_report(y_test, modelo_reg.predict(X_test)))

# ============================================================
# 6. CURVAS DE APRENDIZAJE (helper)
# ============================================================
def curva_aprendizaje(modelo, X, y):
    sizes, tr, val = learning_curve(
        modelo, X, y,
        train_sizes=np.linspace(0.10, 1.0, 10),
        cv=5, scoring="accuracy", n_jobs=-1,
    )[:3]
    return sizes, tr.mean(1), val.mean(1), tr.std(1), val.std(1)


# ============================================================
# 7. CURVA DE VALIDACIÓN — efecto del alpha sobre el accuracy
# ============================================================
alphas = np.logspace(-7, 1, 18)
tr_vc, val_vc = validation_curve(
    MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=250, random_state=42),
    X_train, y_train,
    param_name="alpha", param_range=alphas,
    cv=5, scoring="accuracy", n_jobs=-1,
)

# ============================================================
# 8. FIGURA PRINCIPAL
# ============================================================
C_BASE  = "#E74C3C"   # rojo  → modelo sobreajustado
C_REG   = "#2ECC71"   # verde → modelo regularizado
C_TRAIN = "#3498DB"   # azul  → entrenamiento
C_VAL   = "#F39C12"   # naranja → validación

fig = plt.figure(figsize=(18, 15))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ---- 8.1  Curva de loss durante entrenamiento ----
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(modelo_base.loss_curve_, color=C_BASE, lw=2, label="Base (sin reg.)")
ax1.plot(modelo_reg.loss_curve_,  color=C_REG,  lw=2, label="Regularizado")
ax1.set_title("Pérdida durante entrenamiento", fontweight="bold")
ax1.set_xlabel("Época")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

# ---- 8.2  Accuracy comparativo (barras) ----
ax2 = fig.add_subplot(gs[0, 1])
labels   = ["Base\n(sin reg.)", "Regularizado"]
tr_vals  = [acc_train_base, acc_train_reg]
te_vals  = [acc_test_base,  acc_test_reg]
x = np.arange(2)
w = 0.35
b1 = ax2.bar(x - w/2, tr_vals, w, label="Entrenamiento", color=C_TRAIN, alpha=0.85)
b2 = ax2.bar(x + w/2, te_vals, w, label="Prueba",        color=C_VAL,   alpha=0.85)
ax2.set_title("Accuracy: Entrenamiento vs Prueba", fontweight="bold")
ax2.set_ylabel("Accuracy")
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_ylim(0.5, 1.08)
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")
for b in list(b1) + list(b2):
    ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
             f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=9)

# ---- 8.3  Brecha de generalización ----
ax3 = fig.add_subplot(gs[0, 2])
brechas = [acc_train_base - acc_test_base, acc_train_reg - acc_test_reg]
bars = ax3.bar(labels, brechas, color=[C_BASE, C_REG], alpha=0.85, edgecolor="black")
ax3.set_title("Brecha de Generalización\n(Train − Test Accuracy)", fontweight="bold")
ax3.set_ylabel("Brecha")
ax3.grid(True, alpha=0.3, axis="y")
for b in bars:
    ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 0.001,
             f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

# ---- 8.4  Curva de aprendizaje — Modelo base ----
ax4 = fig.add_subplot(gs[1, 0:2])
clf_base_lc = MLPClassifier(hidden_layer_sizes=CAPAS, alpha=1e-9, max_iter=250, random_state=42)
ts, tr_m, vl_m, tr_s, vl_s = curva_aprendizaje(clf_base_lc, X_train, y_train)
ax4.plot(ts, tr_m, "o-",  color=C_TRAIN, lw=2, label="Train")
ax4.fill_between(ts, tr_m - tr_s, tr_m + tr_s, alpha=0.12, color=C_TRAIN)
ax4.plot(ts, vl_m, "o--", color=C_BASE,  lw=2, label="Validación cruzada")
ax4.fill_between(ts, vl_m - vl_s, vl_m + vl_s, alpha=0.12, color=C_BASE)
ax4.annotate(
    "Alta brecha\n= Alta Varianza\n(Overfitting)",
    xy=(ts[-1], (tr_m[-1] + vl_m[-1]) / 2),
    xytext=(ts[4], min(vl_m) - 0.03),
    arrowprops=dict(arrowstyle="->", color="red"),
    fontsize=9, color="red", ha="center",
)
ax4.set_title("Curva de Aprendizaje — Modelo Base\n(Alta Varianza / Overfitting)", fontweight="bold")
ax4.set_xlabel("Tamaño del conjunto de entrenamiento")
ax4.set_ylabel("Accuracy")
ax4.legend()
ax4.grid(True, alpha=0.3)

# ---- 8.5  Curva de aprendizaje — Modelo regularizado ----
ax5 = fig.add_subplot(gs[1, 2])
clf_reg_lc = MLPClassifier(hidden_layer_sizes=CAPAS, alpha=0.15, max_iter=250,
                            early_stopping=True, random_state=42)
ts_r, tr_mr, vl_mr, tr_sr, vl_sr = curva_aprendizaje(clf_reg_lc, X_train, y_train)
ax5.plot(ts_r, tr_mr, "o-",  color=C_TRAIN, lw=2, label="Train")
ax5.fill_between(ts_r, tr_mr - tr_sr, tr_mr + tr_sr, alpha=0.12, color=C_TRAIN)
ax5.plot(ts_r, vl_mr, "o--", color=C_REG,   lw=2, label="Validación cruzada")
ax5.fill_between(ts_r, vl_mr - vl_sr, vl_mr + vl_sr, alpha=0.12, color=C_REG)
ax5.set_title("Curva de Aprendizaje\nModelo Regularizado", fontweight="bold")
ax5.set_xlabel("Tamaño del entrenamiento")
ax5.set_ylabel("Accuracy")
ax5.legend()
ax5.grid(True, alpha=0.3)

# ---- 8.6  Curva de validación: efecto del alpha ----
ax6 = fig.add_subplot(gs[2, :])
tr_vc_m   = tr_vc.mean(1)
val_vc_m  = val_vc.mean(1)
tr_vc_s   = tr_vc.std(1)
val_vc_s  = val_vc.std(1)

ax6.semilogx(alphas, tr_vc_m,  "o-",  color=C_TRAIN, lw=2, label="Train accuracy")
ax6.fill_between(alphas, tr_vc_m - tr_vc_s, tr_vc_m + tr_vc_s, alpha=0.12, color=C_TRAIN)
ax6.semilogx(alphas, val_vc_m, "o--", color=C_VAL,   lw=2, label="Validación accuracy")
ax6.fill_between(alphas, val_vc_m - val_vc_s, val_vc_m + val_vc_s, alpha=0.12, color=C_VAL)

best_idx = val_vc_m.argmax()
ax6.axvline(alphas[best_idx], color="green", ls="--", lw=1.8,
            label=f"Mejor α ≈ {alphas[best_idx]:.1e}")
ax6.axvspan(alphas[0],   alphas[4],  alpha=0.07, color="red",  label="Zona overfitting (alta varianza)")
ax6.axvspan(alphas[-4],  alphas[-1], alpha=0.07, color="blue", label="Zona underfitting (alto sesgo)")

ax6.text(alphas[1],    val_vc_m[0]  - 0.025, "Alta Varianza\n(Overfitting)",  ha="center", color="red",  fontsize=9, fontweight="bold")
ax6.text(alphas[-2],   val_vc_m[-1] - 0.025, "Alto Sesgo\n(Underfitting)",    ha="center", color="blue", fontsize=9, fontweight="bold")

ax6.set_title("Curva de Validación: Diagnóstico Bias-Variance\n"
              "¿Cómo afecta la fuerza de regularización (α) al accuracy?",
              fontweight="bold", fontsize=12)
ax6.set_xlabel("Alpha — escala logarítmica (fuerza de regularización L2)")
ax6.set_ylabel("Accuracy")
ax6.legend(loc="lower left", fontsize=9)
ax6.grid(True, alpha=0.3)

fig.suptitle(
    "Análisis Bias-Variance: Sobreajuste vs Regularización en Redes Neuronales\n"
    "Dataset sintético · sklearn MLPClassifier",
    fontsize=14, fontweight="bold", y=1.01,
)

plt.savefig("bias_variance_sklearn.png", dpi=150, bbox_inches="tight")
print("\nGráfica guardada como 'bias_variance_sklearn.png'")
plt.show()
