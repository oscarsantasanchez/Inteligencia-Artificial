"""
ML_REGRESION_05 - Modelo polinómico para predicción del IBEX35 (Apertura) - Septiembre 2018
Script en Python bastante comentado que:
 - lee un Excel con los datos (por defecto 'IBEX35_Sept2018.xls')
 - busca el mejor grado polinómico (por validación cruzada, ECM)
 - ajusta el modelo final y muestra coeficientes, grado y ECM
 - plotea los datos y la curva ajustada

Requisitos:
 pip install pandas numpy scikit-learn matplotlib openpyxl

Ajustes:
 - Cambia 'ruta_excel' si tu archivo está en otra ruta.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

# -----------------------------
# Configuración / Parámetros
# -----------------------------
ruta_excel = "IBEX35_Sept2018.xls"   # Cambia aquí si el archivo está en otra carpeta
columna_fecha = None   # Si hay una columna con fecha, pon su nombre. Si no, usaremos índice/orden de filas.
columna_apertura = "Apertura"  # Nombre esperado de la columna de valores de apertura; cámbialo si tu Excel usa otro nombre
min_grado = 1
max_grado = 10
k_folds = 5
random_state = 42

# -----------------------------
# 1) Cargar datos
# -----------------------------
if not os.path.exists(ruta_excel):
    raise FileNotFoundError(f"No se encuentra el archivo Excel en la ruta '{ruta_excel}'."
                            " Asegúrate de que IBEX35_Sept2018.xls esté en el directorio actual"
                            " o cambia la variable 'ruta_excel' a la ruta correcta.")

# Leemos el Excel (intenta leer la primera hoja)
df = pd.read_excel(ruta_excel, engine="openpyxl")

# Mostrar las primeras filas para comprobar (opcional)
print("Primeras filas del archivo leído:")
print(df.head(), "\n")

# -----------------------------
# 2) Extraer X (tiempo/día) e y (apertura)
# -----------------------------
# Si el Excel tiene columna de fecha, úsala para X; si no, usa el índice ordinal (1..n)
if columna_fecha and columna_fecha in df.columns:
    # Convertir a tipo fecha si hace falta
    df[columna_fecha] = pd.to_datetime(df[columna_fecha], errors="coerce")
    # Crear una variable numérica de tiempo (por ejemplo días desde el primer día)
    df = df.dropna(subset=[columna_fecha, columna_apertura])
    X_time = (df[columna_fecha] - df[columna_fecha].min()).dt.days.values.reshape(-1, 1)
else:
    # Usar índice o una columna numérica de "día". Tomamos el índice +1 para tener días 1..n
    df = df.dropna(subset=[columna_apertura])
    X_time = (np.arange(len(df)) + 1).reshape(-1, 1)  # 1,2,3,... por cada fila (asume orden temporal)

# Extraer y
y = df[columna_apertura].values.reshape(-1, 1)

# Convertir a arrays 1D para sklearn
X = X_time.astype(float)
y = y.ravel().astype(float)

print(f"Datos cargados: {len(X)} observaciones.\n")

# -----------------------------
# 3) Buscar el mejor grado polinómico (por CV, ECM)
# -----------------------------
kf = KFold(n_splits=min(k_folds, len(X)), shuffle=True, random_state=random_state)

mejores_resultados = []
for grado in range(min_grado, max_grado + 1):
    # Construir pipeline manualmente: PolynomialFeatures -> LinearRegression
    poly = PolynomialFeatures(degree=grado, include_bias=True)  # incluye término constante
    X_poly = poly.fit_transform(X)

    model = LinearRegression()

    # cross_val_score por defecto devuelve score (R^2). Queremos ECM -> usamos neg_mean_squared_error
    scores_neg_mse = cross_val_score(model, X_poly, y, cv=kf, scoring="neg_mean_squared_error")
    mse_cv = -scores_neg_mse.mean()  # ECM medio sobre folds
    std_mse_cv = scores_neg_mse.std()  # desviación en los folds (negativa)
    mejores_resultados.append((grado, mse_cv, std_mse_cv))

# Ordenamos por ECM ascendente
mejores_resultados.sort(key=lambda t: t[1])

print("Resultados de validación cruzada (grado, ECM medio, desviación):")
for grado, mse_cv, std_cv in mejores_resultados:
    print(f" grado={grado:2d}  ECM_cv={mse_cv:.6f}  std_fold={abs(std_cv):.6f}")
print()

# Elegir el grado con menor ECM medio
grado_optimo = mejores_resultados[0][0]
print(f"Grado polinómico seleccionado (ECM mínimo en CV): {grado_optimo}\n")

# -----------------------------
# 4) Ajustar el modelo final con todo el conjunto y mostrar coeficientes
# -----------------------------
poly_final = PolynomialFeatures(degree=grado_optimo, include_bias=True)
X_poly_final = poly_final.fit_transform(X)

reg = LinearRegression()
reg.fit(X_poly_final, y)

# Predicción y ECM en todo el conjunto
y_pred = reg.predict(X_poly_final)
ecm_final = mean_squared_error(y, y_pred)

print(f"ECM del modelo final ajustado sobre TODO el conjunto: {ecm_final:.6f}\n")

# Obtener coeficientes ordenados por potencia descendente para una presentación clásica
# reg.coef_ -> coeficientes de cada término (excepto intercept); reg.intercept_ -> término constante
# PolynomialFeatures genera columnas: [1, x, x^2, ..., x^grado] -> coef ordenados ascendente en reg.coef_
coef_ascendente = np.hstack(([reg.intercept_], reg.coef_))  # ahora incluye intercept al principio
# Preparar presentación coeficiente por potencia
coef_por_potencia = {}
for i, c in enumerate(coef_ascendente):
    if i == 0:
        coef_por_potencia[0] = c  # término independiente
    else:
        coef_por_potencia[i] = c  # coeficiente de x^i

# Mostrar coeficientes en forma legible
print("Coeficientes del modelo polinómico ajustado (forma: coeficiente asociado a x^k):")
for k in sorted(coef_por_potencia.keys()):
    print(f" x^{k:2d} : {coef_por_potencia[k]: .8e}")
print()

# -----------------------------
# 5) Gráfica: puntos reales y curva polinómica ajustada
# -----------------------------
# Generamos una malla fina en el rango de X para dibujar la curva suave
x_min, x_max = X.min(), X.max()
x_plot = np.linspace(x_min, x_max, 500).reshape(-1, 1)
x_plot_poly = poly_final.transform(x_plot)
y_plot = reg.predict(x_plot_poly)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Datos reales (Apertura)", zorder=3)
plt.plot(x_plot, y_plot, label=f"Ajuste polinómico grado {grado_optimo}", linewidth=2, zorder=4)
plt.title(f"Ajuste polinómico sobre IBEX35 - Apertura (grado {grado_optimo})")
plt.xlabel("Día (ordinal)")
plt.ylabel("Valor Apertura IBEX35")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
# Guardamos la figura
plt.savefig("ajuste_ibex35.png", dpi=300)
print("Gráfica guardada como 'ajuste_ibex35.png' en el directorio actual.")
plt.show()

# -----------------------------
# 6) (Opcional) Imprimir forma explícita del polinomio (más legible)
# -----------------------------
# Mostramos el polinomio en forma legible: p(x) = a0 + a1*x + a2*x^2 + ...
terms = [f"{coef_por_potencia[0]:+.6e}"]
for k in range(1, grado_optimo + 1):
    terms.append(f"{coef_por_potencia[k]:+.6e}*x^{k}")
polinomio_str = " ".join([""] + ["".join(terms)])  # Formato simple
print("Polinomio ajustado (forma aproximada):")
print("p(x) =", polinomio_str)
