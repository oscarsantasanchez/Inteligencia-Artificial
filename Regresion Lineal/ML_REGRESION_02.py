import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- Generador de datos ---
def generador_datos_simple(beta, muestras, desviacion):
    X = np.random.random(muestras) * 100              # valores de X entre 0 y 100
    e = np.random.randn(muestras) * desviacion        # error gaussiano
    y = X * beta + e                                  # modelo lineal con ruido
    return X.reshape((muestras, 1)), y.reshape((muestras, 1))

# Parámetros
desviacion = 200
beta = 10
n = 50
X, y = generador_datos_simple(beta, n, desviacion)

# --- División en entrenamiento y test (70/30) ---
n_train = int(n * 0.7)
X_train, y_train = X[:n_train], y[:n_train]
X_test, y_test = X[n_train:], y[n_train:]

# --- Modelo de regresión lineal ---
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

# Predicciones entrenamiento y test
y_pred_train = regr.predict(X_train)
y_pred_test = regr.predict(X_test)

# --- Resultados ---
print("Coeficiente (pendiente):", regr.coef_)
print("Intercepto:", regr.intercept_)
print("Recta: y = %.2f + %.2f*X" % (regr.intercept_, regr.coef_))

print("\n--- Rendimiento Entrenamiento ---")
print("ECM:", mean_squared_error(y_train, y_pred_train))
print("R^2:", r2_score(y_train, y_pred_train))

print("\n--- Rendimiento Test ---")
print("ECM:", mean_squared_error(y_test, y_pred_test))
print("R^2:", r2_score(y_test, y_pred_test))

# --- Gráficas ---
plt.scatter(X_train, y_train, color="red", label="Train")
plt.scatter(X_test, y_test, color="blue", label="Test")
plt.plot(X_train, y_pred_train, color="black", label="Regresión")
plt.title("Regresión Lineal")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# --- Predicción ejemplo ---
X_new = np.array([[50]])
y_new = regr.predict(X_new)
print(f"\nPredicción para X=50: {y_new}")
