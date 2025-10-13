# Importamos las librerías necesarias
import numpy as np
import matplotlib.pyplot as plt

# Definimos la función teórica que queremos aproximar
# f(x) = x * sin(x)
def f(x):
    """Función teórica usada para generar los puntos"""
    return x * np.sin(x)

# Generamos puntos para representar la función real en un rango de 0 a 10
x_plot = np.linspace(0, 10, 100)   # 100 puntos uniformes entre 0 y 10

# Generamos datos de entrenamiento (x,y) a partir de la función teórica
x = np.linspace(0, 10, 100)        # 100 puntos base
rng = np.random.RandomState(0)     # Semilla para reproducibilidad
rng.shuffle(x)                     # Desordenamos los puntos
x = np.sort(x[:20])                # Seleccionamos 20 puntos y los ordenamos
y = f(x)                           # Valores y reales según la función

# Configuración de colores y grosor de línea
colors = ['red', 'orange', 'green']
lw = 2

# Graficamos la función real y los puntos de entrenamiento
plt.plot(x_plot, f(x_plot), color='blue', linewidth=lw, label="Función teórica")
plt.scatter(x, y, color='navy', s=30, marker='o', label="Datos entrenamiento")
plt.title("Función teórica y puntos de entrenamiento")
plt.legend()
plt.show()

# --------------------------
# AJUSTE DE REGRESIÓN POLINÓMICA
# --------------------------
print("Ajuste de Regresión Polinómica")

# Vamos a probar con polinomios de grado 3, 4 y 5
for count, degree in enumerate([3, 4, 5]):
    # Ajustamos el polinomio de grado 'degree' a los datos
    coeffs = np.polyfit(x, y, deg=degree)  # devuelve los coeficientes
    p = np.poly1d(coeffs)                  # construye el polinomio como objeto

    print(f"\nPolinomio de grado {degree}:")
    print(p)  # imprime la ecuación del polinomio

    # Calculamos las predicciones para los puntos de entrenamiento
    y_pred = np.polyval(coeffs, x)

    # Calculamos el Error Cuadrático Medio (ECM)
    ECM = np.mean((y - y_pred) ** 2)
    print("Error cuadrático medio (ECM):", ECM)

    # Graficamos la aproximación polinómica junto a la función teórica
    y_plot = np.polyval(coeffs, x_plot)
    plt.plot(x_plot, y_plot, color=colors[count], linewidth=lw, label=f"Grado {degree}")

# Volvemos a graficar la función real y los datos
plt.plot(x_plot, f(x_plot), color='blue', linewidth=lw, label="Función teórica")
plt.scatter(x, y, color='navy', s=30, marker='o', label="Datos entrenamiento")
plt.title("Comparación Regresiones Polinómicas")
plt.legend(loc='lower left')
plt.show()

# --------------------------
# PREDICCIÓN CON EL MODELO DE GRADO 5
# --------------------------
coeffs = np.polyfit(x, y, deg=5)
y_pred = np.polyval(coeffs, 6)
print("\nPredicción con polinomio de grado 5 para X=6: y =", y_pred)
