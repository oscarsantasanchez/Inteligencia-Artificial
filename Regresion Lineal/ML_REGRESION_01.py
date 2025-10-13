# ================================================================
#  REGRESIÓN LINEAL: TEMPERATURA vs CO2
#  Descripción: Este notebook implementa un modelo de regresión lineal
#  que analiza la relación entre la temperatura global media y la
#  concentración de CO₂ en la atmósfera.
# ================================================================

# ------------------------------------------------
# Importación de librerías necesarias
# ------------------------------------------------
import urllib.request, json                   # Para descargar y leer archivos JSON desde Internet
import matplotlib.pyplot as plt               # Para crear gráficos y visualizar los datos
import pandas as pd                           # Para manipulación de datos en tablas (DataFrames)
import numpy as np                            # Para operaciones numéricas y con arreglos/matrices
from sklearn import linear_model              # Para modelos de regresión lineal
from sklearn.metrics import mean_squared_error, r2_score  # Para evaluar el rendimiento del modelo

# ------------------------------------------------
# 1️⃣ Descarga y lectura de los datasets
# ------------------------------------------------
# URLs con los datos de temperatura y CO2, respectivamente
temp_url = "https://pkgstore.datahub.io/core/global-temp/annual_json/data/529e69dbd597709e36ce11a5d0bb7243/annual_json.json"
co2_url = "https://pkgstore.datahub.io/core/co2-ppm/co2-annmean-mlo_json/data/31185d494d1a6f6431aee8b8004b6164/co2-annmean-mlo_json.json"

# Descarga y decodificación de los datos de temperatura (JSON)
with urllib.request.urlopen(temp_url) as url:
    temp_data = json.loads(url.read().decode())

# Descarga y decodificación de los datos de CO2 (JSON)
with urllib.request.urlopen(co2_url) as url:
    co2_data = json.loads(url.read().decode())

# ------------------------------------------------
# 2️⃣ Registro y preparación de datos
# ------------------------------------------------
temp, co2, year = [], [], []                  # Inicializamos las listas para almacenar temperatura, CO2 y año

# Extraer temperaturas promedio globales SOLO del dataset "GISTEMP" (NASA)
for d in temp_data:
    if d["Source"] == "GISTEMP":             # Nos aseguramos de trabajar con datos homogéneos
        temp.append(d["Mean"])                # Guardamos la temperatura media del año (anomalía)
        year.append(d["Year"])                # Guardamos el año correspondiente

# Los datos están de más reciente a más antiguo (2016 a 1880), así que los invertimos (ascendente)
temp.reverse()
year.reverse()

# Filtramos los años y temperaturas de 1959 a 2016 (ambos inclusive)
temp = temp[1959 - 1880 : 2016 - 1880 + 1]    # Restamos para obtener el índice inicial y final
year = year[1959 - 1880 : 2016 - 1880 + 1]

# Extraemos las concentraciones de CO2 desde 1959 en adelante
for d in co2_data:
    co2.append(d["Mean"])                     # Guardamos el valor medio anual de CO2 (ppm)

# ------------------------------------------------
# 3️⃣ Visualización de los datos
# ------------------------------------------------
fig, axs = plt.subplots(2, 1, figsize=(8,6))      # Creamos una figura con dos gráficas (subplots) verticales
axs[0].plot(year, temp, label="Temperatura (anomalía)")  # Primera gráfica: temperatura vs año
axs[0].set_title("Temperatura global promedio")
axs[0].set_xlabel("Año")
axs[0].set_ylabel("Temperatura (°C)")
axs[0].grid(True)                                # Agregamos cuadrícula

axs[1].plot(year, co2, color='orange', label="CO₂ (ppm)")  # Segunda gráfica: CO2 vs año
axs[1].set_title("Concentración de CO₂ atmosférico")
axs[1].set_xlabel("Año")
axs[1].set_ylabel("CO₂ (ppm)")
axs[1].grid(True)                                # Agregamos cuadrícula
plt.tight_layout()                               # Espaciado automático para evitar superposición de etiquetas
plt.show()                                       # Mostramos las gráficas

# ------------------------------------------------
# 4️⃣ Creación del modelo de regresión lineal
# ------------------------------------------------
# Creamos un DataFrame (tabla) con dos columnas: temperatura y CO2
df = pd.DataFrame({'temp': temp, 'co2': co2})

# Definimos las variables:
# independiente (X): temperatura (feature/input)
# dependiente (y): CO2 (target/output)
X = df[['temp']]      # NOTA: Debe ser DataFrame, por eso doble corchete
y = df[['co2']]

# Se divide el conjunto de datos en entrenamiento y test
# Primeros 40 registros para entrenamiento (~70%), el resto para test (~30%)
X_train = np.array(X[:40])   # Desde el inicio hasta el elemento 39
y_train = np.array(y[:40])
X_test = np.array(X[40:])    # Desde el elemento 40 al final
y_test = np.array(y[40:])

# ------------------------------------------------
# 5️⃣ Entrenamiento del modelo
# ------------------------------------------------
regr = linear_model.LinearRegression()    # Instanciamos el modelo de regresión lineal
regr.fit(X_train, y_train)                # Ajustamos (entrenamos) el modelo con los datos

# ------------------------------------------------
# 6️⃣ Parámetros del modelo
# ------------------------------------------------
t1 = regr.coef_       # Coeficiente/s (pendiente, cuánto varía Y por cada unidad de X)
t0 = regr.intercept_  # Intercepto (valor de Y cuando X=0)
print("Pendiente (coef_):", t1)
print("Intercepto (intercept_):", t0)
print(f"Ecuación: y = {t0[0]:.3f} + {t1[0][0]:.3f} * X") # Se imprime la ecuación de la recta ajustada

# ------------------------------------------------
# 7️⃣ Evaluación del modelo
# ------------------------------------------------
# Predicciones del modelo con los datos de entrenamiento
y_pred = regr.predict(X_train)

# Cálculo de métricas de evaluación en entrenamiento
print("\n--- RESULTADOS EN ENTRENAMIENTO ---")
print("Error Cuadrático Medio (MSE): %.2f" % mean_squared_error(y_train, y_pred))
print("Coeficiente de determinación (R²): %.2f" % r2_score(y_train, y_pred))

# Predicciones del modelo con los datos de test
y_pred_test = regr.predict(X_test)
print("\n--- RESULTADOS EN TEST ---")
print("Error Cuadrático Medio (MSE): %.2f" % mean_squared_error(y_test, y_pred_test))
print("Coeficiente de determinación (R²): %.2f" % r2_score(y_test, y_pred_test))

# ------------------------------------------------
# 8️⃣ Visualización de la regresión
# ------------------------------------------------
plt.figure(figsize=(7,5))
plt.scatter(X_train, y_train, color="red", label="Entrenamiento") # Muestra datos de entrenamiento
plt.scatter(X_test, y_test, color="blue", label="Test")           # Muestra datos de test
plt.plot(X_train, regr.predict(X_train), color="black", label="Recta de regresión") # Recta del modelo
plt.xlabel("Temperatura (anomalía)")
plt.ylabel("CO₂ (ppm)")
plt.title("Regresión Lineal: Temperatura vs CO₂")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------
# 9️⃣ Predicción de nuevos valores
# ------------------------------------------------
# Ejemplo de predicción: concentración estimada de CO2 para una anomalía de 0.8°C
new_temp = np.array([[0.8]])         # Nueva entrada: temperatura = 0.8
prediccion = regr.predict(new_temp)  # Se calcula la predicción usando el modelo entrenado
print(f"\nPredicción para una anomalía de {new_temp[0][0]}°C: {prediccion[0][0]:.2f} ppm") # Se muestra el resultado
