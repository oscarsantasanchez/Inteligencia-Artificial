# -----------------------------------------------------------
# ML - Regresión Lineal Múltiple
# Caso: Predicción del CO2 en función de:
#       - Temperatura media global (Temp)
#       - Nivel medio del mar (NMM)
#       - Masa glaciar (Glaciar)
#
# Objetivo: aprender un modelo de regresión lineal múltiple
#           y comprobar si es capaz de explicar y predecir
#           los valores de concentración de CO2 en la atmósfera.
# -----------------------------------------------------------

# --- Importamos las librerías necesarias ---
import urllib.request, json      # Para descargar y leer datasets en formato JSON desde URLs
import matplotlib.pyplot as plt  # Para visualización de datos (gráficas)
import pandas as pd              # Para manejo de datos en forma de tablas (DataFrame)
import numpy as np               # Para cálculos numéricos y arrays
from sklearn import linear_model # Para crear el modelo de regresión lineal múltiple
from sklearn.metrics import mean_squared_error, r2_score  # Para métricas de evaluación

# --- URLs de los datasets en formato JSON ---
# Fuente: datahub.io (datasets públicos de clima y medio ambiente)
temp_url    = "https://pkgstore.datahub.io/core/global-temp/annual_json/data/529e69dbd597709e36ce11a5d0bb7243/annual_json.json"
co2_url     = "https://pkgstore.datahub.io/core/co2-ppm/co2-annmean-mlo_json/data/31185d494d1a6f6431aee8b8004b6164/co2-annmean-mlo_json.json"
sea_url     = "https://pkgstore.datahub.io/core/sea-level-rise/epa-sea-level_json/data/ac016d75688136c47a04ac70298e42ec/epa-sea-level_json.json"
glaciar_url = "https://pkgstore.datahub.io/core/glacier-mass-balance/glaciers_json/data/6270342ca6134dadf8f94221be683bc6/glaciers_json.json"

# --- Lectura de ficheros JSON desde las URLs ---
# Se descargan los datos y se convierten en estructuras de Python (listas/diccionarios)
with urllib.request.urlopen(temp_url) as url:
    temp_data = json.loads(url.read().decode())

with urllib.request.urlopen(co2_url) as url:
    co2_data = json.loads(url.read().decode())

with urllib.request.urlopen(sea_url) as url:
    sea_data = json.loads(url.read().decode())

with urllib.request.urlopen(glaciar_url) as url:
    glaciar_data = json.loads(url.read().decode())

# --- Procesamiento de los datos ---
# Inicializamos listas vacías donde guardaremos los valores de cada variable
temp, co2, sea, glaciar, year = [], [], [], [], []

# 1) Procesamos Temperatura (NASA GISTEMP) desde 1880
for i in range(len(temp_data)):
    if temp_data[i]["Source"] == "GISTEMP":
        temp.append(temp_data[i]["Mean"])   # Anomalía de temperatura media global
        year.append(temp_data[i]["Year"])   # Año correspondiente
# Los datos vienen en orden inverso (de más reciente a más antiguo), así que los invertimos
temp.reverse()
year.reverse()
# Seleccionamos el rango de años 1959-2013 (55 registros)
temp = temp[1959-1880:2013-1880+1]
year = year[1959-1880:2013-1880+1]

# 2) Procesamos CO2 desde 1959
for i in range(len(co2_data)):
    co2.append(co2_data[i]["Mean"])
# Nos quedamos solo con 1959-2013
co2 = co2[:2013-1959+1]

# 3) Procesamos Nivel medio del mar desde 1880
for i in range(len(sea_data)):
    sea.append(sea_data[i]["CSIRO Adjusted Sea Level"])
# Seleccionamos de 1959 a 2013
sea = sea[1959-1880:2013-1880+1]

# 4) Procesamos Masa glaciar desde 1945
for i in range(len(glaciar_data)):
    glaciar.append(glaciar_data[i]["Mean cumulative mass balance"])
# Seleccionamos de 1959 a 2013
glaciar = glaciar[1959-1945:2013-1945+1]

# --- Visualización de los datos originales ---
# Creamos 4 subgráficas, una por cada variable, para ver su evolución temporal
fig, axs = plt.subplots(4, 1, figsize=(8, 10))

axs[0].plot(year, temp)
axs[0].set_title("Anomalía de Temperatura (°C)")

axs[1].plot(year, co2)
axs[1].set_title("Concentración de CO2 (ppm)")

axs[2].plot(year, sea)
axs[2].set_title("Nivel Medio del Mar (mm)")

axs[3].plot(year, glaciar)
axs[3].set_title("Masa Glaciar (Gt)")

plt.tight_layout()
plt.show()

# --- Construcción de DataFrame ---
# Guardamos las variables en una tabla con Pandas para facilitar el análisis
datos = {"temp": temp, "co2": co2, "sea": sea, "glaciar": glaciar}
df = pd.DataFrame(datos, columns=["temp", "co2", "sea", "glaciar"])

# --- Definición de variables ---
X = df[["temp", "sea", "glaciar"]]  # Variables independientes (predictoras)
y = df[["co2"]]                     # Variable dependiente (respuesta)

# --- División en entrenamiento y test ---
# Usamos 70% de los datos para entrenamiento (40 registros)
# y 30% para test (15 registros)
X_train, y_train = np.array(X[:40]), np.array(y[:40])
X_test, y_test   = np.array(X[40:]), np.array(y[40:])

# --- Creación y entrenamiento del modelo de regresión lineal múltiple ---
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)  # Ajustamos el modelo a los datos de entrenamiento

# --- Predicciones ---
y_pred_train = regr.predict(X_train)  # Predicciones sobre datos de entrenamiento
y_pred_test  = regr.predict(X_test)   # Predicciones sobre datos de test

# --- Resultados del modelo ---
print("Coeficientes (pendientes):", regr.coef_)
print("Intercepto:", regr.intercept_)
print("Modelo de regresión:")
print("CO2 = %.2f + %.2f*Temp + %.2f*NMM + %.2f*Glaciar" %
      (regr.intercept_, regr.coef_[0][0], regr.coef_[0][1], regr.coef_[0][2]))

# --- Evaluación del modelo ---
print("\n--- Rendimiento en Entrenamiento ---")
print("Error cuadrático medio (ECM):", mean_squared_error(y_train, y_pred_train))
print("Coeficiente de determinación (R^2):", r2_score(y_train, y_pred_train))

print("\n--- Rendimiento en Test ---")
print("Error cuadrático medio (ECM):", mean_squared_error(y_test, y_pred_test))
print("Coeficiente de determinación (R^2):", r2_score(y_test, y_pred_test))

# --- Predicción de ejemplo ---
# Usamos valores hipotéticos de entrada:
# Temp = 0.8 °C, Nivel Mar = 0.3 mm, Masa Glaciar = -6.8 Gt
ejemplo = [[0.8, 0.3, -6.8]]
prediccion = regr.predict(ejemplo)
print(f"\nPredicción de CO2 para {ejemplo}: {prediccion}")
