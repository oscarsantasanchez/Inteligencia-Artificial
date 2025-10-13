# **PREGUNTAS A RESPONDER**

## **1.1 ¿Cuál es el objetivo principal de la regresión lineal en este ejercicio?**

El objetivo principal de la regresión lineal es modelar la relación entre la temperatura (variable independiente X) y la concentración de CO₂ (variable dependiente y) para poder predecir valores futuros de CO₂ a partir de la temperatura.

## **1.2 ¿Qué representan las variables X e y en el código y cómo se formalizan (tipo de estructura de datos)?**

**X:** representa las anomalías de temperatura (entrada o predictor).

**y** representa la concentración de CO₂ (salida o respuesta).
Ambas se formalizan como estructuras DataFrame de Pandas (columnas individuales), y luego como arrays de NumPy para el modelo.

## **1.3 ¿Qué hace el siguiente código?**
```
regr = LinearRegression()
regr.fit(X_train, y_train)
```
Crea un modelo de regresión lineal (LinearRegression) y lo entrena (ajusta) usando los datos de entrenamiento X_train (temperaturas) y y_train (CO₂).


## **1.4 ¿Qué significan los parámetros coef_ e intercept_ del modelo?**

**coef_ →** pendiente de la recta (cuánto cambia y por cada unidad que cambia X).

**intercept_ →** valor de y cuando X = 0 (punto de corte con el eje Y).

## **1.5 Ecuación matemática del modelo de regresión obtenida. Si vuelvo a ejecutar el código, ¿varían los coeficientes de la ecuación? ¿por qué?**

Ecuación:

```
𝑦=𝑡0+𝑡1⋅𝑋
y=t0+t1⋅X
```
Por ejemplo:
```
𝑦=327.33 + 68.69 ⋅ 𝑋
y=327.33 + 68.69 ⋅ X
```

Si vuelves a ejecutar el código no cambian los coeficientes, porque los datos son los mismos y no hay aleatoriedad (no se usa train_test_split aleatorio).

## **1.6 ¿Qué hace el siguiente bloque de código y qué representa la gráfica resultante? ¿qué diferencia hay entre y_train e y_test, y por qué se separan estos dos tipos de datos?**
```
plt.scatter(X_train, y_train, color="red")
plt.scatter(X_train, y_test, color="blue")
plt.plot(X_train, regr.predict(X_train), color="black")
```
Grafica:
- **En rojo:** puntos de entrenamiento.
- **En azul:** puntos de prueba.
- **En negro:** recta del modelo.

y_train se usa para ajustar el modelo, y_test se usa para comprobar su capacidad de generalización. Se separan para evitar sobreajuste.

## **1.7 ¿Qué miden las métricas MSE y R^2 que aparecen en el código?**

- MSE (Mean Squared Error) mide el error promedio cuadrático entre valores reales y predichos.

- R² (Coeficiente de determinación) mide qué tan bien el modelo explica la variabilidad de los datos (1 = perfecto, 0 = sin relación).

## **1.8 Explica los resultados de R^2 en entrenamiento y de test.**
- Un R² alto en entrenamiento indica buen ajuste a los datos usados para entrenar.
- Si R² en test baja (o es negativo), significa mala generalización: el modelo no predice bien datos nuevos.

## **1.9 ¿Cómo se haría una predicción nueva con el modelo entrenado? Formalízalo en código con un ejemplo.**
Ejemplo:
```
nueva_temp = np.array([[1.2]])  # 1.2 ºC
pred = regr.predict(nueva_temp)
print(pred)
```

## **1.10 ¿Qué parámetros o configuraciones se podrían cambiar para mejorar el modelo?**
Para mejorar el modelo:

- Usar un conjunto de datos mayor o más reciente.

- Probar modelos no lineales o polinomiales.

- Aplicar normalización o regularización (Ridge, Lasso).

- Dividir datos aleatoriamente (train_test_split con random_state).