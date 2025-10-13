# **PREGUNTAS A RESPONDER**
## **3.1 Diferencia entre regresión lineal simple y múltiple**

- **Simple:** la salida depende de una sola variable independiente (ej: CO2 = f(Temp)).

- **Múltiple:** la salida depende de varias variables independientes (ej: CO2 = f(Temp, Mar, Glaciar)).


## **3.2 Ecuación del modelo obtenido. ¿Qué significa el término independiente?**
Ejemplo del PDF:

- ```𝐶𝑂2= 252.84 + 4.39 ⋅ 𝑇𝑒𝑚𝑝 + 7.21 ⋅ 𝑁𝑀𝑀 − 3.83 ⋅ 𝐺𝑙𝑎𝑐𝑖𝑎𝑟```
- ```CO2= 252.84 + 4.39 ⋅ Temp + 7.21 ⋅ NMM − 3.83 ⋅ Glaciar```

**Matemáticamente:** el término independiente es el valor esperado de y cuando todas las variables x valen 0.

**Físicamente:** sería la concentración base de CO2 si la temperatura, el nivel del mar y la masa glaciar no tuvieran variaciones (situación inicial de referencia).

## **3.3 ¿De cuántas variables de entrada depende la salida? ¿Podríamos hacerlo de una sola? ¿De qué depende?**

* Depende de 3 variables de entrada: temperatura, nivel medio del mar y masa glaciar.

* Sí, se podría hacer con una sola (ej: solo temperatura).

* Depende del objetivo y disponibilidad de datos: más variables = modelo más completo, pero también mayor riesgo de ruido.

## **3.4 ¿Qué significa que el coeficiente de la masa glaciar sea negativo?**
Significa que a mayor masa glaciar, menor concentración de CO2 (relación inversa). Al derretirse los glaciares (valor de masa más bajo), el CO2 tiende a aumentar, lo que tiene sentido en el contexto del cambio climático.

## **3.5 Interpreta los valores obtenidos de R² en entrenamiento y test.**

Entrenamiento: R² ≈ 0.98 → el modelo explica casi toda la variabilidad en los datos de entrenamiento.

Test: R² ≈ -4.09 → el modelo no generaliza nada bien; el ajuste sobre test es peor que una predicción constante.
Esto indica sobreajuste o falta de representatividad de los datos.

## **3.6 Al aumentar el número de variables de entrada, ventajas e inconvenientes.**

Ventajas:
Más información → potencialmente mayor precisión, modelo más realista.

Inconvenientes:
- Riesgo de sobreajuste.
- Más complejidad y mayor necesidad de datos.
- Posible multicolinealidad (variables redundantes).

Ejemplo: incluir deforestación podría aportar valor, pero también introducir ruido si los datos no son fiables.

## **3.7 ¿Es adecuada la regresión lineal múltiple para predecir CO2 en este caso?**

- **A favor:** es simple, interpretable y permite ver cómo influyen varios factores a la vez.

- **En contra:** los malos resultados en test (R² negativo) muestran que el modelo no generaliza bien. Esto puede deberse a que:

La relación no es estrictamente lineal. Hay factores externos no considerados (actividad industrial, uso de combustibles fósiles, etc.).

```La regresión lineal múltiple sirve como aproximación inicial, pero probablemente no sea la mejor herramienta para predecir CO2 de forma fiable en este caso.```