# **PREGUNTAS A RESPONDER**

## **2.1 ¿Qué propósito tiene la función generador_datos_simple()?**
Genera datos sintéticos para simular un problema de regresión lineal. Devuelve pares (X, y) donde ```y = X*beta + error```. Sirve para entrenar y evaluar el modelo sin necesitar un dataset real.

## **2.2 ¿Por qué se introduce un término de error aleatorio en la generación de datos?**
Para imitar la variabilidad del mundo real. En la práctica, los datos nunca siguen exactamente una línea recta; siempre hay ruido o incertidumbre.


## **2.3 ¿Qué papel tien el parámetro beta en la simulación?**

``` beta ``` es la pendiente de la recta real (el coeficiente verdadero). Controla la relación entre X e y:

- Si es alto → y crece más rápido respecto a X.
- Si es bajo → la relación es más débil.

## **2.4 ¿Por qué se realiza la división de los datos 70% / 30% en entrenamiento y test? ¿harías otra división? ¿en función de qué se cogen esos porcentajes?**

Separa los datos para:

- Entrenar el modelo (70%).
- Evaluarlo con datos no vistos (30%).

Otros porcentajes comunes: 80/20 o 60/40, dependiendo del tamaño de los datos. Cuantos menos datos tengas, más conviene dar más al entrenamiento.

## **2.5 ¿Qué información proporcionan los atributos coef_ e intercept_ después del entrenamiento? Semejanzas y diferencias respecto del código del ej01.**

- ```coef_:``` la pendiente estimada por el modelo (aproxima a beta).
- ```intercept_:``` el valor de y cuando X=0.

En ej01 (regresión simple), la interpretación es la misma, solo cambia el dataset.

## **2.6 Cuánto vale R^2. Interprétalo y compáralo con el ej01.**

En el ejemplo del PDF:

- Entrenamiento → R² = 0.70
- Test → R² = 0.81

Interpretación: mide qué proporción de la variabilidad de y explica el modelo (1 = ajuste perfecto). En comparación con el ej01, puede ser mayor o menor según el ruido. Si R² es bajo, significa que el error aleatorio domina.

## **2.7 ¿Por qué son diferentes los valores de R^2 del test y del entrenamiento? ¿Qué valores desearíamos tener en ellos?**

Porque se evalúan en conjuntos distintos de datos.

- En entrenamiento, el modelo ajusta lo que ya vio.
- En test, mide la capacidad de generalizar.

Lo ideal sería que ambos sean altos y similares (cercanos a 1). Si hay mucha diferencia → posible sobreajuste o subajuste.

## **2.8 ¿Qué pasaría si aumentamos el parámetro desviacion en el generador de datos? ¿para qué querríamos hacer esto?**

Más desviación = más ruido = los puntos se dispersan más.

- Consecuencia → peor ajuste, menor R².
- Útil para probar la robustez del modelo frente a ruido.

## **2.9 ¿Por qué el código hace reshape((muestras,1)) al generar X e y?**

Porque scikit-learn espera entradas como matrices 2D ```((n_samples, n_features))```.
Si no se hace, se obtendría un array 1D que da error en ```.fit()```.

## **2.10 Si yo hago X=50, ¿qué significaría respecto al ejemplo y al modelo calculado?**

Significa que estamos usando el modelo aprendido para predecir el valor de y cuando la variable independiente X = 50. En el ejemplo, da un valor alrededor de 461.91, según la recta calculada