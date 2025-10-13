# **PREGUNTAS A RESPONDER**

## **4.1 Diferencias entre regresión lineal y polinómica**
- La regresión lineal ajusta una recta de la forma y = a*x + b. Solo puede modelar relaciones lineales.
- La regresión polinómica ajusta un polinomio de grado N ```(y = θ0 + θ1 x + θ2 x² + … + θN x^N)```, lo que permite capturar relaciones no lineales y más complejas.

## **4.2 ¿Para qué se usa la función ```np.poly()``` en el código?**
En realidad, en el código se usa ```np.polyfit()``` (para calcular los coeficientes) y ```np.poly1d()``` (para construir el polinomio).

- ```np.polyfit(x, y, deg)``` → ajusta un polinomio de grado deg a los datos.

- ```np.poly1d(coeffs)``` → crea un objeto polinomial que se puede imprimir y evaluar fácilmente.

## **4.3 Explica la métrica utilizada para evaluar la calidad de ajuste de cada modelo polinómico**
Se utiliza el Error Cuadrático Medio (ECM):

𝐸𝐶𝑀=1/𝑛*n∑𝑖=1(𝑦𝑖−𝑦^𝑖)2

Mide la diferencia promedio al cuadrado entre los valores reales y y los predichos ŷ. Cuanto menor es el ECM, mejor es el ajuste.

## **4.4 Por qué el modelo de grado 5 tiene menos error que los de grado 3 y 4**
Porque un polinomio de grado mayor tiene más flexibilidad para ajustarse a la forma de la función real. En este caso, grado 5 captura mejor la oscilación de ```f(x) = x*sin(x)```.

## **4.5 Explica el overfitting o sobreajuste en el contexto del ejemplo**

Si seguimos aumentando el grado del polinomio (por ejemplo, grado 15 o 20), el modelo se ajustaría perfectamente a los puntos de entrenamiento, pero perdería capacidad de generalización. Es decir, funcionaría muy bien en los datos conocidos pero mal en valores nuevos (predicciones fuera de los puntos de entrenamiento).

## **4.6 ¿Para qué se usa la función np.polyval() en el código? Diferencias con np.poly() ¿Por qué empiezan por np. ambas?**

- ```np.polyval(coeffs, x)``` → evalúa un polinomio (definido por sus coeficientes) en los valores x.

- ```np.polyfit() (no np.poly)``` → ajusta un polinomio a los datos y devuelve los coeficientes.

- Empiezan por np. porque pertenecen a la librería NumPy, abreviada como np al importarla.

## **4.7 Si siguiésemos aumentando el grado del polinomio del modelo qué efectos podrían observarse. ¿Con qué grado te quedarías tú y por qué?**

- Efecto: el modelo se ajustaría cada vez más a los datos de entrenamiento, pero caería en sobreajuste (oscilaciones muy bruscas entre puntos, pérdida de generalización).

- Elección: yo me quedaría con grado 4 o 5, porque muestran un buen equilibrio: bajo error cuadrático medio y curvas suaves que siguen bien la tendencia de f(x). Grados superiores ya no aportarían mejoras significativas y aumentarían el riesgo de overfitting.