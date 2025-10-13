**1. Tipo de problema**  
Es un problema de **clasificación supervisada**, porque la variable objetivo (“Categoría”) indica si el paciente tiene o no diabetes, y el modelo aprende a partir de ejemplos con etiqueta.

**2. Atributos y etiqueta**  
- Los 8 atributos son medidas clínicas (ej. embarazos, glucosa, presión arterial, índice de masa corporal, etc.).  
- La etiqueta **Categoría** es una variable **categórica binaria**:  
  - 0 = No diabético  
  - 1 = Diabético  

**3. División de datos**  
Se divide en **entrenamiento y test** para evitar sobreajuste y evaluar el modelo en datos nuevos.  
Proporciones usadas: **70 % entrenamiento / 30 % test**.

**4. Valor de k en kNN**  
- **k bajo (1-2):** muy sensible al ruido → sobreajuste.  
- **k alto (50-100):** demasiado general → infraajuste.  
- Máxima precisión con **k = 7** → equilibrio entre ajuste y generalización.

**5. Matriz de confusión**  
- **VP (Verdaderos Positivos):** diabéticos correctamente detectados.  
- **VN (Verdaderos Negativos):** sanos correctamente detectados.  
- **FP (Falsos Positivos):** sanos clasificados como diabéticos (falsas alarmas).  
- **FN (Falsos Negativos):** diabéticos clasificados como sanos (riesgo grave).

**6. Métricas**  
- **Precisión (Precision):** de los predichos como diabéticos, cuántos lo son.  
- **Recall (Sensibilidad):** de los diabéticos reales, cuántos detecté.  
- **F1-score:** balance entre precisión y recall.  
Conclusión:  
- Recall bajo → se escapan muchos diabéticos.  
- Precisión baja → demasiados falsos positivos.

**7. AUC = 0.7345**  
El modelo tiene un rendimiento **aceptable** (mejor que el azar), pero aún lejos de un clasificador excelente.

**8. Test al 20 %**  
Más datos para entrenar → el modelo podría mejorar.  
Menos datos para test → evaluación menos confiable.

**9. Nuevo paciente**  
El modelo predice "diabético" o "no diabético" según vecinos.  
Confiabilidad:  
- **Mayor** si hay muchos vecinos similares en el dataset.  
- **Menor** si está aislado o cerca de pacientes contradictorios.