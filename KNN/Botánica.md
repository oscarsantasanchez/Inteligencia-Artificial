**1. Objetivo del botánico**  
Clasificar una flor en una de las 3 especies de Iris a partir de sus medidas.

**2. Variables y problema**  
- Variables: longitud y ancho de sépalos, longitud y ancho de pétalos.  
- Problema: **clasificación multiclase (3 especies)**.

**3. División de datos**  
Entrenamiento y test (ej. 80/20).  
Importante: mantener **todas las clases representadas en ambos conjuntos**.

**4. Uso de k = 1**  
- Asigna la clase del vecino más cercano.  
- **Ventaja:** funciona muy bien con datos limpios y bien separados.  
- **Desventaja:** sensible al ruido y valores atípicos.

**5. Precisión = 97 %**  
Indica que el modelo clasifica casi siempre bien.  
Puede sugerir **ligero sobreajuste**, pero en Iris es habitual esa alta precisión.

**6. Identificación de especies**  
El modelo diferencia especies según medidas.  
Los **pétalos (longitud y ancho)** suelen tener mayor peso que los sépalos.

**7. Ruido o atípicos**  
kNN es sensible → el ruido puede alterar vecinos y reducir precisión.

**8. Predicción para (sépalo: 7,3 cm; pétalo: 5,1 cm)**  
El modelo clasifica en la especie con vecinos más cercanos (probablemente *Iris-virginica*).  
Interpretación: la flor se parece a los ejemplos de esa clase.

**9. Posibles mejoras**  
- Normalización de datos.  
- Ajustar k con validación cruzada.  
- Usar **ponderación por distancia**.  
- Eliminar outliers.  
- Probar reducción de dimensiones (PCA).