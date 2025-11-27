# Examen Parcial - IA: **Reconocimiento de Cartas de Poker**

Este proyecto implementa un sistema de reconocimiento automático de cartas de póker utilizando **visión artificial clásica**, sin redes neuronales.  
El sistema captura vídeo en tiempo real, detecta automáticamente cartas en la escena y utiliza **template matching** junto con **detección híbrida de palos (color + forma)** para identificar **valor y palo**.

---

## 1. Componentes Utilizados

|**Componente**|**Características Técnicas**|**Justificación**|
|-|-|-|
|Webcam Logitech C270|Resolución HD 720p, 30fps lente fija|Proporciona nitidez suficiente y estabilidad para la detección en tiempo real.|
|PC con Windows 11|CPU multinúcleo, 8GB RAM mínimo|- OpenCV necesita capacidad de cómputo continua para procesar vídeo y warping de cartas.|
|Iluminación artificial|Luz ambiente suave|Evita sombras duras que pueden alterar la detección de bordes y análisis de color.|

### Requisitos físicos
- Un puerto **USB libre**  
- Superficie plana y uniforme (ideal: tapete verde)  
- Baraja francesa estándar (52 cartas)

---

## 2. Software

|**Software**|**Versión**|**Justificación**|
|-|-|-|
|Python 3.10+|Dependiente del sistema|Lenguaje principal para visión artificial clásica|
|OpenCV|4.x|Biblioteca estándar para procesamiento de imágenes|
|NumPy|Última versión|Manejo eficiente de matrices e imágenes|
|PlantUML| - | Generación de diagramas UML de la documentación|

### Instalación de dependencias
```
pip install opencv-python numpy
```

---

## 3. Hoja de Ruta del Desarrollo

1. **Estudio del problema**
   - Análisis de restricciones del examen
   - Requisitos de reconocimiento de cartas (valor y palo)
2. **Captura y preprocesado de imagen**
   - Conversión a escala de grises  
   - Suavizado con **GaussianBlur (5x5)**  
   - Detección de bordes con **Canny (60–150)**
3. **Detección de la carta principal**
   - Búsqueda del contorno más grande  
   - Filtrado por área mínima (2000 px)  
   - Aproximación poligonal a 4 vértices
4. **Corrección de perspectiva**
   - Warping a **200×300 px**  
   - Carta normalizada para comparación
5. **Clasificación del palo (híbrida)**
   - Análisis de color: rojo vs negro (HSV y binarización)  
   - Análisis de forma: circularidad, aspect ratio, solidity, distribución vertical  
   - Palo resultante: corazón, diamante, pica o trébol
6. **Detección del valor mediante Template Matching**
   - Comparación solo con plantillas del palo detectado  
   - Métrica de similitud: **cv2.TM_CCOEFF_NORMED**
7. **Integración en tiempo real**
   - Dibujo de contornos en la imagen original  
   - Superposición de texto con nombre y palo  
   - Visualización de carta normalizada en ventana aparte
8. **Optimización y refinado**
   - Ajuste de umbrales y filtros  
   - Redimensionamiento de templates a 200x300 px
9. **Documentación**
   - Diagramas PUML  
   - README y explicación técnica completa  

---

## 4. Solución Propuesta

### 4.1 Diagrama de Decisión – Clasificación del Palo
![Diagrama de decision](/Examen%20Parcial/Diagramas/DiagramaDecision.svg)

### 4.2 Secuenciación de operaciones sobre la imagen
![Diagrama de operaciones](/Examen%20Parcial/Diagramas/SecuenciaOperaciones.svg)

## 5. Organización de Carpetas
```
examen_final/
│── CardRecognizer.py
│── README.md
└── templates/
     ├── corazon/
     │     ├── A.jpg
     │     ├── 2.jpg
     │     └── ...
     ├── diamante/
     ├── pica/
     └── trebol/
```

## 6. Código Fuente

El script principal es CardRecognizer.py, que incluye:
- Carga de plantillas
- Detección híbrida de palo
- Corrección de perspectiva (Warp)
- Template Matching para valores
- Visualización en tiempo real
- Comentarios detallados explicando cada función

## 7. Instrucciones de Uso

1. Ejecutar el script principal: ``` python CardRecognizer.py ```
2. Colocar una carta frente a la webcam.
3. El sistema detecta automáticamente:
   - Contorno de la carta  
   - Valor  
   - Palo  
   - Score de coincidencia  


> **Nota:** No es necesario pulsar ninguna tecla para iniciar la detección; todo se realiza automáticamente en tiempo real.

## 8. Código
Este es el archivo principal a ejecutar, [Card Recognizer](/Examen%20Parcial/CardRecognizer.py). para ello tienes que entrar dentro de la carpeta de Examen parcial. Dentro de la carpeta [Templates](/Examen%20Parcial/templates/) encontramos las imágenes con las que el programa realiza la comparación para detectar el valor de la carta. 