# Examen Parcial - IA: **Reconocimiento de cartas de Poker**

Este proyecto inplementa un sistema de reconocimiento automático de cartas de póker utilizando técnicas clásicas de vision artificial, sin redes neuronales. El sistema captura video en tiempo real, detecta automáticamente cartas en la escena y utiliza template matching para identificar su valor y su palo.

## Componentes utilizados
|**Componente**|**Características Técnicas**|**Justificación**|
|-|-|-|
|Webcam Logitech C270|Resolución HD 720p, 30fps lente fija| Suficiente para obtener imágenes nítidas de las cartas. Buena relación calidad/precio.|
|PC con Windows 11|-|OpenCV necesita procesamiento de imágenes en tiempo real, estos requisitos garantizan fluidez.|
|Iluminación artificial |Luz ambiente suave| Evita sobreas duras que dificultan la detección del entorno|

Los requisitos que se tienen para este proyecto son:
- USB libre para conectar la webcam
- Superficie plana y preferiblemente uniforme: tapete verde.
- Baraja de cartas francesa de poker

## Software
|**Software**|**Versión**|**Justificación**|
|-|-|-|
|Python 2.10+|Dependiente del sistema|Principal lenguaje utilizado para CV|
|OpenCV|4.x|Librería estándar para visión artificial clásica|
|NumPy|Última versión|Manejo eficiente de matrices e imágenes|
|PlantUML| - | Uso general para los diagramas dentro de la documentación|

**Requisitos de instalación dentro de la carpeta del proyecto**
```
pip install opencv-python numpy
```

## Hoja de ruta del desarrollo
1. Estudio del problema
   1. Requisitos de reconocimiento
   2. Restricciones del examen
2. Captura y procesamiento
   1. Conversión a gris, suavizado y Canny
3. Detección de contorno
   1. Identificación de la carta más grande
4. Corrección de perspectiva
   1. Warping a tamaño normalizado 200x300
5. Template Marching
   1. Comparación con todas las plantillas cargadas
6. Integración en tiempo real
   1. Cámara + overlay del resultado
7. Optimización y refinado
   1. Umbrales
   2. Normalización de tamaños
8. Documentación

## Solución Propuesta
### Diagrama de Decisión

<div align=center>

![Diamgrama de decision](/Examen%20Parcial/Diagramas/DiagramaDecision.svg)

</div>

### Secuenciación de operaciones sobre la imagen
<div align=center>

![Secuanciacion de imagenes](/Examen%20Parcial/Diagramas/SecuenciaOperaciones.svg)

</div>

## Organización de carpetas
```
examen_final/
│── CardRecognizer.py
│── README.md
└── templates/
     ├── Corazon/
     │     ├── A.jpg
     │     ├── 2.jpg
     │     └── ...
     ├── Pica/
     ├── Diamante/
     └── Trebol/
```

## Código Fuente
El Script del proyecto es la clase [CardRecognizer.py](/Examen%20Parcial/CardRecognizer.py).Esta clase está asociada a la carpeta [Templates](/Examen%20Parcial/templates/). En donde dentro del mismo código se encuentra comentado cada función con la acción que realiza. Lo que hay que destacar es que para poder hacer la detección tienes que pulsar la teca `C`, si no lo haces no pocederá a realizar la detección del valor y el palo.

> [!NOTE] Las imagenes de las carpetas son de las cartas completas

Se entrega un sistema completo de reconocimiento de cartas mediante visión artificial clásica, totalmente funcional y documentado acompañado de diagramas PUML para la documentación del proyecto.