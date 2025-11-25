# Trabajo para el Examen Final: Proyecto OCR  
## Boceto General del Proyecto

---

# 1. Visión General del Proyecto

El objetivo es construir un **OCR propio**, sin usar librerías externas que ya resuelvan el reconocimiento, capaz de:

## **Funcionalidades Obligatorias**
- Reconocer texto **impreso** (tipografía estándar).
- Reconocer texto **manuscrito** (letra humana).

## **Funcionalidades Opcionales (para subir nota)**
- Identificación y recorte de imágenes.
- Detección y digitalización de tablas.
- Recorte de códigos QR / códigos de barras (sin interpretarlos).
- Detección de características (caras, ojos, manos, elementos simples).
- Características extra propuestas al profesor.

**Restricción:** No utilizar Tesseract, EasyOCR ni OCRs ya preparados.

---

# 2. Plan Técnico: Cómo Construir tu OCR Desde Cero

## **A. Preprocesado de Imagen**
- Conversión a escala de grises  
- Binarización (Otsu, Sauvola)  
- Eliminación de ruido (blur, mediana, morfológicas)  
- Normalización del tamaño  
- Corrección de inclinación (deskew)  
- Detección de líneas  
- Segmentación de caracteres  

Librerías permitidas: **OpenCV**, **NumPy**, **Pillow**.

---

## **B. Segmentación del Texto**
- Proyección horizontal → detección de líneas  
- Proyección vertical → detección de caracteres  
- Connected Components (CCLabels)  

---

## **C. Clasificación de Caracteres**
Métodos posibles:

### **1. Template Matching (recomendado para empezar)**
- Crear plantillas de cada carácter (impreso y manuscrito).
- Comparar cada carácter segmentado con las plantillas → `cv2.matchTemplate`, SSIM, distancia euclídea.

### **2. Clasificador KNN propio**
- Extraer características: HuMoments, HOG, perfiles horizontales/verticales.
- Entrenar KNN con tus muestras manuscritas + impresas.
- Guardar modelo entrenado en `.npz`.

### **3. Red neuronal simple (opcional extra)**
- MLP entrenado desde cero con TensorFlow.  
- Solo válido si el entrenamiento lo haces tú mismo.

---

## **D. Reconstrucción del Texto**
- Ordenar caracteres por coordenadas (izquierda→derecha).  
- Agrupar en líneas.  
- Exportar en:
  - `.txt`
  - `.md`
  - `.json`

---

## **E. Extras Opcionales para Subir Nota**
### **1. Detección de imágenes**
- Encontrar contornos grandes.
- Recortar elementos.
- Guardarlos como `.png`.

### **2. Detección de tablas**
- Detección de líneas (morfología).
- Identificación de celdas.
- Exportación a Markdown:

