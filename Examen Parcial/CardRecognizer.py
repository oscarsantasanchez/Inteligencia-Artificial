import cv2
import numpy as np
import os

# =========================================================
#                     CARGA DE TEMPLATES
# =========================================================
def load_templates(folder="templates"):
    """
    Carga todas las cartas de la carpeta 'templates', organizadas por palo,
    convirtiéndolas a escala de grises y redimensionándolas a 200x300 px.

    Estructura esperada:
        templates/
            hearts/
            spades/
            diamonds/
            clubs/

    Retorna:
        templates (dict): templates[palo][nombre] = imagen_gris_200x300
    """
    templates = {}  # Diccionario que guardará los templates por palo
    for suit in os.listdir(folder):  # Recorre todos los subdirectorios (palos)
        suit_path = os.path.join(folder, suit)
        if not os.path.isdir(suit_path):
            continue  # Ignorar archivos que no sean carpetas

        templates[suit] = {}  # Crear diccionario para ese palo
        for filename in os.listdir(suit_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):  # Solo imágenes
                card_name = filename.split(".")[0]  # Nombre sin extensión
                # Leer imagen en escala de grises
                img = cv2.imread(os.path.join(suit_path, filename), 0)
                if img is None:
                    print(f"[ERROR] No se pudo leer template {filename}")
                    continue
                # Redimensionar para un tamaño uniforme
                img = cv2.resize(img, (200, 300))
                templates[suit][card_name] = img  # Guardar en diccionario
    return templates

# =========================================================
#                 DETECCIÓN DE LA CARTA
# =========================================================
def extract_card(frame):
    """
    Detecta la carta más grande en el frame y corrige su perspectiva.

    Pasos:
        1. Conversión a gris.
        2. Suavizado con GaussianBlur.
        3. Detección de bordes con Canny.
        4. Detección de contornos.
        5. Selección del contorno más grande.
        6. Aproximación a polígono y verificación de 4 vértices.
        7. Warp perspective a tamaño normalizado 200x300.

    Retorna:
        Imagen warpeada de la carta o None si no se detecta.
    """
    # Convertir a escala de grises (reduce cálculo y ruido de color)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Aplicar suavizado gaussiano (kernel 5x5, sigma=0)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detectar bordes con Canny (umbral_min=60, umbral_max=150)
    edges = cv2.Canny(blur, 60, 150)
    # Buscar contornos externos simples
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None  # No hay contornos, no hay carta

    # Elegir el contorno más grande (asumimos que es la carta)
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 2000:
        return None  # Contorno demasiado pequeño → ruido

    # Aproximar contorno a polígono de 4 vértices
    peri = cv2.arcLength(cnt, True)  # Perímetro del contorno
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # Precisión 2% del perímetro

    if len(approx) != 4:
        return None  # No es rectángulo, ignorar

    pts = approx.reshape(4, 2)  # Convertir a array 4x2 (x,y)
    # Aplicar warp perspective para normalizar la carta
    return warp_card(frame, pts)

# =========================================================
#                 WARP PERSPECTIVE
# =========================================================
def warp_card(image, pts):
    """
    Corrige la perspectiva de la carta y la normaliza a 200x300 px.

    Parámetros:
        image (np.array): Imagen original
        pts (np.array): Coordenadas de las 4 esquinas de la carta

    Retorna:
        warp (np.array): Imagen de la carta normalizada
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Ordenar puntos para warping:
    # rect[0] = top-left, rect[1] = top-right, rect[2] = bottom-right, rect[3] = bottom-left
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # esquina superior izquierda
    rect[2] = pts[np.argmax(s)]  # esquina inferior derecha

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # esquina superior derecha
    rect[3] = pts[np.argmax(diff)]  # esquina inferior izquierda

    # Coordenadas destino (200x300)
    dst = np.array([
        [0, 0],
        [200, 0],
        [200, 300],
        [0, 300]
    ], dtype="float32")

    # Calcular matriz de transformación
    M = cv2.getPerspectiveTransform(rect, dst)
    # Aplicar warp
    warp = cv2.warpPerspective(image, M, (200, 300))

    return warp

# =========================================================
#                 RECONOCIMIENTO DE CARTAS
# =========================================================
def recognize_card(card_img, templates):
    """
    Compara la carta con todas las plantillas utilizando template matching.

    Parámetros:
        card_img (np.array): Imagen de la carta warpeada
        templates (dict): Diccionario con templates por palo y nombre

    Retorna:
        best_name (str): Nombre de la carta
        best_suit (str): Palo de la carta
        best_score (float): Score de similitud
    """
    # Convertir a gris y redimensionar por seguridad
    card_gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    card_gray = cv2.resize(card_gray, (200, 300))

    best_score = -1
    best_name = "???"
    best_suit = "???"

    # Comparar con todos los templates
    for suit in templates:
        for name, templ in templates[suit].items():
            # Template matching normalizado (coeficiente correlación)
            res = cv2.matchTemplate(card_gray, templ, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(res)  # Obtener máximo score

            if score > best_score:
                best_score = score
                best_name = name
                best_suit = suit

    return best_name, best_suit, best_score

# =========================================================
#                PROGRAMA PRINCIPAL
# =========================================================
def main():
    """
    Programa principal en tiempo real:
    - Captura vídeo
    - Detecta cartas al pulsar 'c'
    - Muestra resultados
    """
    print("Cargando templates...")
    templates = load_templates("templates")
    print("Templates cargados.")

    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    print("Presiona 'c' para reconocer carta | 'q' para salir")

    detected_text = ""  # Último resultado para mostrar en pantalla

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Si no se puede leer frame, salir

        # Mostrar texto de reconocimiento sobre el frame
        if detected_text != "":
            cv2.putText(frame, detected_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # Mostrar ventana de cámara
        cv2.imshow("Camara", frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break  # Salir

        if key == ord('c'):
            # Intentar extraer carta
            card = extract_card(frame)
            if card is None:
                detected_text = "No se detectó carta"
                continue

            # Reconocer carta
            name, suit, score = recognize_card(card, templates)
            detected_text = f"{name} | {suit} ({score:.2f})"

            # Mostrar carta warpeada con texto
            card_display = card.copy()
            cv2.putText(card_display, f"{name} - {suit}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
            cv2.imshow("Carta Detectada", card_display)

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

# =========================================================
#                 EJECUCIÓN PRINCIPAL
# =========================================================
if __name__ == "__main__":
    main()
