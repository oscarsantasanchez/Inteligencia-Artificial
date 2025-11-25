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
    
    Además, extrae la región superior izquierda (símbolo del palo) para
    comparación de palos.

    Estructura esperada:
        templates/
            hearts/
            spades/
            diamonds/
            clubs/

    Retorna:
        templates (dict): templates[palo][nombre] = imagen_gris_200x300
        suit_symbols (dict): suit_symbols[palo] = lista de regiones del símbolo
    """
    templates = {}
    suit_symbols = {}
    
    for suit in os.listdir(folder):
        suit_path = os.path.join(folder, suit)
        if not os.path.isdir(suit_path):
            continue

        templates[suit] = {}
        suit_symbols[suit] = []
        
        for filename in os.listdir(suit_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                card_name = filename.split(".")[0]
                img = cv2.imread(os.path.join(suit_path, filename), 0)
                if img is None:
                    print(f"[ERROR] No se pudo leer template {filename}")
                    continue
                
                img = cv2.resize(img, (200, 300))
                templates[suit][card_name] = img
                
                # Extraer región del símbolo del palo (esquina superior izquierda)
                # Región aproximada: 60x90 píxeles desde la esquina
                suit_region = img[10:100, 10:70]
                suit_symbols[suit].append(suit_region)
    
    return templates, suit_symbols

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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 2000:
        return None

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    if len(approx) != 4:
        return None

    pts = approx.reshape(4, 2)
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

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    dst = np.array([
        [0, 0],
        [200, 0],
        [200, 300],
        [0, 300]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, M, (200, 300))

    return warp

# =========================================================
#                 RECONOCIMIENTO DE CARTAS
# =========================================================
def recognize_card(card_img, templates, suit_symbols):
    """
    Compara la carta con todas las plantillas utilizando template matching.
    Primero detecta el PALO usando solo la esquina superior izquierda,
    luego detecta el NÚMERO dentro de ese palo.

    Parámetros:
        card_img (np.array): Imagen de la carta warpeada
        templates (dict): Diccionario con templates por palo y nombre
        suit_symbols (dict): Diccionario con regiones de símbolos por palo

    Retorna:
        best_name (str): Nombre de la carta
        best_suit (str): Palo de la carta
        best_score (float): Score de similitud
    """
    card_gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    card_gray = cv2.resize(card_gray, (200, 300))
    
    # PASO 1: Detectar el PALO usando solo la esquina superior izquierda
    card_suit_region = card_gray[10:100, 10:70]
    
    best_suit_score = -1
    detected_suit = "???"
    
    for suit, symbol_list in suit_symbols.items():
        for symbol in symbol_list:
            # Comparar región del palo
            res = cv2.matchTemplate(card_suit_region, symbol, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(res)
            
            if score > best_suit_score:
                best_suit_score = score
                detected_suit = suit
    
    # PASO 2: Detectar el NÚMERO solo dentro del palo detectado
    best_number_score = -1
    detected_name = "???"
    
    if detected_suit in templates:
        for name, templ in templates[detected_suit].items():
            res = cv2.matchTemplate(card_gray, templ, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(res)
            
            if score > best_number_score:
                best_number_score = score
                detected_name = name
    
    return detected_name, detected_suit, best_number_score

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
    templates, suit_symbols = load_templates("templates")
    print("Templates cargados.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    print("Presiona 'c' para reconocer carta | 'q' para salir")

    detected_text = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if detected_text != "":
            cv2.putText(frame, detected_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Camara", frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

        if key == ord('c'):
            card = extract_card(frame)
            if card is None:
                detected_text = "No se detectó carta"
                continue

            name, suit, score = recognize_card(card, templates, suit_symbols)
            detected_text = f"{name} | {suit} ({score:.2f})"

            card_display = card.copy()
            cv2.putText(card_display, f"{name} - {suit}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
            cv2.imshow("Carta Detectada", card_display)

    cap.release()
    cv2.destroyAllWindows()

# =========================================================
#                 EJECUCIÓN PRINCIPAL
# =========================================================
if __name__ == "__main__":
    main()