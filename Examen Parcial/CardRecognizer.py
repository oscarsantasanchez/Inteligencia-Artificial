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
                suit_region = img[10:100, 10:70]
                suit_symbols[suit].append(suit_region)
    
    return templates, suit_symbols

# =========================================================
#                 WARP PERSPECTIVE
# =========================================================
def warp_card(image, pts):
    """
    Corrige la perspectiva de la carta y la normaliza a 200x300 px.
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
#                 DETECCIÓN DE LA CARTA
# =========================================================
def extract_card_with_contour(frame):
    """
    Detecta la carta más grande en el frame y corrige su perspectiva,
    devolviendo también el contorno encontrado.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None, None

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 2000:
        return None, None

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

    if len(approx) != 4:
        return None, None

    pts = approx.reshape(4, 2)
    warp = warp_card(frame, pts)
    return warp, approx  # Devuelve la carta y su contorno

# =========================================================
#                 RECONOCIMIENTO DE CARTAS
# =========================================================
def recognize_card(card_img, templates, suit_symbols):
    """
    Reconoce el valor y el palo de una carta.
    Mejorada detección de palos con binarización y escalado de templates.
    """
    card_gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    card_gray = cv2.resize(card_gray, (200, 300))  # Normalizamos tamaño

    h, w = card_gray.shape

    # --------------------------------------------------
    # DETECCIÓN DEL PALO
    # --------------------------------------------------
    # Región proporcional a la esquina superior izquierda
    card_suit_region = card_gray[int(0.03*h):int(0.33*h), int(0.03*w):int(0.35*w)]

    # Binarizamos la región para mejorar contraste
    _, card_suit_bin = cv2.threshold(card_suit_region, 127, 255, cv2.THRESH_BINARY_INV)

    best_suit_score = -1
    detected_suit = "???"

    for suit, symbol_list in suit_symbols.items():
        for symbol in symbol_list:
            # Escalar el template al tamaño de la región
            symbol_resized = cv2.resize(symbol, (card_suit_bin.shape[1], card_suit_bin.shape[0]))
            _, symbol_bin = cv2.threshold(symbol_resized, 127, 255, cv2.THRESH_BINARY_INV)

            # Comparación
            res = cv2.matchTemplate(card_suit_bin, symbol_bin, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(res)

            if score > best_suit_score:
                best_suit_score = score
                detected_suit = suit

    # --------------------------------------------------
    # DETECCIÓN DEL NÚMERO / LETRA
    # --------------------------------------------------
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
    - Detecta cartas automáticamente
    - Muestra resultados con contorno y renderización aparte
    """
    print("Cargando templates...")
    templates, suit_symbols = load_templates("templates")
    print("Templates cargados.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    print("Detección de cartas en directo. Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detectar carta
        card, contour = extract_card_with_contour(frame)

        if card is not None:
            # Reconocer carta
            name, suit, score = recognize_card(card, templates, suit_symbols)

            # Dibujar contorno en el frame original
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)

            # Mostrar texto en el frame original
            detected_text = f"{name} | {suit} ({score:.2f})"
            cv2.putText(frame, detected_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Mostrar carta detectada en ventana aparte
            card_display = card.copy()
            cv2.putText(card_display, f"{name} - {suit}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
            cv2.imshow("Carta Detectada", card_display)

        cv2.imshow("Camara", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================================================
#                 EJECUCIÓN PRINCIPAL
# =========================================================
if __name__ == "__main__":
    main()
