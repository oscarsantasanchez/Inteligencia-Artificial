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
    comparación de palos (se mantiene por compatibilidad, aunque ya no se usa
    para detectar el palo).
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
#          DETECCIÓN HÍBRIDA DE PALOS (COLOR + FORMA)
# =========================================================
def detect_suit_hybrid(card_img):
    """
    Detecta el palo usando color (rojos) y análisis de forma (negros).
    Devuelve: palo_detectado, score_confianza
    """
    card_resized = cv2.resize(card_img, (200, 300))
    h, w = card_resized.shape[:2]
    roi = card_resized[int(0.04*h):int(0.38*h), int(0.04*w):int(0.38*w)]

    # Color rojo en HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100]);   upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100]); upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                              cv2.inRange(hsv, lower_red2, upper_red2))

    # Negro por binarización
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask_black = cv2.threshold(gray_roi, 90, 255, cv2.THRESH_BINARY_INV)

    red_pixels = cv2.countNonZero(mask_red)
    black_pixels = cv2.countNonZero(mask_black)
    is_red = red_pixels > black_pixels * 1.2

    # --- ROJOS: corazón vs diamante ---
    if is_red:
        kernel = np.ones((3, 3), np.uint8)
        mask_red_clean = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_red_clean = cv2.morphologyEx(mask_red_clean, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask_red_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(main_contour)
            if area > 100:
                perimeter = cv2.arcLength(main_contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                x, y, w_box, h_box = cv2.boundingRect(main_contour)
                aspect_ratio = float(w_box) / h_box if h_box > 0 else 0

                hull = cv2.convexHull(main_contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0

                mask_bottom = mask_red_clean[int(roi.shape[0] * 0.5):, :]
                bottom_ratio = cv2.countNonZero(mask_bottom) / max(cv2.countNonZero(mask_red_clean), 1)

                corazon_score = diamante_score = 0
                # Circularidad
                if circularity > 0.6: corazon_score += 3
                elif circularity < 0.5: diamante_score += 3
                # Aspect ratio
                if 0.85 < aspect_ratio < 1.15: corazon_score += 2
                elif aspect_ratio < 0.85: diamante_score += 2
                # Solidity
                if solidity > 0.87: corazon_score += 2
                elif solidity < 0.83: diamante_score += 2
                # Distribución vertical
                if bottom_ratio > 0.45: diamante_score += 3
                else: corazon_score += 2
                # Extent
                extent = area / (w_box * h_box) if (w_box * h_box) > 0 else 0
                if extent > 0.75: corazon_score += 1
                else: diamante_score += 1

                best_suit = 'corazon' if corazon_score > diamante_score else 'diamante'
                best_score = max(corazon_score, diamante_score)
            else:
                best_suit = "diamante"
                best_score = 0.5
        else:
            best_suit = "diamante"
            best_score = 0.5

    # --- NEGROS: pica vs trébol ---
    else:
        kernel = np.ones((3, 3), np.uint8)
        mask_black_clean = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_black_clean = cv2.morphologyEx(mask_black_clean, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask_black_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(main_contour)

            if area > 100:
                perimeter = cv2.arcLength(main_contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                x, y, w_box, h_box = cv2.boundingRect(main_contour)
                aspect_ratio = float(w_box) / h_box if h_box > 0 else 0

                hull = cv2.convexHull(main_contour)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0

                roi_h = roi.shape[0]
                top_half = mask_black_clean[:roi_h // 2, :]
                top_ratio = cv2.countNonZero(top_half) / max(cv2.countNonZero(mask_black_clean), 1)

                pica_score = trebol_score = 0
                # Circularidad
                if circularity < 0.55: pica_score += 2
                else: trebol_score += 2
                # Aspect ratio
                if aspect_ratio < 0.9: pica_score += 2
                else: trebol_score += 2
                # Solidity (trébol suele tener menor)
                if solidity < 0.85: trebol_score += 2
                else: pica_score += 2
                # Distribución vertical (pica tiene más arriba)
                if top_ratio > 0.55: pica_score += 3
                else: trebol_score += 3

                best_suit = 'pica' if pica_score > trebol_score else 'trebol'
                best_score = max(pica_score, trebol_score)
            else:
                best_suit = "pica"
                best_score = 0.5
        else:
            best_suit = "pica"
            best_score = 0.5

    return best_suit, best_score

# =========================================================
#                 RECONOCIMIENTO DE CARTAS
# =========================================================
def recognize_card(card_img, templates, suit_symbols):
    """
    Reconoce el valor y el palo de una carta.
    Palo: híbrido color+forma.
    Número/letra: template matching (por palo detectado).
    """
    # Palo con lógica híbrida
    detected_suit, suit_score = detect_suit_hybrid(card_img)

    # Número/letra con templates por palo
    card_gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    card_gray = cv2.resize(card_gray, (200, 300))  # Normalizamos tamaño

    best_number_score = -1
    detected_name = "???"

    if detected_suit in templates:
        for name, templ in templates[detected_suit].items():
            res = cv2.matchTemplate(card_gray, templ, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(res)
            if score > best_number_score:
                best_number_score = score
                detected_name = name

    # Devolvemos el score del número (mantener compatibilidad con tu UI actual)
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
