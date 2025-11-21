import cv2
import numpy as np
import os

# ------------------------------------------
# CARGA DE TEMPLATES DESDE /templates
# ------------------------------------------
def load_templates(folder="templates"):
    templates = {}
    for suit in os.listdir(folder):
        suit_path = os.path.join(folder, suit)
        if not os.path.isdir(suit_path):
            continue

        templates[suit] = {}
        for filename in os.listdir(suit_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                card_name = filename.split(".")[0]  # ejemplo: "7H"
                img = cv2.imread(os.path.join(suit_path, filename), 0)
                img = cv2.resize(img, (200, 300))  # normalizamos tamaño
                templates[suit][card_name] = img
    return templates


# ------------------------------------------
# DETECCIÓN Y CORRECCIÓN DE PERSPECTIVA
# ------------------------------------------
def extract_card(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 60, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # coger contorno más grande
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


# ------------------------------------------
# WARP PERSPECTIVE → carta vertical 200x300
# ------------------------------------------
def warp_card(image, pts):
    rect = np.zeros((4, 2), dtype="float32")

    # ordenar esquinas
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    dst = np.array([
        [0, 0],
        [200, 0],
        [200, 300],
        [0, 300]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, M, (200, 300))

    return warp


# -----------------------------------------------------
# MATCHING MEDIANTE TEMPLATE MATCHING (cv2.TM_CCOEFF_NORMED)
# -----------------------------------------------------
def recognize_card(card_img, templates):
    card_gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    card_gray = cv2.resize(card_gray, (200, 300))

    best_score = -1
    best_name = "???"
    best_suit = "???"

    for suit in templates:
        for name, templ in templates[suit].items():
            res = cv2.matchTemplate(card_gray, templ, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(res)

            if score > best_score:
                best_score = score
                best_name = name
                best_suit = suit

    return best_name, best_suit, best_score


# ============================================================
#                   PROGRAMA PRINCIPAL EN DIRECTO
# ============================================================
def main():
    print("Cargando templates...")
    templates = load_templates("templates")
    print("Templates cargados.")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    print("Presiona 'c' para reconocer carta | 'q' para salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Camara", frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

        if key == ord('c'):
            card = extract_card(frame)

            if card is None:
                print("❌ No se detectó ninguna carta.")
                continue

            name, suit, score = recognize_card(card, templates)

            print(f"✔ Carta detectada: {name} | Palo: {suit} | score={score:.3f}")

            cv2.imshow("Carta Detectada", card)

    cap.release()
    cv2.destroyAllWindows()


# Iniciar
if __name__ == "__main__":
    main()
