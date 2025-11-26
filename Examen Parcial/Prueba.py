import cv2
import numpy as np
import os

# =========================================================
#            CARGA DE PLANTILLAS DE RANGO (A-10,J,Q,K)
# =========================================================
def load_rank_templates(folder="rank_templates"):
    ranks = {}
    for filename in os.listdir(folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            name = filename.split(".")[0]  # nombre tal cual: '2','3','0','J','Q','K'
            img = cv2.imread(os.path.join(folder, filename), 0)
            if img is None:
                print(f"[ERROR] No se pudo leer rank template {filename}")
                continue
            tpl = cv2.resize(img, (70, 90))
            tpl_bin = preprocess_rank_roi(tpl)
            ranks[name] = tpl_bin
    print("=== Plantillas de rango cargadas ===")
    print(f"Rangos: {sorted(list(ranks.keys()), key=lambda x: (len(x), x))}")
    return ranks

# =========================================================
#           PREPROCESADO DE ROI DE RANGO (robustez)
# =========================================================
def preprocess_rank_roi(roi):
    eq = cv2.equalizeHist(roi)
    thr = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)
    if np.mean(thr) > 127:
        thr = cv2.bitwise_not(thr)
    return thr

# =========================================================
#                 DETECCIÓN DE LA CARTA
# =========================================================
def extract_card(frame):
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

    return cnt, frame

# =========================================================
#                 WARP A VERTICAL 200x300
# =========================================================
def warp_card(image, cnt):
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)

    s = box.sum(axis=1)
    diff = np.diff(box, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = box[np.argmin(s)]
    ordered[2] = box[np.argmax(s)]
    ordered[1] = box[np.argmin(diff)]
    ordered[3] = box[np.argmax(diff)]

    dst = np.array([[0,0],[200,0],[200,300],[0,300]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered, dst)
    warp = cv2.warpPerspective(image, M, (200, 300))
    return warp

# =========================================================
#       DETECCIÓN DEL NÚMERO/FIGURA (USANDO SOLO LA ESQUINA)
# =========================================================
def recognize_rank(card_img, rank_templates):
    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (200, 300))
    roi = gray[10:100, 10:80]   # esquina superior izquierda
    roi = cv2.resize(roi, (70, 90))
    roi_bin = preprocess_rank_roi(roi)

    best_name = "???"
    best_score = -1.0

    for name, tpl_bin in rank_templates.items():
        res = cv2.matchTemplate(roi_bin, tpl_bin, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(res)
        if score > best_score:
            best_score = score
            best_name = name

    # Mapear '0' a '10' si se usa esa convención
    if best_name == "0":
        best_name = "10"

    return best_name, best_score

# =========================================================
#                PROGRAMA PRINCIPAL
# =========================================================
def main():
    rank_templates = load_rank_templates("rank_templates")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    print("Detección de carta y reconocimiento de número/figura | 'q' para salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cnt, _ = extract_card(frame)
        if cnt is not None:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

            card = warp_card(frame, cnt)
            rank, score = recognize_rank(card, rank_templates)

            card_display = card.copy()
            cv2.putText(card_display, f"{rank} ({score:.2f})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
            cv2.imshow("Carta Normalizada", card_display)

        cv2.imshow("Camara", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
