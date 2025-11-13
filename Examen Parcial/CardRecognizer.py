"""
card_recognition_c270.py
Reconocimiento clásico de cartas de póker sobre tapete verde, optimizado para Logitech C270.

MODALIDAD entregada: Modo combinado (live + foto) — No guarda resultados (solo muestra en pantalla).
Controles del teclado (ventana principal):
    - 'c' : capturar un frame y procesarlo (modo foto)
    - 'p' : pausar/reanudar (en live)
    - 'q' : salir y cerrar
    - 'h' : mostrar/ocultar ayuda en pantalla

REQUISITOS:
    pip install opencv-python numpy

PLANTILLAS:
    Crear carpeta `templates/` con archivos:
      rank_A.png, rank_2.png, ..., rank_10.png, rank_J.png, rank_Q.png, rank_K.png
      suit_spades.png, suit_hearts.png, suit_clubs.png, suit_diamonds.png
    (plantillas binarias o en gris de los símbolos, recortadas de las esquinas de cartas).

NOTAS:
 - No usa redes neuronales ni clasificadores entrenados.
 - Parametrizable: si tu tapete es distinto, ejecuta la calibración al inicio.
"""

import cv2
import numpy as np
import os
import time

# ----------------------------
# Parámetros iniciales (ajustables)
# ----------------------------
CAM_DEVICE_ID = 0
CAP_WIDTH = 1280
CAP_HEIGHT = 720
FPS = 30

# Tamaño del warp para cada carta (proporción 2:3)
CARD_WIDTH = 200
CARD_HEIGHT = 300

# Valores por defecto para el verde del tapete (se sobreescriben tras calibración)
GREEN_LOWER_DEFAULT = np.array([35, 40, 40], dtype=np.int32)
GREEN_UPPER_DEFAULT = np.array([90, 255, 255], dtype=np.int32)

# MIN_CARD_AREA por defecto (en px), para 1280x720 un valor inicial razonable
MIN_CARD_AREA_DEFAULT = 8000

# Umbrales para matching
MATCH_RANK_THRESH = 0.55
MATCH_SUIT_THRESH = 0.50

# Carpeta de plantillas
TEMPLATES_FOLDER = "templates"

# ----------------------------
# Utilidades: cámara y calibración
# ----------------------------
def init_camera(device_id=CAM_DEVICE_ID, width=CAP_WIDTH, height=CAP_HEIGHT):
    # Intenta usar CAP_DSHOW en Windows para menor latencia; si falla, usa default
    try:
        cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
    except:
        cap = cv2.VideoCapture(device_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    # Intento de estabilizar exposición si el backend lo permite (no garantizado)
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    except:
        pass
    return cap

def calibrate_green_range(cap, sample_frames=8, debug=False):
    """Estima rango HSV del verde tomando varios frames del tapete (sin cartas)."""
    hsv_vals = []
    # Ignorar algunos frames iniciales para estabilizar
    for _ in range(3):
        cap.read()
    for i in range(sample_frames):
        ret, frame = cap.read()
        if not ret:
            continue
        blur = cv2.GaussianBlur(frame, (5,5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        hsv_vals.append(hsv.reshape(-1,3))
    if len(hsv_vals) == 0:
        return GREEN_LOWER_DEFAULT.copy(), GREEN_UPPER_DEFAULT.copy()
    allvals = np.vstack(hsv_vals)
    lower = np.percentile(allvals, 1, axis=0)
    upper = np.percentile(allvals, 99, axis=0)
    lower = np.maximum(lower - np.array([8, 40, 40]), [0,0,0]).astype(int)
    upper = np.minimum(upper + np.array([8, 40, 40]), [179,255,255]).astype(int)
    if debug:
        print("Calibración verde -> lower:", lower, "upper:", upper)
    return lower, upper

def auto_estimate_min_area(mask, scale_factor=0.02, debug=False):
    """Estimación heurística de MIN_CARD_AREA basada en el contorno mayor."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return MIN_CARD_AREA_DEFAULT
    max_area = max(cv2.contourArea(c) for c in cnts)
    est = int(max_area * scale_factor)
    est = max(3000, min(est, 50000))
    if debug:
        print("max_area fondo:", max_area, "MIN_CARD_AREA estimado:", est)
    return est

# ----------------------------
# Procesamiento de imagen (segmentación + detección de cartas)
# ----------------------------
def segment_green(image, lower_hsv, upper_hsv):
    """Devuelve máscara limpia de objetos no-verdes (cartas, etc.)."""
    blur = cv2.GaussianBlur(image, (5,5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask_objects = cv2.bitwise_not(mask_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    clean = cv2.morphologyEx(mask_objects, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)
    return clean

def find_card_contours(mask, min_area):
    """Encuentra contornos candidatos a ser cartas (devuelve lista de quads de 4 puntos)."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            candidates.append(approx.reshape(4,2))
        else:
            # fallback: bounding rect
            x,y,w,h = cv2.boundingRect(c)
            ratio = w / float(h) if h>0 else 0
            if 0.4 < ratio < 2.5:
                quad = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])
                candidates.append(quad)
    return candidates

def order_points(pts):
    """Ordena 4 puntos en (tl, tr, br, bl)."""
    pts = pts.reshape(4,2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype='float32')

def four_point_transform(image, pts, w=CARD_WIDTH, h=CARD_HEIGHT):
    pts2 = order_points(pts)
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype='float32')
    M = cv2.getPerspectiveTransform(pts2, dst)
    warp = cv2.warpPerspective(image, M, (w,h))
    return warp

# ----------------------------
# Plantillas: carga y matching
# ----------------------------
def load_templates(templates_folder):
    ranks = {}
    suits = {}
    if not os.path.isdir(templates_folder):
        print(f"[WARN] templates folder '{templates_folder}' no encontrado. Crea la carpeta y añade plantillas.")
        return ranks, suits
    for fname in os.listdir(templates_folder):
        path = os.path.join(templates_folder, fname)
        key = os.path.splitext(fname)[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if key.startswith('rank_'):
            ranks[key.replace('rank_', '').upper()] = img_bin
        elif key.startswith('suit_'):
            suits[key.replace('suit_', '').lower()] = img_bin
    return ranks, suits

def match_template_scores(roi_gray, templates):
    best_key, best_score = None, -1.0
    if roi_gray is None or roi_gray.size == 0:
        return None, -1.0
    for key, tpl in templates.items():
        th, tw = tpl.shape
        rh, rw = roi_gray.shape
        # Escalar plantilla si mayor que ROI
        if th > rh or tw > rw:
            scale = min(rh/th, rw/tw) * 0.95
            if scale <= 0:
                continue
            tpl_resized = cv2.resize(tpl, (int(tw*scale), int(th*scale)))
        else:
            tpl_resized = tpl
        try:
            res = cv2.matchTemplate(roi_gray, tpl_resized, cv2.TM_CCOEFF_NORMED)
            _, maxv, _, _ = cv2.minMaxLoc(res)
        except:
            continue
        if maxv > best_score:
            best_score = maxv
            best_key = key
    return best_key, best_score

# ----------------------------
# Reconocimiento de rank y suit por heurísticas
# ----------------------------
def count_pips(card_gray):
    h,w = card_gray.shape
    cx1, cy1 = int(0.2*w), int(0.2*h)
    cx2, cy2 = int(0.8*w), int(0.8*h)
    roi = card_gray[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        return None
    _, thr = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = cv2.bitwise_not(thr)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pip_count = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 50:
            continue
        if area > 0.001 * (roi.shape[0]*roi.shape[1]) and area < 0.05 * (roi.shape[0]*roi.shape[1]):
            peri = cv2.arcLength(c, True)
            if peri == 0:
                continue
            circularity = 4*np.pi*area / (peri*peri)
            if circularity > 0.15:
                pip_count += 1
    if pip_count >= 1 and pip_count <= 10:
        return pip_count
    return None

def extract_rank_suit(card_bgr, ranks_tpl, suits_tpl):
    card_gray = cv2.cvtColor(card_bgr, cv2.COLOR_BGR2GRAY)
    h, w = card_gray.shape

    # ROIs: Esquina superior izquierda
    rank_roi = card_gray[int(0.05*h):int(0.20*h), int(0.05*w):int(0.25*w)]
    suit_roi = card_gray[int(0.20*h):int(0.35*h), int(0.05*w):int(0.25*w)]

    # ROIs invertidas (por si la carta está girada 180°)
    rank_roi_b = card_gray[int(0.80*h):int(0.95*h), int(0.75*w):int(0.95*w)]
    suit_roi_b = card_gray[int(0.65*h):int(0.80*h), int(0.75*w):int(0.95*w)]

    rank, suit, confidence = None, None, 0.0

    # --- Procesamiento helper ---
    def process_region(roi, templates, threshold):
        if roi is None or roi.size == 0:
            return None, -1
        _, thr = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        key, score = match_template_scores(thr, templates)
        if score >= threshold:
            return key, score
        return None, score

    # --- Matching de rank ---
    rk1, rk1_score = process_region(rank_roi, ranks_tpl, MATCH_RANK_THRESH)
    rk2, rk2_score = process_region(rank_roi_b, ranks_tpl, MATCH_RANK_THRESH)
    if rk1_score >= rk2_score and rk1:
        rank = rk1.upper()
        confidence = rk1_score
    elif rk2:
        rank = rk2.upper()
        confidence = rk2_score

    # --- Matching de suit ---
    su1, su1_score = process_region(suit_roi, suits_tpl, MATCH_SUIT_THRESH)
    su2, su2_score = process_region(suit_roi_b, suits_tpl, MATCH_SUIT_THRESH)
    if su1_score >= su2_score and su1:
        suit = su1.lower()
        confidence = max(confidence, su1_score)
    elif su2:
        suit = su2.lower()
        confidence = max(confidence, su2_score)

    return rank, suit, confidence


# ----------------------------
# Pipeline principal de procesamiento de una imagen (frame)
# ----------------------------
def process_frame(frame, lower_hsv, upper_hsv, min_card_area, ranks_tpl, suits_tpl, draw=True):
    orig = frame.copy()
    mask = segment_green(frame, lower_hsv, upper_hsv)
    candidates = find_card_contours(mask, min_card_area)
    results = []
    for quad in candidates:
        try:
            warp = four_point_transform(orig, quad, CARD_WIDTH, CARD_HEIGHT)
        except Exception:
            continue
        rank, suit, conf = extract_rank_suit(warp, ranks_tpl, suits_tpl)
        x,y,w,h = cv2.boundingRect(quad.astype(int))
        results.append({'bbox':(x,y,w,h), 'quad':quad.astype(int), 'rank':rank, 'suit':suit, 'conf':conf, 'warp':warp})
        if draw:
            label = f"{rank or '?'}{(' ' + suit) if suit else ''} ({conf:.2f})"
            cv2.drawContours(orig, [quad.astype(int)], -1, (0,255,0), 2)
            cv2.putText(orig, label, (x, max(15, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return orig, mask, results

# ----------------------------
# UI / Main loop
# ----------------------------
def print_help():
    print("Controles:")
    print("  c : capturar y procesar un frame (modo foto)")
    print("  p : pausar/reanudar live")
    print("  q : salir")
    print("  h : mostrar/ocultar ayuda en pantalla")

def main():
    print("Inicializando cámara...")
    cap = init_camera()
    # Calibración inicial: indicamos al usuario que muestre solo el tapete
    print("Atención: para calibrar, sitúa la webcam mirando únicamente el tapete verde y pulsa ENTER.")
    input("Pulsa ENTER para iniciar calibración (si no quieres calibrar, pulsa ENTER igualmente).")
    lower_hsv, upper_hsv = calibrate_green_range(cap, sample_frames=8, debug=True)
    # Obtener un frame para estimar MIN_CARD_AREA
    ret, frame = cap.read()
    if not ret:
        print("ERROR: no se ha podido leer de la cámara.")
        cap.release()
        return
    mask_sample = segment_green(frame, lower_hsv, upper_hsv)
    min_card_area = auto_estimate_min_area(mask_sample, scale_factor=0.02, debug=True)
    print(f"Calibración final -> lower: {lower_hsv}, upper: {upper_hsv}, MIN_CARD_AREA: {min_card_area}")

    # Cargar plantillas
    ranks_tpl, suits_tpl = load_templates(TEMPLATES_FOLDER)
    if not ranks_tpl or not suits_tpl:
        print("[WARN] Plantillas rank o suit faltantes o carpeta vacía. El reconocimiento por templates fallará.")
        print("Asegúrate de crear la carpeta 'templates/' con archivos rank_* y suit_*.")

    show_help = True
    paused = False
    print_help()

    window_name = "CardRec - C270 (Live: presiona 'c' para foto, 'q' para salir)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Frame no disponible. Reintentando...")
                time.sleep(0.1)
                continue
            display_frame, mask, results = process_frame(frame, lower_hsv, upper_hsv, min_card_area, ranks_tpl, suits_tpl, draw=True)
            # Mostrar ayuda textual en la ventana si corresponde
            if show_help:
                help_lines = [
                    "c: captura | p: pausar | q: salir | h: mostrar/ocultar ayuda",
                    f"Plantillas rank: {len(ranks_tpl)} | suits: {len(suits_tpl)}"
                ]
                y0 = 20
                for i,line in enumerate(help_lines):
                    cv2.putText(display_frame, line, (10, y0 + i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow(window_name, display_frame)

        # lee teclas (espera 1 ms para estar responsive)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Saliendo...")
            break
        elif key == ord('p'):
            paused = not paused
            print("Pausado." if paused else "Reanudado.")
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('c'):
            # Captura y procesa un frame (modo foto)
            ret, frame_cap = cap.read()
            if not ret:
                print("No se pudo capturar el frame.")
                continue
            frame_res, mask_res, results_res = process_frame(frame_cap, lower_hsv, upper_hsv, min_card_area, ranks_tpl, suits_tpl, draw=True)
            # Mostrar resultado de la foto en otra ventana
            cv2.imshow("Captured - Processed", frame_res)
            print("Resultados de la captura:")
            if not results_res:
                print("  No se detectaron cartas.")
            for i,r in enumerate(results_res):
                print(f"  Carta {i+1}: Rank={r['rank']}, Suit={r['suit']}, Conf={r['conf']:.2f}")
            # Esperar hasta que usuario cierre ventana o pulse tecla
            print("Pulsa cualquier tecla en la ventana de la captura para volver al live.")
            cv2.waitKey(0)
            cv2.destroyWindow("Captured - Processed")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
