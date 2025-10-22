import cv2
import os
import numpy as np

# ================================
# CONFIGURACIÓN
# ================================
TEMPLATES_PATH = "templates"
VALORES_PATH = os.path.join(TEMPLATES_PATH, "valores")
PALOS_PATH = os.path.join(TEMPLATES_PATH, "palos")
CAM_INDEX = 0  # cambia a 1 si usas otra cámara (por ejemplo, iPhone conectado por cable)

os.makedirs(VALORES_PATH, exist_ok=True)
os.makedirs(PALOS_PATH, exist_ok=True)

# ================================
# FUNCIONES AUXILIARES
# ================================
def preprocesar(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def encontrar_carta(frame):
    edges = preprocesar(frame)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mejor = None
    max_area = 0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area and area > 10000:
                max_area = area
                mejor = approx
    return mejor

def warp_card(frame, approx, w=200, h=300):
    pts = approx.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, M, (w, h))
    return warped

# ================================
# CAPTURA CON GUÍA VISUAL
# ================================
cap = cv2.VideoCapture(CAM_INDEX)
print("🎴 Coloca una carta frente a la cámara.")
print("Pulsa 'v' para capturar valor, 'p' para palo, 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo leer frame de la cámara.")
        break

    frame_display = frame.copy()
    approx = encontrar_carta(frame)

    if approx is not None:
        cv2.drawContours(frame_display, [approx], -1, (0, 255, 0), 2)
        carta = warp_card(frame, approx)
        h, w, _ = carta.shape

        # Zonas ROI — fijas dentro de la carta enderezada
        x_val_ini, y_val_ini = int(0.05*w), int(0.05*h)
        x_val_fin, y_val_fin = int(0.25*w), int(0.22*h)
        x_palo_ini, y_palo_ini = int(0.05*w), int(0.23*h)
        x_palo_fin, y_palo_fin = int(0.25*w), int(0.43*h)

        roi_valor = carta[y_val_ini:y_val_fin, x_val_ini:x_val_fin]
        roi_palo = carta[y_palo_ini:y_palo_fin, x_palo_ini:x_palo_fin]

        # Mostrar la carta recortada y sus ROIs
        cv2.imshow("ROI Valor", roi_valor)
        cv2.imshow("ROI Palo", roi_palo)

        # Añadir guías visuales en la ventana principal
        overlay = frame_display.copy()
        cv2.rectangle(overlay, (x_val_ini*2, y_val_ini*2), (x_val_fin*2, y_val_fin*2), (0, 255, 0), 2)
        cv2.rectangle(overlay, (x_palo_ini*2, y_palo_ini*2), (x_palo_fin*2, y_palo_fin*2), (255, 0, 0), 2)
        frame_display = cv2.addWeighted(overlay, 0.7, frame_display, 0.3, 0)

    cv2.imshow("Captura", frame_display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('v') and approx is not None:
        nombre = input("🂡 Nombre del valor (A, 2, 3, 10, J, Q, K): ").strip()
        ruta = os.path.join(VALORES_PATH, f"{nombre}.jpg")
        cv2.imwrite(ruta, roi_valor)
        print(f"✅ Valor guardado: {ruta}")

    elif key == ord('p') and approx is not None:
        nombre = input("🂾 Nombre del palo (corazones, picas, diamantes, treboles): ").strip()
        ruta = os.path.join(PALOS_PATH, f"{nombre}.jpg")
        cv2.imwrite(ruta, roi_palo)
        print(f"✅ Palo guardado: {ruta}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Cámara cerrada correctamente.")
