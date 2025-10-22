import cv2
import numpy as np
import os
import time

class Matcher:
    def __init__(self, templates_path="templates", debug=False):
        self.debug = debug
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.templates_valores = {}
        self.templates_palos = {}

        # Cargar plantillas desde carpetas
        self.load_templates(os.path.join(templates_path, "valores"), tipo="valor")
        self.load_templates(os.path.join(templates_path, "palos"), tipo="palo")

    # ===============================================================
    def load_templates(self, folder, tipo="valor"):
        if not os.path.exists(folder):
            print(f"[WARN] Carpeta de plantillas no encontrada: {folder}")
            return

        for file in os.listdir(folder):
            path = os.path.join(folder, file)
            if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            name = os.path.splitext(file)[0]
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Preprocesamiento uniforme
            img = cv2.resize(img, (70, 100))
            img = cv2.bitwise_not(img)  # invertir (fondo negro, símbolo blanco)

            kp, des = self.orb.detectAndCompute(img, None)
            if tipo == "valor":
                self.templates_valores[name] = (kp, des)
            else:
                self.templates_palos[name] = (kp, des)

            print(f"[INFO] Plantilla cargada: {tipo} {name}")

    # ===============================================================
    def recognize(self, card_img):
        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # === 1. Recortes de ROI ajustados ===
        roi_val = gray[int(0.02*h):int(0.18*h), int(0.05*w):int(0.20*w)]
        roi_palo = gray[int(0.18*h):int(0.40*h), int(0.05*w):int(0.25*w)]

        # === 2. Preprocesamiento ===
        roi_val = cv2.GaussianBlur(roi_val, (3,3), 0)
        roi_val = cv2.adaptiveThreshold(
            roi_val, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10
        )

        # Si está demasiado blanco, invertir
        if np.mean(roi_val) > 127:
            roi_val = cv2.bitwise_not(roi_val)

        # Limpieza morfológica
        kernel = np.ones((2,2), np.uint8)
        roi_val = cv2.morphologyEx(roi_val, cv2.MORPH_OPEN, kernel)

        # --- Palo ---
        roi_palo = cv2.GaussianBlur(roi_palo, (3,3), 0)
        roi_palo = cv2.adaptiveThreshold(
            roi_palo, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10
        )
        if np.mean(roi_palo) > 127:
            roi_palo = cv2.bitwise_not(roi_palo)
        roi_palo = cv2.morphologyEx(roi_palo, cv2.MORPH_OPEN, kernel)

        # === 3. Guardado de ROI para depuración ===
        if self.debug:
            os.makedirs("debug_rois", exist_ok=True)
            ts = int(time.time())
            cv2.imwrite(os.path.join("debug_rois", f"roi_val_{ts}.jpg"), roi_val)
            cv2.imwrite(os.path.join("debug_rois", f"roi_palo_{ts}.jpg"), roi_palo)
            cv2.imshow("ROI Valor procesado", roi_val)
            cv2.imshow("ROI Palo procesado", roi_palo)

        # === 4. Extracción de descriptores ===
        kp1v, des1v = self.orb.detectAndCompute(roi_val, None)
        kp1p, des1p = self.orb.detectAndCompute(roi_palo, None)

        best_val, best_val_score = self.match_with_templates(des1v, self.templates_valores)
        best_palo, best_palo_score = self.match_with_templates(des1p, self.templates_palos)

        # === 5. Filtrado de confianza ===
        if best_val_score < 8:
            best_val = "?"
        if best_palo_score < 6:
            best_palo = "?"

        if self.debug:
            print(f"[DEBUG] Valor={best_val}({best_val_score}) Palo={best_palo}({best_palo_score})")

        return best_val, best_palo, best_val_score, best_palo_score

    # ===============================================================
    def match_with_templates(self, des_query, templates_dict):
        if des_query is None:
            return "?", 0

        best_name = "?"
        best_score = 0

        for name, (kp_tmpl, des_tmpl) in templates_dict.items():
            if des_tmpl is None:
                continue
            matches = self.bf.match(des_query, des_tmpl)
            good = [m for m in matches if m.distance < 60]
            score = len(good)
            if score > best_score:
                best_score = score
                best_name = name

        return best_name, best_score
