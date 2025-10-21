# DetectorCartaUnico.py
import cv2
import numpy as np
import os
import time

class DetectorCarta:
    
    def __init__(self, cam_index=1, templates_path="Imagenes/", frame_w=640, frame_h=480,
             min_area=1000, match_method=cv2.TM_CCOEFF_NORMED, debug=False):

        """
        Inicializa capturador y plantillas.
        - cam_index: índice de la cámara (0,1,...)
        - templates_path: carpeta con plantillas para valores y palos
        - frame_w/frame_h: resolución deseada
        - min_area: área mínima para considerar contornos
        - match_method: método de matchTemplate (por defecto TM_SQDIFF)
        - debug: muestra ventanas adicionales si True
        """
        self.cam_index = cam_index
        self.templates_path = templates_path
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.min_area = min_area
        self.match_method = match_method
        self.debug = debug

        # Inicializar cámara
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se puede abrir la cámara index {self.cam_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)

        # Cargar plantillas
        self.templates_valores = {}
        self.templates_palos = {}
        self._cargar_plantillas()

    def _cargar_plantillas(self):
        valores_map = {
            "A": "As.jpg", "2": "Dos.jpg", "3": "Tres.jpg", "4": "Cuatro.jpg",
            "5": "Cinco.jpg", "6": "Seis.jpg", "7": "Siete.jpg", "8": "Ocho.jpg",
            "9": "Nueve.jpg", "0": "Cero.jpg", "J": "Jota.jpg", "Q": "Qu.jpg", "K": "Ka.jpg"
        }
        palos_map = {
            "Corazon": "Corazon.jpg", "Pica": "Pica.jpg",
            "Trebol": "Trebor.jpg", "Diamante": "Diamante.jpg"
        }

        missing = []
        for key, fname in valores_map.items():
            path = os.path.join(self.templates_path, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.templates_valores[key] = img
            else:
                missing.append(path)

        for key, fname in palos_map.items():
            path = os.path.join(self.templates_path, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.templates_palos[key] = img
            else:
                missing.append(path)

        if self.debug:
            print(f"Plantillas valores cargadas: {list(self.templates_valores.keys())}")
            print(f"Plantillas palos cargadas: {list(self.templates_palos.keys())}")
            if missing:
                print("Plantillas no encontradas:", missing)

    def _preprocesar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        return blur

    def _es_rect_carta(self, approx):
        # Filtrado por número de lados y área
        if len(approx) != 4:
            return False
        area = cv2.contourArea(approx)
        if area < self.min_area:
            return False
        x, y, w, h = cv2.boundingRect(approx)
        if w == 0 or h == 0:
            return False
        aspect_ratio = float(h) / float(w)
        # Proporción aproximada carta ≈ 1.39 (ajusta tolerancia si hace falta)
        return 1.2 < aspect_ratio < 2.5

    def _warp_card(self, frame, approx, width=200, height=300):
        # Reordenar puntos y aplicar warp perspective para normalizar la carta
        pts = approx.reshape(4, 2).astype(np.float32)
        # ordenar los puntos: top-left, top-right, bottom-right, bottom-left
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(4)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        src = np.array([tl, tr, br, bl], dtype=np.float32)
        dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(frame, M, (width, height))
        return warped

    def reconocer_valor_palo(self, card_img):
        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        roi_valor = gray[int(0.01*h):int(0.28*h), int(0.02*w):int(0.28*w)]
        roi_palo = gray[int(0.12*h):int(0.5*h), int(0.02*w):int(0.32*w)]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        roi_valor_proc = clahe.apply(roi_valor)
        roi_palo_proc  = clahe.apply(roi_palo)
        _, roi_val_bin = cv2.threshold(roi_valor_proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, roi_pal_bin = cv2.threshold(roi_palo_proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        match_method = cv2.TM_CCOEFF_NORMED
        best_val_score, best_val_name = -1.0, "?"
        best_suit_score, best_suit_name = -1.0, "?"
        scales = [0.7, 0.85, 1.0, 1.15, 1.3]

        for name, tmpl in self.templates_valores.items():
            if tmpl is None:
                continue
            for s in scales:
                nh = max(1, int(roi_val_bin.shape[0] * s))
                nw = max(1, int(roi_val_bin.shape[1] * s))
                tmpl_r = cv2.resize(tmpl, (nw, nh), interpolation=cv2.INTER_AREA)
                tmpl_proc = clahe.apply(tmpl_r)
                _, tmpl_bin = cv2.threshold(tmpl_proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if tmpl_bin.shape[0] > roi_val_bin.shape[0] or tmpl_bin.shape[1] > roi_val_bin.shape[1]:
                    continue
                res = cv2.matchTemplate(roi_val_bin, tmpl_bin, match_method)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > best_val_score:
                    best_val_score = max_val
                    best_val_name = name

        for name, tmpl in self.templates_palos.items():
            if tmpl is None:
                continue
            for s in scales:
                nh = max(1, int(roi_pal_bin.shape[0] * s))
                nw = max(1, int(roi_pal_bin.shape[1] * s))
                tmpl_r = cv2.resize(tmpl, (nw, nh), interpolation=cv2.INTER_AREA)
                tmpl_proc = clahe.apply(tmpl_r)
                _, tmpl_bin = cv2.threshold(tmpl_proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if tmpl_bin.shape[0] > roi_pal_bin.shape[0] or tmpl_bin.shape[1] > roi_pal_bin.shape[1]:
                    continue
                res = cv2.matchTemplate(roi_pal_bin, tmpl_bin, match_method)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > best_suit_score:
                    best_suit_score = max_val
                    best_suit_name = name

        accept_val = 0.55
        accept_suit = 0.55

        if best_val_score < accept_val:
            try:
                orb = cv2.ORB_create(500)
                kp1, des1 = orb.detectAndCompute(roi_valor_proc, None)
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                best_orb_name, best_orb_matches = "?", 0
                for name, tmpl in self.templates_valores.items():
                    tmpl_proc = clahe.apply(cv2.resize(tmpl, (int(roi_valor.shape[1]*0.9), int(roi_valor.shape[0]*0.9))))
                    kp2, des2 = orb.detectAndCompute(tmpl_proc, None)
                    if des1 is None or des2 is None:
                        continue
                    matches = bf.match(des1, des2)
                    matches = sorted(matches, key=lambda x: x.distance)
                    good = [m for m in matches if m.distance < 60]
                    if len(good) > best_orb_matches:
                        best_orb_matches = len(good)
                        best_orb_name = name
                if best_orb_matches >= 8:
                    best_val_name = best_orb_name
                    best_val_score = 0.5 + best_orb_matches/100.0
            except Exception:
                pass

        if best_suit_score < accept_suit:
            try:
                orb = cv2.ORB_create(300)
                kp1, des1 = orb.detectAndCompute(roi_palo_proc, None)
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                best_orb_name, best_orb_matches = "?", 0
                for name, tmpl in self.templates_palos.items():
                    tmpl_proc = clahe.apply(cv2.resize(tmpl, (int(roi_palo.shape[1]*0.9), int(roi_palo.shape[0]*0.9))))
                    kp2, des2 = orb.detectAndCompute(tmpl_proc, None)
                    if des1 is None or des2 is None:
                        continue
                    matches = bf.match(des1, des2)
                    matches = sorted(matches, key=lambda x: x.distance)
                    good = [m for m in matches if m.distance < 60]
                    if len(good) > best_orb_matches:
                        best_orb_matches = len(good)
                        best_orb_name = name
                if best_orb_matches >= 6:
                    best_suit_name = best_orb_name
                    best_suit_score = 0.5 + best_orb_matches/100.0
            except Exception:
                pass

        if self.debug:
            print(f"Valor: {best_val_name} score={best_val_score:.3f}  Palo: {best_suit_name} score={best_suit_score:.3f}")

        return best_val_name, best_suit_name, best_val_score, best_suit_score

    def run(self):
        print("Cámara abierta correctamente. Pulsa 'q' para salir.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("No se puede recibir frame. Revisa la cámara o el índice.")
                break

            pre = self._preprocesar(frame)
            edges = cv2.Canny(pre, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detected = False
            for cnt in contours:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if self._es_rect_carta(approx):
                    # Warp para obtener vista normalizada de la carta
                    try:
                        warped = self._warp_card(frame, approx, width=200, height=300)
                    except Exception:
                        continue

                    # Reconocer valor y palo
                    val_name, suit_name, val_score, suit_score = self.reconocer_valor_palo(warped)

                    # Dibujar contorno y etiqueta en el frame original
                    cv2.drawContours(frame, [approx], -1, (0,255,0), 2)
                    x, y, w, h = cv2.boundingRect(approx)
                    label = f"{val_name} {suit_name}"
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                    # Después de reconocer_valor_palo(...)
                    val_name, suit_name, val_score, suit_score = val_name, suit_name, val_score, suit_score  # deja nombres existentes
                    # Solo mostrar si ambos pasan umbral; si no, mostrar '?'
                    display_val = val_name if val_score >= 0.58 else "?"
                    display_suit = suit_name if suit_score >= 0.58 else "?"
                    label = f"{display_val} {display_suit}"
                    if self.debug:
                        print(f"[DEBUG] Carta detectada ROI -> Valor: {val_name} score={val_score:.3f}; Palo: {suit_name} score={suit_score:.3f}")

                    # Si debug, mostrar la carta warpeada y ROIs
                    if self.debug:
                        cv2.imshow("Carta warpeada", warped)
                        gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                        h2, w2 = gray_w.shape
                        roi_val = gray_w[int(0.02*h2):int(0.23*h2), int(0.03*w2):int(0.2*w2)]
                        roi_pal = gray_w[int(0.18*h2):int(0.38*h2), int(0.03*w2):int(0.2*w2)]
                        cv2.imshow("ROI Valor", roi_val)
                        cv2.imshow("ROI Palo", roi_pal)

                    detected = True
                    # Opcional: guardar plantilla detectada rápidamente con 's' (cuando se pulse)
                    # No romper el bucle; puede haber varias cartas en la escena

            if not detected and self.debug:
                cv2.putText(frame, "No detectada", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow("Detector de cartas - Unico", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                # Guardar la primera carta warpeada visible (si debug)
                # En modo no-debug tomará la primera carta encontrada en el siguiente ciclo que no guardó nada
                print("Guardado manual no implementado en este modo. Ejecuta con debug=True para ver cartas warpeadas y guardarlas manualmente.")

        self.cap.release()
        cv2.destroyAllWindows()
        print("Cámara cerrada correctamente.")

if __name__ == "__main__":
    # Ajusta cam_index o templates_path si es necesario
    detector = DetectorCarta(cam_index=0, templates_path="Imagenes/", debug=True)
    detector.run()