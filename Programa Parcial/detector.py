import cv2
from matcher import Matcher
import numpy as np

class Detector:
    def __init__(self, cam_index=0, debug=False):
        self.cap = cv2.VideoCapture(cam_index)
        self.matcher = Matcher("templates", debug)
        self.debug = debug

    def preprocesar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges

    def detectar_cartas(self, frame):
        edges = self.preprocesar(frame)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cartas = []

        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(approx) > 5000:
                cartas.append(approx)
        return cartas

    def warp_card(self, frame, approx, w=200, h=300):
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

    def run(self):
        print("📷 Detector iniciado. Pulsa 'q' para salir.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Error al leer cámara.")
                break

            cartas = self.detectar_cartas(frame)

            for approx in cartas:
                cv2.drawContours(frame, [approx], -1, (0,255,0), 2)
                warped = self.warp_card(frame, approx)
                val, palo, vscore, pscore = self.matcher.recognize(warped)
                x, y, w, h = cv2.boundingRect(approx)
                label = f"{val} {palo}"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            cv2.imshow("Detector de cartas", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
