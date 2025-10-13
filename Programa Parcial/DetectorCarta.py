# Import necessary packages
import cv2
import numpy as np
import ValorCarta;

def main():
    # Abrir la cámara USB (0 suele ser la primera cámara, la del USB es 1)
    cap = cv2.VideoCapture(1)

 # Verificamos que la cámara se haya abierto correctamente
    if not cap.isOpened():
        print("No se puede abrir la cámara")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Cámara abierta correctamente. Pulsa 'q' para salir.")

    while True:
        # Capturamos frame a frame
        ret, frame = cap.read()
        if not ret:
            print("No se puede recibir frame. Revisa la cámara o el índice.")
            break

        # Convertir a gris y suavizar
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        # Detectar bordes
        edges = cv2.Canny(blur, 50, 150)

        # Encontrar contornos
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            # Aproximar contorno a un polígono
            approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
            
            # Solo polígonos con 4 lados (rectángulos)
            if len(approx) == 4 and cv2.contourArea(approx) > 1000:
                # Obtener bounding box
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(h) / float(w) if w > 0 else 0

                # Comprobamos si la proporción se parece a la de una carta (≈1.39)
                if 1.3 < aspect_ratio < 2.45:  # tolerancia de ±0.2
                    cv2.drawContours(frame, [approx], 0, (0,255,0), 2)
                    cv2.putText(frame, "Carta detectada", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                else:
                    # Dibuja en rojo si no cumple proporciones
                    cv2.drawContours(frame, [approx], 0, (0,0,255), 2)
                    cv2.putText(frame, "No coincide con medidas", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        # mostrar el frame en una ventana
        cv2.imshow("Detector de cartas", frame)

 # Salir si el usuario pulsa la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

 # Liberar la cámara y cerrar ventanas
    cap.release()
    cv2.destroyAllWindows()
    print("Cámara cerrada correctamente.")

if __name__ == "__main__":
    main()
