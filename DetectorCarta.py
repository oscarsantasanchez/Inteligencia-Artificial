# Import necessary packages
import cv2
import numpy as np


def main():
    # Abrir la cámara USB (0 suele ser la primera cámara, cámbialo si tienes varias)
    cap = cv2.VideoCapture(1)

    # Verificamos que la cámara se haya abierto correctamente
    if not cap.isOpened():
        print("No se puede abrir la cámara")
        return

    print("Cámara abierta correctamente. Pulsa 'q' para salir.")

    while True:
        # Capturamos frame a frame
        ret, frame = cap.read()
        if not ret:
            print("No se puede recibir frame. Saliendo...")
            break

        # Mostrar el frame en una ventana
        cv2.imshow('Cámara USB', frame)

        # Salir si el usuario pulsa la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar ventanas
    cap.release()
    cv2.destroyAllWindows()
    print("Cámara cerrada correctamente.")

if __name__ == "__main__":
    main()
