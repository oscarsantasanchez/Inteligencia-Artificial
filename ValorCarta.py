import cv2
import numpy as np

class ValorCarta:
    def __init__(self, templates_path="Imagenes/"):
        """
        Inicializa el reconocedor de cartas.
        Separa plantillas de valores (A-K) y palos (♥♠♣♦).
        """
        # Plantillas de números/figuras
        self.templates_valores = {
            "A": cv2.imread(templates_path + "As.jpg", 0),
            "2": cv2.imread(templates_path + "Dos.jpg", 0),
            "3": cv2.imread(templates_path + "Tres.jpg", 0),
            "4": cv2.imread(templates_path + "Cuatro.jpg", 0),
            "5": cv2.imread(templates_path + "Cinco.jpg", 0),
            "6": cv2.imread(templates_path + "Seis.jpg", 0),
            "7": cv2.imread(templates_path + "Siete.jpg", 0),
            "8": cv2.imread(templates_path + "Ocho.jpg", 0),
            "9": cv2.imread(templates_path + "Nueve.jpg", 0),
            "10": cv2.imread(templates_path + "Diez.jpg", 0),
            "J": cv2.imread(templates_path + "Jota.jpg", 0),
            "Q": cv2.imread(templates_path + "Qu.jpg", 0),
            "K": cv2.imread(templates_path + "Ka.jpg", 0)
        }

        # Plantillas de palos
        self.templates_palos = {
            "Corazon": cv2.imread(templates_path + "Corazon.jpg", 0),
            "Pica": cv2.imread(templates_path + "Pica.jpg", 0),
            "Trebol": cv2.imread(templates_path + "Trebol.jpg", 0),
            "Diamante": cv2.imread(templates_path + "Diamante.jpg", 0)
        }

    def reconocer(self, card_img):
        """
        Reconoce el valor y el palo de la carta.
        - card_img: imagen normalizada de la carta (ej: 200x300 px).
        """
        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)

        # ROI del valor (esquina superior izquierda)
        roi_valor = gray[5:70, 5:40]

        # ROI del palo (debajo del valor)
        roi_palo = gray[40:80, 5:40]

        # --- Reconocimiento del valor ---
        best_val, best_val_match = 1e6, "?"
        for name, tmpl in self.templates_valores.items():
            if tmpl is None:
                continue
            tmpl_resized = cv2.resize(tmpl, (roi_valor.shape[1], roi_valor.shape[0]))
            res = cv2.matchTemplate(roi_valor, tmpl_resized, cv2.TM_SQDIFF)
            min_val, _, _, _ = cv2.minMaxLoc(res)
            if min_val < best_val:
                best_val, best_val_match = min_val, name

        # --- Reconocimiento del palo ---
        best_suit, best_suit_match = 1e6, "?"
        for name, tmpl in self.templates_palos.items():
            if tmpl is None:
                continue
            tmpl_resized = cv2.resize(tmpl, (roi_palo.shape[1], roi_palo.shape[0]))
            res = cv2.matchTemplate(roi_palo, tmpl_resized, cv2.TM_SQDIFF)
            min_val, _, _, _ = cv2.minMaxLoc(res)
            if min_val < best_suit:
                best_suit, best_suit_match = min_val, name

        return best_val_match, best_suit_match
