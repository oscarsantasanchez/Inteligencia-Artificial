import cv2
import os

# Tamaño de normalización de caracteres
CHAR_SIZE = (32, 32)

# ==========================
# Texto Impreso
# ==========================
def load_print_templates(folder):
    """
    Carga plantillas de texto impreso desde folder/
    Devuelve diccionario {label: imagen}
    """
    templates = {}
    for fname in os.listdir(folder):
        if fname.lower().endswith((".png", ".jpg", ".bmp")):
            label = os.path.splitext(fname)[0]
            img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
            _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            img_bin = cv2.resize(img_bin, CHAR_SIZE)
            templates[label] = img_bin
    return templates


def match_character_print(char_img, templates):
    """
    Compara un carácter con plantillas de texto impreso.
    Devuelve la etiqueta con mejor similitud.
    """
    char_resized = cv2.resize(char_img, CHAR_SIZE)
    best_label = None
    best_score = -1

    for label, tmpl in templates.items():
        result = cv2.matchTemplate(char_resized, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


# ==========================
# Texto Manuscrito
# ==========================
def load_handwritten_templates(root_folder):
    """
    Carga plantillas manuscritas desde la estructura:
    root_folder/
        uppercase/
            A/
                A_Juan.png
                ...
        lowercase/
        numbers/
    Devuelve diccionario {label: [imagenes]}
    """
    templates = {}
    for category in os.listdir(root_folder):
        category_path = os.path.join(root_folder, category)
        if not os.path.isdir(category_path):
            continue

        for char_folder in os.listdir(category_path):
            char_path = os.path.join(category_path, char_folder)
            if not os.path.isdir(char_path):
                continue

            for fname in os.listdir(char_path):
                if fname.lower().endswith((".png", ".jpg", ".bmp")):
                    img_path = os.path.join(char_path, fname)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    img_bin = cv2.resize(img_bin, CHAR_SIZE)

                    label = char_folder
                    if label not in templates:
                        templates[label] = []
                    templates[label].append(img_bin)
    return templates


def match_character_handwritten(char_img, templates):
    """
    Compara un carácter con todas las plantillas manuscritas.
    Devuelve la etiqueta con mejor similitud.
    """
    char_resized = cv2.resize(char_img, CHAR_SIZE)
    best_label = None
    best_score = -1

    for label, imgs in templates.items():
        for tmpl in imgs:
            result = cv2.matchTemplate(char_resized, tmpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)
            if score > best_score:
                best_score = score
                best_label = label
    return best_label
