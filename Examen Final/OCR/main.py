import os
from preprocessing import load_image, binarize, deskew
from segmentation import segment_lines, segment_characters
from recognizer import (
    load_print_templates, match_character_print,
    load_handwritten_templates, match_character_handwritten
)
from utils import save_text

# ==========================
# Carpetas de plantillas
# ==========================
PRINT_TEMPLATES_FOLDER = "data/print_templates"
HAND_TEMPLATES_FOLDER = "data/handwritten_templates"
INPUT_IMAGE = "data/input_images/test1.png"
OUTPUT_FOLDER = "outputs/recognized_text/"

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================
# Cargar plantillas
# ==========================
print("Cargando plantillas impresas...")
print_templates = load_print_templates(PRINT_TEMPLATES_FOLDER)
print("Cargando plantillas manuscritas...")
hand_templates = load_handwritten_templates(HAND_TEMPLATES_FOLDER)

# ==========================
# Procesar imagen
# ==========================
original, gray = load_image(INPUT_IMAGE)
binary = binarize(gray)
binary = deskew(binary)

lines = segment_lines(binary)
final_text = ""

print("Iniciando reconocimiento OCR...")

for line in lines:
    chars = segment_characters(line)
    for ch in chars:
        label_print = match_character_print(ch, print_templates)
        label_hand = match_character_handwritten(ch, hand_templates)

        # Prioridad: impreso, luego manuscrito
        final_char = label_print if label_print else label_hand
        final_text += final_char if final_char else "?"
    final_text += "\n"

# ==========================
# Guardar resultado
# ==========================
output_file = os.path.join(OUTPUT_FOLDER, "output.txt")
save_text(final_text, output_file)
print("OCR COMPLETADO. Archivo generado en:", output_file)
