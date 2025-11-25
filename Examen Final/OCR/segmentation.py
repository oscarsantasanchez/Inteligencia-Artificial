import numpy as np

def segment_lines(binary_img):
    h_proj = np.sum(binary_img, axis=1)
    lines = []
    start = None
    for i, val in enumerate(h_proj):
        if val > 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            lines.append(binary_img[start:i, :])
            start = None
    return lines

def segment_characters(line_img):
    v_proj = np.sum(line_img, axis=0)
    chars = []
    start = None
    for j, val in enumerate(v_proj):
        if val > 0 and start is None:
            start = j
        elif val == 0 and start is not None:
            chars.append(line_img[:, start:j])
            start = None
    return chars
