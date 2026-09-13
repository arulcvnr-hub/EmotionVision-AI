from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def uploaded_file_to_bgr(uploaded_file) -> np.ndarray:
    """Decode a Streamlit upload into an OpenCV BGR image."""
    data = uploaded_file.getvalue()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a valid image.")
    return image


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr_to_rgb(image))


def draw_prediction(image: np.ndarray, box, label: str, confidence: float) -> np.ndarray:
    x, y, w, h = [int(v) for v in box]
    color = (53, 211, 153)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    text = f"{label} | {confidence:.1f}%"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    top = max(y - th - 12, 0)
    cv2.rectangle(image, (x, top), (x + tw + 10, y), color, -1)
    cv2.putText(image, text, (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 20, 30), 2, cv2.LINE_AA)
    return image
