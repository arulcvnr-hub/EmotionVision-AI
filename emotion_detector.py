from __future__ import annotations

from pathlib import Path
import numpy as np

from utils.constants import EMOTIONS, MODEL_EMOTIONS, MODEL_PATH

try:
    from tf_keras.models import load_model
except Exception:
    try:
        from tensorflow.keras.models import load_model
    except Exception:  # Allows the UI to show a useful message if TensorFlow is unavailable.
        load_model = None


class EmotionDetector:
    """Adapter for a 7-class Keras facial-expression model."""
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self.model = None
        self.error = None
        if not self.model_path.exists():
            self.error = f"Model file not found: {self.model_path}"
        elif load_model is None:
            self.error = "TensorFlow is not installed. Install requirements.txt first."
        else:
            try:
                self.model = load_model(self.model_path, compile=False)
            except Exception as exc:
                self.error = f"Model could not be loaded: {exc}"

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, face_bgr: np.ndarray) -> dict:
        if not self.available:
            raise RuntimeError(self.error or "Emotion model is unavailable.")
        import cv2
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        input_shape = self.model.input_shape
        height = int(input_shape[1] or 48)
        width = int(input_shape[2] or 48)
        resized = cv2.resize(gray, (width, height)).astype("float32") / 255.0
        if len(input_shape) == 4 and input_shape[3] == 1:
            batch = resized[np.newaxis, ..., np.newaxis]
        else:
            batch = np.repeat(resized[np.newaxis, ..., np.newaxis], 3, axis=3)
        probabilities = np.asarray(self.model.predict(batch, verbose=0)[0]).reshape(-1)
        if probabilities.size != len(MODEL_EMOTIONS):
            raise ValueError(f"Model returned {probabilities.size} classes; expected {len(MODEL_EMOTIONS)}.")
        probabilities = probabilities / probabilities.sum() if probabilities.sum() else probabilities
        index = int(np.argmax(probabilities))
        model_probabilities = dict(zip(MODEL_EMOTIONS, (probabilities * 100).round(3)))
        probabilities_by_ui_label = {label: float(model_probabilities.get(label, 0.0)) for label in EMOTIONS}
        return {"label": MODEL_EMOTIONS[index], "confidence": float(probabilities[index] * 100),
                "probabilities": probabilities_by_ui_label}
