from __future__ import annotations

import cv2


class FaceDetector:
    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5, min_size=(40, 40)):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError("OpenCV could not load its Haar cascade face detector.")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.cascade.detectMultiScale(gray, self.scale_factor, self.min_neighbors, minSize=self.min_size)
