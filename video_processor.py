from __future__ import annotations

import cv2
import tempfile
from pathlib import Path

from utils.image_utils import draw_prediction


def process_video(uploaded_file, face_detector, emotion_detector, frame_stride=2, max_frames=900):
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        input_path = tmp.name
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise ValueError("The uploaded video could not be opened.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    timeline, counts = [], {emotion: 0 for emotion in __import__('utils.constants', fromlist=['EMOTIONS']).EMOTIONS}
    total_faces = analyzed = 0
    try:
        frame_no = 0
        while analyzed < max_frames:
            ok, frame = capture.read()
            if not ok: break
            if frame_no % frame_stride == 0:
                analyzed += 1
                faces = face_detector.detect(frame)
                total_faces += len(faces)
                labels = []
                for box in faces:
                    x, y, w, h = box
                    try:
                        result = emotion_detector.predict(frame[y:y+h, x:x+w])
                        labels.append(result["label"]); counts[result["label"]] += 1
                        draw_prediction(frame, box, result["label"], result["confidence"])
                    except (RuntimeError, ValueError):
                        pass
                timeline.append({"frame": frame_no, "emotion": labels[0] if labels else "No face"})
            writer.write(frame); frame_no += 1
    finally:
        capture.release(); writer.release()
    return output_path, {"timeline": timeline, "counts": counts, "faces": total_faces, "frames": analyzed}
