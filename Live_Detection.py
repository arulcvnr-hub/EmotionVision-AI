import streamlit as st
from database.database import add_detection
from utils.image_utils import uploaded_file_to_bgr, bgr_to_rgb, draw_prediction


def render(face_detector, emotion_detector):
    st.title("Live Webcam Emotion Detection")
    st.caption("Streamlit's native camera input provides a secure browser camera snapshot. For continuous streaming, deploy with a WebRTC component.")
    if not emotion_detector.available:
        st.error(emotion_detector.error); st.code("Place your trained 7-class Keras model at models/emotion_model.h5")
    active = st.toggle("Start Camera", value=False)
    if not active:
        st.info("Enable Start Camera to grant camera access."); return
    camera = st.camera_input("Capture a frame")
    if not camera: return
    try: image = uploaded_file_to_bgr(camera)
    except ValueError as exc: st.error(str(exc)); return
    faces = face_detector.detect(image); results=[]
    for box in faces:
        x,y,w,h = box
        try:
            result = emotion_detector.predict(image[y:y+h,x:x+w]); results.append(result)
            add_detection("Webcam", result["label"], result["confidence"]); draw_prediction(image, box, result["label"], result["confidence"])
        except (RuntimeError, ValueError) as exc: st.warning(str(exc))
    st.image(bgr_to_rgb(image), caption=f"Detection status: {len(faces)} face(s)", use_container_width=True)
    if not faces: st.warning("No face detected in the captured frame.")
    else: st.success(f"Detected {len(faces)} face(s)")
