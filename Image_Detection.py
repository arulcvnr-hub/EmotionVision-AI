import streamlit as st
from services.analytics import distribution_chart
from database.database import add_detection
from utils.constants import SUPPORTED_IMAGE_TYPES
from utils.image_utils import uploaded_file_to_bgr, bgr_to_rgb, draw_prediction


def render(face_detector, emotion_detector):
    st.title("Image Emotion Detection")
    if not emotion_detector.available:
        st.error(emotion_detector.error); st.code("Place your trained 7-class Keras model at models/emotion_model.h5")
    upload = st.file_uploader("Upload JPG, JPEG, or PNG", type=SUPPORTED_IMAGE_TYPES)
    if not upload: return
    try: image = uploaded_file_to_bgr(upload)
    except ValueError as exc: st.error(str(exc)); return
    faces = face_detector.detect(image); counts = {e: 0 for e in __import__('utils.constants', fromlist=['EMOTIONS']).EMOTIONS}; results=[]
    for box in faces:
        x,y,w,h = box
        try:
            result = emotion_detector.predict(image[y:y+h,x:x+w]); results.append(result); counts[result['label']] += 1
            add_detection("Image", result["label"], result["confidence"]); draw_prediction(image, box, result["label"], result["confidence"])
        except (RuntimeError, ValueError) as exc: st.warning(str(exc))
    st.image(bgr_to_rgb(image), caption=f"Detected {len(faces)} face(s)", use_container_width=True)
    if not faces: st.warning("No face detected. Try a well-lit, front-facing image.")
    if results:
        st.subheader("Emotion probabilities")
        st.plotly_chart(distribution_chart(counts), use_container_width=True)
        st.dataframe([{"Face": i+1, "Emotion": r["label"], "Confidence": f'{r["confidence"]:.2f}%'} for i,r in enumerate(results)], hide_index=True, use_container_width=True)
