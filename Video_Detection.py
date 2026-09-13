import streamlit as st
import pandas as pd
from services.video_processor import process_video
from services.analytics import timeline_chart
from utils.constants import SUPPORTED_VIDEO_TYPES, EMOTIONS


def render(face_detector, emotion_detector):
    st.title("Video Emotion Detection")
    if not emotion_detector.available:
        st.error(emotion_detector.error); st.code("Place your trained 7-class Keras model at models/emotion_model.h5")
    upload = st.file_uploader("Upload MP4, AVI, or MOV", type=SUPPORTED_VIDEO_TYPES)
    stride = st.slider("Analyze every Nth frame", 1, 10, 2, help="Higher values process faster with fewer timeline points.")
    if upload and st.button("Process video", type="primary"):
        if not emotion_detector.available: st.stop()
        with st.spinner("Analyzing video frames…"):
            try: output, data = process_video(upload, face_detector, emotion_detector, stride)
            except ValueError as exc: st.error(str(exc)); return
        st.session_state.video_result = (output, data)
    if "video_result" not in st.session_state: return
    output, data = st.session_state.video_result
    st.video(output)
    c1,c2,c3 = st.columns(3); c1.metric("Frames analyzed", data["frames"]); c2.metric("Faces detected", data["faces"]); c3.metric("Dominant emotion", max(data["counts"], key=data["counts"].get) if data["faces"] else "—")
    chart = timeline_chart(data["timeline"])
    if chart: st.plotly_chart(chart, use_container_width=True)
