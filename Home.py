import streamlit as st
from services.emotion_detector import EmotionDetector
from database.database import get_history


def render():
    st.markdown('<div class="hero"><h1>EmotionVision AI</h1><p>Understanding Human Emotions Through Artificial Intelligence</p></div>', unsafe_allow_html=True)
    history = get_history()
    total = len(history); current = history.iloc[0] if total else None
    cols = st.columns(4)
    cols[0].metric("Total faces detected", total)
    cols[1].metric("Current emotion", current["emotion"] if current is not None else "—")
    cols[2].metric("Emotion confidence", f"{current['confidence']:.1f}%" if current is not None else "—")
    cols[3].metric("Detection status", "Active" if current is not None else "Ready")
    st.subheader("Get started")
    a,b,c = st.columns(3)
    a.markdown('<div class="card"><h3>Live camera</h3><p>Analyze expressions in real time with bounding boxes and confidence scores.</p></div>', unsafe_allow_html=True)
    b.markdown('<div class="card"><h3>Image analysis</h3><p>Upload an image and inspect every detected face individually.</p></div>', unsafe_allow_html=True)
    c.markdown('<div class="card"><h3>Video analytics</h3><p>Process recorded footage and explore emotion timelines.</p></div>', unsafe_allow_html=True)
    st.info("The application never invents predictions. Add a compatible trained model at models/emotion_model.h5 to enable inference.")
