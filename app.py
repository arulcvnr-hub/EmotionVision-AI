from __future__ import annotations
import streamlit as st

from database.database import init_db
from services.face_detector import FaceDetector
from services.emotion_detector import EmotionDetector

st.set_page_config(page_title="EmotionVision AI", page_icon="◉", layout="wide", initial_sidebar_state="expanded")
init_db()

@st.cache_resource
def get_face_detector(): return FaceDetector()

@st.cache_resource
def get_emotion_detector(): return EmotionDetector()

st.markdown("""<style>
:root { --accent:#35d399; }
.stApp { background: radial-gradient(circle at 15% 0%, #17233d 0%, #0b1120 45%, #070b14 100%); color:#e6edf7; }
[data-testid="stSidebar"] { background:#0a1020; border-right:1px solid #263652; }
[data-testid="stMetric"] { background:rgba(21,34,58,.75); border:1px solid #29415c; padding:18px; border-radius:16px; }
.hero { padding: 30px 34px; border:1px solid #29415c; border-radius:20px; background:linear-gradient(115deg,rgba(24,43,73,.92),rgba(13,24,45,.65)); margin-bottom:22px; }
.hero h1 { color:#f2f7ff; font-size:2.5rem; margin:0; } .hero p { color:#9db0ca; font-size:1.1rem; }
.card { background:rgba(18,30,52,.85); border:1px solid #29415c; border-radius:16px; padding:20px; }
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ◉ EmotionVision AI")
    st.caption("AI-powered facial expression analysis")
    choice = st.radio("Navigate", ["Home", "Live Detection", "Image Detection", "Video Detection", "Analytics", "History"], index=0)
    st.divider(); st.caption("Model status")
    detector = get_emotion_detector()
    st.success("Model ready") if detector.available else st.error("Model missing")

if choice == "Home":
    from pages.Home import render; render()
elif choice == "Live Detection":
    from pages.Live_Detection import render; render(get_face_detector(), detector)
elif choice == "Image Detection":
    from pages.Image_Detection import render; render(get_face_detector(), detector)
elif choice == "Video Detection":
    from pages.Video_Detection import render; render(get_face_detector(), detector)
elif choice == "Analytics":
    from pages.Analytics import render; render()
elif choice == "History":
    from pages.History import render; render()
