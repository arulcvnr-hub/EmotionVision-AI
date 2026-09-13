import streamlit as st
from database.database import get_history, clear_history
from utils.constants import EMOTIONS


def render():
    st.title("Detection History")
    c1,c2 = st.columns(2)
    kind = c1.selectbox("Detection type", ["All", "Webcam", "Image", "Video"])
    emotion = c2.selectbox("Emotion", ["All"] + EMOTIONS)
    data = get_history(kind, emotion)
    if st.button("Clear all history", type="secondary"):
        clear_history(); st.success("History cleared."); st.rerun()
    st.metric("Matching detections", len(data))
    if data.empty: st.info("No detections match the selected filters.")
    else:
        data.columns = ["Timestamp", "Type", "Emotion", "Confidence (%)"]
        st.dataframe(data, hide_index=True, use_container_width=True)
