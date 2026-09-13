import streamlit as st
from database.database import get_history
from services.analytics import confidence_chart
from utils.constants import EMOTIONS, EMOTION_COLORS
import plotly.express as px
import pandas as pd


def render():
    st.title("Emotion Analytics")
    data = get_history()
    if data.empty: st.info("Analyze an image, webcam frame, or video to populate analytics."); return
    counts = data["emotion"].value_counts().reindex(EMOTIONS, fill_value=0).to_dict()
    dominant = data["emotion"].mode().iloc[0]
    c1,c2,c3 = st.columns(3); c1.metric("Total faces", len(data)); c2.metric("Dominant emotion", dominant); c3.metric("Average confidence", f'{data.confidence.mean():.1f}%')
    left,right = st.columns(2)
    with left:
        st.subheader("Emotion distribution")
        chart = px.pie(pd.DataFrame({"Emotion":list(counts),"Detections":list(counts.values())}), names="Emotion", values="Detections", hole=.45, color="Emotion", color_discrete_map=EMOTION_COLORS)
        st.plotly_chart(chart, use_container_width=True)
    with right:
        st.subheader("Average confidence")
        chart = confidence_chart(data)
        if chart: st.plotly_chart(chart, use_container_width=True)
    st.subheader("Detection timeline")
    timeline = data.sort_values("timestamp").reset_index(); timeline["sequence"] = timeline.index + 1
    st.plotly_chart(px.scatter(timeline, x="sequence", y="emotion", color="emotion", color_discrete_map=EMOTION_COLORS, labels={"sequence":"Detection sequence"}), use_container_width=True)
