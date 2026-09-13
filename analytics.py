from __future__ import annotations
import pandas as pd
import plotly.express as px
from utils.constants import EMOTION_COLORS


def distribution_chart(counts):
    frame = pd.DataFrame({"Emotion": list(counts), "Detections": list(counts.values())})
    return px.pie(frame, names="Emotion", values="Detections", hole=.48, color="Emotion", color_discrete_map=EMOTION_COLORS)


def confidence_chart(df):
    if df.empty: return None
    grouped = df.groupby("emotion", as_index=False)["confidence"].mean().sort_values("confidence", ascending=False)
    return px.bar(grouped, x="emotion", y="confidence", color="emotion", color_discrete_map=EMOTION_COLORS,
                  labels={"confidence": "Average confidence (%)", "emotion": "Emotion"})


def timeline_chart(timeline):
    frame = pd.DataFrame(timeline)
    if frame.empty: return None
    return px.scatter(frame, x="frame", y="emotion", color="emotion", color_discrete_map=EMOTION_COLORS,
                      labels={"frame": "Frame", "emotion": "Detected emotion"})
