from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "emotion_model.h5"
DB_PATH = BASE_DIR / "data" / "emotion_history.db"
ASSETS_DIR = BASE_DIR / "assets"
MODEL_EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOTION_COLORS = {
    "Angry": "#ef4444", "Disgust": "#84cc16", "Fear": "#a855f7",
    "Happy": "#fbbf24", "Sad": "#38bdf8", "Surprise": "#fb923c", "Neutral": "#94a3b8",
}
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
SUPPORTED_VIDEO_TYPES = ["mp4", "avi", "mov"]
