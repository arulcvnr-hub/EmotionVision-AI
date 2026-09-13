# EmotionVision AI

**Advanced real-time face emotion detection with Python, OpenCV, TensorFlow/Keras, and Streamlit.** The application is designed to analyze faces from uploaded images, video files, and browser camera snapshots while preserving detection history in SQLite.

## Features

- Dark AI-themed dashboard with model readiness status and summary metrics.
- Haar-cascade face detection with support for multiple faces per frame.
- Seven-class Keras emotion inference: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
- Image analysis with annotated output and interactive probability distribution.
- Video processing with frame stride control, annotated MP4 output, dominant emotion, and timeline.
- Browser webcam snapshot analysis with secure native Streamlit camera access.
- Plotly analytics and filterable SQLite detection history.
- Defensive handling for missing models, invalid files, unavailable cameras, and no-face cases.

## Technology

Python 3.10+, Streamlit, OpenCV, TensorFlow/Keras, NumPy, Pandas, Plotly, Pillow, and SQLite.

## Folder structure

```text
EmotionVision-AI/
├── app.py                 # Application shell and navigation
├── requirements.txt
├── models/emotion_model.h5 # Add a real compatible trained model here
├── services/              # Face, emotion, video, and analytics services
├── database/              # SQLite persistence
├── utils/                 # Constants and image helpers
├── pages/                 # Streamlit page renderers
├── assets/styles.css
└── data/                  # Created database location
```

## Installation

```bash
git clone <repository-url>
cd EmotionVision-AI
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Model setup (required for predictions)

This repository intentionally does **not** include a fake or random model. Add a genuine TensorFlow/Keras model at `models/emotion_model.h5`. It must accept a grayscale or RGB face tensor and return exactly seven output probabilities in this order:

`Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral`

The included model uses the output order `Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise`. The adapter reads the model input size, resizes face crops, validates the output class count, and converts the model order into the UI’s display order.

## Run

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit. Use the sidebar to switch between Home, Live Detection, Image Detection, Video Detection, Analytics, and History.

## Screenshots

_Add screenshots of the dashboard, annotated image, and analytics views here._

## Troubleshooting

- **Model missing:** place the real file at `models/emotion_model.h5` and restart Streamlit.
- **Model class-count error:** verify that the model returns seven values and that label order matches `utils/constants.py`.
- **No face detected:** use a well-lit, front-facing image with a sufficiently large face.
- **Video will not open:** convert the file to MP4/H.264 or install the operating system's required codecs.
- **Camera unavailable:** grant browser camera permission and ensure another application is not using the camera.
- **TensorFlow installation issues:** use Python 3.10–3.12 in a fresh virtual environment and follow the TensorFlow platform installation guidance.

## Future improvements

A production deployment could add a WebRTC streaming component for continuous webcam video, GPU inference, model version metadata, user authentication, and exportable analytics reports.
