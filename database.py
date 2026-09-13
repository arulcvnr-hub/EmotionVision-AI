from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from utils.constants import DB_PATH


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    with _connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            detection_type TEXT NOT NULL,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL
        )""")
        conn.commit()


def add_detection(detection_type: str, emotion: str, confidence: float):
    with _connect() as conn:
        conn.execute("INSERT INTO detections(timestamp,detection_type,emotion,confidence) VALUES (?,?,?,?)",
                     (datetime.now(timezone.utc).isoformat(timespec="seconds"), detection_type, emotion, confidence))
        conn.commit()


def get_history(detection_type: str | None = None, emotion: str | None = None) -> pd.DataFrame:
    query = "SELECT timestamp, detection_type, emotion, confidence FROM detections WHERE 1=1"
    params = []
    if detection_type and detection_type != "All": query += " AND detection_type=?"; params.append(detection_type)
    if emotion and emotion != "All": query += " AND emotion=?"; params.append(emotion)
    query += " ORDER BY timestamp DESC"
    with _connect() as conn:
        return pd.read_sql_query(query, conn, params=params)


def clear_history():
    with _connect() as conn:
        conn.execute("DELETE FROM detections")
        conn.commit()
