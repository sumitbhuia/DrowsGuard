"""
inference.py — Model loading + per-session prediction logic
Loads MobileNetV2 + LSTM once at startup, holds a ring buffer per session.
"""

import cv2
import numpy as np
from collections import deque
from pathlib import Path

SEQ_LEN = 30  # must match training

# ── Lazy globals (loaded once on first use) ───────────────────────────────────
_face_cascade = None
_cnn          = None
_lstm         = None

# Per-session ring buffer:  session_id → deque of (1280,) feature vectors
_session_buffers: dict[str, deque] = {}


def _load_models():
    """Load all models into memory once."""
    global _face_cascade, _cnn, _lstm

    if _face_cascade is not None:
        return  # already loaded

    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2

    print("⏳ Loading face cascade...")
    _face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Load MobileNetV2 — prefer saved .h5 if available, else download weights
    cnn_path = Path("mobilenetv2_feature_extractor.h5")
    if cnn_path.exists():
        print("⏳ Loading MobileNetV2 from disk...")
        _cnn = tf.keras.models.load_model(str(cnn_path))
    else:
        print("⏳ Downloading MobileNetV2 weights (first run only)...")
        _cnn = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
    _cnn.trainable = False

    # Load LSTM
    lstm_path = Path("best_model.h5")
    if not lstm_path.exists():
        raise FileNotFoundError(
            "best_model.h5 not found. Export it from Colab first.\n"
            "See README.md → Step 1."
        )
    print("⏳ Loading LSTM model...")
    _lstm = tf.keras.models.load_model(str(lstm_path))

    print("✅ All models loaded.")


def get_buffer(session_id: str) -> deque:
    if session_id not in _session_buffers:
        _session_buffers[session_id] = deque(maxlen=SEQ_LEN)
    return _session_buffers[session_id]


def clear_session(session_id: str):
    _session_buffers.pop(session_id, None)


def process_frame(jpeg_bytes: bytes, session_id: str) -> dict:
    """
    Process one JPEG frame for a given session.

    Returns a dict with keys:
      status      : "ok" | "no_face" | "buffering"
      drowsy      : bool or None
      confidence  : float 0-1 or None
      buffer_fill : int (how many frames buffered so far)
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    _load_models()  # no-op after first call

    # ── Decode JPEG ───────────────────────────────────────────────────────────
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"status": "decode_error", "drowsy": None,
                "confidence": None, "buffer_fill": 0}

    # ── Face detection ────────────────────────────────────────────────────────
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        buf = get_buffer(session_id)
        return {"status": "no_face", "drowsy": None,
                "confidence": None, "buffer_fill": len(buf)}

    # Crop largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = frame[y:y+h, x:x+w]
    face = cv2.resize(face, (224, 224))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # ── MobileNetV2 feature extraction ───────────────────────────────────────
    inp  = preprocess_input(face.astype(np.float32))[np.newaxis]  # (1,224,224,3)
    feat = _cnn.predict(inp, verbose=0)[0]                         # (1280,)

    # ── Update ring buffer ───────────────────────────────────────────────────
    buf = get_buffer(session_id)
    buf.append(feat)

    if len(buf) < SEQ_LEN:
        return {
            "status":      "buffering",
            "drowsy":      None,
            "confidence":  None,
            "buffer_fill": len(buf),
        }

    # ── LSTM inference ────────────────────────────────────────────────────────
    sequence = np.array(buf)[np.newaxis]           # (1, 30, 1280)
    prob     = float(_lstm.predict(sequence, verbose=0)[0][0])

    return {
        "status":      "ok",
        "drowsy":      prob > 0.5,
        "confidence":  round(prob, 4),
        "buffer_fill": SEQ_LEN,
    }
