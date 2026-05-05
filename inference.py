"""
inference.py — Model loading, calibration, and per-session prediction logic

Changes from v1:
  - Per-session calibration: 30s of alert-state LSTM outputs → personal threshold
  - Overlapping stride windows (3 × SEQ_LEN windows, majority vote) to smooth
    timestamp boundary irregularities from training data
  - Calibration frames that miss a face trigger a full reset of the countdown
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────
SEQ_LEN        = 30      # frames per LSTM window — must match training
LARGE_BUF_LEN  = 60      # holds enough frames for 3 overlapping windows
STRIDE         = 15      # window stride: windows start at 0, 15, 30 in the 60-buf
K_SIGMA        = 2.5     # threshold = cal_mean + K_SIGMA * cal_std
DEFAULT_THRESH = 0.50    # fallback if no calibration
CAL_WARMUP     = 30      # discard first 30 probs (buffer not full = noisy)
CAL_TARGET     = 150     # 150 good frames — ~30s at ~5fps on Render free tier

# ── Lazy globals (loaded once on first use) ───────────────────────────────────
_face_cascade = None
_cnn          = None
_lstm         = None


@dataclass
class SessionState:
    # Large ring buffer for overlapping windows
    feature_buf: deque = field(default_factory=lambda: deque(maxlen=LARGE_BUF_LEN))

    # Calibration
    cal_probs:          list  = field(default_factory=list)
    cal_done:           bool  = False
    threshold:          float = DEFAULT_THRESH
    consecutive_no_face: int  = 0   # only reset after N consecutive misses


# Per-session state store
_sessions: dict[str, SessionState] = {}


def _load_models():
    """Load all models into memory once — no-op on subsequent calls."""
    global _face_cascade, _cnn, _lstm

    # Fix for postmortem issue #2: guard ALL three, not just face_cascade.
    # Previously, a partial load left _cnn/_lstm as None causing AttributeError.
    if _face_cascade is not None and _cnn is not None and _lstm is not None:
        return

    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2

    print("⏳ Loading MediaPipe face detector...")
    import mediapipe as mp
    _face_cascade = mp.solutions.face_detection.FaceDetection(
        model_selection=0,       # 0 = short range (< 2m) — webcam use case
        min_detection_confidence=0.5
    )

    # MobileNetV2 — prefer saved file, else download weights
    cnn_path = Path("mobilenetv2_feature_extractor.h5")
    if cnn_path.exists():
        print("⏳ Loading MobileNetV2 from disk...")
        _cnn = tf.keras.models.load_model(str(cnn_path))
    else:
        print("⏳ Downloading MobileNetV2 weights (first run only)...")
        _cnn = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
    _cnn.trainable = False

    # LSTM — detect format: Keras 3.x saves as a FOLDER named best_model.keras
    # (containing config.json + model.weights.h5), not a single file.
    # tf.keras.models.load_model handles both folder and single-file formats
    # as long as we pass the correct path type.
    lstm_path = None
    for candidate in ["best_model.h5","best_model.keras"]:
        p = Path(candidate)
        if p.exists():          # exists() is True for both files AND folders
            lstm_path = str(p)
            break

    if lstm_path is None:
        raise FileNotFoundError(
            "Model not found. Expected 'best_model.keras' (folder) or "
            "'best_model.h5' in the project root. See README Step 1."
        )

    print(f"⏳ Loading LSTM from {lstm_path} ...")
    # For Keras 3.x folder format, keras.saving.load_model handles it natively.
    # We call keras directly (not tf.keras) to bypass the TF wrapper's
    # open() call which fails on directories.
    import keras
    _lstm = keras.saving.load_model(lstm_path, safe_mode=False)
    print("✅ All models loaded.")


def _get_session(session_id: str) -> SessionState:
    if session_id not in _sessions:
        _sessions[session_id] = SessionState()
    return _sessions[session_id]


def clear_session(session_id: str):
    _sessions.pop(session_id, None)


def _extract_face_feature(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect face using MediaPipe (handles tilts, nods, partial faces),
    crop, run MobileNetV2.
    Returns (1280,) feature vector or None if no face found.
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results   = _face_cascade.process(frame_rgb)

    if not results.detections:
        return None

    # Pick detection with highest confidence
    det = max(results.detections, key=lambda d: d.score[0])
    bb  = det.location_data.relative_bounding_box

    # Convert relative coords to absolute — clamp to frame bounds
    x1 = max(0, int(bb.xmin * w))
    y1 = max(0, int(bb.ymin * h))
    x2 = min(w, int((bb.xmin + bb.width)  * w))
    y2 = min(h, int((bb.ymin + bb.height) * h))

    if x2 <= x1 or y2 <= y1:
        return None

    face = frame_bgr[y1:y2, x1:x2]
    face = cv2.resize(face, (224, 224))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    inp  = preprocess_input(face.astype(np.float32))[np.newaxis]
    feat = _cnn.predict(inp, verbose=0)[0]
    return feat


def _run_lstm(feature_sequence: np.ndarray) -> float:
    """Run LSTM on (SEQ_LEN, 1280) sequence → scalar probability."""
    seq = feature_sequence[np.newaxis]                              # (1,30,1280)
    return float(_lstm.predict(seq, verbose=0)[0][0])


def _majority_vote_prediction(session: SessionState) -> tuple[float, float]:
    """
    Run 3 overlapping LSTM windows over the 60-frame buffer.
    Strides: frames[0:30], frames[15:45], frames[30:60].
    Returns (mean_prob, voted_prob) — voted is the median of 3 windows.

    This smooths the timestamp boundary irregularity from training:
    contaminated boundary frames get outvoted by clean central frames.
    """
    buf = np.array(session.feature_buf)   # (60, 1280)

    probs = []
    for start in range(0, LARGE_BUF_LEN - SEQ_LEN + 1, STRIDE):  # 0, 15, 30
        window = buf[start:start + SEQ_LEN]                        # (30, 1280)
        probs.append(_run_lstm(window))

    mean_prob   = float(np.mean(probs))
    voted_prob  = float(np.median(probs))   # median of 3 = robust majority
    return mean_prob, voted_prob


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_frame(jpeg_bytes: bytes, session_id: str) -> dict:
    """
    Called during the 30-second calibration phase.

    Each frame:
      - Extracts face feature → runs LSTM on a growing buffer
      - On NO FACE: resets calibration entirely (caller restarts countdown)
      - Accumulates LSTM alert-state probabilities
      - On completion (n_frames reached): computes personal threshold

    Returns:
      status        : "collecting" | "no_face_reset" | "complete"
      collected     : int  — how many good frames so far
      target        : int  — total frames needed
      threshold     : float | None  — set only when status == "complete"
    """
    _load_models()

    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"status": "decode_error", "collected": 0, "target": SEQ_LEN * 10}

    session = _get_session(session_id)

    feat = _extract_face_feature(frame)

    # ── No face → only reset after 5 consecutive misses ──────────────────────
    if feat is None:
        session.consecutive_no_face += 1
        if session.consecutive_no_face >= 5:
            session.cal_probs.clear()
            session.feature_buf.clear()
            session.consecutive_no_face = 0
            return {
                "status":    "no_face_reset",
                "collected": 0,
                "target":    CAL_TARGET,
                "threshold": None,
            }
        # Not enough consecutive misses yet — keep collecting, return current state
        return {
            "status":    "collecting",
            "collected": len(session.cal_probs),
            "target":    CAL_TARGET,
            "threshold": None,
        }

    # Face found — reset consecutive counter
    session.consecutive_no_face = 0

    # Accumulate feature in buffer (we need SEQ_LEN to run LSTM)
    session.feature_buf.append(feat)

    # Only start collecting probs once buffer is full AND warmup passed
    if len(session.feature_buf) >= SEQ_LEN:
        buf_arr = np.array(session.feature_buf)
        prob = _run_lstm(buf_arr[-SEQ_LEN:])
        # Skip first CAL_WARMUP predictions — buffer not full, very noisy
        if len(session.cal_probs) >= CAL_WARMUP or \
           len(session.feature_buf) >= LARGE_BUF_LEN:
            session.cal_probs.append(prob)

    collected = len(session.cal_probs)

    # ── Calibration complete ──────────────────────────────────────────────────
    if collected >= CAL_TARGET:
        probs_arr = np.array(session.cal_probs)
        cal_mean  = float(probs_arr.mean())
        cal_std   = float(probs_arr.std())

        # Personal threshold: alert baseline + K_SIGMA standard deviations
        # Clipped to [0.35, 0.92] — wide ceiling for high-baseline users
        personal_thresh = float(np.clip(
            cal_mean + K_SIGMA * cal_std,
            0.35, 0.92
        ))
        session.threshold = personal_thresh
        session.cal_done  = True
        session.cal_probs.clear()
        session.feature_buf.clear()

        print(f"✅ Calibration complete — session={session_id[:8]} "
              f"mean={cal_mean:.3f} std={cal_std:.3f} "
              f"threshold={personal_thresh:.3f}")

        return {
            "status":    "complete",
            "collected": collected,
            "target":    CAL_TARGET,
            "threshold": round(personal_thresh, 4),
        }

    return {
        "status":    "collecting",
        "collected": collected,
        "target":    CAL_TARGET,
        "threshold": None,
    }

    return {
        "status":    "collecting",
        "collected": collected,
        "target":    target,
        "threshold": None,
    }


def process_frame(jpeg_bytes: bytes, session_id: str) -> dict:
    """
    Live inference — called after calibration is complete.

    Pipeline:
      face detect → MobileNetV2 → 60-frame ring buffer →
      3 overlapping LSTM windows → median vote → threshold comparison

    Returns:
      status      : "ok" | "no_face" | "buffering"
      drowsy      : bool | None
      confidence  : float | None   (median probability across 3 windows)
      buffer_fill : int
      threshold   : float          (personal or default)
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    _load_models()

    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"status": "decode_error", "drowsy": None,
                "confidence": None, "buffer_fill": 0, "threshold": DEFAULT_THRESH}

    session = _get_session(session_id)
    feat    = _extract_face_feature(frame)

    if feat is None:
        return {
            "status":      "no_face",
            "drowsy":      None,
            "confidence":  None,
            "buffer_fill": len(session.feature_buf),
            "threshold":   session.threshold,
        }

    session.feature_buf.append(feat)

    # Need full 60-frame buffer for 3 overlapping windows
    if len(session.feature_buf) < LARGE_BUF_LEN:
        return {
            "status":      "buffering",
            "drowsy":      None,
            "confidence":  None,
            "buffer_fill": len(session.feature_buf),
            "threshold":   session.threshold,
        }

    # ── Overlapping stride prediction ─────────────────────────────────────────
    _, voted_prob = _majority_vote_prediction(session)

    return {
        "status":      "ok",
        "drowsy":      voted_prob > session.threshold,
        "confidence":  round(voted_prob, 4),
        "buffer_fill": LARGE_BUF_LEN,
        "threshold":   round(session.threshold, 4),
    }