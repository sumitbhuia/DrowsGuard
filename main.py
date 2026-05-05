"""
main.py — FastAPI server
Serves the webcam frontend AND handles /predict inference.

Run locally:
    uvicorn main:app --reload

Then open:  http://localhost:8000
API docs:   http://localhost:8000/docs
"""

import uuid
from fastapi import FastAPI, File, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from inference import process_frame, calibrate_frame, clear_session

app = FastAPI(title="Driver Drowsiness Detection API")

# ── CORS (needed if you ever separate frontend from backend) ──────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve the static frontend ─────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ── Health check (used by UptimeRobot to keep Render warm) ───────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Calibration endpoint ──────────────────────────────────────────────────────
@app.post("/calibrate")
async def calibrate(
    frame: UploadFile = File(..., description="JPEG frame during alert calibration"),
    x_session_id: str = Header(default=None),
):
    """
    Called during the 30-second calibration phase (before live monitoring).
    Send frames at 10fps. Backend accumulates 300 alert-state LSTM probabilities
    to compute a personal drowsiness threshold.

    On no-face: resets calibration — frontend restarts countdown.
    On complete: returns personal threshold used for the rest of the session.
    """
    session_id = x_session_id or str(uuid.uuid4())
    jpeg_bytes = await frame.read()
    result     = calibrate_frame(jpeg_bytes, session_id)
    return result


# ── Main inference endpoint ───────────────────────────────────────────────────
@app.post("/predict")
async def predict(
    frame: UploadFile = File(..., description="JPEG frame from webcam"),
    x_session_id: str = Header(default=None, description="Stable browser session ID"),
):
    """
    Accepts one JPEG frame, runs face detection → MobileNetV2 → LSTM.
    Returns drowsiness prediction once 30 frames are buffered.
    """
    session_id = x_session_id or str(uuid.uuid4())
    jpeg_bytes = await frame.read()
    result     = process_frame(jpeg_bytes, session_id)
    return result


# ── Clear session buffer (call when user stops the camera) ───────────────────
@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    clear_session(session_id)
    return {"cleared": session_id}
