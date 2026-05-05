# DrowsGuard — Driver Drowsiness Detection

Real-time drowsiness detection using MobileNetV2 + LSTM, served via FastAPI.  
No Node.js required. Pure Python backend + vanilla JS frontend.

---

## Project Structure

```
drowsiness-app/
  ├── main.py                          ← FastAPI server (API + serves frontend)
  ├── inference.py                     ← Face detection + CNN + LSTM logic
  ├── requirements.txt                 ← Python dependencies
  ├── .gitignore
  ├── best_model.h5                    ← YOUR LSTM (export from Colab — see Step 1)
  ├── mobilenetv2_feature_extractor.h5 ← Optional: pre-saved MobileNetV2
  └── static/
        ├── index.html                 ← Webcam UI
        └── app.js                     ← Frontend logic
```

---

## Step 1 — Export your model from Colab

Your Colab uses TF 2.20 / Keras 3.13 — save in `.keras` format (native Keras 3.x).
Add this cell at the end of your Colab notebook and run it:

```python
# Save LSTM in native Keras 3.x format
model.save('/content/best_model.keras')

from google.colab import files
files.download('/content/best_model.keras')

# Optional but recommended — saves MobileNetV2 so server loads faster
from tensorflow.keras.applications import MobileNetV2
cnn = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
cnn.save('/content/mobilenetv2_feature_extractor.h5')
files.download('/content/mobilenetv2_feature_extractor.h5')
```

Place the downloaded files in the project root (same folder as `main.py`).
The server will find `best_model.keras` automatically — no config needed.

---

## Step 2 — First-time local setup (macOS)

```bash
cd drowsiness-app

# Create venv with Python 3.11 (installed via pyenv)
python3.11 -m venv venv
source venv/bin/activate

pip install --upgrade pip

# Use the MAC requirements — NOT requirements.txt (that's for Render/Linux)
pip install -r requirements-mac.txt
```

---

## Step 3 — Run locally

```bash
# Make sure venv is active (you'll see "(venv)" in your terminal prompt)
source venv/bin/activate

uvicorn main:app --reload
```

Open your browser:
- **App:**      http://localhost:8000
- **API docs:** http://localhost:8000/docs  ← test /predict here with Postman-like UI

---

## Step 4 — Deploy to Render (free)

1. Push this folder to a GitHub repo
   *(`.h5`/`.keras` model files are in `.gitignore` — upload them via Render's Persistent Disk or store in a private S3 bucket)*
2. Go to [render.com](https://render.com) → **New Web Service** → connect repo
3. Render will auto-detect `render.yaml` — no manual settings needed
4. Or set manually:
   - **Build command:** `pip install -r requirements.txt`  ← Linux, no metal
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance:** Free (upgrade to $7/mo Starter if RAM exceeded)

> ⚠️ Never set Render's build command to `requirements-mac.txt` — that file has `tensorflow-metal` which will crash on Linux.

### Keep Render warm
Set up a free [UptimeRobot](https://uptimerobot.com) monitor pinging:
`https://your-app.onrender.com/health` every 5 minutes.

---

## Daily workflow

```bash
# Start working
cd drowsiness-app
source venv/bin/activate
uvicorn main:app --reload

# Done for the day
deactivate
```

## Free up storage (when project is paused)

```bash
pip freeze > requirements.txt   # save current deps (if you added any)
deactivate
rm -rf venv/                    # removes all installed packages
pip cache purge                 # clears pip download cache

# To restore later:
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

---

## How it works

```
Browser (10fps)
  └── Webcam frame (224×224 JPEG)
        └── POST /predict  +  x-session-id header

FastAPI
  └── Haar Cascade → find largest face
  └── MobileNetV2 → 1280-dim feature vector
  └── Ring buffer (30 frames per session)
  └── LSTM → drowsy probability
        └── { status, drowsy, confidence, buffer_fill }

Frontend
  └── confidence meter + status badge + event log
  └── Red flash overlay when drowsy detected
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `best_model.h5 not found` | Export from Colab (Step 1) and place in project root |
| Camera permission denied | Use `http://localhost` not `127.0.0.1` (Chrome treats them differently) |
| Render OOM crash (free tier) | Upgrade to $7/mo Starter — MobileNetV2 needs ~400MB |
| Cold start slow | Add UptimeRobot ping to `/health` every 5 min |
| `No face detected` | Improve lighting, face the camera directly |
