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

Add this cell at the end of your Colab notebook and run it:

```python
# Save & download the trained LSTM
from google.colab import files
files.download('/content/best_model.h5')

# Optional but recommended — saves MobileNetV2 so server loads faster
from tensorflow.keras.applications import MobileNetV2
cnn = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
cnn.save('/content/mobilenetv2_feature_extractor.h5')
files.download('/content/mobilenetv2_feature_extractor.h5')
```

Place the downloaded `.h5` files in the project root (same folder as `main.py`).

---

## Step 2 — First-time local setup

```bash
# 1. Clone / navigate to project folder
cd drowsiness-app

# 2. Create virtual environment (isolated — won't touch system Python)
python3 -m venv venv

# 3. Activate it
source venv/bin/activate        # Mac / Linux
# venv\Scripts\activate         # Windows

# 4. Install dependencies (~1-1.5GB, one-time)
pip install -r requirements.txt
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
   *(the `.h5` files are in `.gitignore` — upload them manually or use Git LFS)*
2. Go to [render.com](https://render.com) → **New Web Service** → connect repo
3. Settings:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance:** Free (upgrade to $7/mo Starter if you hit RAM limits)
4. Add model files via Render's **Persistent Disk** or store them in a private S3 bucket

### Keep Render warm (avoid cold start latency)
Set up a free [UptimeRobot](https://uptimerobot.com) monitor on:
`https://your-app.onrender.com/health` — ping every 5 minutes.

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
