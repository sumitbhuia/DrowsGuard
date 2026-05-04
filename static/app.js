/**
 * app.js — DrowsGuard frontend logic
 * Captures webcam frames at 10fps, sends to /predict, updates UI.
 * No Node.js, no build step — plain browser JS.
 */

// ── Config ────────────────────────────────────────────────────────────────────
const API_URL     = '';          // empty = same origin (FastAPI serves this file)
const FRAME_MS    = 100;         // 10fps
const SESSION_ID  = crypto.randomUUID();
const SEQ_LEN     = 30;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const video         = document.getElementById('video');
const btnStart      = document.getElementById('btn-start');
const btnStop       = document.getElementById('btn-stop');
const statusText    = document.getElementById('status-text');
const statusIcon    = document.getElementById('status-icon');
const confValue     = document.getElementById('confidence-value');
const meterFill     = document.getElementById('meter-fill');
const bufferFill    = document.getElementById('buffer-fill');
const bufferLabel   = document.getElementById('buffer-label');
const logList       = document.getElementById('log-list');
const alertOverlay  = document.getElementById('alert-overlay');
const statusBarFill = document.getElementById('status-bar-fill');
const fpsLabel      = document.getElementById('fps-label');
const clockEl       = document.getElementById('clock');

// ── State ─────────────────────────────────────────────────────────────────────
let intervalId   = null;
let stream       = null;
let frameCount   = 0;
let fpsTimer     = Date.now();
let lastDrowsy   = false;
let drowsyStart  = null;   // timestamp when drowsiness began
const canvas     = document.createElement('canvas');
canvas.width     = 224;
canvas.height    = 224;
const ctx        = canvas.getContext('2d');

// ── Clock ─────────────────────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  clockEl.textContent = now.toTimeString().slice(0, 8);
}
setInterval(updateClock, 1000);
updateClock();

// ── Log ───────────────────────────────────────────────────────────────────────
function log(message, type = '') {
  const li   = document.createElement('li');
  const time = new Date().toTimeString().slice(0, 8);
  li.innerHTML = `<span class="log-time">${time}</span><span>${message}</span>`;
  if (type) li.classList.add(`log-${type}`);
  logList.prepend(li);
  // Keep only last 50 entries
  while (logList.children.length > 50) logList.removeChild(logList.lastChild);
}

// ── UI update ─────────────────────────────────────────────────────────────────
function setAlert() {
  statusText.textContent = 'ALERT';
  statusText.className   = 'alert';
  statusIcon.textContent = '👁️';
  alertOverlay.classList.remove('active');
  meterFill.style.background = 'var(--alert-color)';
  statusBarFill.style.background = 'var(--alert-color)';
}

function setDrowsy(confidence) {
  statusText.textContent = 'DROWSY!';
  statusText.className   = 'drowsy';
  statusIcon.textContent = '😴';
  alertOverlay.classList.add('active');
  meterFill.style.background = 'var(--danger-color)';
  statusBarFill.style.background = 'var(--danger-color)';
}

function setBuffering(fill) {
  statusText.textContent = 'WARMING UP';
  statusText.className   = 'warn';
  statusIcon.textContent = '⏳';
  alertOverlay.classList.remove('active');
}

function setNoFace() {
  statusText.textContent = 'NO FACE';
  statusText.className   = '';
  statusIcon.textContent = '🔍';
  alertOverlay.classList.remove('active');
}

function setStandby() {
  statusText.textContent = 'STANDBY';
  statusText.className   = '';
  statusIcon.textContent = '😐';
  confValue.textContent  = '--.-%';
  meterFill.style.width  = '0%';
  bufferFill.style.width = '0%';
  bufferLabel.textContent = `BUFFER: 0 / ${SEQ_LEN}`;
  alertOverlay.classList.remove('active');
  statusBarFill.style.width = '0%';
}

function updateConfidence(confidence) {
  const pct = Math.round(confidence * 100);
  confValue.textContent    = `${pct}%`;
  meterFill.style.width    = `${pct}%`;
  statusBarFill.style.width = `${pct}%`;
}

function updateBuffer(fill) {
  const pct = Math.round((fill / SEQ_LEN) * 100);
  bufferFill.style.width  = `${pct}%`;
  bufferLabel.textContent = `BUFFER: ${fill} / ${SEQ_LEN}`;
}

// ── FPS counter ───────────────────────────────────────────────────────────────
function updateFPS() {
  frameCount++;
  const now     = Date.now();
  const elapsed = (now - fpsTimer) / 1000;
  if (elapsed >= 2) {
    fpsLabel.textContent = `${(frameCount / elapsed).toFixed(1)} FPS`;
    frameCount = 0;
    fpsTimer   = now;
  }
}

// ── Send one frame ────────────────────────────────────────────────────────────
async function sendFrame() {
  if (!video.srcObject || video.readyState < 2) return;

  // Draw to canvas (mirrored back for correct face detection)
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(video, -224, 0, 224, 224);
  ctx.restore();

  canvas.toBlob(async (blob) => {
    if (!blob) return;

    try {
      const form = new FormData();
      form.append('frame', blob, 'frame.jpg');

      const res  = await fetch(`${API_URL}/predict`, {
        method:  'POST',
        headers: { 'x-session-id': SESSION_ID },
        body:    form,
      });

      if (!res.ok) { log(`API error ${res.status}`, 'danger'); return; }

      const data = await res.json();
      handleResult(data);
      updateFPS();

    } catch (err) {
      log(`Network error: ${err.message}`, 'danger');
    }
  }, 'image/jpeg', 0.8);
}

// ── Handle API result ─────────────────────────────────────────────────────────
function handleResult(data) {
  const { status, drowsy, confidence, buffer_fill } = data;

  if (status === 'buffering') {
    setBuffering(buffer_fill);
    updateBuffer(buffer_fill ?? 0);
    return;
  }

  if (status === 'no_face') {
    setNoFace();
    updateBuffer(buffer_fill ?? 0);
    return;
  }

  if (status === 'ok') {
    updateBuffer(SEQ_LEN);
    updateConfidence(confidence);

    if (drowsy) {
      setDrowsy(confidence);
      if (!lastDrowsy) {
        drowsyStart = new Date();
        log(`⚠ Drowsiness detected — conf ${Math.round(confidence * 100)}%`, 'danger');
      }
    } else {
      setAlert();
      if (lastDrowsy && drowsyStart) {
        const secs = ((Date.now() - drowsyStart) / 1000).toFixed(1);
        log(`✓ Alert resumed — drowsy for ${secs}s`, 'ok');
        drowsyStart = null;
      }
    }

    lastDrowsy = drowsy;
  }
}

// ── Start monitoring ──────────────────────────────────────────────────────────
async function startMonitor() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    btnStart.disabled = true;
    btnStop.disabled  = false;

    log('▶ Monitoring started', 'ok');
    intervalId = setInterval(sendFrame, FRAME_MS);

  } catch (err) {
    log(`Camera error: ${err.message}`, 'danger');
  }
}

// ── Stop monitoring ───────────────────────────────────────────────────────────
async function stopMonitor() {
  clearInterval(intervalId);
  intervalId = null;

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }

  video.srcObject = null;
  lastDrowsy      = false;
  drowsyStart     = null;

  btnStart.disabled = false;
  btnStop.disabled  = true;
  fpsLabel.textContent = '-- FPS';

  setStandby();
  log('■ Monitoring stopped');

  // Clean up server-side buffer
  fetch(`/session/${SESSION_ID}`, { method: 'DELETE' }).catch(() => {});
}

// ── Events ────────────────────────────────────────────────────────────────────
btnStart.addEventListener('click', startMonitor);
btnStop.addEventListener('click',  stopMonitor);

// Warn before leaving page while monitoring
window.addEventListener('beforeunload', (e) => {
  if (intervalId) {
    e.preventDefault();
    e.returnValue = '';
  }
});
