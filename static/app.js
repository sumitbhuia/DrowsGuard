/**
 * app.js — DrowsGuard v2 frontend
 *
 * Session flow:
 *   1. User clicks Start → camera opens → calibration phase (30s)
 *   2. Calibration: frames → POST /calibrate at 10fps
 *      - No face → backend resets → countdown restarts
 *      - Complete → personal threshold set → live inference begins
 *   3. Live inference: frames → POST /predict
 *      - 3 overlapping LSTM windows → median vote → threshold compare
 */

// ── Config ────────────────────────────────────────────────────────────────────
const API_URL    = '';
const FRAME_MS   = 100;               // 10fps
const SESSION_ID = crypto.randomUUID();
const CAL_TOTAL  = 300;               // 300 frames @ 10fps = 30s
const BUF_TOTAL  = 60;                // large buffer for 3 overlapping windows
const RING_C     = 2 * Math.PI * 50; // SVG circle r=50 circumference ≈ 314

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
const thresholdVal  = document.getElementById('threshold-val');
const calOverlay    = document.getElementById('cal-overlay');
const calRingFill   = document.getElementById('cal-ring-fill');
const calCount      = document.getElementById('cal-count');
const calTitle      = document.getElementById('cal-title');
const calSubtitle   = document.getElementById('cal-subtitle');

// ── State ─────────────────────────────────────────────────────────────────────
let intervalId     = null;
let stream         = null;
let appPhase       = 'idle';   // 'idle' | 'calibrating' | 'monitoring'
let personalThresh = null;
let lastDrowsy     = false;
let drowsyStart    = null;
let frameCount     = 0;
let fpsTimer       = Date.now();
let sending        = false;    // prevent overlapping fetches

const canvas   = document.createElement('canvas');
canvas.width   = 224;
canvas.height  = 224;
const ctx      = canvas.getContext('2d');

// ── Clock ─────────────────────────────────────────────────────────────────────
setInterval(() => {
  clockEl.textContent = new Date().toTimeString().slice(0, 8);
}, 1000);
clockEl.textContent = new Date().toTimeString().slice(0, 8);

// ── Logging ───────────────────────────────────────────────────────────────────
function log(message, type = '') {
  const li   = document.createElement('li');
  const time = new Date().toTimeString().slice(0, 8);
  li.innerHTML = `<span class="log-time">${time}</span><span>${message}</span>`;
  if (type) li.classList.add(`log-${type}`);
  logList.prepend(li);
  while (logList.children.length > 50) logList.removeChild(logList.lastChild);
}

// ── Calibration UI ────────────────────────────────────────────────────────────
function showCalOverlay(visible) {
  if (visible) calOverlay.classList.remove('hidden');
  else         calOverlay.classList.add('hidden');
}

function updateCalRing(collected, target, isReset) {
  const frac   = Math.min(collected / target, 1);
  const offset = (RING_C * (1 - frac)).toFixed(1);
  calRingFill.style.strokeDashoffset = offset;

  const secsLeft = Math.ceil((target - collected) / 10);
  calCount.textContent = isReset ? '!' : Math.max(secsLeft, 0);

  calOverlay.classList.remove('reset', 'done');
  if (isReset) calOverlay.classList.add('reset');
}

function calComplete(threshold) {
  calOverlay.classList.add('done');
  calRingFill.style.strokeDashoffset = '0';
  calCount.textContent    = '✓';
  calTitle.textContent    = 'Calibrated';
  calSubtitle.textContent = `PERSONAL THRESHOLD SET TO ${Math.round(threshold * 100)}%`;
  setTimeout(() => showCalOverlay(false), 1800);
}

// ── Status UI ─────────────────────────────────────────────────────────────────
function setStateAlert() {
  statusText.textContent         = 'ALERT';
  statusText.className           = 'alert';
  statusIcon.textContent         = '👁️';
  meterFill.style.background     = 'var(--alert-color)';
  statusBarFill.style.background = 'var(--alert-color)';
  alertOverlay.classList.remove('active');
}

function setStateDrowsy() {
  statusText.textContent         = 'DROWSY!';
  statusText.className           = 'drowsy';
  statusIcon.textContent         = '😴';
  meterFill.style.background     = 'var(--danger-color)';
  statusBarFill.style.background = 'var(--danger-color)';
  alertOverlay.classList.add('active');
}

function setStateBuffering() {
  statusText.textContent = 'WARMING UP';
  statusText.className   = 'warn';
  statusIcon.textContent = '⏳';
  alertOverlay.classList.remove('active');
}

function setStateNoFace() {
  statusText.textContent = 'NO FACE';
  statusText.className   = '';
  statusIcon.textContent = '🔍';
  alertOverlay.classList.remove('active');
}

function setStateCalibrating() {
  statusText.textContent = 'CALIBRATING';
  statusText.className   = 'warn';
  statusIcon.textContent = '📡';
  alertOverlay.classList.remove('active');
}

function setStateStandby() {
  statusText.textContent   = 'STANDBY';
  statusText.className     = '';
  statusIcon.textContent   = '😐';
  confValue.textContent    = '--.-%';
  meterFill.style.width    = '0%';
  bufferFill.style.width   = '0%';
  bufferLabel.textContent  = `BUFFER: 0 / ${BUF_TOTAL}`;
  statusBarFill.style.width = '0%';
  alertOverlay.classList.remove('active');
}

function updateConfidence(confidence) {
  const pct = Math.round(confidence * 100);
  confValue.textContent     = `${pct}%`;
  meterFill.style.width     = `${pct}%`;
  statusBarFill.style.width = `${pct}%`;
}

function updateBuffer(fill, total) {
  const pct = Math.round((fill / total) * 100);
  bufferFill.style.width  = `${pct}%`;
  bufferLabel.textContent = `BUFFER: ${fill} / ${total}`;
}

function updateThresholdBadge(threshold) {
  if (threshold == null) return;
  const pct = Math.round(threshold * 100);
  const tag = personalThresh ? 'PERSONAL' : 'DEFAULT';
  thresholdVal.textContent = `${tag} (${pct}%)`;
}

// ── FPS ───────────────────────────────────────────────────────────────────────
function tickFPS() {
  frameCount++;
  const now = Date.now();
  if (now - fpsTimer >= 2000) {
    fpsLabel.textContent = `${(frameCount / ((now - fpsTimer) / 1000)).toFixed(1)} FPS`;
    frameCount = 0;
    fpsTimer   = now;
  }
}

// ── Frame capture ─────────────────────────────────────────────────────────────
function captureBlob() {
  return new Promise((resolve) => {
    if (!video.srcObject || video.readyState < 2) return resolve(null);
    // Un-mirror the video so face detection sees correct orientation
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, -224, 0, 224, 224);
    ctx.restore();
    canvas.toBlob(resolve, 'image/jpeg', 0.8);
  });
}

// ── POST frame ────────────────────────────────────────────────────────────────
async function postFrame(endpoint, blob) {
  const form = new FormData();
  form.append('frame', blob, 'frame.jpg');
  const res = await fetch(`${API_URL}${endpoint}`, {
    method:  'POST',
    headers: { 'x-session-id': SESSION_ID },
    body:    form,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ── Calibration tick ──────────────────────────────────────────────────────────
async function calibrationTick() {
  if (sending) return;         // skip if previous frame still in flight
  sending = true;

  try {
    const blob = await captureBlob();
    if (!blob) return;

    const data = await postFrame('/calibrate', blob);
    tickFPS();

    if (data.status === 'no_face_reset') {
      updateCalRing(0, CAL_TOTAL, true);
      log('⚠ Face lost — calibration reset', 'warn');
      return;
    }

    if (data.status === 'collecting') {
      updateCalRing(data.collected, data.target, false);
      return;
    }

    if (data.status === 'complete') {
      personalThresh = data.threshold;
      updateThresholdBadge(data.threshold);
      calComplete(data.threshold);
      log(`✓ Calibrated — threshold ${Math.round(data.threshold * 100)}%`, 'ok');

      // Switch to inference after animation
      setTimeout(() => {
        appPhase    = 'monitoring';
        lastDrowsy  = false;
        drowsyStart = null;
        setStateBuffering();
      }, 1800);
    }

  } catch (err) {
    log(`Calibration error: ${err.message}`, 'danger');
  } finally {
    sending = false;
  }
}

// ── Inference tick ────────────────────────────────────────────────────────────
async function inferenceTick() {
  if (sending) return;         // skip if previous frame still in flight
  sending = true;

  try {
    const blob = await captureBlob();
    if (!blob) return;

    const data = await postFrame('/predict', blob);
    tickFPS();

    const { status, drowsy, confidence, buffer_fill, threshold } = data;
    updateThresholdBadge(threshold);

    if (status === 'buffering') {
      setStateBuffering();
      updateBuffer(buffer_fill ?? 0, BUF_TOTAL);
      return;
    }

    if (status === 'no_face') {
      setStateNoFace();
      updateBuffer(buffer_fill ?? 0, BUF_TOTAL);
      return;
    }

    if (status === 'ok') {
      updateBuffer(BUF_TOTAL, BUF_TOTAL);
      updateConfidence(confidence);

      if (drowsy) {
        setStateDrowsy();
        if (!lastDrowsy) {
          drowsyStart = Date.now();
          log(`⚠ Drowsy detected — ${Math.round(confidence * 100)}% (thresh ${Math.round(threshold * 100)}%)`, 'danger');
        }
      } else {
        setStateAlert();
        if (lastDrowsy && drowsyStart) {
          const secs = ((Date.now() - drowsyStart) / 1000).toFixed(1);
          log(`✓ Alert resumed — episode ${secs}s`, 'ok');
          drowsyStart = null;
        }
      }
      lastDrowsy = drowsy;
    }

  } catch (err) {
    log(`Inference error: ${err.message}`, 'danger');
  } finally {
    sending = false;
  }
}

// ── Master tick ───────────────────────────────────────────────────────────────
async function tick() {
  if (appPhase === 'calibrating') await calibrationTick();
  else if (appPhase === 'monitoring') await inferenceTick();
}

// ── Start session ─────────────────────────────────────────────────────────────
async function startSession() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    btnStart.disabled = true;
    btnStop.disabled  = false;

    // Reset calibration overlay
    calOverlay.classList.remove('hidden', 'reset', 'done');
    calTitle.textContent    = 'Calibrating';
    calSubtitle.textContent = 'SIT NORMALLY & FACE THE CAMERA\nESTABLISHING YOUR ALERT BASELINE';
    calRingFill.style.strokeDashoffset = RING_C.toFixed(1);
    calCount.textContent = '30';

    appPhase = 'calibrating';
    setStateCalibrating();
    log('📡 Calibration started — sit normally for 30s', 'warn');

    intervalId = setInterval(tick, FRAME_MS);

  } catch (err) {
    log(`Camera error: ${err.message}`, 'danger');
  }
}

// ── Stop session ──────────────────────────────────────────────────────────────
async function stopSession() {
  clearInterval(intervalId);
  intervalId     = null;
  appPhase       = 'idle';
  sending        = false;
  personalThresh = null;
  lastDrowsy     = false;
  drowsyStart    = null;

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;

  showCalOverlay(false);
  btnStart.disabled        = false;
  btnStop.disabled         = true;
  fpsLabel.textContent     = '-- FPS';
  thresholdVal.textContent = 'DEFAULT (50%)';

  setStateStandby();
  log('■ Session stopped');

  // Clean up server-side session
  fetch(`/session/${SESSION_ID}`, { method: 'DELETE' }).catch(() => {});
}

// ── Events ────────────────────────────────────────────────────────────────────
btnStart.addEventListener('click', startSession);
btnStop.addEventListener('click',  stopSession);

window.addEventListener('beforeunload', (e) => {
  if (intervalId) { e.preventDefault(); e.returnValue = ''; }
});
