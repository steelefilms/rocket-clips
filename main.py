import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pymediainfo import MediaInfo

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Store analysis results in memory keyed by filename
# Phase 4 will move this to a database
analysis_results = {}


def probe_video(filepath: str) -> dict:
    media_info = MediaInfo.parse(filepath)
    for track in media_info.tracks:
        if track.track_type == "Video":
            duration = round(float(track.duration or 0) / 1000, 2)
            fps = round(float(track.frame_rate or 30), 2)
            return {
                "duration_seconds": duration,
                "width": int(track.width or 0),
                "height": int(track.height or 0),
                "fps": fps,
                "codec": track.codec_id or "unknown",
                "size_mb": round(os.path.getsize(filepath) / (1024 * 1024), 2),
                "estimated_frames": round(duration * fps),
            }
    raise ValueError("No video stream found.")


def detect_highlights(filepath: str, sample_interval: float = 2.0, top_n: int = 10) -> list:
    """
    Opens the video and samples one frame every sample_interval seconds.
    Computes motion score between consecutive frames.
    Returns the top_n highest-motion timestamps.

    Motion score = average pixel difference between two frames.
    High motion = fast movement = likely an exciting moment.
    """
    cap = cv2.VideoCapture(filepath)

    if not cap.isOpened():
        raise RuntimeError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # How many frames to skip between each sample
    frame_step = int(fps * sample_interval)

    scores = []
    prev_gray = None
    frame_index = 0

    while True:
        # Jump directly to the frame we want — much faster than reading every frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if not ret:
            break

        # Convert to grayscale — color doesn't help motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # Absolute difference between this frame and the previous sample
            diff = cv2.absdiff(prev_gray, gray)
            score = float(np.mean(diff))
            timestamp = frame_index / fps
            scores.append({
                "timestamp": round(timestamp, 2),
                "motion_score": round(score, 4),
                "timestamp_formatted": format_timestamp(timestamp),
            })

        prev_gray = gray
        frame_index += frame_step

        # Safety check — don't go past the end
        if frame_index >= total_frames:
            break

    cap.release()

    if not scores:
        return []

    # Normalize scores to 0-100 range so they're easier to understand
    max_score = max(s["motion_score"] for s in scores)
    if max_score > 0:
        for s in scores:
            s["excitement_score"] = round((s["motion_score"] / max_score) * 100, 1)
    else:
        for s in scores:
            s["excitement_score"] = 0.0

    # Sort by excitement, take top N
    top = sorted(scores, key=lambda x: x["excitement_score"], reverse=True)[:top_n]

    # Re-sort by timestamp so clips appear in chronological order
    top = sorted(top, key=lambda x: x["timestamp"])

    return top


def format_timestamp(seconds: float) -> str:
    """Convert 125.4 seconds → '2:05'"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


def run_analysis(filepath: str, filename: str):
    """
    Runs in the background so the upload response is immediate.
    Results are stored in analysis_results dict.
    """
    try:
        analysis_results[filename] = {"status": "processing"}
        metadata = probe_video(filepath)
        highlights = detect_highlights(filepath)
        analysis_results[filename] = {
            "status": "done",
            "metadata": metadata,
            "highlights": highlights,
        }
    except Exception as e:
        analysis_results[filename] = {
            "status": "error",
            "error": str(e),
        }


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
      <head>
        <title>Rocket Clips</title>
        <style>
          * { box-sizing: border-box; margin: 0; padding: 0; }
          body {
            font-family: sans-serif;
            background: #0f0f0f;
            color: #fff;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            padding: 32px;
          }
          h1 { font-size: 2rem; margin-bottom: 8px; }
          .subtitle { color: #888; margin-bottom: 32px; }
          input[type="file"] { color: #fff; margin-bottom: 16px; display: block; }
          button {
            background: #ff4655;
            color: white;
            border: none;
            padding: 12px 32px;
            font-size: 1rem;
            border-radius: 6px;
            cursor: pointer;
          }
          button:hover { background: #e03545; }
          button:disabled { background: #555; cursor: not-allowed; }
          #result { margin-top: 32px; width: 100%; max-width: 560px; display: none; }
          .card { background: #1a1a1a; border-radius: 10px; padding: 20px 24px; margin-bottom: 12px; }
          .card-title { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #555; margin-bottom: 12px; }
          .stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #222; font-size: 0.95rem; }
          .stat-row:last-child { border-bottom: none; }
          .stat-label { color: #888; }
          .stat-value { color: #fff; font-weight: 500; }
          .success { color: #4caf50; font-size: 1.1rem; margin-bottom: 16px; }
          .error-msg { color: #ff4655; }
          #status { color: #aaa; margin-top: 12px; font-size: 0.9rem; text-align: center; }
          .highlight-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #222;
          }
          .highlight-row:last-child { border-bottom: none; }
          .timestamp { color: #ff4655; font-weight: 600; font-size: 1rem; min-width: 48px; }
          .bar-wrap { flex: 1; margin: 0 12px; background: #222; border-radius: 4px; height: 6px; }
          .bar { background: #ff4655; height: 6px; border-radius: 4px; }
          .score { color: #888; font-size: 0.85rem; min-width: 40px; text-align: right; }
        </style>
      </head>
      <body>
        <h1>🎮 Rocket Clips</h1>
        <p class="subtitle">Phase 3 — Highlight Detection</p>
        <input type="file" id="fileInput" accept=".mp4" />
        <button id="uploadBtn" onclick="uploadFile()">Analyze MP4</button>
        <div id="status"></div>
        <div id="result"></div>

        <script>
          let pollInterval = null;
          let currentFilename = null;

          async function uploadFile() {
            const file = document.getElementById('fileInput').files[0];
            if (!file) { alert('Please select an MP4 file first.'); return; }

            const btn = document.getElementById('uploadBtn');
            const status = document.getElementById('status');
            const resultDiv = document.getElementById('result');

            btn.disabled = true;
            btn.textContent = 'Uploading...';
            status.textContent = '⏳ Uploading file...';
            resultDiv.style.display = 'none';

            const formData = new FormData();
            formData.append('file', file);

            try {
              const response = await fetch('/upload', { method: 'POST', body: formData });
              const data = await response.json();

              if (data.success) {
                currentFilename = data.filename;
                status.textContent = '🔍 Analyzing gameplay... this may take a minute for long recordings.';
                btn.textContent = 'Analyzing...';
                // Poll for results every 3 seconds
                pollInterval = setInterval(() => pollResults(currentFilename), 3000);
              } else {
                status.textContent = '';
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = `<div class="card error-msg">❌ ${data.error}</div>`;
                btn.disabled = false;
                btn.textContent = 'Analyze MP4';
              }
            } catch (err) {
              status.textContent = '';
              resultDiv.style.display = 'block';
              resultDiv.innerHTML = `<div class="card error-msg">❌ Upload failed: ${err.message}</div>`;
              btn.disabled = false;
              btn.textContent = 'Analyze MP4';
            }
          }

          async function pollResults(filename) {
            try {
              const response = await fetch(`/results/${encodeURIComponent(filename)}`);
              const data = await response.json();

              if (data.status === 'done') {
                clearInterval(pollInterval);
                showResults(data, filename);
              } else if (data.status === 'error') {
                clearInterval(pollInterval);
                document.getElementById('status').textContent = '';
                document.getElementById('result').style.display = 'block';
                document.getElementById('result').innerHTML = `<div class="card error-msg">❌ ${data.error}</div>`;
                document.getElementById('uploadBtn').disabled = false;
                document.getElementById('uploadBtn').textContent = 'Analyze MP4';
              }
              // if status === 'processing', keep polling
            } catch (err) {
              console.error('Poll error:', err);
            }
          }

          function showResults(data, filename) {
            const m = data.metadata;
            const mins = Math.floor(m.duration_seconds / 60);
            const secs = (m.duration_seconds % 60).toFixed(0);
            const duration = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;

            let highlightsHtml = '';
            if (data.highlights.length === 0) {
              highlightsHtml = '<div style="color:#888;">No highlights detected.</div>';
            } else {
              data.highlights.forEach((h, i) => {
                highlightsHtml += `
                  <div class="highlight-row">
                    <span class="timestamp">${h.timestamp_formatted}</span>
                    <div class="bar-wrap"><div class="bar" style="width:${h.excitement_score}%"></div></div>
                    <span class="score">${h.excitement_score}%</span>
                  </div>`;
              });
            }

            document.getElementById('status').textContent = '';
            document.getElementById('result').style.display = 'block';
            document.getElementById('result').innerHTML = `
              <div class="card">
                <div class="success">✅ ${filename} analyzed</div>
                <div class="card-title">Video Info</div>
                <div class="stat-row"><span class="stat-label">Duration</span><span class="stat-value">${duration}</span></div>
                <div class="stat-row"><span class="stat-label">Resolution</span><span class="stat-value">${m.width} × ${m.height}</span></div>
                <div class="stat-row"><span class="stat-label">FPS</span><span class="stat-value">${m.fps}</span></div>
                <div class="stat-row"><span class="stat-label">File size</span><span class="stat-value">${m.size_mb} MB</span></div>
              </div>
              <div class="card">
                <div class="card-title">🎯 Top Highlight Moments</div>
                ${highlightsHtml}
              </div>`;

            document.getElementById('uploadBtn').disabled = false;
            document.getElementById('uploadBtn').textContent = 'Analyze MP4';
          }
        </script>
      </body>
    </html>
    """


@app.post("/upload")
async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    if not file.filename.endswith(".mp4"):
        return {"success": False, "error": "Only .mp4 files are allowed."}

    save_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(save_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    # Start analysis in the background — don't make the user wait for upload
    background_tasks.add_task(run_analysis, save_path, file.filename)

    return {"success": True, "filename": file.filename}


@app.get("/results/{filename}")
async def get_results(filename: str):
    if filename not in analysis_results:
        return {"status": "processing"}
    return analysis_results[filename]
