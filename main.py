import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from pymediainfo import MediaInfo

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


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
          #result { margin-top: 32px; width: 100%; max-width: 480px; display: none; }
          .card { background: #1a1a1a; border-radius: 10px; padding: 20px 24px; margin-bottom: 12px; }
          .card-title { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: #555; margin-bottom: 12px; }
          .stat-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #222; font-size: 0.95rem; }
          .stat-row:last-child { border-bottom: none; }
          .stat-label { color: #888; }
          .stat-value { color: #fff; font-weight: 500; }
          .success { color: #4caf50; font-size: 1.1rem; margin-bottom: 16px; }
          .error { color: #ff4655; }
          #status { color: #aaa; margin-top: 12px; font-size: 0.9rem; }
        </style>
      </head>
      <body>
        <h1>🎮 Rocket Clips</h1>
        <p class="subtitle">Phase 2 — Video Analysis</p>
        <input type="file" id="fileInput" accept=".mp4" />
        <button id="uploadBtn" onclick="uploadFile()">Analyze MP4</button>
        <div id="status"></div>
        <div id="result"></div>
        <script>
          async function uploadFile() {
            const file = document.getElementById('fileInput').files[0];
            if (!file) { alert('Please select an MP4 file first.'); return; }
            const btn = document.getElementById('uploadBtn');
            const status = document.getElementById('status');
            const resultDiv = document.getElementById('result');
            btn.disabled = true;
            btn.textContent = 'Uploading...';
            status.textContent = '⏳ Uploading file to server...';
            resultDiv.style.display = 'none';
            const formData = new FormData();
            formData.append('file', file);
            try {
              const response = await fetch('/upload', { method: 'POST', body: formData });
              const data = await response.json();
              status.textContent = '';
              resultDiv.style.display = 'block';
              if (data.success) {
                const m = data.metadata;
                const mins = Math.floor(m.duration_seconds / 60);
                const secs = (m.duration_seconds % 60).toFixed(1);
                const duration = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
                resultDiv.innerHTML = `
                  <div class="card">
                    <div class="success">✅ ${data.filename} uploaded</div>
                    <div class="card-title">Video Metadata</div>
                    <div class="stat-row"><span class="stat-label">Duration</span><span class="stat-value">${duration}</span></div>
                    <div class="stat-row"><span class="stat-label">Resolution</span><span class="stat-value">${m.width} × ${m.height}</span></div>
                    <div class="stat-row"><span class="stat-label">Frame rate</span><span class="stat-value">${m.fps} fps</span></div>
                    <div class="stat-row"><span class="stat-label">Codec</span><span class="stat-value">${m.codec}</span></div>
                    <div class="stat-row"><span class="stat-label">File size</span><span class="stat-value">${m.size_mb} MB</span></div>
                    <div class="stat-row"><span class="stat-label">Total frames</span><span class="stat-value">${m.estimated_frames.toLocaleString()}</span></div>
                  </div>`;
              } else {
                resultDiv.innerHTML = `<div class="card error">❌ ${data.error}</div>`;
              }
            } catch (err) {
              status.textContent = '';
              resultDiv.style.display = 'block';
              resultDiv.innerHTML = `<div class="card error">❌ Upload failed: ${err.message}</div>`;
            }
            btn.disabled = false;
            btn.textContent = 'Analyze MP4';
          }
        </script>
      </body>
    </html>
    """


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.endswith(".mp4"):
        return {"success": False, "error": "Only .mp4 files are allowed."}

    save_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(save_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    try:
        metadata = probe_video(save_path)
    except Exception as e:
        return {"success": False, "error": str(e)}

    return {
        "success": True,
        "filename": file.filename,
        "metadata": metadata,
    }
