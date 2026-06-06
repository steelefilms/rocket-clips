import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

app = FastAPI()

# Where uploaded videos get saved on the server
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
      <head>
        <title>Rocket Clips</title>
        <style>
          body {
            font-family: sans-serif;
            background: #0f0f0f;
            color: #ffffff;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
          }
          h1 { font-size: 2rem; margin-bottom: 8px; }
          p  { color: #888; margin-bottom: 32px; }
          input[type="file"] { color: #fff; margin-bottom: 16px; }
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
          #result {
            margin-top: 24px;
            padding: 16px;
            background: #1a1a1a;
            border-radius: 8px;
            display: none;
            max-width: 400px;
            text-align: center;
          }
        </style>
      </head>
      <body>
        <h1>🎮 Rocket Clips</h1>
        <p>Upload your Rocket League recording to get started.</p>

        <input type="file" id="fileInput" accept=".mp4" />
        <br>
        <button onclick="uploadFile()">Upload MP4</button>

        <div id="result"></div>

        <script>
          async function uploadFile() {
            const file = document.getElementById('fileInput').files[0];
            if (!file) { alert('Please select an MP4 file first.'); return; }

            const formData = new FormData();
            formData.append('file', file);

            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = '⏳ Uploading...';

            const response = await fetch('/upload', {
              method: 'POST',
              body: formData
            });

            const data = await response.json();

            if (data.success) {
              resultDiv.innerHTML = `
                <div style="color: #4caf50; font-size: 1.2rem;">✅ Upload successful!</div>
                <div style="margin-top: 8px; color: #aaa;">📁 ${data.filename}</div>
                <div style="color: #666; font-size: 0.85rem; margin-top: 4px;">${data.message}</div>
              `;
            } else {
              resultDiv.innerHTML = `<div style="color: #ff4655;">❌ ${data.error}</div>`;
            }
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

    # Read and write in 1MB chunks — safe for large video files
    with open(save_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    file_size_mb = round(os.path.getsize(save_path) / (1024 * 1024), 2)

    return {
        "success": True,
        "filename": file.filename,
        "saved_to": save_path,
        "size_mb": file_size_mb,
        "message": f"Saved {file_size_mb} MB to server"
    }
