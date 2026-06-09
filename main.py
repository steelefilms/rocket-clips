import os
import re
import uuid
import json
import base64
import sqlite3
import subprocess
import cv2
import numpy as np
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pymediainfo import MediaInfo
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

_api_key = os.getenv("GEMINI_API_KEY", "")
_genai_client = genai.Client(api_key=_api_key) if _api_key else None

app = FastAPI()

UPLOAD_DIR = "uploads"
CLIPS_DIR = "clips"
DB_PATH    = "rocketclips.db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

analysis_results = {}

# ── Database setup ────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            email     TEXT UNIQUE NOT NULL,
            name      TEXT,
            source    TEXT DEFAULT 'signup',
            created_at TEXT DEFAULT (datetime('now'))
        )
    """)
    con.commit()
    con.close()

init_db()


def find_ffmpeg():
    import glob as _glob
    candidates = [
        "ffmpeg",                    # Linux/Railway — installed via apt, on PATH
        "/usr/bin/ffmpeg",           # explicit Linux path
        "/usr/local/bin/ffmpeg",
        # Windows WinGet install (local dev)
        r"C:\Users\steel\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]
    winget_pattern = os.path.expandvars(
        r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*\**\bin\ffmpeg.exe"
    )
    candidates += _glob.glob(winget_pattern, recursive=True)
    for candidate in candidates:
        try:
            subprocess.run([candidate, "-version"], capture_output=True, check=True)
            return candidate
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    return None


FFMPEG = find_ffmpeg()


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


def detect_highlights(filepath: str, sample_interval: float = 1.0, top_n: int = 20) -> dict:
    """
    Pass 1 — motion scoring (fast, every sample_interval seconds).
    Returns all scores + top_n candidates for Gemini to re-rank.
    """
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file.")

    fps        = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = int(fps * sample_interval)

    scores     = []
    prev_gray  = None
    frame_index = 0

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff  = cv2.absdiff(prev_gray, gray)
            score = float(np.mean(diff))
            ts    = frame_index / fps
            scores.append({
                "timestamp":           round(ts, 2),
                "motion_score":        round(score, 4),
                "timestamp_formatted": format_timestamp(ts),
            })
        prev_gray    = gray
        frame_index += frame_step
        if frame_index >= total_frames:
            break

    cap.release()

    if not scores:
        return {"top": [], "all_scores": []}

    max_score = max(s["motion_score"] for s in scores)
    for s in scores:
        s["excitement_score"] = round((s["motion_score"] / max_score) * 100, 1) if max_score > 0 else 0.0

    # Deduplicate — no two moments within 5s of each other (wider window = better candidates)
    top_all = sorted(scores, key=lambda x: x["excitement_score"], reverse=True)
    top     = []
    for moment in top_all:
        if all(abs(moment["timestamp"] - kept["timestamp"]) >= 5.0 for kept in top):
            top.append(moment)
        if len(top) == top_n:
            break

    return {
        "top":        sorted(top, key=lambda x: x["timestamp"]),
        "all_scores": sorted(scores, key=lambda x: x["timestamp"]),
    }


def gemini_score_highlights(filepath: str, candidates: list) -> list:
    """
    Pass 2 — Gemini scores each candidate frame on:
      • Is it car POV?  (discard if not)
      • Is a goal being scored or just happened?
      • Overall excitement 0-100
    Returns only car-POV moments, re-ranked by Gemini score.
    """
    if not _genai_client or not candidates:
        # Fall back: return top 12 motion candidates as-is
        for h in candidates[:12]:
            h.setdefault("label", "highlight")
            h.setdefault("ai_score", h["excitement_score"])
            h.setdefault("is_goal", False)
        return candidates[:12]

    scored = []
    for h in candidates:
        frame_bytes = extract_frame_jpeg(filepath, h["timestamp"])
        if frame_bytes is None:
            continue
        try:
            prompt = """You are analyzing a frame from a Rocket League gameplay recording.

Answer ONLY with a JSON object — no markdown, no explanation:
{
  "is_car_pov": true/false,        // true if camera is from the player's car perspective (not replay/overhead/scorescreen)
  "is_goal": true/false,           // true if a goal was just scored or is being scored right now
  "excitement": 0-100,             // how exciting is this moment from a viewer's perspective
  "label": "short label"           // 3-5 words describing what is happening
}"""
            resp = _genai_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    types.Part.from_bytes(data=frame_bytes, mime_type="image/jpeg"),
                    prompt,
                ]
            )
            raw  = resp.text.strip()
            # Strip any accidental markdown fences
            raw  = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
            data = json.loads(raw)

            if not data.get("is_car_pov", False):
                continue   # skip non-POV frames entirely

            h["label"]    = str(data.get("label", "highlight"))[:60].lower()
            h["ai_score"] = int(data.get("excitement", h["excitement_score"]))
            h["is_goal"]  = bool(data.get("is_goal", False))
            scored.append(h)
        except Exception:
            # If Gemini fails on this frame keep it with motion score
            h.setdefault("label",    "highlight")
            h.setdefault("ai_score", h["excitement_score"])
            h.setdefault("is_goal",  False)
            scored.append(h)

    # Goals always float to the top, then sort by AI excitement
    scored.sort(key=lambda x: (x["is_goal"], x["ai_score"]), reverse=True)

    # Deduplicate again after re-ranking (Gemini may cluster them differently)
    final = []
    for moment in scored:
        if all(abs(moment["timestamp"] - kept["timestamp"]) >= 5.0 for kept in final):
            final.append(moment)
        if len(final) == 12:
            break

    return sorted(final, key=lambda x: x["timestamp"])


def format_timestamp(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


def extract_frame_jpeg(filepath: str, timestamp: float) -> bytes | None:
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp * fps))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def run_analysis(filepath: str, filename: str):
    try:
        analysis_results[filename] = {"status": "processing", "stage": "motion"}

        metadata   = probe_video(filepath)
        motion     = detect_highlights(filepath)
        candidates = motion["top"]

        analysis_results[filename] = {
            "status":     "processing",
            "stage":      "ai_scoring",
            "metadata":   metadata,
            "highlights": candidates,
            "all_scores": motion["all_scores"],
        }

        # Pass 2 — Gemini POV filter + goal detection + excitement re-score
        highlights = gemini_score_highlights(filepath, candidates)

        analysis_results[filename] = {
            "status":      "done",
            "metadata":    metadata,
            "highlights":  highlights,
            "all_scores":  motion["all_scores"],
            "ai_labeling": bool(_api_key),
        }
    except Exception as e:
        analysis_results[filename] = {"status": "error", "error": str(e)}


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.post("/upload")
async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()):
    if not file.filename.lower().endswith(".mp4"):
        return JSONResponse({"success": False, "error": "Only .mp4 files are supported."})
    safe_name = re.sub(r"[^\w\-.]", "_", file.filename)
    save_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(save_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    background_tasks.add_task(run_analysis, save_path, safe_name)
    return {"success": True, "filename": safe_name}


@app.get("/results/{filename}")
async def get_results(filename: str):
    if filename not in analysis_results:
        return {"status": "processing"}
    return analysis_results[filename]


@app.post("/clip")
async def create_clip(body: dict):
    filename = body.get("filename")
    timestamp = float(body.get("timestamp", 0))
    before_secs = float(body.get("before_secs", 3))
    after_secs = float(body.get("after_secs", 7))

    source_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(source_path):
        return JSONResponse({"success": False, "error": "Source video not found."})
    if FFMPEG is None:
        return JSONResponse({"success": False, "error": "ffmpeg not found."})

    start = max(0, timestamp - before_secs)
    duration = before_secs + after_secs
    clip_id = uuid.uuid4().hex[:8]
    clip_filename = f"clip_{clip_id}.mp4"
    clip_path = os.path.join(CLIPS_DIR, clip_filename)

    cmd = [
        FFMPEG, "-y",
        "-ss", str(start),
        "-i", source_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "16",                  # near-lossless (0=lossless, 18=visually lossless, 23=default)
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease:eval=init,pad=1920:1080:-1:-1:color=black",
        "-c:a", "aac", "-b:a", "320k",
        "-movflags", "+faststart",     # optimise for web playback
        clip_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        return JSONResponse({"success": False, "error": "ffmpeg failed to create clip."})

    size_mb = round(os.path.getsize(clip_path) / (1024 * 1024), 2)
    return {
        "success": True,
        "clip_id": clip_id,
        "clip_filename": clip_filename,
        "size_mb": size_mb,
        "start": round(start, 2),
        "duration": round(duration, 2),
        "timestamp": timestamp,
    }


@app.get("/download/{clip_filename}")
async def download_clip(clip_filename: str):
    safe = re.sub(r"[^\w\-.]", "_", clip_filename)
    path = os.path.join(CLIPS_DIR, safe)
    if not os.path.exists(path):
        return JSONResponse({"error": "Clip not found."}, status_code=404)
    return FileResponse(path, media_type="video/mp4", filename=safe)


EMAILS_TXT = os.path.join(os.path.dirname(__file__), "emails.txt")

def append_email_to_txt(email: str, name: str):
    """Append a new signup to emails.txt — one entry per line."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    line = f"{email}"
    if name:
        line += f"  |  {name}"
    line += f"  |  {timestamp}\n"
    with open(EMAILS_TXT, "a", encoding="utf-8") as f:
        f.write(line)


@app.post("/signup")
async def signup(request: Request):
    body  = await request.json()
    email = (body.get("email") or "").strip().lower()
    name  = (body.get("name")  or "").strip()
    if not email or "@" not in email:
        return JSONResponse({"success": False, "error": "Valid email required."})
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "INSERT INTO users (email, name, source) VALUES (?, ?, ?)",
            (email, name, "webapp")
        )
        con.commit()
        con.close()
        append_email_to_txt(email, name)
        return {"success": True, "message": "Welcome to Rocket Clips!"}
    except sqlite3.IntegrityError:
        return {"success": True, "message": "Already signed up — welcome back!"}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


@app.get("/admin/emails.txt")
async def download_emails_txt():
    """Download the raw emails.txt file directly."""
    if not os.path.exists(EMAILS_TXT):
        return JSONResponse({"error": "No signups yet."}, status_code=404)
    return FileResponse(EMAILS_TXT, media_type="text/plain", filename="rocket_clips_emails.txt")


@app.get("/admin/users")
async def list_users():
    """Dev-only endpoint — returns all signed-up emails."""
    con  = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT id, email, name, created_at FROM users ORDER BY id DESC").fetchall()
    con.close()
    return {"count": len(rows), "users": [
        {"id": r[0], "email": r[1], "name": r[2], "joined": r[3]} for r in rows
    ]}


@app.post("/preview")
async def preview_clip(body: dict):
    """
    Same as /clip but uses ultrafast preset — generates in 2-3s for instant modal preview.
    Clips land in the clips/ folder with a 'preview_' prefix.
    """
    filename   = body.get("filename")
    timestamp  = float(body.get("timestamp", 0))
    before_secs = float(body.get("before_secs", 15))
    after_secs  = float(body.get("after_secs", 5))

    source_path = os.path.join(UPLOAD_DIR, re.sub(r"[^\w\-.]", "_", filename))
    if not os.path.exists(source_path):
        return JSONResponse({"success": False, "error": "Source video not found."})
    if FFMPEG is None:
        return JSONResponse({"success": False, "error": "ffmpeg not found."})

    start         = max(0, timestamp - before_secs)
    duration      = before_secs + after_secs
    clip_id       = uuid.uuid4().hex[:8]
    clip_filename = f"preview_{clip_id}.mp4"
    clip_path     = os.path.join(CLIPS_DIR, clip_filename)

    cmd = [
        FFMPEG, "-y",
        "-ss", str(start),
        "-i", source_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "ultrafast",   # fast generation for previewing
        "-crf", "20",
        "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease:eval=init,pad=1920:1080:-1:-1:color=black",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        clip_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        return JSONResponse({"success": False, "error": "ffmpeg preview failed."})

    return {
        "success":       True,
        "clip_filename": clip_filename,
        "start":         round(start, 2),
        "duration":      round(duration, 2),
    }


@app.get("/stream/{filename}")
async def stream_video(filename: str):
    safe = re.sub(r"[^\w\-.]", "_", filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.exists(path):
        return JSONResponse({"error": "File not found."}, status_code=404)
    return FileResponse(path, media_type="video/mp4")


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Rocket Clips</title>
<style>
/* ─── Reset & Tokens ─────────────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#07070a;
  --s1:#0e0e14;
  --s2:#14141c;
  --s3:#1c1c28;
  --border:#252535;
  --border2:#32324a;
  --accent:#ff4655;
  --accent-dim:#ff465522;
  --accent-glow:0 0 24px #ff465540;
  --green:#3de87a;
  --yellow:#ffb547;
  --blue:#5b8fff;
  --text:#eeeef8;
  --sub:#8888aa;
  --muted:#44445a;
  --r:12px;
  --r2:8px;
  --font:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}
html,body{height:100%;overflow:hidden}
body{font-family:var(--font);background:var(--bg);color:var(--text);display:flex;flex-direction:column}

/* ─── Header ─────────────────────────────────────────────── */
.topbar{
  height:52px;min-height:52px;
  background:var(--s1);
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:14px;
  padding:0 20px;
  position:relative;z-index:10;
}
.logo{font-size:1.2rem;font-weight:900;letter-spacing:-0.5px;white-space:nowrap}
.logo em{color:var(--accent);font-style:normal}
.pill{
  font-size:.65rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
  background:var(--accent-dim);color:var(--accent);
  border:1px solid #ff465530;border-radius:99px;padding:3px 10px;
}
.topbar-right{margin-left:auto;display:flex;align-items:center;gap:16px}
.file-info-top{font-size:.8rem;color:var(--sub);display:none;align-items:center;gap:8px}
.file-info-top .fname{color:var(--text);font-weight:600;max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* ─── Layout ─────────────────────────────────────────────── */
.workspace{flex:1;display:grid;grid-template-columns:300px 1fr 300px;overflow:hidden;min-height:0}

/* ─── Panels ─────────────────────────────────────────────── */
.panel{
  background:var(--s1);border-right:1px solid var(--border);
  display:flex;flex-direction:column;overflow:hidden;
}
.panel.right{border-right:none;border-left:1px solid var(--border)}
.panel-header{
  padding:14px 16px 12px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;flex-shrink:0;
}
.panel-title{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--sub)}
.panel-body{flex:1;overflow-y:auto;padding:14px 16px;display:flex;flex-direction:column;gap:12px}
.panel-body::-webkit-scrollbar{width:4px}
.panel-body::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}

/* ─── Center Area ─────────────────────────────────────────── */
.center{display:flex;flex-direction:column;overflow:hidden;min-height:0}

/* ─── Upload Zone ─────────────────────────────────────────── */
.drop-zone{
  margin:16px;
  border:2px dashed var(--border2);border-radius:var(--r);
  background:var(--s2);
  text-align:center;padding:28px 20px;
  cursor:pointer;transition:border-color .2s,background .2s;
  position:relative;flex-shrink:0;
}
.drop-zone.active,.drop-zone:hover{border-color:var(--accent);background:var(--accent-dim)}
.drop-zone input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.drop-icon{font-size:2rem;margin-bottom:8px;display:block}
.drop-text{font-size:.85rem;color:var(--sub);line-height:1.6}
.drop-text strong{color:var(--text)}
.drop-zone.has-file{border-style:solid;border-color:var(--accent);background:var(--accent-dim)}

/* ─── Analyze Button ─────────────────────────────────────── */
.btn-analyze{
  margin:0 16px 16px;
  background:var(--accent);color:#fff;border:none;
  border-radius:var(--r2);padding:13px;
  font-size:.9rem;font-weight:800;letter-spacing:.04em;
  cursor:pointer;transition:background .15s,box-shadow .2s,transform .1s;
  display:flex;align-items:center;justify-content:center;gap:8px;
  flex-shrink:0;
}
.btn-analyze:hover:not(:disabled){background:#e03040;box-shadow:var(--accent-glow)}
.btn-analyze:active:not(:disabled){transform:scale(.98)}
.btn-analyze:disabled{background:var(--s3);color:var(--muted);cursor:not-allowed;box-shadow:none}
.btn-analyze .spinner{width:16px;height:16px;border:2px solid #fff4;border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;display:none}
@keyframes spin{to{transform:rotate(360deg)}}

/* ─── Progress ───────────────────────────────────────────── */
.progress-block{margin:0 16px 16px;display:none;flex-direction:column;gap:6px;flex-shrink:0}
.progress-meta{display:flex;justify-content:space-between;font-size:.75rem;color:var(--sub)}
.progress-track{background:var(--s3);border-radius:99px;height:4px;overflow:hidden}
.progress-fill{
  background:linear-gradient(90deg,var(--accent),#ff8a95);
  height:100%;width:0%;border-radius:99px;
  transition:width .25s;
}
.progress-fill.indeterminate{
  width:40%!important;
  animation:slide 1.4s ease-in-out infinite;
}
@keyframes slide{0%{transform:translateX(-150%)}100%{transform:translateX(350%)}}

/* ─── Status Bar ─────────────────────────────────────────── */
.status-bar{
  margin:0 16px 16px;
  background:var(--s2);border:1px solid var(--border);border-radius:var(--r2);
  padding:10px 14px;
  display:none;align-items:center;gap:10px;
  font-size:.82rem;flex-shrink:0;
}
.status-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.status-dot.idle{background:var(--muted)}
.status-dot.working{background:var(--yellow);animation:pulse 1s infinite}
.status-dot.done{background:var(--green)}
.status-dot.err{background:var(--accent)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.status-text{color:var(--sub);flex:1}
.status-text strong{color:var(--text)}

/* ─── Video Info ─────────────────────────────────────────── */
.info-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.info-cell{
  background:var(--s2);border:1px solid var(--border);border-radius:var(--r2);
  padding:10px 12px;
}
.info-cell-label{font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:4px}
.info-cell-val{font-size:.95rem;font-weight:700;color:var(--text)}

/* ─── Timeline ───────────────────────────────────────────── */
.timeline-wrap{
  margin:0 16px 12px;flex-shrink:0;
  background:var(--s2);border:1px solid var(--border);border-radius:var(--r2);
  padding:10px 14px;display:none;
}
.timeline-label{font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;color:var(--sub);margin-bottom:8px}
.timeline-track{position:relative;height:36px;background:var(--s3);border-radius:6px;overflow:hidden;cursor:crosshair}
.timeline-bar{
  position:absolute;bottom:0;width:3px;border-radius:2px 2px 0 0;
  background:var(--accent);opacity:.5;transition:opacity .15s;
}
.timeline-bar.top{opacity:1;background:var(--accent);box-shadow:0 0 6px var(--accent)}
.timeline-bar:hover{opacity:1}
.timeline-pin{
  position:absolute;top:0;bottom:0;width:2px;background:#fff3;pointer-events:none;
}
.timeline-thumb{
  position:absolute;top:0;left:0;right:0;bottom:0;
  display:flex;align-items:flex-end;padding:0 1px;gap:1px;
}
.timeline-cursor{
  position:absolute;top:-18px;transform:translateX(-50%);
  background:var(--s1);border:1px solid var(--border2);
  border-radius:4px;padding:2px 6px;font-size:.65rem;color:var(--text);
  pointer-events:none;white-space:nowrap;display:none;
}

/* ─── Video Player ───────────────────────────────────────── */
.video-area{
  flex:1;background:var(--bg);overflow:hidden;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  position:relative;min-height:0;
}
.video-empty{text-align:center;color:var(--muted);pointer-events:none}
.video-empty .icon{font-size:3.5rem;margin-bottom:12px;opacity:.3}
.video-empty p{font-size:.85rem;line-height:1.6;max-width:240px;margin:0 auto}
video#player{
  max-width:100%;max-height:100%;width:100%;height:100%;
  object-fit:contain;display:none;background:#000;
}
.video-controls{
  position:absolute;bottom:0;left:0;right:0;
  background:linear-gradient(transparent,#000a);
  padding:20px 16px 12px;
  display:none;flex-direction:column;gap:8px;
}
.vc-seek{
  width:100%;height:4px;background:#fff2;border-radius:99px;
  cursor:pointer;appearance:none;-webkit-appearance:none;
  position:relative;
}
.vc-seek::-webkit-slider-thumb{
  appearance:none;width:14px;height:14px;border-radius:50%;
  background:var(--accent);cursor:pointer;box-shadow:0 0 6px var(--accent);
}
.vc-row{display:flex;align-items:center;gap:10px}
.vc-btn{background:none;border:none;color:#fff;cursor:pointer;font-size:1rem;padding:4px;opacity:.8;transition:opacity .15s}
.vc-btn:hover{opacity:1}
.vc-time{font-size:.75rem;color:#ffffffaa;font-variant-numeric:tabular-nums}
.vc-vol{width:70px;height:3px;cursor:pointer;appearance:none;-webkit-appearance:none;background:#fff3;border-radius:99px}
.vc-vol::-webkit-slider-thumb{appearance:none;width:10px;height:10px;border-radius:50%;background:#fff;cursor:pointer}

/* ─── Highlight Cards ─────────────────────────────────────── */
.hl-list{display:flex;flex-direction:column;gap:8px}
.hl-card{
  background:var(--s2);border:1px solid var(--border);border-radius:var(--r2);
  padding:12px 14px;cursor:pointer;
  transition:border-color .15s,background .15s;
  position:relative;overflow:hidden;
}
.hl-card::before{
  content:'';position:absolute;left:0;top:0;bottom:0;width:3px;
  background:var(--accent);opacity:0;transition:opacity .15s;
}
.hl-card:hover{border-color:var(--border2);background:var(--s3)}
.hl-card:hover::before{opacity:1}
.hl-card.active{border-color:var(--accent);background:var(--accent-dim)}
.hl-card.active::before{opacity:1}
.hl-top{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.hl-num{
  width:22px;height:22px;border-radius:50%;flex-shrink:0;
  background:var(--s3);border:1px solid var(--border2);
  font-size:.65rem;font-weight:700;color:var(--sub);
  display:flex;align-items:center;justify-content:center;
}
.hl-card.active .hl-num{background:var(--accent-dim);border-color:var(--accent);color:var(--accent)}
.hl-ts{font-size:1.05rem;font-weight:800;color:var(--accent);font-variant-numeric:tabular-nums;letter-spacing:-0.5px}
.hl-score-badge{
  margin-left:auto;font-size:.7rem;font-weight:700;
  background:var(--s3);border:1px solid var(--border2);
  border-radius:4px;padding:2px 7px;color:var(--sub);
}
.hl-card.active .hl-score-badge{background:var(--accent-dim);border-color:var(--accent);color:var(--accent)}
.hl-label{
  font-size:.72rem;font-weight:600;color:var(--accent);
  background:var(--accent-dim);border:1px solid #ff465530;
  border-radius:4px;padding:2px 8px;margin-bottom:8px;
  display:inline-block;text-transform:capitalize;
}
.goal-badge{
  font-size:.65rem;font-weight:800;letter-spacing:.04em;
  background:#ffb54720;border:1px solid #ffb54760;color:var(--yellow);
  border-radius:4px;padding:2px 7px;text-transform:uppercase;
  animation:goalPulse 2s ease infinite;
}
@keyframes goalPulse{0%,100%{box-shadow:none}50%{box-shadow:0 0 8px #ffb54760}}
.hl-bar-bg{background:var(--border);border-radius:99px;height:3px}
.hl-bar-fill{background:linear-gradient(90deg,var(--accent),#ff8a95);height:100%;border-radius:99px}

/* ─── Clip Controls ──────────────────────────────────────── */
.clip-row{display:flex;gap:6px;margin-top:10px}
.dur-select{
  flex:1;background:var(--s3);border:1px solid var(--border);
  color:var(--text);border-radius:var(--r2);padding:7px 8px;
  font-size:.78rem;cursor:pointer;
}
.btn-watch{
  width:100%;background:var(--s3);border:1px solid var(--border2);
  color:var(--text);border-radius:var(--r2);padding:9px;
  font-size:.82rem;font-weight:700;cursor:pointer;
  display:flex;align-items:center;justify-content:center;gap:7px;
  transition:all .15s;letter-spacing:.02em;
}
.btn-watch:hover{background:#5b8fff18;border-color:var(--blue);color:var(--blue)}
.btn-clip{
  background:var(--s3);border:1px solid var(--border);color:var(--sub);
  border-radius:var(--r2);padding:7px 12px;font-size:.78rem;font-weight:700;
  cursor:pointer;white-space:nowrap;transition:all .15s;
  display:flex;align-items:center;gap:5px;
}
.btn-clip:hover:not(:disabled){background:var(--accent);border-color:var(--accent);color:#fff;box-shadow:var(--accent-glow)}
.btn-clip:disabled{opacity:.4;cursor:not-allowed}
.btn-clip .clip-spinner{width:10px;height:10px;border:1.5px solid #fff4;border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;display:none}

/* ─── Clips Manager (right panel) ────────────────────────── */
.no-clips{
  flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
  text-align:center;color:var(--muted);gap:8px;
}
.no-clips .icon{font-size:2rem;opacity:.3}
.no-clips p{font-size:.78rem;line-height:1.5;max-width:180px}
.clip-item{
  background:var(--s2);border:1px solid var(--border);border-radius:var(--r2);
  padding:12px;display:flex;flex-direction:column;gap:8px;
  animation:fadeIn .3s ease;
}
@keyframes fadeIn{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.clip-item-head{display:flex;align-items:center;gap:8px}
.clip-thumb{
  width:50px;height:32px;border-radius:4px;object-fit:cover;
  background:var(--s3);flex-shrink:0;border:1px solid var(--border2);
}
.clip-meta{flex:1;min-width:0}
.clip-meta-name{font-size:.75rem;font-weight:700;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.clip-meta-sub{font-size:.68rem;color:var(--sub)}
.clip-actions{display:flex;gap:6px}
.btn-dl{
  flex:1;background:var(--green);color:#000;border:none;
  border-radius:var(--r2);padding:7px;font-size:.75rem;font-weight:800;
  cursor:pointer;text-align:center;text-decoration:none;
  display:flex;align-items:center;justify-content:center;gap:4px;
  transition:opacity .15s;
}
.btn-dl:hover{opacity:.85}
.btn-preview{
  background:var(--s3);border:1px solid var(--border);color:var(--sub);
  border-radius:var(--r2);padding:7px 10px;font-size:.75rem;font-weight:600;
  cursor:pointer;transition:all .15s;white-space:nowrap;
}
.btn-preview:hover{border-color:var(--blue);color:var(--blue)}
.clip-count{
  background:var(--accent);color:#fff;border-radius:99px;
  font-size:.65rem;font-weight:800;padding:2px 7px;min-width:18px;text-align:center;
}

/* ─── Toast ──────────────────────────────────────────────── */
#toast{
  position:fixed;bottom:20px;left:50%;transform:translateX(-50%) translateY(20px);
  background:var(--s1);border:1px solid var(--border2);border-radius:var(--r2);
  padding:10px 18px;font-size:.82rem;color:var(--text);
  box-shadow:0 8px 32px #0008;pointer-events:none;
  opacity:0;transition:opacity .25s,transform .25s;z-index:999;
  display:flex;align-items:center;gap:8px;
}
#toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
#toast .t-icon{font-size:1rem}

/* ─── Scrollbar (global) ─────────────────────────────────── */
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}
</style>
</head>
<body>

<!-- ── TOPBAR ── -->
<div class="topbar">
  <div class="logo">🚀 Rocket<em>Clips</em></div>
  <div class="pill">Phase 3</div>
  <div class="topbar-right">
    <div class="file-info-top" id="fileInfoTop">
      <span>🎬</span>
      <span class="fname" id="topFilename"></span>
    </div>
    <div id="userDisplay" style="display:none;align-items:center;gap:8px;font-size:.82rem">
      <span style="color:var(--green)">●</span>
      <span id="userEmail" style="color:var(--sub)"></span>
    </div>
    <button onclick="openSignIn()" id="signInBtn" style="background:var(--s3);border:1px solid var(--border2);color:var(--text);border-radius:var(--r2);padding:7px 16px;font-size:.8rem;font-weight:700;cursor:pointer;transition:all .15s" onmouseover="this.style.borderColor='var(--accent)';this.style.color='var(--accent)'" onmouseout="this.style.borderColor='var(--border2)';this.style.color='var(--text)'">Sign In</button>
  </div>
</div>

<!-- ── WORKSPACE ── -->
<div class="workspace">

  <!-- LEFT: Upload + Info + Timeline -->
  <div class="panel">
    <div class="panel-header">
      <span class="panel-title">Source</span>
    </div>
    <div class="panel-body" style="gap:0;padding:0">

      <div class="drop-zone" id="dropZone" style="margin:16px 16px 12px">
        <input type="file" id="fileInput" accept=".mp4"/>
        <span class="drop-icon">🎬</span>
        <div class="drop-text"><strong>Drop MP4 here</strong><br>or click to browse</div>
      </div>

      <div class="progress-block" id="progressBlock" style="margin:0 16px 12px">
        <div class="progress-meta">
          <span id="progressLabel">Uploading...</span>
          <span id="progressPct">0%</span>
        </div>
        <div class="progress-track"><div class="progress-fill" id="progressFill"></div></div>
      </div>

      <div class="status-bar" id="statusBar" style="margin:0 16px 12px">
        <div class="status-dot idle" id="statusDot"></div>
        <div class="status-text" id="statusText"></div>
      </div>

      <div style="margin:0 16px 12px;display:none" id="infoBlock">
        <div class="info-grid" id="infoGrid"></div>
      </div>

      <div class="timeline-wrap" id="timelineWrap">
        <div class="timeline-label">Motion Timeline — click to seek</div>
        <div class="timeline-track" id="timelineTrack">
          <div class="timeline-thumb" id="timelineThumb"></div>
          <div class="timeline-cursor" id="timelineCursor"></div>
        </div>
      </div>

      <div style="margin:0 16px 16px;flex-shrink:0">
        <button class="btn-analyze" id="analyzeBtn" onclick="uploadFile()">
          <span class="spinner" id="btnSpinner"></span>
          <span id="btnLabel">✦ Analyze Highlights</span>
        </button>
      </div>

    </div>
  </div>

  <!-- CENTER: Video + Highlights -->
  <div class="center">

    <!-- Video Player -->
    <div class="video-area" id="videoArea">
      <div class="video-empty" id="videoEmpty">
        <div class="icon">🎮</div>
        <p>Upload a video to detect highlight moments</p>
      </div>
      <video id="player" controls preload="metadata"></video>
      <div class="video-controls" id="videoControls" style="display:none"></div>
    </div>

    <!-- Timeline + highlights below player -->
    <div style="flex-shrink:0;border-top:1px solid var(--border);padding:14px 16px;background:var(--s1);display:none" id="hlSection">
      <div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;color:var(--sub);margin-bottom:10px" id="hlCountLabel"></div>
      <div class="hl-list" id="hlList"></div>
    </div>

  </div>

  <!-- RIGHT: Clips Manager -->
  <div class="panel right">
    <div class="panel-header">
      <span class="panel-title">Clips</span>
      <span class="clip-count" id="clipCount" style="display:none">0</span>
    </div>
    <div class="panel-body" id="clipsPanel">
      <div class="no-clips" id="noClips">
        <div class="icon">✂️</div>
        <p>Created clips will appear here</p>
      </div>
    </div>
  </div>

</div>

<!-- Toast -->
<div id="toast"><span class="t-icon" id="toastIcon"></span><span id="toastMsg"></span></div>

<!-- Sign-In Modal -->
<div id="signInModal" style="display:none;position:fixed;inset:0;z-index:1001;background:#000c;align-items:center;justify-content:center">
  <div style="width:min(420px,94vw);background:var(--s1);border:1px solid var(--border2);border-radius:16px;overflow:hidden;box-shadow:0 24px 80px #000a">
    <!-- Header -->
    <div style="padding:28px 28px 0;text-align:center">
      <div style="font-size:2rem;margin-bottom:8px">🚀</div>
      <div style="font-size:1.2rem;font-weight:800;margin-bottom:6px">Join Rocket Clips</div>
      <div style="font-size:.85rem;color:var(--sub);line-height:1.5">Sign up for early access, updates,<br>and exclusive clipping features.</div>
    </div>
    <!-- Form -->
    <div style="padding:24px 28px 28px;display:flex;flex-direction:column;gap:12px">
      <div style="display:flex;flex-direction:column;gap:5px">
        <label style="font-size:.75rem;font-weight:600;color:var(--sub);text-transform:uppercase;letter-spacing:.06em">Name</label>
        <input id="signInName" type="text" placeholder="Your name" style="background:var(--s3);border:1px solid var(--border2);color:var(--text);border-radius:var(--r2);padding:11px 14px;font-size:.9rem;outline:none;transition:border-color .15s" onfocus="this.style.borderColor='var(--accent)'" onblur="this.style.borderColor='var(--border2)'"/>
      </div>
      <div style="display:flex;flex-direction:column;gap:5px">
        <label style="font-size:.75rem;font-weight:600;color:var(--sub);text-transform:uppercase;letter-spacing:.06em">Email</label>
        <input id="signInEmail" type="email" placeholder="you@example.com" style="background:var(--s3);border:1px solid var(--border2);color:var(--text);border-radius:var(--r2);padding:11px 14px;font-size:.9rem;outline:none;transition:border-color .15s" onfocus="this.style.borderColor='var(--accent)'" onblur="this.style.borderColor='var(--border2)'" onkeydown="if(event.key==='Enter')submitSignIn()"/>
      </div>
      <div id="signInError" style="font-size:.8rem;color:var(--accent);display:none"></div>
      <button onclick="submitSignIn()" id="signInSubmit" style="background:var(--accent);color:#fff;border:none;border-radius:var(--r2);padding:13px;font-size:.92rem;font-weight:800;cursor:pointer;margin-top:4px;transition:background .15s;letter-spacing:.02em" onmouseover="this.style.background='#e03040'" onmouseout="this.style.background='var(--accent)'">
        Get Early Access
      </button>
      <div style="text-align:center;font-size:.75rem;color:var(--muted)">No spam. Unsubscribe anytime.</div>
    </div>
    <button onclick="closeSignIn()" style="position:absolute;top:14px;right:16px;background:none;border:none;color:var(--muted);font-size:1.1rem;cursor:pointer;padding:4px 8px" onmouseover="this.style.color='var(--text)'" onmouseout="this.style.color='var(--muted)'">✕</button>
  </div>
</div>

<!-- Preview Modal -->
<div id="previewModal" style="display:none;position:fixed;inset:0;z-index:1000;background:#000c;align-items:center;justify-content:center;flex-direction:column;gap:0">
  <div style="width:min(1080px,96vw);background:var(--s1);border:1px solid var(--border2);border-radius:var(--r);overflow:hidden;box-shadow:0 24px 80px #000a;display:flex;flex-direction:column">
    <!-- Modal header -->
    <div style="display:flex;align-items:center;justify-content:space-between;padding:12px 16px;border-bottom:1px solid var(--border);flex-shrink:0">
      <div style="display:flex;align-items:center;gap:10px">
        <span style="color:var(--accent);font-size:1rem">✂</span>
        <span id="modalTitle" style="font-size:.85rem;font-weight:600;color:var(--text);max-width:400px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"></span>
      </div>
      <div style="display:flex;gap:8px">
        <button onclick="modalFullscreen()" style="background:var(--s3);border:1px solid var(--border2);color:var(--sub);border-radius:var(--r2);padding:6px 14px;font-size:.8rem;font-weight:600;cursor:pointer;transition:all .15s" onmouseover="this.style.borderColor='var(--blue)';this.style.color='var(--blue)'" onmouseout="this.style.borderColor='var(--border2)';this.style.color='var(--sub)'">⛶ Fullscreen</button>
        <button onclick="closeModal()" style="background:var(--s3);border:1px solid var(--border2);color:var(--sub);border-radius:var(--r2);padding:6px 14px;font-size:.8rem;font-weight:600;cursor:pointer;transition:all .15s" onmouseover="this.style.borderColor='var(--accent)';this.style.color='var(--accent)'" onmouseout="this.style.borderColor='var(--border2)';this.style.color='var(--sub)'">✕ Close</button>
      </div>
    </div>
    <!-- Video -->
    <div style="background:#000;position:relative;aspect-ratio:16/9;width:100%">
      <!-- Loading state -->
      <div id="modalBody" style="display:none;position:absolute;inset:0;align-items:center;justify-content:center;flex-direction:column;gap:14px;background:#000">
        <div style="width:40px;height:40px;border:3px solid #ffffff18;border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite"></div>
        <div style="font-size:.85rem;color:var(--sub)">Cutting clip — just a moment...</div>
      </div>
      <video id="modalVideo" controls style="width:100%;height:100%;display:block;background:#000" preload="auto"></video>
    </div>
  </div>
</div>

<script>
// ─── State ────────────────────────────────────────────────
let pollInterval = null;
let currentFilename = null;
let currentResults = null;
let totalClips = 0;
let videoDuration = 0;

// ─── Toast ────────────────────────────────────────────────
function toast(msg, icon='✅', dur=2800) {
  const el = document.getElementById('toast');
  document.getElementById('toastIcon').textContent = icon;
  document.getElementById('toastMsg').textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), dur);
}

// ─── Drag & Drop ──────────────────────────────────────────
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('active'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('active'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('active');
  const f = e.dataTransfer.files[0];
  if (f) { assignFile(f); uploadFile(); }
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) assignFile(fileInput.files[0]);
});

function assignFile(f) {
  const dt = new DataTransfer();
  dt.items.add(f);
  fileInput.files = dt.files;
  dropZone.classList.add('has-file');
  dropZone.querySelector('.drop-text').innerHTML =
    '<strong>' + f.name + '</strong><br>' + (f.size/1048576).toFixed(1) + ' MB';
}

// ─── Status helpers ───────────────────────────────────────
function setStatus(type, html) {
  const bar = document.getElementById('statusBar');
  const dot = document.getElementById('statusDot');
  bar.style.display = 'flex';
  dot.className = 'status-dot ' + type;
  document.getElementById('statusText').innerHTML = html;
}
function showProgress(show) {
  document.getElementById('progressBlock').style.display = show ? 'flex' : 'none';
}
function setProgress(pct, label) {
  document.getElementById('progressFill').style.width = pct + '%';
  document.getElementById('progressPct').textContent = pct + '%';
  if (label) document.getElementById('progressLabel').textContent = label;
}
function setBtnState(loading, label) {
  const btn = document.getElementById('analyzeBtn');
  const spinner = document.getElementById('btnSpinner');
  const lbl = document.getElementById('btnLabel');
  btn.disabled = loading;
  spinner.style.display = loading ? 'block' : 'none';
  lbl.textContent = label;
}

// ─── Upload ───────────────────────────────────────────────
async function uploadFile() {
  const file = fileInput.files[0];
  if (!file) { toast('Select an MP4 first', '⚠️', 2000); return; }

  setBtnState(true, 'Uploading...');
  showProgress(true);
  setProgress(0, 'Uploading...');
  setStatus('working', 'Uploading <strong>' + file.name + '</strong>...');

  // Reset results
  document.getElementById('hlSection').style.display = 'none';
  document.getElementById('hlList').innerHTML = '';
  document.getElementById('timelineWrap').style.display = 'none';
  document.getElementById('infoBlock').style.display = 'none';

  const formData = new FormData();
  formData.append('file', file);

  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/upload');

  xhr.upload.onprogress = e => {
    if (e.lengthComputable) {
      const pct = Math.round(e.loaded / e.total * 100);
      setProgress(pct, pct < 100 ? 'Uploading...' : 'Processing...');
    }
  };

  xhr.onload = () => {
    const data = JSON.parse(xhr.responseText);
    if (data.success) {
      currentFilename = data.filename;
      document.getElementById('topFilename').textContent = data.filename;
      document.getElementById('fileInfoTop').style.display = 'flex';

      // Load video for preview
      const playerEl = document.getElementById('player');
      playerEl.src = '/stream/' + encodeURIComponent(data.filename);
      playerEl.style.display = 'block';
      document.getElementById('videoEmpty').style.display = 'none';
      playerEl.onloadedmetadata = () => { videoDuration = playerEl.duration; };

      setProgress(100, 'Analyzing...');
      document.getElementById('progressFill').classList.add('indeterminate');
      setStatus('working', 'Detecting highlights...');
      setBtnState(true, 'Analyzing...');
      pollInterval = setInterval(() => pollResults(currentFilename), 2500);
    } else {
      setStatus('err', data.error || 'Upload failed');
      setBtnState(false, '✦ Analyze Highlights');
      showProgress(false);
    }
  };
  xhr.onerror = () => {
    setStatus('err', 'Network error');
    setBtnState(false, '✦ Analyze Highlights');
    showProgress(false);
  };
  xhr.send(formData);
}

// ─── Poll ─────────────────────────────────────────────────
async function pollResults(filename) {
  try {
    const res = await fetch('/results/' + encodeURIComponent(filename));
    const data = await res.json();
    if (data.status === 'done') {
      clearInterval(pollInterval);
      document.getElementById('progressFill').classList.remove('indeterminate');
      showProgress(false);
      const aiNote = data.ai_labeling ? ' &middot; <span style="color:var(--blue)">AI labeled</span>' : '';
      setStatus('done', '<strong>' + data.highlights.length + ' highlights</strong> found' + aiNote);
      currentResults = data;
      renderResults(data, filename);
      setBtnState(false, '✦ Analyze Another');
    } else if (data.stage === 'ai_scoring') {
      setStatus('working', 'AI scoring — filtering POV frames &amp; detecting goals...');
    } else if (data.status === 'error') {
      clearInterval(pollInterval);
      document.getElementById('progressFill').classList.remove('indeterminate');
      showProgress(false);
      setStatus('err', data.error || 'Analysis failed');
      setBtnState(false, '✦ Analyze Highlights');
    }
  } catch(e) { console.error(e); }
}

// ─── Render Results ───────────────────────────────────────
function renderResults(data, filename) {
  const m = data.metadata;
  const dur = fmtDuration(m.duration_seconds);

  // Info grid
  const infoBlock = document.getElementById('infoBlock');
  infoBlock.style.display = 'block';
  document.getElementById('infoGrid').innerHTML = `
    <div class="info-cell"><div class="info-cell-label">Duration</div><div class="info-cell-val">${dur}</div></div>
    <div class="info-cell"><div class="info-cell-label">Resolution</div><div class="info-cell-val">${m.width}×${m.height}</div></div>
    <div class="info-cell"><div class="info-cell-label">FPS</div><div class="info-cell-val">${m.fps}</div></div>
    <div class="info-cell"><div class="info-cell-label">Size</div><div class="info-cell-val">${m.size_mb} MB</div></div>
  `;

  // Timeline
  buildTimeline(data.all_scores, data.highlights, m.duration_seconds);

  // Highlight cards
  const hlSection = document.getElementById('hlSection');
  const hlList = document.getElementById('hlList');
  document.getElementById('hlCountLabel').textContent =
    data.highlights.length + ' highlight moment' + (data.highlights.length !== 1 ? 's' : '') + ' detected';
  hlSection.style.display = 'block';

  const aiOn = data.ai_labeling;
  hlList.innerHTML = data.highlights.map((h, i) => `
    <div class="hl-card" id="hl-${i}" onclick="seekTo(${h.timestamp}, ${i})">
      <div class="hl-top">
        <div class="hl-num">${i+1}</div>
        <div class="hl-ts">${h.timestamp_formatted}</div>
        <div style="margin-left:auto;display:flex;gap:5px;align-items:center">
          ${h.is_goal ? '<span class="goal-badge">⚽ GOAL</span>' : ''}
          <div class="hl-score-badge">${h.ai_score !== undefined ? h.ai_score : h.excitement_score}%</div>
        </div>
      </div>
      ${h.label && h.label !== 'highlight' ? `<div class="hl-label">✦ ${h.label}</div>` : ''}
      <div class="hl-bar-bg"><div class="hl-bar-fill" style="width:${h.ai_score !== undefined ? h.ai_score : h.excitement_score}%"></div></div>
      <div class="clip-row" style="margin-top:10px">
        <button class="btn-watch" onclick="event.stopPropagation();watchHighlight('${filename}',${h.timestamp})">
          ▶ Watch
        </button>
      </div>
      <div class="clip-row" style="margin-top:6px">
        <select class="dur-select" id="dur-${i}">
          <option value="15,5" selected>15s before · 5s after</option>
          <option value="10,5">10s before · 5s after</option>
          <option value="20,5">20s before · 5s after</option>
          <option value="15,10">15s before · 10s after</option>
        </select>
        <button class="btn-clip" id="clipBtn-${i}" onclick="event.stopPropagation();createClip(${i},'${filename}',${h.timestamp})">
          <span class="clip-spinner" id="cs-${i}"></span>✂ Clip
        </button>
      </div>
    </div>
  `).join('');

  toast('Analysis complete — ' + data.highlights.length + ' highlights found', '🎯');
}

// ─── Timeline ─────────────────────────────────────────────
function buildTimeline(allScores, topMoments, duration) {
  const wrap = document.getElementById('timelineWrap');
  const thumb = document.getElementById('timelineThumb');
  const track = document.getElementById('timelineTrack');
  const cursor = document.getElementById('timelineCursor');

  if (!allScores || allScores.length === 0) return;

  wrap.style.display = 'block';
  const maxScore = Math.max(...allScores.map(s => s.motion_score));
  const topTs = new Set(topMoments.map(h => h.timestamp));

  thumb.innerHTML = allScores.map(s => {
    const left = (s.timestamp / duration * 100).toFixed(2);
    const h = Math.max(8, Math.round(s.motion_score / maxScore * 36));
    const isTop = topTs.has(s.timestamp);
    return `<div class="timeline-bar ${isTop ? 'top' : ''}" style="left:${left}%;height:${h}px" data-ts="${s.timestamp}" title="${fmtSeconds(s.timestamp)}"></div>`;
  }).join('');

  // Click to seek
  track.onclick = e => {
    const rect = track.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    const ts = pct * duration;
    seekTo(ts, -1);
  };

  // Hover tooltip
  track.onmousemove = e => {
    const rect = track.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    const ts = Math.max(0, Math.min(duration, pct * duration));
    cursor.style.display = 'block';
    cursor.style.left = (pct * 100) + '%';
    cursor.textContent = fmtSeconds(ts);
  };
  track.onmouseleave = () => { cursor.style.display = 'none'; };
}

// ─── Seek ─────────────────────────────────────────────────
function seekTo(ts, cardIndex) {
  const player = document.getElementById('player');
  if (player.src) {
    player.currentTime = ts;
    player.play();
  }
  // Activate card
  document.querySelectorAll('.hl-card').forEach(c => c.classList.remove('active'));
  if (cardIndex >= 0) {
    const card = document.getElementById('hl-' + cardIndex);
    if (card) {
      card.classList.add('active');
      card.scrollIntoView({behavior:'smooth',block:'nearest'});
    }
  }
}

// ─── Create Clip ──────────────────────────────────────────
async function createClip(index, filename, timestamp) {
  const btn = document.getElementById('clipBtn-' + index);
  const spinner = document.getElementById('cs-' + index);
  const durVal = document.getElementById('dur-' + index).value;
  const [before, after] = durVal.split(',').map(Number);

  btn.disabled = true;
  spinner.style.display = 'inline-block';
  btn.lastChild.textContent = ' Cutting...';

  try {
    const res = await fetch('/clip', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({filename, timestamp, before_secs: before, after_secs: after}),
    });
    const data = await res.json();

    if (data.success) {
      addClipToPanel(data, index + 1);
      toast('Clip created — ' + data.size_mb + ' MB', '✂️');
    } else {
      toast(data.error || 'Clip failed', '❌', 3000);
    }
  } catch(e) {
    toast('Request failed', '❌', 3000);
  }

  btn.disabled = false;
  spinner.style.display = 'none';
  btn.lastChild.textContent = ' Re-clip';
}

// ─── Add Clip to Panel ────────────────────────────────────
function addClipToPanel(data, hlNum) {
  totalClips++;
  const countEl = document.getElementById('clipCount');
  countEl.style.display = 'inline';
  countEl.textContent = totalClips;

  document.getElementById('noClips').style.display = 'none';

  const item = document.createElement('div');
  item.className = 'clip-item';
  item.id = 'clipItem-' + data.clip_id;
  item.innerHTML = `
    <div class="clip-item-head">
      <div class="clip-meta">
        <div class="clip-meta-name">Highlight #${hlNum}</div>
        <div class="clip-meta-sub">${fmtSeconds(data.start)} → ${fmtSeconds(data.start + data.duration)} · ${data.size_mb} MB</div>
      </div>
    </div>
    <div class="clip-actions">
      <button class="btn-preview" onclick="previewClip('${data.clip_filename}')">▶ Preview</button>
      <a class="btn-dl" href="/download/${data.clip_filename}" download="${data.clip_filename}">⬇ Save</a>
    </div>
  `;
  document.getElementById('clipsPanel').prepend(item);
}

// ─── Watch Highlight in Modal ─────────────────────────────
async function watchHighlight(filename, timestamp) {
  const modal      = document.getElementById('previewModal');
  const modalVideo = document.getElementById('modalVideo');
  const modalTitle = document.getElementById('modalTitle');
  const modalBody  = document.getElementById('modalBody');

  // Show modal immediately with loading state
  modalTitle.textContent = formatModalTime(timestamp) + '  ·  cutting clip...';
  modalVideo.style.display = 'none';
  modalBody.style.display  = 'flex';
  modal.style.display = 'flex';
  document.body.style.overflow = 'hidden';

  try {
    const res  = await fetch('/preview', {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({ filename, timestamp, before_secs: 15, after_secs: 5 }),
    });
    const data = await res.json();

    if (data.success) {
      modalTitle.textContent = formatModalTime(timestamp) + '  ·  ' + filename;
      modalBody.style.display  = 'none';
      modalVideo.style.display = 'block';
      modalVideo.src = '/download/' + encodeURIComponent(data.clip_filename);
      modalVideo.load();
      modalVideo.play();
    } else {
      modalBody.innerHTML = '<div style="color:var(--accent);font-size:.9rem">❌ ' + (data.error || 'Failed to generate preview') + '</div>';
    }
  } catch(e) {
    modalBody.innerHTML = '<div style="color:var(--accent);font-size:.9rem">❌ Network error</div>';
  }
}

function formatModalTime(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60).toString().padStart(2,'0');
  return m + ':' + sec;
}

// ─── Preview Clip Modal ───────────────────────────────────
function previewClip(clipFilename) {
  const modal = document.getElementById('previewModal');
  const modalVideo = document.getElementById('modalVideo');
  const modalTitle = document.getElementById('modalTitle');
  modalVideo.src = '/download/' + encodeURIComponent(clipFilename);
  modalTitle.textContent = clipFilename;
  modal.style.display = 'flex';
  document.body.style.overflow = 'hidden';
  modalVideo.load();
  modalVideo.play();
}

function closeModal() {
  const modal      = document.getElementById('previewModal');
  const modalVideo = document.getElementById('modalVideo');
  const modalBody  = document.getElementById('modalBody');
  modalVideo.pause();
  modalVideo.src          = '';
  modalVideo.style.display = 'block';
  modalBody.style.display  = 'none';
  modal.style.display      = 'none';
  document.body.style.overflow = '';
}

function modalFullscreen() {
  const modalVideo = document.getElementById('modalVideo');
  if (modalVideo.requestFullscreen) modalVideo.requestFullscreen();
  else if (modalVideo.webkitRequestFullscreen) modalVideo.webkitRequestFullscreen();
  else if (modalVideo.mozRequestFullScreen) modalVideo.mozRequestFullScreen();
}

// Close on backdrop click
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('previewModal').addEventListener('click', e => {
    if (e.target === document.getElementById('previewModal')) closeModal();
  });
});

// Close on Escape
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeModal();
});

// ─── Helpers ──────────────────────────────────────────────
function fmtSeconds(s) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60).toString().padStart(2, '0');
  return m + ':' + sec;
}
function fmtDuration(s) {
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return m > 0 ? m + 'm ' + sec + 's' : sec + 's';
}

// ─── Sign In ──────────────────────────────────────────────
function openSignIn() {
  document.getElementById('signInModal').style.display = 'flex';
  document.body.style.overflow = 'hidden';
  setTimeout(() => document.getElementById('signInEmail').focus(), 100);
}
function closeSignIn() {
  document.getElementById('signInModal').style.display = 'none';
  document.body.style.overflow = '';
}
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') { closeSignIn(); }
});

async function submitSignIn() {
  const email = document.getElementById('signInEmail').value.trim();
  const name  = document.getElementById('signInName').value.trim();
  const errEl = document.getElementById('signInError');
  const btn   = document.getElementById('signInSubmit');

  if (!email || !email.includes('@')) {
    errEl.textContent = 'Please enter a valid email.';
    errEl.style.display = 'block';
    return;
  }
  errEl.style.display = 'none';
  btn.textContent = 'Signing up...';
  btn.disabled = true;

  try {
    const res  = await fetch('/signup', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ email, name }),
    });
    const data = await res.json();
    if (data.success) {
      closeSignIn();
      // Update header
      document.getElementById('signInBtn').style.display = 'none';
      const disp = document.getElementById('userDisplay');
      disp.style.display = 'flex';
      document.getElementById('userEmail').textContent = email;
      localStorage.setItem('rc_user', JSON.stringify({ email, name }));
      toast(data.message, '🎉', 3500);
    } else {
      errEl.textContent = data.error || 'Something went wrong.';
      errEl.style.display = 'block';
    }
  } catch(e) {
    errEl.textContent = 'Network error — try again.';
    errEl.style.display = 'block';
  }
  btn.textContent = 'Get Early Access';
  btn.disabled = false;
}

// Restore session from localStorage
(function restoreSession() {
  const saved = localStorage.getItem('rc_user');
  if (saved) {
    try {
      const u = JSON.parse(saved);
      document.getElementById('signInBtn').style.display = 'none';
      const disp = document.getElementById('userDisplay');
      disp.style.display = 'flex';
      document.getElementById('userEmail').textContent = u.email;
    } catch(e) {}
  }
})();
</script>
</body>
</html>"""
