"""
FastAPI backend: live video analysis pipeline.
  1. Upload video → Gemini file API
  2. As video plays, analyze each 5-sec segment live (Gemini)
  3. Generate ads from each event (Groq / DeepSeek R1)
  4. Append everything to results.json + ad_results.json in real time
"""
import json
import shutil
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from understanding import analyze_video, analyze_video_stream, upload_video
from ad_generator import generate_ad, generate_ad_stream, process_all_events

app = FastAPI(title="SuperBowl Ad Pulse API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory state: uploaded Gemini video URI ──
_state = {
    "video_uri": None,       # Gemini file URI after upload
    "uploading": False,
}
_state_lock = threading.Lock()

RESULTS_FILE = Path("results.json")
AD_RESULTS_FILE = Path("ad_results.json")


# ─────────────────────────────────────────────
# 1. Upload video to Gemini
# ─────────────────────────────────────────────

@app.post("/api/upload-video")
async def upload_video_endpoint(file: UploadFile = File(...)):
    """
    Accept a video file from the browser, save locally, upload to Gemini.
    Returns the Gemini file URI so the frontend can trigger per-segment analysis.
    """
    with _state_lock:
        if _state["uploading"]:
            raise HTTPException(status_code=409, detail="Upload already in progress")
        _state["uploading"] = True

    try:
        # Save uploaded file locally
        local_path = Path("uploaded_video.mp4")
        with open(local_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Upload to Gemini and wait for processing
        uri = upload_video(str(local_path))

        with _state_lock:
            _state["video_uri"] = uri

        # Clear previous results
        RESULTS_FILE.write_text("[]")
        AD_RESULTS_FILE.write_text("[]")

        return {"video_uri": uri, "status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        with _state_lock:
            _state["uploading"] = False


@app.get("/api/upload-status")
def upload_status():
    """Check if a video has been uploaded and is ready for analysis."""
    with _state_lock:
        return {
            "video_uri": _state["video_uri"],
            "uploading": _state["uploading"],
            "ready": _state["video_uri"] is not None and not _state["uploading"],
        }


# ─────────────────────────────────────────────
# 2. Live segment analysis: Gemini → JSON → Groq → ad
# ─────────────────────────────────────────────

class LiveSegmentRequest(BaseModel):
    start_sec: int
    end_sec: int
    business_name: str = ""
    business_type: str = ""
    video_uri: Optional[str] = None  # override if provided


@app.post("/api/live-segment")
def live_segment(request: LiveSegmentRequest):
    """
    Analyze ONE 5-second segment live:
      1. Gemini analyzes the video segment → event text
      2. Append event to results.json
      3. Groq generates an ad from the event
      4. Append ad to ad_results.json
      5. Return both event + ad to the frontend
    """
    # Determine video URI
    uri = request.video_uri
    if not uri:
        with _state_lock:
            uri = _state["video_uri"]
    if not uri:
        raise HTTPException(status_code=400, detail="No video uploaded yet. Upload first.")

    start = request.start_sec
    end = request.end_sec
    window = f"{start // 60}:{start % 60:02d} – {end // 60}:{end % 60:02d}"

    # Step 1: Gemini video analysis
    try:
        analysis_text = analyze_video(uri, start, end)
    except Exception as e:
        analysis_text = f"[Error: {e}]"

    event = {
        "start_sec": start,
        "end_sec": end,
        "window": window,
        "analysis": analysis_text,
    }

    # Step 2: Append event to results.json
    _append_to_json(RESULTS_FILE, event)

    # Step 3: Groq ad generation
    try:
        ad = generate_ad(event, request.business_name, request.business_type)
    except Exception as e:
        ad = {"is_significant": False, "error": str(e)}

    ad["source_event"] = event

    # Step 4: Append ad to ad_results.json
    _append_to_json(AD_RESULTS_FILE, ad)

    # Step 5: Return both
    return {"event": event, "ad": ad}


def _append_to_json(path: Path, item: dict):
    """Thread-safe append a single item to a JSON array file."""
    with _state_lock:
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, ValueError):
                data = []
        else:
            data = []
        data.append(item)
        path.write_text(json.dumps(data, indent=2))


# ─────────────────────────────────────────────
# 3. Reset — clear all results for a fresh run
# ─────────────────────────────────────────────

@app.post("/api/reset")
def reset():
    """Clear all results and start fresh."""
    RESULTS_FILE.write_text("[]")
    AD_RESULTS_FILE.write_text("[]")
    return {"status": "cleared"}


# ─────────────────────────────────────────────
# 4. Read current results
# ─────────────────────────────────────────────

@app.get("/api/events")
def get_events():
    """Return all detected events so far."""
    if not RESULTS_FILE.exists():
        return []
    return json.loads(RESULTS_FILE.read_text())


@app.get("/api/ad-results")
def get_ad_results():
    """Return all generated ads so far."""
    if not AD_RESULTS_FILE.exists():
        return []
    return json.loads(AD_RESULTS_FILE.read_text())


# ─────────────────────────────────────────────
# 5. Legacy / batch endpoints (kept for flexibility)
# ─────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    video_url: str
    start_time: str
    end_time: str


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest):
    """Stream Gemini analysis for a given video and time window."""
    video_url = (request.video_url or "").strip()
    if not video_url:
        raise HTTPException(status_code=400, detail="video_url is required")

    def generate():
        try:
            for chunk in analyze_video_stream(video_url, request.start_time, request.end_time):
                yield chunk
        except Exception as e:
            yield f"\n[Error: {e}]"

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class GenerateAdRequest(BaseModel):
    event: dict
    business_name: str = ""
    business_type: str = ""


@app.post("/api/generate-ad")
def generate_single_ad(request: GenerateAdRequest):
    """Generate an ad for a single game event."""
    try:
        ad = generate_ad(request.event, request.business_name, request.business_type)
        return ad
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health():
    return {"status": "ok"}
