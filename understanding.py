"""
Gemini video understanding: find significant match events (knockdowns, fouls, goals, etc.)
in a given time window. Uses Gemini 2.5 Flash with native video clipping and streaming.
Used by the React app via the API.
"""
import json
import subprocess
import time
from pathlib import Path
from typing import Tuple, Union

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env from project root so GOOGLE_API_KEY (or GEMINI_API_KEY) can be set there
load_dotenv(Path(__file__).resolve().parent / ".env")

# Gemini 2.5 Flash has stronger video understanding and supports video_metadata clipping
VIDEO_MODEL = "models/gemini-2.5-flash"


def _format_time_range(start_min: int, start_sec: int, end_min: int, end_sec: int) -> Tuple[str, str]:
    start_str = f"{start_min}:{start_sec:02d}" if start_min > 0 else f"0:{start_sec}"
    end_str = f"{end_min}:{end_sec:02d}" if end_min > 0 else f"0:{end_sec}"
    return start_str, end_str


def _parse_time_to_min_sec(t: Union[str, int]) -> Tuple[int, int]:
    """Parse '10:20' or 620 (seconds) -> (min, sec)."""
    if isinstance(t, int):
        return t // 60, t % 60
    s = (t or "").strip()
    if not s:
        return 0, 0
    if s.isdigit():
        sec = int(s)
        return sec // 60, sec % 60
    parts = s.split(":")
    if len(parts) == 2:
        return int(parts[0].strip()), int(parts[1].strip())
    if len(parts) == 3:
        h, m, s = int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip())
        total_sec = h * 3600 + m * 60 + s
        return total_sec // 60, total_sec % 60
    return 0, 0


def _time_to_offset_seconds(t: Union[str, int]) -> int:
    """Convert start/end time to total seconds for VideoMetadata (e.g. 600 for 10:00)."""
    min_, sec = _parse_time_to_min_sec(t)
    return min_ * 60 + sec


def build_prompt(start_time: Union[str, int], end_time: Union[str, int]) -> str:
    """Build the analysis prompt. Video is already clipped to this window via video_metadata."""
    smin, ssec = _parse_time_to_min_sec(start_time)
    emin, esec = _parse_time_to_min_sec(end_time)
    start_str, end_str = _format_time_range(smin, ssec, emin, esec)
    return (
        "This clip is from a match. List every significant event in this clip that changes the course of the match: "
        "knockdowns, fouls (who committed / who was fouled), goals (scorer/assist if visible), "
        "cards (yellow/red), penalties, major chances (big saves, misses), injuries/stoppages, "
        "and any other pivotal moment (tactical change, big substitution). "
        f"For each event give the approximate timestamp (MM:SS within {start_str}–{end_str}) and a short description. "
        "Format each as: TIMESTAMP - Description (e.g. 10:05 - Goal by #9)."
    )


def _build_video_content(
    video_uri: str, start_time: Union[str, int], end_time: Union[str, int]
) -> types.Content:
    """Build Content with video part and optional video_metadata clipping (Gemini video understanding)."""
    start_sec = _time_to_offset_seconds(start_time)
    end_sec = _time_to_offset_seconds(end_time)
    prompt = build_prompt(start_time, end_time)

    # Clip video to the requested window so the model only sees that segment (better accuracy)
    video_metadata = None
    if end_sec > start_sec:
        video_metadata = types.VideoMetadata(
            start_offset=f"{start_sec}s",
            end_offset=f"{end_sec}s",
        )

    return types.Content(
        parts=[
            types.Part(
                file_data=types.FileData(file_uri=video_uri),
                video_metadata=video_metadata,
            ),
            types.Part(text=prompt),
        ]
    )


def analyze_video(video_uri: str, start_time: Union[str, int], end_time: Union[str, int]) -> str:
    """Run Gemini video understanding once and return the full analysis text."""
    client = genai.Client()
    response = client.models.generate_content(
        model=VIDEO_MODEL,
        contents=_build_video_content(video_uri, start_time, end_time),
    )
    return (response.text or "").strip()


def analyze_video_stream(video_uri: str, start_time: Union[str, int], end_time: Union[str, int]):
    """Stream Gemini video understanding response chunks for live updates."""
    client = genai.Client()
    stream = client.models.generate_content_stream(
        model=VIDEO_MODEL,
        contents=_build_video_content(video_uri, start_time, end_time),
    )
    for chunk in stream:
        if getattr(chunk, "text", None):
            yield chunk.text


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def upload_video(video_path: str) -> str:
    """Upload a local video file to Gemini and return its URI."""
    client = genai.Client()
    uploaded = client.files.upload(file=video_path)
    # Wait until the file is fully processed
    while uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state.name != "ACTIVE":
        raise RuntimeError(f"Upload failed, state: {uploaded.state.name}")
    return uploaded.uri


def analyze_full_video(video_path: str, interval: int = 5, output_json: str = "results.json"):
    """Analyze a local video in N-second chunks and save all results to a JSON file."""
    duration = get_video_duration(video_path)
    print(f"Video duration: {duration:.1f}s — analyzing every {interval}s")

    print("Uploading video to Gemini...")
    video_uri = upload_video(video_path)
    print(f"Upload complete: {video_uri}")

    results = []
    start = 0
    while start < duration:
        end = min(start + interval, int(duration))
        window = f"{start // 60}:{start % 60:02d} – {end // 60}:{end % 60:02d}"
        print(f"\nAnalyzing {window} ...")

        try:
            text = analyze_video(video_uri, start, end)
        except Exception as e:
            text = f"[Error: {e}]"

        results.append({
            "start_sec": start,
            "end_sec": end,
            "window": window,
            "analysis": text,
        })
        print(text if text else "(no events)")

        # Write incrementally so partial results are saved if interrupted
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)

        start += interval

    print(f"\nDone — {len(results)} segments saved to {output_json}")


if __name__ == "__main__":
    analyze_full_video("test.mp4", interval=5, output_json="results.json")
