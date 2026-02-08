"""
Live match event detection and speech-to-text using Gemini Live API.

- Video (--video <file.mp4>): extract audio from video and run live speech-to-text
  (transcribe commentary, crowd, etc.). Requires ffmpeg on PATH.
- Audio: send raw 16-bit PCM at 16 kHz via send_realtime_input(audio=...) for
  event detection or transcription.
"""
import asyncio
import struct
import warnings
from contextlib import aclosing
from pathlib import Path
from typing import AsyncIterator, Optional

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).resolve().parent / ".env")


def _gemini_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "Missing Gemini API key. Set GOOGLE_API_KEY or GEMINI_API_KEY in .env or environment. "
            "Get a key at https://aistudio.google.com/apikey"
        )
    return key

# Optional: play model's audio (24 kHz 16-bit PCM from Live API)
def _play_audio_pcm(data: bytes, sample_rate: int = 24000) -> None:
    """Play raw 16-bit PCM mono. No-op if sounddevice/numpy not available."""
    if not data:
        return
    try:
        import numpy as np
        import sounddevice as sd
        samples = np.frombuffer(data, dtype=np.int16)
        sd.play(samples, samplerate=sample_rate, blocking=True)
    except Exception:
        pass

# Live API: only gemini-2.5-flash-native-audio-preview-12-2025 is available on v1alpha.
# It requires response_modalities=["AUDIO"]. We request output_audio_transcription to get text.
# See https://ai.google.dev/gemini-api/docs/live-guide
LIVE_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"

MATCH_EVENTS_SYSTEM_INSTRUCTION_AUDIO = """You are analyzing live audio from a sports match (e.g. football/soccer, or similar).
Listen for anything significant that would change the course of the match or the crowd's reaction:
- Goals (and who scored if mentioned)
- Fouls, cards (yellow/red), penalties
- Knockdowns, injuries, stoppages
- Big chances, saves, near-misses
- Major substitutions or tactical shifts
- Crowd reactions (celebrations, protests)

Respond briefly in real time when you detect such an event. Give a short line: what happened and when if you can infer it.
Do not narrate every pass; only call out pivotal moments."""

MATCH_EVENTS_SYSTEM_INSTRUCTION_VIDEO = """You are analyzing live video frames from a sports match (e.g. football/soccer, or similar).
Watch for anything significant that would change the course of the match or the crowd's reaction:
- Goals (and who scored if visible)
- Fouls, cards (yellow/red), penalties
- Knockdowns, injuries, stoppages
- Big chances, saves, near-misses
- Major substitutions or tactical shifts
- Crowd reactions (celebrations, protests)

Respond briefly in real time when you detect such an event. Give a short line: what happened in the frame(s).
Do not describe every touch; only call out pivotal moments."""

# Live speech-to-text: transcribe only what's in the audio (e.g. from video), no added commentary
LIVE_TRANSCRIBE_SYSTEM_INSTRUCTION = """You are a live speech-to-text transcriber. Transcribe only what you hear in the audio: commentary, crowd noise, referee, players, any speech or distinct sounds. Output only the transcription. Do not add any response, commentary, or description. If you hear nothing or only silence, say nothing."""


async def run_live_session(
    audio_chunks: AsyncIterator[bytes],
    *,
    model: str = LIVE_MODEL,
    system_instruction: Optional[str] = None,
    response_modalities: Optional[list[str]] = None,
) -> AsyncIterator[str]:
    """
    Run a Live API session: send audio chunks and yield model text (via output transcription).

    Uses the native audio Live model with response_modalities=["AUDIO"] and
    output_audio_transcription so you get text of what the model says.

    Args:
        audio_chunks: Async iterator of raw 16-bit PCM audio at 16 kHz (mono).
        model: Live model name (default supports AUDIO + transcription).
        system_instruction: Override default match-events instruction.
        response_modalities: Default ["AUDIO"]; transcription gives text.

    Yields:
        Text from the model (transcript of its speech or msg.text).
    """
    # Live API is on v1alpha. Native audio model requires AUDIO; we get text via transcription.
    client = genai.Client(
        api_key=_gemini_api_key(),
        http_options=types.HttpOptions(api_version="v1alpha"),
    )
    modalities = response_modalities or ["AUDIO"]
    config = types.LiveConnectConfig(
        system_instruction=system_instruction or MATCH_EVENTS_SYSTEM_INSTRUCTION_AUDIO,
        response_modalities=modalities,
        output_audio_transcription={},  # so we get text transcript of model's speech
    )

    async with client.aio.live.connect(model=model, config=config) as session:
        # Task 1: send audio chunks
        async def send_audio() -> None:
            try:
                async for chunk in audio_chunks:
                    if not chunk:
                        continue
                    await session.send_realtime_input(
                        audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
                    )
                    await asyncio.sleep(0)  # yield to receive loop
                # Signal end of audio stream so model can finalize
                await session.send_realtime_input(audio_stream_end=True)
            except Exception:
                raise

        # Task 2: receive model responses; play audio and yield text
        loop = asyncio.get_event_loop()

        async def receive() -> AsyncIterator[str]:
            try:
                async for msg in session.receive():
                    # Play model's speech (24 kHz PCM) so you can hear it
                    audio_data = getattr(msg, "data", None)
                    if audio_data:
                        await loop.run_in_executor(
                            None, lambda d=audio_data: _play_audio_pcm(d)
                        )
                    text = getattr(msg, "text", None)
                    if text and text.strip():
                        yield text.strip()
                    sc = getattr(msg, "server_content", None)
                    if sc is not None:
                        ot = getattr(sc, "output_transcription", None)
                        if ot is not None and getattr(ot, "text", None):
                            t = (ot.text or "").strip()
                            if t:
                                yield t
            except Exception:
                pass

        send_task = asyncio.create_task(send_audio())
        try:
            async for text in receive():
                yield text
        finally:
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass


async def run_live_session_video(
    frame_chunks: AsyncIterator[bytes],
    *,
    model: str = LIVE_MODEL,
    system_instruction: Optional[str] = None,
    response_modalities: Optional[list[str]] = None,
    mime_type: str = "image/jpeg",
) -> AsyncIterator[str]:
    """
    Run a Live API session: send video frames (JPEG) and yield model text responses.

    Each chunk is one image frame sent as:
        await session.send_realtime_input(
            media=types.Blob(data=chunk_data, mime_type="image/jpeg")
        )

    Args:
        frame_chunks: Async iterator of JPEG (or other image) bytes, one frame per chunk.
        model: Live model name.
        system_instruction: Override default match-events instruction.
        response_modalities: e.g. ["TEXT"]; default ["TEXT"].
        mime_type: Image MIME type; default "image/jpeg".

    Yields:
        Text from the model (each yield is one server message's text, if any).
    """
    # Same model/config as audio: AUDIO + transcription (native audio model rejects TEXT for video).
    client = genai.Client(
        api_key=_gemini_api_key(),
        http_options=types.HttpOptions(api_version="v1alpha"),
    )
    modalities = response_modalities or ["AUDIO"]
    config = types.LiveConnectConfig(
        system_instruction=system_instruction or MATCH_EVENTS_SYSTEM_INSTRUCTION_VIDEO,
        response_modalities=modalities,
        output_audio_transcription={},
    )

    async with client.aio.live.connect(model=model, config=config) as session:
        async def send_frames() -> None:
            try:
                async for chunk in frame_chunks:
                    if not chunk:
                        continue
                    await session.send_realtime_input(
                        media=types.Blob(data=chunk, mime_type=mime_type)
                    )
                    await asyncio.sleep(0)
            except Exception:
                raise

        loop = asyncio.get_event_loop()

        async def receive() -> AsyncIterator[str]:
            try:
                async for msg in session.receive():
                    # Play model's speech so you can hear it
                    audio_data = getattr(msg, "data", None)
                    if audio_data:
                        await loop.run_in_executor(
                            None, lambda d=audio_data: _play_audio_pcm(d)
                        )
                    if getattr(msg, "text", None) and msg.text.strip():
                        yield msg.text.strip()
                    sc = getattr(msg, "server_content", None)
                    if sc is not None:
                        ot = getattr(sc, "output_transcription", None)
                        if ot is not None and getattr(ot, "text", None):
                            t = (ot.text or "").strip()
                            if t:
                                yield t
            except Exception:
                pass

        send_task = asyncio.create_task(send_frames())
        try:
            async for text in receive():
                yield text
        finally:
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass


async def audio_chunks_from_file(
    path: Path,
    *,
    chunk_size: int = 3200,
) -> AsyncIterator[bytes]:
    """
    Read a raw PCM file (16-bit, 16 kHz) and yield chunks.
    chunk_size=3200 gives 100 ms at 16 kHz mono 16-bit.
    """
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk
            await asyncio.sleep(0)


async def audio_from_video_file(
    path: Path,
    *,
    chunk_size: int = 3200,
) -> AsyncIterator[bytes]:
    """
    Extract audio from a video file as 16-bit PCM 16 kHz mono, yield in chunks.
    Requires ffmpeg on PATH.
    """
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-i", str(path),
        "-vn",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(chunk_size)
            if not chunk:
                break
            yield chunk
    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass
        except OSError:
            pass


async def demo_audio_chunks(
    duration_sec: float = 3.0,
    chunk_size: int = 3200,
) -> AsyncIterator[bytes]:
    """
    Yield silent 16-bit PCM at 16 kHz for demo (no file needed).
    duration_sec: how many seconds of audio to send.
    """
    num_samples = int(16000 * duration_sec)
    sent = 0
    while sent < num_samples:
        n = min(chunk_size // 2, num_samples - sent)
        chunk = struct.pack(f"<{n}h", *([0] * n))
        sent += n
        yield chunk
        await asyncio.sleep(0)


async def video_frames_from_file(
    path: Path,
    *,
    every_n_frames: int = 1,
    jpeg_quality: int = 85,
) -> AsyncIterator[bytes]:
    """
    Yield JPEG-encoded video frames from a file. Requires opencv-python.

    Args:
        path: Path to video file (e.g. .mp4, .avi).
        every_n_frames: Send every Nth frame (1 = every frame).
        jpeg_quality: JPEG quality 1–100.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("video_frames_from_file requires opencv-python: pip install opencv-python") from None

    cap = cv2.VideoCapture(str(path))
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    frame_index = [0]  # mutable so closure can update

    def get_next_jpeg() -> Optional[bytes]:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None
            if frame_index[0] % every_n_frames == 0:
                _, jpeg = cv2.imencode(".jpg", frame, encode_params)
                frame_index[0] += 1
                return jpeg.tobytes()
            frame_index[0] += 1

    loop = asyncio.get_event_loop()
    while True:
        chunk = await loop.run_in_executor(None, get_next_jpeg)
        if chunk is None:
            break
        yield chunk


# Common video extensions for CLI
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

if __name__ == "__main__":
    import sys

    # Reduce noise from google-auth and urllib3 when run as script
    warnings.filterwarnings("ignore", category=FutureWarning, module="google.auth")
    warnings.filterwarnings("ignore", category=FutureWarning, module="google.oauth2")
    warnings.filterwarnings("ignore", message=".*OpenSSL.*", category=UserWarning)

    async def _main() -> None:
        argv = sys.argv[1:]
        use_video = "--video" in argv or "-v" in argv
        if use_video:
            argv = [x for x in argv if x not in ("--video", "-v")]
        path = Path(argv[0]) if argv else None
        is_video = path and path.suffix.lower() in VIDEO_EXTENSIONS

        if use_video or is_video:
            if path and path.exists():
                print("Live speech-to-text: extracting audio from video...", flush=True)
                print("Connecting to Live API. Transcribing video audio...", flush=True)
                print("--- Live transcript ---", flush=True)
                try:
                    async with aclosing(audio_from_video_file(path)) as chunks:
                        async for text in run_live_session(
                            chunks,
                            system_instruction=LIVE_TRANSCRIBE_SYSTEM_INSTRUCTION,
                        ):
                            print("Transcript:", text, flush=True)
                except FileNotFoundError:
                    print("ffmpeg not found. Install it (e.g. brew install ffmpeg) for video audio extraction.", flush=True)
            else:
                print("Video path missing or not found.")
                print("Usage: python live_streaming.py [--video] <video.mp4>")
            print("Done.")
            return

        # Audio mode
        if path and path.exists():
            chunks = audio_chunks_from_file(path)
            print("Live session (audio). Sending PCM and listening for match events...")
            demo_mode = False
        else:
            if path:
                print("File not found, using demo (silent) audio instead.")
            else:
                print("No file given. Using demo (silent) audio to test the Live session.")
            print("Usage: python live_streaming.py <file.pcm>   (audio)")
            print("       python live_streaming.py --video <video.mp4>   (video frames as JPEG)")
            chunks = demo_audio_chunks(duration_sec=4.0)
            print("Live session started. Sending demo audio...")
            demo_mode = True

        try:
            print("--- Transcript (what the model says) ---", flush=True)
            if demo_mode:
                # Silent audio often gets no response; don't hang forever
                async def consume():
                    async for text in run_live_session(chunks):
                        print("Transcript:", text, flush=True)
                await asyncio.wait_for(consume(), timeout=25.0)
            else:
                async for text in run_live_session(chunks):
                    print("Transcript:", text, flush=True)
        except asyncio.TimeoutError:
            print("(No event in 25s — expected with silent demo audio. Try a real .pcm file.)")
        print("Done.")

    asyncio.run(_main())
