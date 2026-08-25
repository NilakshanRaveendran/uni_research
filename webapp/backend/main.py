"""FastAPI service for the dubbing web app.

Jobs run on a background thread with a single global lock: XTTS on CPU already saturates the
machine, so running two dubs concurrently would make both slower and risk memory pressure. Models
are loaded once, lazily, and reused across jobs -- the first request pays ~30 s of model loading,
every later one does not.

    .venv/bin/python -m uvicorn webapp.backend.main:app --port 8000
"""

from __future__ import annotations

import shutil
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bilingual_voice.pipeline import LocalModels  # noqa: E402

from .dubbing import DIRECTIONS, dub, ffmpeg_available  # noqa: E402

JOBS_DIR = ROOT / "webapp" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_MB = 200
ALLOWED_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".wav", ".mp3", ".m4a", ".flac"}

app = FastAPI(title="Bilingual Voice Dubbing", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()
_MODELS: LocalModels | None = None
_MODELS_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()  # only one dub at a time


def get_models() -> LocalModels:
    global _MODELS
    with _MODELS_LOCK:
        if _MODELS is None:
            _MODELS = LocalModels(whisper_size="small", mt_device="cpu",
                                  model_root=ROOT / "models")
        return _MODELS


def _set(job_id: str, **fields) -> None:
    with _LOCK:
        _JOBS.setdefault(job_id, {}).update(fields)


def _worker(job_id: str, video: Path, direction: str) -> None:
    job_dir = JOBS_DIR / job_id
    try:
        _set(job_id, status="running", progress=1, message="Waiting for a free worker")
        with _RUN_LOCK:
            def progress(pct: int, message: str) -> None:
                _set(job_id, progress=pct, message=message)

            result = dub(video, direction, job_dir, get_models(), progress)
        _set(job_id, status="done", progress=100, message="Complete",
             result=result.to_dict())
    except Exception as exc:  # noqa: BLE001 - surface the failure to the client
        _set(job_id, status="error", message=f"{type(exc).__name__}: {exc}"[:400])


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "ffmpeg": ffmpeg_available(),
        "directions": list(DIRECTIONS),
        "models_loaded": _MODELS is not None,
        "max_upload_mb": MAX_UPLOAD_MB,
    }


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), direction: str = Form("es-en")) -> JSONResponse:
    if direction not in DIRECTIONS:
        raise HTTPException(400, f"direction must be one of {list(DIRECTIONS)}")
    if not ffmpeg_available():
        raise HTTPException(500, "ffmpeg/ffprobe not found on the server")

    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"unsupported file type '{suffix}'")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    target = job_dir / f"input{suffix}"

    size = 0
    with target.open("wb") as handle:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB * 1024 * 1024:
                handle.close()
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(413, f"file exceeds {MAX_UPLOAD_MB} MB")
            handle.write(chunk)

    _set(job_id, status="queued", progress=0, message="Queued",
         direction=direction, filename=file.filename, size_bytes=size)
    threading.Thread(target=_worker, args=(job_id, target, direction), daemon=True).start()
    return JSONResponse({"job_id": job_id, "status": "queued"}, status_code=202)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    with _LOCK:
        job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return {"job_id": job_id, **job}


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str) -> FileResponse:
    with _LOCK:
        job = _JOBS.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "result not ready")
    path = Path(job["result"]["video_out"])
    if not path.exists():
        raise HTTPException(404, "output missing")
    media = "video/mp4" if path.suffix == ".mp4" else "audio/wav"
    return FileResponse(path, media_type=media, filename=path.name)


@app.get("/api/jobs/{job_id}/audio")
def job_audio(job_id: str) -> FileResponse:
    with _LOCK:
        job = _JOBS.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "result not ready")
    path = Path(job["result"]["audio_out"])
    if not path.exists():
        raise HTTPException(404, "output missing")
    return FileResponse(path, media_type="audio/wav", filename=path.name)
