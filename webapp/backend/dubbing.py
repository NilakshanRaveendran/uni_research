"""Video dubbing pipeline: extract audio, transcribe, translate, synthesise, retime, remux.

This is the piece the research pipeline was missing. `bvt run` evaluates pre-segmented DRAL
fragments; a real video has continuous speech, so we need Whisper's *segment timestamps* to know
where each utterance sits on the timeline.

That requirement turns out to give us prosody preservation for free. Each synthesised segment is
time-scaled to fit the duration of the original segment it replaces, so the dubbed track keeps the
source's rhythm and stays aligned with the picture. Measured on the DRAL test split, raw XTTS output
runs ~1.5x longer than the human reference; without retiming, a dubbed video drifts out of sync
within a few utterances.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
import wave
from dataclasses import dataclass, field, asdict
from pathlib import Path

TARGET_SR = 16000  # Whisper / ECAPA operate at 16 kHz
MIN_SEGMENT_S = 0.30
# Bounds on how far a segment may be stretched or compressed. Beyond this the audio degrades
# audibly, so we clamp and report the clamp rather than produce something unlistenable.
MIN_SCALE, MAX_SCALE = 0.5, 2.0

LANGUAGES = {"en": "English", "es": "Spanish"}
DIRECTIONS = {"en-es": ("en", "es"), "es-en": ("es", "en")}


@dataclass
class SegmentResult:
    index: int
    start: float
    end: float
    source_text: str
    translated_text: str
    original_duration: float
    raw_tts_duration: float
    final_duration: float
    scale_applied: float
    clamped: bool
    error: str = ""


@dataclass
class DubResult:
    job_id: str
    direction: str
    video_out: str
    audio_out: str
    segments: list[SegmentResult] = field(default_factory=list)
    speaker_similarity: float | None = None
    duration_match_ratio: float | None = None
    raw_duration_ratio: float | None = None
    clamped_segments: int = 0
    failed_segments: int = 0
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["segments"] = [asdict(s) if not isinstance(s, dict) else s for s in self.segments]
        return data


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {result.stderr[-500:]}")


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=False,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def extract_audio(video: Path, out_wav: Path) -> Path:
    """Pull a 16 kHz mono WAV out of any container ffmpeg can read."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", str(TARGET_SR),
          "-c:a", "pcm_s16le", str(out_wav)])
    return out_wav


def has_video_stream(path: Path) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=False,
    )
    return "video" in result.stdout


def merge_audio_into_video(video: Path, audio: Path, out_video: Path) -> Path:
    """Replace the video's audio track. Video is stream-copied, so this is fast and lossless."""
    out_video.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-i", str(video), "-i", str(audio),
          "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
          "-shortest", str(out_video)])
    return out_video


def _read_wav(path: Path):
    import numpy as np

    with wave.open(str(path), "rb") as handle:
        sr = handle.getframerate()
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sr


def _write_wav(path: Path, audio, sr: int) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes((clipped * 32767.0).astype("<i2").tobytes())


def time_scale(audio, sr: int, factor: float):
    """Stretch (factor>1) or compress (factor<1) without shifting pitch.

    WORLD analysis/resynthesis: decompose into F0, spectral envelope and aperiodicity, resample the
    frame sequence, resynthesise. This preserves formants, so a retimed voice still sounds like the
    same person -- which matters because we measure speaker similarity on the result.
    """
    import numpy as np

    if abs(factor - 1.0) < 1e-3 or len(audio) < sr // 20:
        return audio
    try:
        import pyworld

        contiguous = np.ascontiguousarray(audio, dtype=np.float64)
        f0, t = pyworld.dio(contiguous, sr)
        f0 = pyworld.stonemask(contiguous, f0, t, sr)
        spectrum = pyworld.cheaptrick(contiguous, f0, t, sr)
        aperiodicity = pyworld.d4c(contiguous, f0, t, sr)
        frames = len(f0)
        if frames < 4:
            return audio
        index = np.linspace(0, frames - 1, max(4, int(round(frames * factor))))
        source = np.arange(frames)
        f0_s = np.interp(index, source, f0)
        sp_s = np.stack([np.interp(index, source, spectrum[:, k])
                         for k in range(spectrum.shape[1])], axis=1)
        ap_s = np.stack([np.interp(index, source, aperiodicity[:, k])
                         for k in range(aperiodicity.shape[1])], axis=1)
        return pyworld.synthesize(
            np.ascontiguousarray(f0_s),
            np.ascontiguousarray(sp_s),
            np.ascontiguousarray(ap_s),
            sr,
        )
    except Exception:  # noqa: BLE001 - fall back rather than fail the whole job
        import librosa

        return librosa.effects.time_stretch(
            np.asarray(audio, dtype=float), rate=1.0 / max(factor, 1e-6)
        )


def transcribe_segments(models, audio_path: Path, language: str) -> list[dict]:
    """Whisper transcription WITH timestamps, which is what makes timeline placement possible."""
    result = models.asr.transcribe(
        str(audio_path), language=language, task="transcribe", verbose=False
    )
    segments = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        start, end = float(seg["start"]), float(seg["end"])
        if text and end - start >= MIN_SEGMENT_S:
            segments.append({"start": start, "end": end, "text": text})
    return segments


def dub(
    video_path: Path,
    direction: str,
    job_dir: Path,
    models,
    progress=None,
) -> DubResult:
    """Full dubbing pass. `progress(pct, message)` is called as work completes."""
    import numpy as np

    started = time.time()
    source_lang, target_lang = DIRECTIONS[direction]
    job_dir.mkdir(parents=True, exist_ok=True)

    def note(pct: int, message: str) -> None:
        if progress:
            progress(pct, message)

    note(3, "Extracting audio from video")
    source_wav = extract_audio(video_path, job_dir / "source.wav")
    total_duration = ffprobe_duration(source_wav)

    note(10, f"Transcribing {LANGUAGES[source_lang]} speech")
    segments = transcribe_segments(models, source_wav, source_lang)
    if not segments:
        raise RuntimeError("No speech detected in the uploaded file")
    note(20, f"Found {len(segments)} speech segments")

    source_audio, source_sr = _read_wav(source_wav)
    timeline_sr = 24000  # XTTS-v2 output rate; the assembled track uses this throughout
    timeline = np.zeros(int(math.ceil(total_duration * timeline_sr)) + timeline_sr, dtype=np.float64)

    results: list[SegmentResult] = []
    segment_dir = job_dir / "segments"
    segment_dir.mkdir(exist_ok=True)

    for i, seg in enumerate(segments):
        span = 20 + int(65 * i / max(1, len(segments)))
        note(span, f"Segment {i + 1}/{len(segments)}: translating and synthesising")
        original = seg["end"] - seg["start"]
        record = SegmentResult(
            index=i, start=seg["start"], end=seg["end"], source_text=seg["text"],
            translated_text="", original_duration=original, raw_tts_duration=0.0,
            final_duration=0.0, scale_applied=1.0, clamped=False,
        )
        try:
            translated = models.translate(seg["text"], source_lang, target_lang)
            record.translated_text = translated
            if not translated.strip():
                raise ValueError("translation was empty")

            out_wav = segment_dir / f"seg_{i:04d}.wav"
            # The speaker reference is the FULL source audio, not the segment: XTTS produces a
            # better-conditioned voice from several seconds of reference than from one short clip.
            models.synthesize(translated, source_wav, target_lang, out_wav, seed=498 + i)
            generated, gen_sr = _read_wav(out_wav)
            record.raw_tts_duration = len(generated) / gen_sr

            # --- prosody preservation: fit the dub into the slot it replaces ---
            wanted = original / record.raw_tts_duration if record.raw_tts_duration > 0 else 1.0
            scale = min(max(wanted, MIN_SCALE), MAX_SCALE)
            record.clamped = abs(scale - wanted) > 1e-6
            record.scale_applied = scale
            retimed = time_scale(generated, gen_sr, scale)
            record.final_duration = len(retimed) / gen_sr

            if gen_sr != timeline_sr:
                import librosa

                retimed = librosa.resample(
                    np.asarray(retimed, dtype=float), orig_sr=gen_sr, target_sr=timeline_sr
                )
            offset = int(seg["start"] * timeline_sr)
            end = min(offset + len(retimed), len(timeline))
            timeline[offset:end] += retimed[: end - offset]
            _write_wav(segment_dir / f"seg_{i:04d}_retimed.wav", retimed, timeline_sr)
        except Exception as exc:  # noqa: BLE001 - one bad segment must not lose the whole job
            record.error = f"{type(exc).__name__}: {exc}"[:200]
        results.append(record)

    note(88, "Assembling dubbed audio track")
    peak = float(np.max(np.abs(timeline))) if timeline.size else 0.0
    if peak > 1.0:
        timeline = timeline / peak * 0.97
    dubbed_wav = job_dir / "dubbed.wav"
    _write_wav(dubbed_wav, timeline, timeline_sr)

    note(92, "Measuring speaker similarity")
    similarity = None
    try:
        similarity = models.speaker_similarity(source_wav, dubbed_wav)
    except Exception:  # noqa: BLE001 - metric is informative, not load-bearing
        similarity = None

    note(95, "Merging audio back into video")
    if has_video_stream(video_path):
        out_video = merge_audio_into_video(video_path, dubbed_wav, job_dir / "dubbed.mp4")
    else:
        out_video = dubbed_wav  # audio-only upload: the dubbed track *is* the deliverable

    ok = [r for r in results if not r.error]
    raw_total = sum(r.raw_tts_duration for r in ok)
    final_total = sum(r.final_duration for r in ok)
    original_total = sum(r.original_duration for r in ok)

    result = DubResult(
        job_id=job_dir.name,
        direction=direction,
        video_out=str(out_video),
        audio_out=str(dubbed_wav),
        segments=results,
        speaker_similarity=similarity,
        duration_match_ratio=(final_total / original_total) if original_total else None,
        raw_duration_ratio=(raw_total / original_total) if original_total else None,
        clamped_segments=sum(1 for r in results if r.clamped),
        failed_segments=sum(1 for r in results if r.error),
        elapsed_s=round(time.time() - started, 1),
    )
    (job_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    note(100, "Done")
    return result


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
