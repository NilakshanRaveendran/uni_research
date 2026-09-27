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

import inspect
import json
import math
import shutil
import statistics
import subprocess
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import background

TARGET_SR = 16000  # Whisper / ECAPA operate at 16 kHz
MIN_SEGMENT_S = 0.30
# Bounds on how far a segment may be stretched or compressed. Beyond this the audio degrades
# audibly, so we clamp and report the clamp rather than produce something unlistenable.
MIN_SCALE, MAX_SCALE = 0.5, 2.0

LANGUAGES = {"en": "English", "es": "Spanish"}
DIRECTIONS = {"en-es": ("en", "es"), "es-en": ("es", "en")}
# Whisper is told the source language rather than asked, so a Tamil video submitted as Spanish
# would otherwise be "transcribed" as Spanish gibberish and dubbed without complaint. Below this
# confidence the detection is treated as inconclusive (silence, music intros) and the job proceeds.
LANGUAGE_CONFIDENCE = 0.5


class UnsupportedLanguageError(RuntimeError):
    """The upload's spoken language does not match the selected direction."""


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
    # Per-word timings on the output timeline, for highlighting words as they are spoken.
    # `words` is the dub (what the viewer hears); `source_words` is the original speech.
    words: list[dict] = field(default_factory=list)
    source_words: list[dict] = field(default_factory=list)


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
    background_mode: str = "none"
    background_model: str = ""
    detected_language: str = ""
    language_confidence: float | None = None
    mix_info: dict = field(default_factory=dict)
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
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def extract_audio(video: Path, out_wav: Path) -> Path:
    """Pull a 16 kHz mono WAV out of any container ffmpeg can read."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(TARGET_SR),
            "-c:a",
            "pcm_s16le",
            str(out_wav),
        ]
    )
    return out_wav


def has_video_stream(path: Path) -> bool:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return "video" in result.stdout


def merge_audio_into_video(video: Path, audio: Path, out_video: Path) -> Path:
    """Replace the video's audio track. Video is stream-copied, so this is fast and lossless."""
    out_video.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(out_video),
        ]
    )
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
        index = np.linspace(0, frames - 1, max(4, round(frames * factor)))
        source = np.arange(frames)
        # F0 must be resampled by NEAREST NEIGHBOUR, not linearly. WORLD stores unvoiced frames
        # as f0 = 0, so linear interpolation between a voiced frame and an unvoiced one invents
        # pitch values that were never in the signal and smears the voiced/unvoiced boundary.
        # Measured effect of getting this wrong: a 3.5 semitone shift on a segment that was only
        # supposed to be stretched in time.
        f0_s = f0[np.clip(np.round(index).astype(int), 0, frames - 1)]
        # Spectral envelope and aperiodicity are smooth and continuous, so linear is correct there.
        sp_s = np.stack(
            [np.interp(index, source, spectrum[:, k]) for k in range(spectrum.shape[1])], axis=1
        )
        ap_s = np.stack(
            [np.interp(index, source, aperiodicity[:, k]) for k in range(aperiodicity.shape[1])],
            axis=1,
        )
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


def detect_language(models, audio_path: Path) -> tuple[str, float]:
    """Whisper's language ID on the first 30 s of audio: (language code, probability)."""
    import whisper

    model = models.asr
    audio = whisper.pad_or_trim(whisper.load_audio(str(audio_path)))
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    _, probs = model.detect_language(mel)
    code = max(probs, key=probs.get)
    return code, float(probs[code])


def language_name(code: str) -> str:
    if code in LANGUAGES:
        return LANGUAGES[code]
    try:
        from whisper.tokenizer import LANGUAGES as WHISPER_LANGUAGES
    except ImportError:
        return code
    return WHISPER_LANGUAGES.get(code, code).title()


def check_language(detected: str, confidence: float, direction: str) -> None:
    """Raise UnsupportedLanguageError when the speech confidently isn't the source language."""
    source_lang, target_lang = DIRECTIONS[direction]
    if detected == source_lang or confidence < LANGUAGE_CONFIDENCE:
        return
    heard = f"{language_name(detected)} ({confidence:.0%} confidence)"
    if detected == target_lang:
        raise UnsupportedLanguageError(
            f"This video is in {heard}, but {LANGUAGES[source_lang]} → "
            f"{LANGUAGES[target_lang]} was selected. Choose {LANGUAGES[target_lang]} → "
            f"{LANGUAGES[source_lang]} and upload again."
        )
    raise UnsupportedLanguageError(
        f"This video appears to be in {heard}. Only English and Spanish speech is supported."
    )


def _word_list(timings) -> list[dict]:
    words = []
    for w in timings:
        text = (w["word"] if isinstance(w, dict) else w.word).strip()
        start = w["start"] if isinstance(w, dict) else w.start
        end = w["end"] if isinstance(w, dict) else w.end
        if text:  # merged-away punctuation leaves empty entries
            words.append(
                {"word": text, "start": round(float(start), 3), "end": round(float(end), 3)}
            )
    return words


def transcribe_segments(models, audio_path: Path, language: str) -> list[dict]:
    """Whisper transcription WITH timestamps, which is what makes timeline placement possible."""
    result = models.asr.transcribe(
        str(audio_path),
        language=language,
        task="transcribe",
        verbose=False,
        word_timestamps=True,
    )
    segments = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        start, end = float(seg["start"]), float(seg["end"])
        if text and end - start >= MIN_SEGMENT_S:
            words = _word_list(seg.get("words") or [])
            segments.append({"start": start, "end": end, "text": text, "words": words})
    return segments


def align_words(models, audio_16k, text: str, language: str) -> list[dict]:
    """Forced-align KNOWN text to audio: when is each word of the dub actually spoken?

    Transcribing the dub again would return Whisper's words, which can differ from the text we
    synthesised; aligning the exact text we sent to XTTS keeps the highlighted words identical to
    the translation shown on the page. Times are seconds from the start of `audio_16k`.
    """
    import numpy as np
    import torch
    from whisper.audio import HOP_LENGTH, N_FRAMES, log_mel_spectrogram, pad_or_trim
    from whisper.timing import find_alignment, merge_punctuations
    from whisper.tokenizer import get_tokenizer

    model = models.asr
    tokenizer = get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language=language,
        task="transcribe",
    )
    audio = torch.from_numpy(np.asarray(audio_16k, dtype=np.float32))
    num_frames = min(len(audio) // HOP_LENGTH, N_FRAMES)
    mel = log_mel_spectrogram(pad_or_trim(audio), n_mels=model.dims.n_mels).to(model.device)
    alignment = find_alignment(
        model, tokenizer, tokenizer.encode(" " + text.strip()), mel, num_frames
    )
    # Same punctuation sets Whisper's own word_timestamps uses, so "acento." is one word.
    merge_punctuations(alignment, "\"'“¿([{-", "\"'.。,，!！?？:：”)]}、")
    duration = len(audio) / TARGET_SR
    words = _word_list(alignment)
    for w in words:
        w["start"], w["end"] = min(w["start"], duration), min(w["end"], duration)
    return words


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

    # The progress callback gained an ETA argument. Older callers pass a two-argument function,
    # so the arity is inspected once rather than letting the extra argument raise mid-job.
    accepts_eta = False
    if progress is not None:
        try:
            accepts_eta = len(inspect.signature(progress).parameters) >= 3
        except (TypeError, ValueError):  # builtins and C callables have no inspectable signature
            accepts_eta = False

    def note(pct: int, message: str, eta: float | None = None) -> None:
        if not progress:
            return
        if accepts_eta:
            progress(pct, message, eta)
        else:
            progress(pct, message)

    note(3, "Extracting audio from video")
    source_wav = extract_audio(video_path, job_dir / "source.wav")
    total_duration = ffprobe_duration(source_wav)

    # Checked before separation and synthesis, which take minutes, so a wrong-language upload
    # fails within seconds instead of producing a confidently nonsensical dub.
    note(4, "Detecting spoken language")
    detected_lang, lang_confidence = detect_language(models, source_wav)
    check_language(detected_lang, lang_confidence, direction)
    note(5, f"Detected {language_name(detected_lang)} speech")

    # Full-quality stereo copy, kept for separation and for the final mix. The 16 kHz mono file
    # above is what Whisper and ECAPA want; mixing music at 16 kHz mono would discard most of
    # what the background-preservation step exists to save.
    background_mode = background.requested_mode()
    separated = None
    full_wav = None
    if background_mode != "none":
        try:
            full_wav = background.extract_full_audio(video_path, job_dir / "source_full.wav")
        except Exception as exc:  # noqa: BLE001 - fall back to speech-only rather than fail
            note(4, f"Full-quality extraction failed ({type(exc).__name__}); speech only")
            background_mode = "none"

    if background_mode == "separate" and full_wav is not None:
        note(5, "Separating voice from music and effects")
        separated = background.separate(
            full_wav, job_dir / "stems", progress=lambda message: note(6, message)
        )
        if separated is None:
            # Not an error: ducking still returns the music and effects, at the cost of leaving
            # the original voice faintly audible underneath.
            note(8, "Separation unavailable; keeping the original audio ducked underneath")
            background_mode = "duck"

    # Transcribe the isolated voice when we have it: Whisper on a track with the music and
    # gunfire removed makes fewer errors than Whisper on the full mix.
    asr_wav = source_wav
    if separated is not None:
        try:
            asr_wav = extract_audio(separated["vocals"], job_dir / "vocals_16k.wav")
        except Exception:  # noqa: BLE001 - the original mix is a fine fallback
            asr_wav = source_wav

    note(10, f"Transcribing {LANGUAGES[source_lang]} speech")
    segments = transcribe_segments(models, asr_wav, source_lang)
    if not segments:
        raise RuntimeError("No speech detected in the uploaded file")
    note(20, f"Found {len(segments)} speech segments")

    # One batched call rather than one per segment: translation is cheap next to synthesis, and
    # doing it up front means the per-segment timing used for the ETA measures synthesis alone.
    try:
        translations = models.translate_many(
            [seg["text"] for seg in segments], source_lang, target_lang
        )
    except Exception:  # noqa: BLE001 - fall back to per-segment translation
        translations = [None] * len(segments)

    timeline_sr = 24000  # XTTS-v2 output rate; the assembled track uses this throughout
    timeline = np.zeros(math.ceil(total_duration * timeline_sr) + timeline_sr, dtype=np.float64)

    results: list[SegmentResult] = []
    segment_dir = job_dir / "segments"
    segment_dir.mkdir(exist_ok=True)

    # The ETA is measured on THIS machine rather than assumed from a hardcoded multiplier: after
    # the first segment we know what synthesis actually costs here, and the estimate self-corrects
    # as more segments finish.
    segment_times: list[float] = []
    tail_fraction = 0.14  # mixing, similarity and muxing, as a share of the segment work

    for i, seg in enumerate(segments):
        started_segment = time.time()
        span = 20 + int(65 * i / max(1, len(segments)))
        remaining = len(segments) - i
        eta = None
        # The FIRST segment carries one-off warm-up (lazy allocation, first touch of the model's
        # weights) and runs about twice as slow as the rest. Extrapolating from it alone reported
        # 135s remaining on a job that finished in 68s, so nothing is published until a second
        # segment has been timed -- the UI shows "estimating..." until then, which is honest.
        # From then on a median of recent segments keeps one slow outlier from dominating.
        if len(segment_times) >= 2:
            per = statistics.median(segment_times[1:][-5:])
            eta = per * remaining * (1.0 + tail_fraction)
        note(span, f"Segment {i + 1} of {len(segments)}", eta)
        original = seg["end"] - seg["start"]
        record = SegmentResult(
            index=i,
            start=seg["start"],
            end=seg["end"],
            source_text=seg["text"],
            translated_text="",
            original_duration=original,
            raw_tts_duration=0.0,
            final_duration=0.0,
            scale_applied=1.0,
            clamped=False,
            source_words=seg.get("words", []),
        )
        try:
            translated = translations[i]
            if translated is None:
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

            try:
                import librosa

                speech_16k = librosa.resample(
                    np.asarray(retimed, dtype=float), orig_sr=timeline_sr, target_sr=TARGET_SR
                )
                record.words = [
                    {
                        **w,
                        "start": round(seg["start"] + w["start"], 3),
                        "end": round(seg["start"] + w["end"], 3),
                    }
                    for w in align_words(models, speech_16k, translated, target_lang)
                ]
            except Exception:  # noqa: BLE001 - highlighting is a nicety; the page estimates instead
                record.words = []
        except Exception as exc:  # noqa: BLE001 - one bad segment must not lose the whole job
            record.error = f"{type(exc).__name__}: {exc}"[:200]
        results.append(record)
        segment_times.append(time.time() - started_segment)

    note(88, "Assembling dubbed audio track")
    peak = float(np.max(np.abs(timeline))) if timeline.size else 0.0
    if peak > 1.0:
        timeline = timeline / peak * 0.97
    dubbed_wav = job_dir / "dubbed.wav"
    _write_wav(dubbed_wav, timeline, timeline_sr)  # speech only, kept for inspection

    mix_info: dict = {}
    background_model = ""
    final_wav = dubbed_wav
    if background_mode != "none" and full_wav is not None:
        note(89, "Mixing music and effects back in")
        try:
            if separated is not None:
                bed, bed_sr = background.read_stereo(separated["background"])
                reference, _ref_sr = background.read_stereo(separated["vocals"])
                background_model = separated["model"]
            else:
                bed, bed_sr = background.read_stereo(full_wav)
                bed = background.duck(bed, bed_sr, segments)
                reference = None
                background_model = "ducked original (no separation)"
            mixed, mix_sr, mix_info = background.mix(
                timeline, timeline_sr, bed, bed_sr, reference=reference
            )
            final_wav = background.write_stereo(job_dir / "dubbed_mixed.wav", mixed, mix_sr)
        except Exception as exc:  # noqa: BLE001 - a failed mix must not lose the dub
            mix_info = {"error": f"{type(exc).__name__}: {exc}"[:200]}
            background_mode = "none"
            final_wav = dubbed_wav

    note(92, "Measuring speaker similarity")
    similarity = None
    try:
        # Compare against the isolated voice when we have it: measuring against the full mix would
        # charge the dub for music it was never supposed to reproduce.
        reference_wav = separated["vocals"] if separated is not None else source_wav
        similarity = models.speaker_similarity(reference_wav, dubbed_wav)
    except Exception:  # noqa: BLE001 - metric is informative, not load-bearing
        similarity = None

    note(95, "Merging audio back into video")
    if has_video_stream(video_path):
        out_video = merge_audio_into_video(video_path, final_wav, job_dir / "dubbed.mp4")
    else:
        out_video = final_wav  # audio-only upload: the dubbed track *is* the deliverable

    ok = [r for r in results if not r.error]
    raw_total = sum(r.raw_tts_duration for r in ok)
    final_total = sum(r.final_duration for r in ok)
    original_total = sum(r.original_duration for r in ok)

    result = DubResult(
        job_id=job_dir.name,
        direction=direction,
        video_out=str(out_video),
        audio_out=str(final_wav),
        segments=results,
        speaker_similarity=similarity,
        duration_match_ratio=(final_total / original_total) if original_total else None,
        raw_duration_ratio=(raw_total / original_total) if original_total else None,
        clamped_segments=sum(1 for r in results if r.clamped),
        failed_segments=sum(1 for r in results if r.error),
        background_mode=background_mode,
        background_model=background_model,
        mix_info=mix_info,
        detected_language=detected_lang,
        language_confidence=round(lang_confidence, 3),
        elapsed_s=round(time.time() - started, 1),
    )
    (job_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    note(100, "Done")
    return result


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
