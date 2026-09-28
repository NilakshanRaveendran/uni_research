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
import itertools
import json
import math
import re
import shutil
import statistics
import subprocess
import time
import traceback
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import background, shorten, speakers, wsola

TARGET_SR = 16000  # Whisper / ECAPA operate at 16 kHz
MIN_SEGMENT_S = 0.30
# What is left after XTTS's own speed is compressed with WSOLA (wsola.py), which copies the voice's
# real waveform. WORLD, used before, rebuilt every sample through a vocoder and damaged the voice
# even when it compressed nothing (at x0.999: ECAPA -0.033, 3 dB spectral distortion, 13.5 % of
# frames an octave off) -- the "synthetic" sound. Measured on 48 real XTTS takes, WSOLA stays close
# to the untouched take down to 0.75 and is near-transparent at 0.85 (similarity 0.96); below 0.67
# words start to go, so 0.67 is the floor. Nothing is ever stretched: slowing speech down costs
# naturalness for nothing.
MIN_SCALE, MAX_SCALE = 0.67, 1.0
WSOLA_TRANSPARENT = 0.85

# XTTS pads every sentence with 0.417 s of digital silence (coqui's PAD_SILENCE_SAMPLES) and pauses
# up to ~3.9 s between the sentences it synthesises separately. In a dub every such pause is time
# taken from the next line. Measured on 309 past lines: trimming at 35 dB with pauses capped at
# 0.25 s removes ~15 % of the audio and no words (Whisper check); 30 dB starts cutting soft speech.
TRIM_TOP_DB = 35.0
MAX_PAUSE_S = 0.25
LEAD_KEEP_S, TAIL_KEEP_S = 0.05, 0.10  # air kept before and after speech
# XTTS's own speed control is free up to 1.25x (17 lines: Whisper WER 0.034 vs 0.040 at 1.0,
# speaker similarity -0.007, not significant). At 1.5x it starts dropping words ("me encanta tu
# acento" -> "me encantan los acento") and above 1.75x it is worse than any time-scaling.
MAX_XTTS_SPEED = 1.25
# Compression that costs nothing audible: XTTS's free range times WSOLA's transparent range. A line
# that would need more than this gets a shorter translation first (shorten.py).
FREE_COMPRESSION = MAX_XTTS_SPEED / WSOLA_TRANSPARENT
# XTTS speech per character at speed 1 (voiced seconds), measured per output language; a learned
# per-speaker rate did not predict better than these.
CHARS_PER_VOICED_S = {"es": 12.0, "en": 13.2}
SENTENCE_BREAK_S = 0.25
# XTTS synthesises each sentence separately, and a one- or two-word sentence on its own often comes
# out hallucinated ("¿Acento? No tengo acento." -> "Acento. Acento. Acento. No tengo acento.";
# Whisper WER 0.25 over 3 seeds, 0.08 synthesised in one piece). Lines holding such a sentence are
# synthesised whole; others keep XTTS's split, which did better on longer sentences (0.07 vs 0.22).
SHORT_SENTENCE_WORDS = 2
# A take slower than this share of the language's rate is a runaway generation (e.g. "Algo es
# difícil." rendered as 16 s): sample again with another seed and keep the fastest take.
RUNAWAY_RATE_SHARE = 0.6
RUNAWAY_RETRIES = 2
# XTTS samples its speech, so a take can say something else ("¿Acento?" came out "¿Entonces?").
# Each take is heard back with Whisper, which is loaded anyway, and one whose character error rate
# against the text exceeds this is sampled again (sharing the retries above); the clearest wins.
# Accents and punctuation are ignored, so a regional pronunciation Whisper spells "asento" for
# "acento" costs one character, not a word.
MAX_TAKE_CER = 0.2
# The pause kept between lines. At a change of speaker 0.2 s: the lower quartile of natural silent
# gaps between line-length turns in DRAL conversations (0.18 s) and what the actors in the app's
# videos leave (median 0.24 s). Within one speaker 0.1 s. Without them a new voice started the
# instant the last one stopped -- one character still talking while the other's lips move.
TURN_GAP_S = 0.2
SAME_SPEAKER_GAP_S = 0.1
# How far a line may run past the end of the speech it replaces (Whisper's line end is itself a
# little late, median +0.05 s). 1 s left a Short's last line sounding 0.6 s after the actor stopped.
SPILL_S = 0.3
MIN_WINDOW_S = 0.3
FADE_S = 0.05

LANGUAGES = {"en": "English", "es": "Spanish"}
DIRECTIONS = {"en-es": ("en", "es"), "es-en": ("es", "en")}
# Whisper is told the source language rather than asked, so a Tamil video submitted as Spanish
# would otherwise be "transcribed" as Spanish gibberish and dubbed without complaint. Below this
# confidence the detection is treated as inconclusive (silence, music intros) and the job proceeds.
LANGUAGE_CONFIDENCE = 0.5
# Before separation the language is heard through any music, and Whisper labels music as a
# language too (a music intro came back as Norwegian Nynorsk at 76 %). So an unsupported language
# only fails the job this early when it is near certain; otherwise it is checked again on the
# separated vocals, where the music is gone.
EARLY_UNSUPPORTED_CONFIDENCE = 0.9


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
    # XTTS speed used, the synthesised length after silence was cut, and any seconds faded out
    # because the line would otherwise have run into the next one.
    tts_speed: float = 1.0
    trimmed_duration: float = 0.0
    truncated_s: float = 0.0
    # Set when the translation was reworded to fit: the first, longer translation.
    unshortened_text: str = ""
    # How many XTTS takes were sampled, and the kept take's character error rate heard back.
    takes: int = 1
    take_cer: float = 0.0
    # Which voice this segment was cloned from, numbered from 1 in order of first appearance.
    speaker: int = 1
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
    truncated_segments: int = 0
    background_mode: str = "none"
    background_model: str = ""
    detected_language: str = ""
    language_confidence: float | None = None
    mix_info: dict = field(default_factory=dict)
    # One entry per detected speaker: segments assigned and voiced, reference seconds, similarity.
    speakers: list[dict] = field(default_factory=list)
    # Set when speaker detection failed and every segment fell back to one shared voice.
    speaker_detection_error: str = ""
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


def check_language(
    detected: str, confidence: float, direction: str, final: bool = True
) -> bool:
    """Raise UnsupportedLanguageError when the speech confidently isn't the source language.

    With `final=False` (the early check, heard through any music) an unsupported language below
    EARLY_UNSUPPORTED_CONFIDENCE is not decided yet: returns False so the caller checks again on
    the separated vocals. Returns True when the language is settled.
    """
    source_lang, target_lang = DIRECTIONS[direction]
    if detected == source_lang or confidence < LANGUAGE_CONFIDENCE:
        return True
    heard = f"{language_name(detected)} ({confidence:.0%} confidence)"
    if detected == target_lang:
        raise UnsupportedLanguageError(
            f"This video is in {heard}, but {LANGUAGES[source_lang]} → "
            f"{LANGUAGES[target_lang]} was selected. Choose {LANGUAGES[target_lang]} → "
            f"{LANGUAGES[source_lang]} and upload again."
        )
    if not final and confidence < EARLY_UNSUPPORTED_CONFIDENCE:
        return False
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


def _speech_spans(audio, top_db: float = TRIM_TOP_DB):
    import librosa

    return librosa.effects.split(audio, top_db=top_db, frame_length=1024, hop_length=256)


def _plain(text: str) -> str:
    """Lowercase letters and digits only, accents removed: what intelligibility is judged on."""
    import unicodedata

    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def take_cer(models, audio, sr: int, text: str, language: str) -> float:
    """Character error rate of a synthesised take heard back by Whisper, against its text."""
    import jiwer
    import librosa
    import numpy as np

    audio16 = librosa.resample(np.asarray(audio, dtype=float), orig_sr=sr, target_sr=TARGET_SR)
    heard = models.asr.transcribe(
        audio16.astype(np.float32), language=language, task="transcribe", temperature=0.0,
        verbose=None,
    )["text"]
    reference = _plain(text)
    return float(jiwer.cer(reference, _plain(heard) or "-")) if reference else 0.0


def voiced_seconds(audio, sr: int) -> float:
    import numpy as np

    audio = np.asarray(audio, dtype=float)
    if audio.size == 0 or not np.any(audio):
        return 0.0
    return float(sum(int(end) - int(start) for start, end in _speech_spans(audio)) / sr)


def tighten(audio, sr: int, max_pause: float = MAX_PAUSE_S):
    """Cut leading and trailing silence from a synthesised line and shorten long inner pauses."""
    import numpy as np

    audio = np.asarray(audio, dtype=float)
    if audio.size == 0 or not np.any(audio):
        return audio
    spans = _speech_spans(audio)
    if len(spans) == 0:
        return audio
    merged = [[int(start), int(end)] for start, end in spans]
    keep = int(max_pause * sr)
    pieces = [audio[max(0, merged[0][0] - int(LEAD_KEEP_S * sr)) : merged[0][1]]]
    for (_, prev_end), (start, end) in itertools.pairwise(merged):
        pieces.append(audio[prev_end:start] if start - prev_end <= keep else np.zeros(keep))
        pieces.append(audio[start:end])
    pieces.append(audio[merged[-1][1] : merged[-1][1] + int(TAIL_KEEP_S * sr)])
    return np.concatenate(pieces)


def place(timeline, audio, offset: int, limit: int, sr: int) -> float:
    """Add a line to the timeline at `offset`, never past `limit` (sample index).

    Whatever would run past the limit -- the next line's start, or the end of the media -- is faded
    out and dropped instead of summed with the next voice. Returns the seconds dropped.
    """
    import numpy as np

    limit = min(limit, len(timeline))
    usable = max(0, min(len(audio), limit - offset))
    piece = np.array(audio[:usable], dtype=float)
    dropped = (len(audio) - usable) / sr
    if dropped > 0 and usable > 0:
        fade = min(int(FADE_S * sr), usable)
        piece[usable - fade :] *= np.linspace(1.0, 0.0, fade)
    timeline[offset : offset + usable] += piece
    return dropped


def line_windows(segments, labels, total_duration: float) -> list[tuple[float, float]]:
    """For each line, the seconds it may occupy and the time its audio must be gone by.

    A line may run SPILL_S past the speech it replaces, but always leaves the next line its pause:
    TURN_GAP_S before another speaker, SAME_SPEAKER_GAP_S before the same one.
    """
    windows = []
    for i, seg in enumerate(segments):
        start, end = float(seg["start"]), float(seg["end"]) + SPILL_S
        if i + 1 < len(segments):
            gap = SAME_SPEAKER_GAP_S if labels[i] == labels[i + 1] else TURN_GAP_S
            next_start = float(segments[i + 1]["start"])
            end = min(end, next_start - gap)
            hard_stop = next_start  # never over the next line, whatever the minimum window
        else:
            end = min(end, total_duration)
            hard_stop = total_duration
        window = max(end - start, MIN_WINDOW_S)
        windows.append((window, min(start + window, hard_stop)))
    return windows


def xtts_speed(text: str, language: str, window: float) -> float:
    """XTTS speed for a line: enough to fit `window` seconds, within the range that costs nothing."""
    chars = max(1, len(text.replace(" ", "")))
    predicted = chars / CHARS_PER_VOICED_S.get(language, 12.0)
    predicted += SENTENCE_BREAK_S * (len(split_sentences(text)) - 1)
    return round(min(max(predicted / max(window, 0.1), 1.0), MAX_XTTS_SPEED), 3)


_SENTENCE_BREAK = re.compile(r"(?<=[?!…])\s+|(?<=\.)\s+(?=[A-ZÁÉÍÓÚÑÜ¿¡])")


def split_sentences(text: str) -> list[str]:
    """Split a transcript line into sentences for translation.

    MarianMT given a whole multi-sentence line drops sentences and question marks (the pretrained
    model dropped a sentence in 5 of 8 such English lines); one sentence at a time it drops none.
    A full stop only ends a sentence before a capital or an opening ¿/¡, and a trailing one-word
    fragment -- a Whisper cut-off like "...I agree. I" -- stays with the sentence before it.
    """
    parts = [part.strip() for part in _SENTENCE_BREAK.split(text.strip()) if part.strip()]
    if len(parts) > 1 and len(parts[-1].split()) == 1 and parts[-1][-1] not in ".?!…":
        fragment = parts.pop()
        parts[-1] = f"{parts[-1]} {fragment}"
    return parts or [text.strip()]


def synthesise_whole(text: str) -> bool:
    """True when XTTS should get the line in one piece (it holds a very short sentence)."""
    return any(len(sentence.split()) <= SHORT_SENTENCE_WORDS for sentence in split_sentences(text))


def shorten_translations(
    models, texts: list[str], translations: list[str], windows, source: str, target: str
) -> list[str]:
    """Re-translate, with a shorter wording, the lines predicted not to fit their window.

    Professional dubbing shortens the words rather than speeding the voice. For flagged lines each
    sentence is re-chosen from MarianMT's beam-8 candidates: within 0.3 of the best score, keeping
    every question, negation, number and sentence ending, and passing a round-trip check (its
    back-translation at most 15 chrF worse). On the lines of 35 past jobs that did not fit this
    removed ~10 % of the characters with no meaning errors in the 35 changed sentences judged.
    Lines that fit are returned untouched.
    """
    flagged = [
        i for i, (text, (window, _)) in enumerate(zip(translations, windows, strict=True))
        if text and shorten.needs_shortening(text, window, target, FREE_COMPRESSION)
    ]
    if not flagged:
        return list(translations)
    pieces = [split_sentences(texts[i]) for i in flagged]
    sentences = [sentence for group in pieces for sentence in group]
    currents = models.translate_many(sentences, source, target)
    budgets = []
    for i, group in zip(flagged, pieces, strict=True):
        line_budget = shorten.budget_chars(windows[i][0], target, FREE_COMPRESSION)
        k = len(budgets)
        chars = [shorten.spoken_chars(c) for c in currents[k : k + len(group)]]
        budgets += [int(line_budget * c / max(1, sum(chars))) for c in chars]
    tokenizer, model = models.mt_model(source, target)
    chosen = shorten.translate_fitted(
        model, tokenizer, sentences, currents, target, [True] * len(sentences), budgets,
        back_translate=lambda xs: models.translate_many(xs, target, source),
    )
    out, k = list(translations), 0
    for i, group in zip(flagged, pieces, strict=True):
        out[i] = " ".join(t.strip() for t in chosen[k : k + len(group)] if t.strip())
        k += len(group)
    return out


def translate_lines(models, texts: list[str], source: str, target: str) -> list[str]:
    """Translate each line sentence by sentence, in one batch, and rejoin."""
    pieces = [split_sentences(text) for text in texts]
    flat = [sentence for sentences in pieces for sentence in sentences]
    out = models.translate_many(flat, source, target)
    joined, k = [], 0
    for sentences in pieces:
        joined.append(" ".join(t.strip() for t in out[k : k + len(sentences)] if t.strip()))
        k += len(sentences)
    return joined


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
    language_settled = check_language(detected_lang, lang_confidence, direction, final=False)
    if language_settled:
        note(5, f"Detected {language_name(detected_lang)} speech")

    # Full-quality stereo copy, kept for separation and for the final mix. The 16 kHz mono file
    # above is what Whisper and ECAPA want; mixing music at 16 kHz mono would discard most of
    # what the background-preservation step exists to save.
    background_mode = background.requested_mode()
    # Checked with the other settings, before minutes of separation and transcription.
    forced_speakers = speakers.requested_speakers()
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

    if not language_settled:
        # Heard as another language through the music: listen again to the voices alone.
        detected_lang, lang_confidence = detect_language(models, asr_wav)
        check_language(detected_lang, lang_confidence, direction)
        note(9, f"Detected {language_name(detected_lang)} speech")

    note(10, f"Transcribing {LANGUAGES[source_lang]} speech")
    segments = transcribe_segments(models, asr_wav, source_lang)
    if not segments:
        raise RuntimeError("No speech detected in the uploaded file")
    note(20, f"Found {len(segments)} speech segments")

    # One batched call rather than one per segment: translation is cheap next to synthesis, and
    # doing it up front means the per-segment timing used for the ETA measures synthesis alone.
    try:
        translations = translate_lines(
            models, [seg["text"] for seg in segments], source_lang, target_lang
        )
    except Exception:  # noqa: BLE001 - fall back to per-segment translation
        translations = [None] * len(segments)

    # Zero-shot cloning is per speaker, so each actor is voiced from a reference cut from their own
    # segments. One reference for the whole soundtrack gave every actor the same voice: XTTS reads
    # only its first 30 s, so everyone sounded like whoever spoke first.
    note(21, "Identifying speakers")
    voice_path = separated["vocals"] if separated is not None else (full_wav or source_wav)
    speaker_error = ""
    try:
        labels = speakers.assign_speakers(models, asr_wav, segments, forced_speakers)
        references = speakers.build_references(voice_path, segments, labels, job_dir / "speakers")
        count = len(set(labels))
        note(22, f"Found {count} speaker{'s' if count != 1 else ''}")
    except Exception as exc:  # noqa: BLE001 - one shared voice is worse, not a reason to fail
        traceback.print_exc()
        speaker_error = f"{type(exc).__name__}: {exc}"[:200]
        labels, references = [0] * len(segments), {}
        note(22, "Speaker detection failed; using one voice for everyone")

    # A speaker without a reference of their own (detection failed, or too little clean audio)
    # is voiced from one reference built from all the speech -- still the cleaned vocal stem, never
    # the raw soundtrack with its music and its first-30-s bias.
    fallback_reference = source_wav
    if any(label not in references for label in set(labels)):
        try:
            shared = speakers.build_references(
                voice_path, segments, [0] * len(segments), job_dir / "speakers" / "shared"
            )
            fallback_reference = shared.get(0, {}).get("xtts_path", source_wav)
        except Exception:  # noqa: BLE001 - the soundtrack itself is the last resort
            traceback.print_exc()

    # Each line's window, now that we know who speaks next -- then shorter wording for the lines
    # that cannot fit it even with the compression that costs nothing.
    windows = line_windows(segments, labels, total_duration)
    unshortened = list(translations)
    if all(t is not None for t in translations):
        note(23, "Fitting the translation to the timing")
        try:
            translations = shorten_translations(
                models, [seg["text"] for seg in segments], translations, windows,
                source_lang, target_lang,
            )
        except Exception:  # noqa: BLE001 - shortening improves a dub, it never blocks one
            traceback.print_exc()

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
        span = 24 + int(61 * i / max(1, len(segments)))
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
            speaker=labels[i] + 1,
            source_words=seg.get("words", []),
        )
        if translations[i] is not None and translations[i] != unshortened[i]:
            record.unshortened_text = unshortened[i]
        try:
            translated = translations[i]
            if translated is None:
                translated = " ".join(
                    models.translate(sentence, source_lang, target_lang).strip()
                    for sentence in split_sentences(seg["text"])
                )
            record.translated_text = translated
            if not translated.strip():
                raise ValueError("translation was empty")

            out_wav = segment_dir / f"seg_{i:04d}.wav"
            # The reference is the speaker's whole reference file, not this segment alone: XTTS
            # conditions far better on several seconds of a voice than on one short clip.
            reference = references.get(labels[i], {}).get("xtts_path", fallback_reference)
            window, stop = windows[i]
            speed = xtts_speed(translated, target_lang, window)
            chars = max(1, len(translated.replace(" ", "")))
            slow = RUNAWAY_RATE_SHARE * CHARS_PER_VOICED_S.get(target_lang, 12.0)
            best = None
            for attempt in range(1 + RUNAWAY_RETRIES):
                take = segment_dir / f"seg_{i:04d}_take{attempt}.wav"
                models.synthesize(
                    translated, reference, target_lang, take, seed=498 + i + 7919 * attempt,
                    speed=speed, split_sentences=not synthesise_whole(translated),
                )
                generated, gen_sr = _read_wav(take)
                spoken = tighten(generated, gen_sr)
                voiced = voiced_seconds(generated, gen_sr) * speed  # at speed 1, for the rate
                runaway = voiced > 0 and chars / voiced < slow
                try:
                    cer = take_cer(models, spoken, gen_sr, translated, target_lang)
                except Exception:  # noqa: BLE001 - no verdict: judge on length alone
                    cer = 0.0
                # Best take: not a runaway, then clearest, then shortest.
                rank = (runaway, round(cer, 2), len(spoken))
                if best is None or rank < best[0]:
                    best = (rank, generated, spoken, take, cer)
                if not runaway and cer <= MAX_TAKE_CER:
                    break  # a normal, intelligible take
            _, generated, spoken, take, record.take_cer = best
            record.takes = attempt + 1
            take.replace(out_wav)
            for extra in segment_dir.glob(f"seg_{i:04d}_take*.wav"):
                extra.unlink()
            record.tts_speed = speed
            record.raw_tts_duration = len(generated) / gen_sr
            record.trimmed_duration = len(spoken) / gen_sr

            # --- fit the dub into its window: WSOLA compresses what XTTS's speed did not ---
            wanted = min(1.0, window / record.trimmed_duration) if record.trimmed_duration else 1.0
            scale = min(max(wanted, MIN_SCALE), MAX_SCALE)
            record.clamped = wanted < MIN_SCALE - 1e-6
            record.scale_applied = scale
            retimed = wsola.stretch(spoken, gen_sr, scale) if scale < 1.0 else spoken
            record.final_duration = len(retimed) / gen_sr

            if gen_sr != timeline_sr:
                import librosa

                retimed = librosa.resample(
                    np.asarray(retimed, dtype=float), orig_sr=gen_sr, target_sr=timeline_sr
                )
            offset = int(seg["start"] * timeline_sr)
            limit = round(stop * timeline_sr)
            record.truncated_s = round(place(timeline, retimed, offset, limit, timeline_sr), 3)
            if record.truncated_s:
                retimed = retimed[: max(0, limit - offset)]
                record.final_duration = len(retimed) / timeline_sr
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
    # Per speaker: that speaker's dubbed segments against their own reference. Comparing the whole
    # dub with the whole vocal track would score a blend against a blend, and could not show
    # whether each actor kept their own voice.
    speaker_rows = []
    for label in sorted(set(labels)):
        ok_members = [r for r in results if r.speaker == label + 1 and not r.error]
        row = {
            "speaker": label + 1,
            "segments": sum(1 for r in results if r.speaker == label + 1),
            "voiced": len(ok_members),
            "reference_s": references.get(label, {}).get("seconds"),
            "similarity": None,
        }
        if label in references and ok_members:
            try:
                clips = [
                    _read_wav(segment_dir / f"seg_{r.index:04d}_retimed.wav")[0]
                    for r in ok_members
                ]
                own_dub = job_dir / "speakers" / f"speaker_{label + 1}_dub.wav"
                _write_wav(own_dub, np.concatenate(clips), timeline_sr)
                row["similarity"] = models.speaker_similarity(references[label]["path"], own_dub)
            except Exception:  # noqa: BLE001 - metric is informative, not load-bearing
                row["similarity"] = None
        speaker_rows.append(row)

    similarity = None
    # Weighted by segments actually voiced: a speaker's score is measured on those alone.
    scored = [(r["similarity"], r["voiced"]) for r in speaker_rows if r["similarity"] is not None]
    if scored:
        similarity = sum(s * n for s, n in scored) / sum(n for _, n in scored)
    else:
        try:
            # Compare against the isolated voice when we have it: measuring against the full mix
            # would charge the dub for music it was never supposed to reproduce.
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
        truncated_segments=sum(1 for r in results if r.truncated_s > 0),
        failed_segments=sum(1 for r in results if r.error),
        background_mode=background_mode,
        background_model=background_model,
        mix_info=mix_info,
        speakers=speaker_rows,
        speaker_detection_error=speaker_error,
        detected_language=detected_lang,
        language_confidence=round(lang_confidence, 3),
        elapsed_s=round(time.time() - started, 1),
    )
    (job_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    note(100, "Done")
    return result


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
