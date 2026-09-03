"""Keep the music and sound effects when dubbing.

The dubbing pass replaces the video's audio track entirely: it builds a silent timeline and adds
only synthesised speech to it. Anything that was not speech -- background music, gunshots, room
tone, laughter -- is therefore lost. This module puts it back.

Two strategies, tried in order:

1. SEPARATE (preferred). Split the original audio into a voice track and a non-voice track using
   Hybrid Demucs, keep the non-voice track untouched, and mix the dubbed speech on top of it. The
   original voice is gone and everything else survives. Uses
   `torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS`, so it needs NO new dependency -- torchaudio is
   already installed -- and it is pure PyTorch, no TensorFlow anywhere.

2. DUCK (fallback). No separation: keep the ORIGINAL audio underneath at reduced volume, dipping
   further while the dubbed voice speaks. Music and effects return immediately, but the original
   voice stays faintly audible. Used automatically when separation is unavailable or fails, because
   a faint original voice is still much better than total silence.

Honest limitation of strategy 1: Hybrid Demucs was trained to separate SINGING from instruments,
not dialogue from film effects. A gunshot is a sharp broadband transient and some of it can leak
into the "vocals" stem and be discarded with it. The result is a large improvement, not a perfect
dialogue/effects split. Models trained for cinematic audio source separation would do better and
are the obvious upgrade path.
"""

from __future__ import annotations

import os
import subprocess
import wave
from pathlib import Path

MIX_SR = 44100  # music deserves better than the 24 kHz speech timeline
SEPARATION_SEGMENT_S = 10.0  # chunk length for Demucs inference
SEPARATION_OVERLAP = 0.1  # fraction of a chunk crossfaded into the next
DUCK_GAIN = 0.25  # background level while the dubbed voice is speaking
DUCK_BED_GAIN = 0.8  # background level elsewhere
DUCK_FADE_S = 0.15  # ramp in/out of a duck, so it does not click
SPEECH_GAIN_LIMITS = (0.25, 4.0)  # clamp on the loudness match, so nothing explodes
MODES = ("separate", "duck", "none")


def requested_mode() -> str:
    """`BVT_BACKGROUND` = separate (default) | duck | none."""
    mode = os.environ.get("BVT_BACKGROUND", "separate").strip().lower()
    if mode not in MODES:
        raise RuntimeError(f"BVT_BACKGROUND must be one of {MODES}, got {mode!r}")
    return mode


# --------------------------------------------------------------------------------------
# Extraction and I/O
# --------------------------------------------------------------------------------------


def extract_full_audio(video: Path, out_wav: Path, sample_rate: int = MIX_SR) -> Path:
    """Full-quality stereo extraction, for separation and for the final mix.

    Separate from `dubbing.extract_audio`, which produces the 16 kHz mono file Whisper and ECAPA
    want. Mixing music at 16 kHz mono would throw away most of what this module exists to save.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(out_wav),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")
    return out_wav


def read_stereo(path: Path):
    """Read a wav as a (2, n) float array in [-1, 1]. Mono input is duplicated to both channels."""
    import numpy as np
    import soundfile

    audio, sample_rate = soundfile.read(str(path), dtype="float64", always_2d=True)
    audio = audio.T  # soundfile gives (n, channels)
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    return np.ascontiguousarray(audio), int(sample_rate)


def write_stereo(path: Path, audio, sample_rate: int) -> Path:
    """Write a (2, n) or (n,) float array as 16-bit PCM."""
    import numpy as np

    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    channels = audio.shape[0]
    interleaved = np.clip(audio.T.reshape(-1), -1.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes((interleaved * 32767.0).astype("<i2").tobytes())
    return path


def _resample(audio, source_sr: int, target_sr: int):
    """Resample a (channels, n) array. No-op when the rates already match."""
    import librosa
    import numpy as np

    if source_sr == target_sr:
        return np.asarray(audio, dtype=np.float64)
    audio = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    return np.stack(
        [librosa.resample(channel, orig_sr=source_sr, target_sr=target_sr) for channel in audio]
    )


# --------------------------------------------------------------------------------------
# Strategy 1: separation
# --------------------------------------------------------------------------------------


def separation_device() -> str:
    """CPU by default: Demucs runs at roughly 0.3x realtime there, which is fast enough, and MPS
    has historically been the source of silent numerical surprises in this project."""
    return os.environ.get("BVT_SEPARATION_DEVICE", "cpu").strip().lower()


def separate(wav_path: Path, out_dir: Path, progress=None) -> dict | None:
    """Split into a voice track and a background track.

    Returns {"background": Path, "vocals": Path, "sample_rate": int, "model": str} or None when
    separation is not possible -- the caller then falls back to ducking.
    """
    try:
        import numpy as np
        import torch
        import torchaudio
    except ImportError:
        return None

    try:
        bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
    except AttributeError:
        # Older/newer torchaudio without the bundled Demucs weights.
        return None

    try:
        if progress:
            progress("Loading separation model")
        model = bundle.get_model()
        device = torch.device(separation_device())
        model = model.to(device).eval()
        model_sr = int(bundle.sample_rate)

        audio, sample_rate = read_stereo(wav_path)
        # Demucs expects its own training rate; resample in, and back out afterwards.
        mix = torch.tensor(_resample(audio, sample_rate, model_sr), dtype=torch.float32)

        # Demucs was trained on loudness-normalised input.
        reference = mix.mean(dim=0)
        scale = float(reference.std()) or 1.0
        centre = float(reference.mean())
        normalised = (mix - centre) / scale

        if progress:
            progress("Separating voice from music and effects")
        stems = _separate_chunked(model, normalised.unsqueeze(0).to(device), model_sr)
        stems = stems.squeeze(0) * scale + centre  # (sources, channels, n)

        names = list(model.sources)
        vocal_index = names.index("vocals")
        vocals = stems[vocal_index].cpu().numpy()
        # Background is the sum of every stem that is not the voice: drums + bass + other.
        background = np.sum(
            np.stack([stems[i].cpu().numpy() for i in range(len(names)) if i != vocal_index]),
            axis=0,
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        background_path = write_stereo(
            out_dir / "background.wav", _resample(background, model_sr, sample_rate), sample_rate
        )
        vocals_path = write_stereo(
            out_dir / "vocals.wav", _resample(vocals, model_sr, sample_rate), sample_rate
        )
        return {
            "background": background_path,
            "vocals": vocals_path,
            "sample_rate": sample_rate,
            "model": "torchaudio HDEMUCS_HIGH_MUSDB_PLUS",
        }
    except Exception:  # noqa: BLE001 - separation is an enhancement; never fail the whole job
        return None


def _separate_chunked(model, mix, sample_rate: int):
    """Run Demucs over a long recording in overlapping chunks with a linear crossfade.

    A whole video cannot go through the model in one pass without exhausting memory. Chunks are
    faded into one another so no seam is audible; this is the approach from torchaudio's own
    source-separation tutorial.
    """
    import torch
    from torchaudio.transforms import Fade

    batch, channels, length = mix.shape
    chunk = int(sample_rate * SEPARATION_SEGMENT_S * (1 + SEPARATION_OVERLAP))
    overlap_frames = int(SEPARATION_OVERLAP * sample_rate)
    fade = Fade(fade_in_len=0, fade_out_len=overlap_frames, fade_shape="linear")

    output = torch.zeros(batch, len(model.sources), channels, length, device=mix.device)
    start, end = 0, chunk
    while start < length - overlap_frames:
        with torch.inference_mode():
            separated = model.forward(mix[:, :, start:end])
        separated = fade(separated)
        output[:, :, :, start:end] += separated
        if start == 0:
            fade.fade_in_len = overlap_frames
            start += chunk - overlap_frames
        else:
            start += chunk
        end += chunk
        if end >= length:
            fade.fade_out_len = 0
    return output


# --------------------------------------------------------------------------------------
# Strategy 2: ducking
# --------------------------------------------------------------------------------------


def duck(audio, sample_rate: int, segments, gain: float = DUCK_GAIN, bed: float = DUCK_BED_GAIN):
    """Attenuate the original audio underneath the dubbed speech.

    Used when separation is unavailable. The original voice remains faintly audible -- that is the
    known cost of this route, and it is why separation is preferred.
    """
    import numpy as np

    audio = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    n = audio.shape[1]
    envelope = np.full(n, float(bed))
    for seg in segments or []:
        start = max(0, int(float(seg["start"]) * sample_rate))
        end = min(n, int(float(seg["end"]) * sample_rate))
        if end > start:
            envelope[start:end] = float(gain)

    # Smooth the envelope so the level changes are ramps, not steps that click. The envelope is
    # padded with its EDGE values first: convolving with mode="same" alone zero-pads the ends, which
    # would fade the music out at the very start and end of every video -- measured, and caught by
    # a test asserting that a no-speech bed comes back uniform.
    width = max(1, int(DUCK_FADE_S * sample_rate))
    if width > 1 and n > width:
        half = width // 2
        kernel = np.ones(width) / width
        padded = np.pad(envelope, half, mode="edge")
        envelope = np.convolve(padded, kernel, mode="same")[half : half + n]
    return audio * envelope


# --------------------------------------------------------------------------------------
# Mixing
# --------------------------------------------------------------------------------------


def _active_rms(audio, floor_ratio: float = 0.05) -> float:
    """RMS over the loud part of a signal only.

    A plain RMS over a track that is mostly silence is dominated by the silence, so matching on it
    would make the dubbed voice far too loud. Only samples above a fraction of the peak count.
    """
    import numpy as np

    flat = np.abs(np.asarray(audio, dtype=np.float64)).mean(axis=0)
    if flat.size == 0:
        return 0.0
    peak = float(flat.max())
    if peak <= 0:
        return 0.0
    active = flat[flat > peak * floor_ratio]
    return float(np.sqrt(np.mean(active**2))) if active.size else 0.0


def speech_gain(speech, reference) -> float:
    """How much to scale the dubbed speech so it sits where the original voice sat."""
    import numpy as np

    target = _active_rms(reference)
    current = _active_rms(speech)
    if target <= 0 or current <= 0:
        return 1.0
    return float(np.clip(target / current, *SPEECH_GAIN_LIMITS))


def mix(
    speech,
    speech_sr: int,
    background,
    background_sr: int,
    reference=None,
    target_sr: int = MIX_SR,
) -> tuple:
    """Combine the dubbed speech with the preserved background.

    Returns (stereo_mix, sample_rate, info). The speech is loudness-matched to `reference` (the
    separated voice track) when one is given, so the dub sits at the level the original voice did
    rather than floating over or under the music.
    """
    import numpy as np

    speech = np.atleast_2d(np.asarray(speech, dtype=np.float64))
    if speech.shape[0] == 1:
        speech = np.repeat(speech, 2, axis=0)
    speech = _resample(speech, speech_sr, target_sr)
    background = _resample(np.atleast_2d(background), background_sr, target_sr)
    if background.shape[0] == 1:
        background = np.repeat(background, 2, axis=0)

    gain = 1.0
    if reference is not None:
        reference = _resample(np.atleast_2d(reference), background_sr, target_sr)
        gain = speech_gain(speech, reference)
    speech = speech * gain

    length = max(speech.shape[1], background.shape[1])
    padded_speech = np.zeros((2, length))
    padded_background = np.zeros((2, length))
    padded_speech[:, : speech.shape[1]] = speech[:, :length]
    padded_background[:, : background.shape[1]] = background[:, :length]

    mixed = padded_speech + padded_background
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    limiter = 1.0
    if peak > 0.99:
        limiter = 0.99 / peak
        mixed = mixed * limiter
    return (
        mixed,
        target_sr,
        {"speech_gain": round(gain, 4), "limiter_gain": round(limiter, 4), "peak": round(peak, 4)},
    )
