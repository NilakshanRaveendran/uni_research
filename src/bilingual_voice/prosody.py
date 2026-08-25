"""Reference-friendly acoustic feature extraction."""

from __future__ import annotations

from pathlib import Path


def extract_prosody(path: Path, transcript: str = "") -> dict[str, float]:
    import librosa
    import numpy as np

    audio, sample_rate = librosa.load(path, sr=16000, mono=True)
    duration = len(audio) / sample_rate if sample_rate else 0.0
    f0, _, _ = librosa.pyin(
        audio,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sample_rate,
    )
    voiced = f0[~np.isnan(f0)]
    rms = librosa.feature.rms(y=audio)[0]
    word_count = len(transcript.split()) if transcript else 0
    return {
        "f0_mean": float(np.mean(voiced)) if voiced.size else float("nan"),
        "f0_std": float(np.std(voiced)) if voiced.size else float("nan"),
        "energy_mean": float(np.mean(rms)) if rms.size else float("nan"),
        "energy_std": float(np.std(rms)) if rms.size else float("nan"),
        "duration": float(duration),
        "speaking_rate": float(word_count / duration) if duration and word_count else float("nan"),
    }
