"""Reference-friendly acoustic feature extraction."""

from __future__ import annotations

from pathlib import Path

# Speech F0 search range. This deliberately does NOT use the musical C2-C7 band (65-2093 Hz):
# adult conversational F0 means sit around 80-250 Hz, and a 2 kHz ceiling lets pyin lock onto
# octave-doubled harmonics. Measured on this corpus with C2-C7, the 99th percentile of mean F0
# reached 1284 Hz and the maximum 2048 Hz -- not speech. Narrowing the band raised the
# generated-vs-human F0 correlation from 0.327 to 0.514 on a matched sample, i.e. the wide band
# was destroying real signal. These bounds match analysis.SPEECH_F0_MIN / SPEECH_F0_MAX.
SPEECH_F0_MIN = 60.0
SPEECH_F0_MAX = 400.0


def extract_prosody(path: Path, transcript: str = "") -> dict[str, float]:
    import librosa
    import numpy as np

    audio, sample_rate = librosa.load(path, sr=16000, mono=True)
    duration = len(audio) / sample_rate if sample_rate else 0.0
    f0, _, _ = librosa.pyin(
        audio,
        fmin=SPEECH_F0_MIN,
        fmax=SPEECH_F0_MAX,
        sr=sample_rate,
    )
    voiced = f0[~np.isnan(f0)]
    voiced = voiced[voiced > 0]
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
