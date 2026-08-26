"""Reference-friendly acoustic feature extraction."""

from __future__ import annotations

from pathlib import Path

# F0 SEARCH band vs PLAUSIBILITY band -- these are deliberately different, and conflating them
# was a measured mistake.
#
# The SEARCH band must stay wide. pyin's voiced/unvoiced decision depends on how many frequency
# candidates it has; narrowing the band lowers the aggregate voiced probability and it stops
# detecting voicing at all on quiet recordings. Measured on this corpus, generated-vs-human F0
# correlation (semitone space, median statistic, implausible values gated) by search band:
#     60-400  Hz : en-es 0.371, es-en 0.589      <- too narrow, loses voicing, WORST
#     65-600  Hz : en-es 0.778, es-en 0.739
#     65-1000 Hz : en-es 0.888, es-en 0.856      <- best
#     65-2093 Hz : en-es 0.822, es-en 0.819      <- more octave errors
#
# The PLAUSIBILITY band (60-400 Hz) is applied downstream, in metrics/analysis, as a data-quality
# gate: a median F0 outside it is a tracking failure, not a voice. Gating there rather than
# restricting the search here keeps pyin's voicing detection intact while still excluding the
# octave-doubled values that dominated the old numbers.
F0_SEARCH_MIN = 65.0
F0_SEARCH_MAX = 1000.0

# Downstream gate, exported so metrics.py and analysis.py agree on one definition.
F0_PLAUSIBLE_MIN = 60.0
F0_PLAUSIBLE_MAX = 400.0


def extract_prosody(path: Path, transcript: str = "") -> dict[str, float]:
    import librosa
    import numpy as np

    audio, sample_rate = librosa.load(path, sr=16000, mono=True)
    duration = len(audio) / sample_rate if sample_rate else 0.0
    f0, _, _ = librosa.pyin(
        audio,
        fmin=F0_SEARCH_MIN,
        fmax=F0_SEARCH_MAX,
        sr=sample_rate,
    )
    voiced = f0[~np.isnan(f0)]
    voiced = voiced[voiced > 0]
    rms = librosa.feature.rms(y=audio)[0]
    word_count = len(transcript.split()) if transcript else 0
    return {
        "f0_mean": float(np.mean(voiced)) if voiced.size else float("nan"),
        # The median is the statistic to prefer: a wide search band admits occasional octave
        # doublings, and one doubled frame moves the mean far more than the median.
        "f0_median": float(np.median(voiced)) if voiced.size else float("nan"),
        "f0_std": float(np.std(voiced)) if voiced.size else float("nan"),
        "energy_mean": float(np.mean(rms)) if rms.size else float("nan"),
        "energy_std": float(np.std(rms)) if rms.size else float("nan"),
        "duration": float(duration),
        "speaking_rate": float(word_count / duration) if duration and word_count else float("nan"),
    }
