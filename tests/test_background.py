"""Tests for keeping music and sound effects when dubbing.

The failure this guards against is silent and total: before this module existed, the dubbing pass
built a silent timeline and added only speech to it, so every video came back with its music and
sound effects deleted. Nothing errored -- the output was simply wrong.
"""

import numpy as np
import pytest

from webapp.backend import background as bg

SR = 44100


def _tone(seconds: float, hz: float, amplitude: float = 0.2, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return amplitude * np.sin(2 * np.pi * hz * t)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


def test_default_mode_is_separation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BVT_BACKGROUND", raising=False)
    assert bg.requested_mode() == "separate"


@pytest.mark.parametrize("mode", ["separate", "duck", "none"])
def test_every_documented_mode_is_accepted(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BVT_BACKGROUND", mode.upper())  # case-insensitive
    assert bg.requested_mode() == mode


def test_unknown_mode_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo must not silently disable background preservation -- that is the whole bug."""
    monkeypatch.setenv("BVT_BACKGROUND", "seperate")  # deliberate misspelling
    with pytest.raises(RuntimeError, match="BVT_BACKGROUND"):
        bg.requested_mode()


# --------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------


def test_stereo_round_trip_preserves_both_channels(tmp_path) -> None:
    left, right = _tone(0.4, 220.0), _tone(0.4, 440.0)
    path = bg.write_stereo(tmp_path / "s.wav", np.stack([left, right]), SR)

    audio, sample_rate = bg.read_stereo(path)

    assert sample_rate == SR
    assert audio.shape[0] == 2
    # 16-bit quantisation is the only loss.
    assert np.abs(audio[0] - left).max() < 1e-3
    assert np.abs(audio[1] - right).max() < 1e-3
    assert not np.allclose(audio[0], audio[1]), "channels must stay distinct"


def test_mono_input_is_duplicated_to_stereo(tmp_path) -> None:
    mono = _tone(0.3, 300.0)
    path = bg.write_stereo(tmp_path / "m.wav", mono, SR)
    audio, _sr = bg.read_stereo(path)
    assert audio.shape[0] == 2
    assert np.allclose(audio[0], audio[1])


# --------------------------------------------------------------------------------------
# Ducking (the fallback)
# --------------------------------------------------------------------------------------


def test_duck_lowers_the_background_only_under_speech() -> None:
    audio = np.stack([_tone(3.0, 220.0)] * 2)
    segments = [{"start": 1.0, "end": 2.0}]

    ducked = bg.duck(audio, SR, segments)

    def rms(signal, lo, hi):
        return float(np.sqrt(np.mean(signal[:, int(lo * SR) : int(hi * SR)] ** 2)))

    # Sample away from the segment edges, which are deliberately ramped rather than stepped.
    under_speech = rms(ducked, 1.3, 1.7)
    outside = rms(ducked, 2.4, 2.9)

    assert under_speech < outside, "the bed must dip while the dubbed voice speaks"
    assert under_speech == pytest.approx(rms(audio, 1.3, 1.7) * bg.DUCK_GAIN, rel=0.15)
    assert outside == pytest.approx(rms(audio, 2.4, 2.9) * bg.DUCK_BED_GAIN, rel=0.15)


def test_duck_ramps_instead_of_stepping() -> None:
    """A hard gain step is an audible click. The envelope must be smoothed."""
    audio = np.stack([np.ones(int(3.0 * SR))] * 2)
    ducked = bg.duck(audio, SR, [{"start": 1.0, "end": 2.0}])
    boundary = ducked[0, int(0.9 * SR) : int(1.1 * SR)]
    # A step would show one huge jump; a ramp spreads the change over many samples.
    assert np.abs(np.diff(boundary)).max() < 0.01


def test_duck_with_no_segments_leaves_a_uniform_bed() -> None:
    audio = np.stack([_tone(1.0, 220.0)] * 2)
    ducked = bg.duck(audio, SR, [])
    assert np.allclose(ducked, audio * bg.DUCK_BED_GAIN, atol=1e-6)


# --------------------------------------------------------------------------------------
# Loudness matching and mixing
# --------------------------------------------------------------------------------------


def test_active_rms_ignores_silence() -> None:
    """A plain RMS over a mostly-silent track is dominated by the silence, which would make the
    dubbed voice far too loud."""
    loud = _tone(0.5, 200.0, amplitude=0.5)
    padded = np.concatenate([np.zeros(int(4.5 * SR)), loud])

    plain = float(np.sqrt(np.mean(padded**2)))
    active = bg._active_rms(np.stack([padded, padded]))

    assert active > plain * 2, "silence must not drag the measurement down"
    assert active == pytest.approx(float(np.sqrt(np.mean(loud**2))), rel=0.25)


def test_speech_gain_matches_a_quiet_dub_to_the_original_voice() -> None:
    reference = np.stack([_tone(1.0, 200.0, amplitude=0.40)] * 2)
    quiet_dub = np.stack([_tone(1.0, 200.0, amplitude=0.10)] * 2)

    gain = bg.speech_gain(quiet_dub, reference)

    assert gain == pytest.approx(4.0, rel=0.2)


def test_speech_gain_is_clamped_both_ways() -> None:
    """An extreme match would either bury the dub or blow it up; both are worse than a rough level."""
    reference = np.stack([_tone(1.0, 200.0, amplitude=0.9)] * 2)
    silent_ish = np.stack([_tone(1.0, 200.0, amplitude=0.001)] * 2)
    assert bg.speech_gain(silent_ish, reference) == pytest.approx(bg.SPEECH_GAIN_LIMITS[1])
    assert bg.speech_gain(reference, silent_ish) == pytest.approx(bg.SPEECH_GAIN_LIMITS[0])


def test_speech_gain_survives_a_silent_input() -> None:
    silence = np.zeros((2, SR))
    assert bg.speech_gain(silence, np.stack([_tone(1.0, 200.0)] * 2)) == 1.0
    assert bg.speech_gain(np.stack([_tone(1.0, 200.0)] * 2), silence) == 1.0


def test_mix_keeps_the_background_and_resamples_the_speech() -> None:
    """The whole point: the music must still be there afterwards."""
    music = np.stack([_tone(2.0, 440.0, amplitude=0.25)] * 2)
    speech_24k = _tone(2.0, 150.0, amplitude=0.2, sr=24000)

    mixed, sample_rate, info = bg.mix(speech_24k, 24000, music, SR)

    assert sample_rate == bg.MIX_SR
    assert mixed.shape[0] == 2
    assert mixed.shape[1] == pytest.approx(2.0 * bg.MIX_SR, rel=0.02)
    # The 440 Hz music must survive into the output.
    spectrum = np.abs(np.fft.rfft(mixed[0]))
    freqs = np.fft.rfftfreq(mixed.shape[1], 1.0 / sample_rate)
    assert spectrum[np.argmin(np.abs(freqs - 440.0))] > spectrum.mean() * 20
    assert info["limiter_gain"] <= 1.0


def test_mix_limits_the_peak_instead_of_clipping() -> None:
    loud_music = np.stack([_tone(1.0, 300.0, amplitude=0.95)] * 2)
    loud_speech = _tone(1.0, 150.0, amplitude=0.95, sr=24000)

    mixed, _sr, info = bg.mix(loud_speech, 24000, loud_music, SR)

    assert np.abs(mixed).max() <= 1.0
    assert info["limiter_gain"] < 1.0, "a mix that would clip must be scaled down"


def test_mix_reconciles_mismatched_lengths() -> None:
    """The speech timeline and the video's audio never have exactly the same length."""
    music = np.stack([_tone(3.0, 440.0)] * 2)
    short_speech = _tone(1.0, 150.0, sr=24000)

    mixed, sample_rate, _info = bg.mix(short_speech, 24000, music, SR)

    assert mixed.shape[1] == pytest.approx(3.0 * sample_rate, rel=0.02)
    tail = mixed[:, int(2.0 * sample_rate) :]
    assert float(np.sqrt(np.mean(tail**2))) > 0, "the music must continue past the speech"


def test_mono_background_is_accepted() -> None:
    mixed, _sr, _info = bg.mix(_tone(1.0, 150.0, sr=24000), 24000, _tone(1.0, 440.0), SR)
    assert mixed.shape[0] == 2


# --------------------------------------------------------------------------------------
# Fallback safety
# --------------------------------------------------------------------------------------


def test_separate_returns_none_instead_of_raising(tmp_path) -> None:
    """Separation is an enhancement, never a requirement. It must signal failure by returning
    None so the caller can fall back to ducking -- an exception here would lose the whole dubbing
    job over background music."""
    garbage = tmp_path / "not_audio.wav"
    garbage.write_bytes(b"this is not a wav file at all")
    assert bg.separate(garbage, tmp_path / "stems") is None

    missing = tmp_path / "does_not_exist.wav"
    assert bg.separate(missing, tmp_path / "stems") is None


def test_dub_result_reports_what_happened_to_the_background() -> None:
    """The UI and result.json must be able to say which strategy ran; a silent fallback to
    speech-only is exactly the failure this work fixed."""
    import dataclasses

    from webapp.backend.dubbing import DubResult

    names = {f.name for f in dataclasses.fields(DubResult)}
    assert {"background_mode", "background_model", "mix_info"} <= names


def test_dubbing_uses_the_background_module() -> None:
    """Guards against the mixing step being removed or bypassed in a refactor."""
    from pathlib import Path

    source = Path("webapp/backend/dubbing.py").read_text(encoding="utf-8")
    assert "from . import background" in source
    assert "background.requested_mode()" in source
    assert "background.mix(" in source
    # The final mix, not the speech-only timeline, is what gets muxed into the video.
    assert "merge_audio_into_video(video_path, final_wav" in source
