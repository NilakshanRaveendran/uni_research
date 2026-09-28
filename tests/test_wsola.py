"""WSOLA time-scaling (webapp/backend/wsola.py): exact length, pitch kept, no clicks or dropouts."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from webapp.backend.wsola import _wsola, stretch

SR = 24000
FACTORS = [0.5, 0.6, 0.67, 0.75, 0.85, 1.0]
SHORT_TAKE = Path(
    "/Users/nilakshanraveendran/Documents/research/webapp/jobs/97d43f12e04a/segments/seg_0000.wav"
)


def tone(freq, seconds=2.0, sr=SR, amp=0.5):
    t = np.arange(int(seconds * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def peak_hz(x, sr=SR):
    """FFT peak with parabolic interpolation, on a Hann-windowed, zero-padded middle chunk."""
    mid = x[len(x) // 4 : 3 * len(x) // 4]
    spec = np.abs(np.fft.rfft(mid * np.hanning(len(mid)), n=1 << 20))
    k = int(np.argmax(spec))
    a, b, c = np.log(spec[k - 1 : k + 2] + 1e-20)
    k = k + 0.5 * (a - c) / (a - 2 * b + c)
    return k * sr / (1 << 20)


@pytest.mark.parametrize("factor", FACTORS + [1.25])
@pytest.mark.parametrize("length", [1, 7, 100, 599, 1000, 24000, 3 * 24000 + 7])
def test_length_is_exact(length, factor):
    x = np.random.default_rng(length).standard_normal(length) * 0.1
    y = stretch(x, SR, factor)
    assert len(y) == round(length * factor)
    assert np.all(np.isfinite(y))


@pytest.mark.parametrize("sr", [16000, 22050, 24000, 44100])
def test_length_other_rates(sr):
    x = np.random.default_rng(0).standard_normal(sr * 2 + 3)
    for f in FACTORS:
        assert len(stretch(x, sr, f)) == round(len(x) * f)


@pytest.mark.parametrize("freq", [110.0, 220.0, 1000.0])
@pytest.mark.parametrize("factor", FACTORS)
def test_tone_keeps_pitch_fft(freq, factor):
    y = stretch(tone(freq), SR, factor)
    assert abs(peak_hz(y) - freq) < 0.005 * freq  # within 0.1 semitone


@pytest.mark.parametrize("factor", [0.5, 0.67, 0.85])
def test_tone_keeps_pitch_pyin(factor):
    import librosa

    y = stretch(tone(220.0), SR, factor)
    f0, voiced, _ = librosa.pyin(y.astype(np.float32), fmin=80, fmax=500, sr=SR, frame_length=2048)
    med = np.nanmedian(f0[voiced])
    assert abs(12 * np.log2(med / 220.0)) < 0.1


@pytest.mark.parametrize("freq", [110.0, 220.0, 1000.0])
@pytest.mark.parametrize("factor", FACTORS)
def test_tone_no_clicks_no_dropouts(freq, factor):
    x = tone(freq)
    y = stretch(x, SR, factor)
    # No clicks: the largest sample-to-sample step is no bigger than the input's.
    assert np.max(np.abs(np.diff(y))) <= 1.02 * np.max(np.abs(np.diff(x)))
    # No dropouts (phase cancellation at a splice): RMS over ~20 ms (a whole number of periods,
    # so a clean tone reads flat) stays within 3 %, edges included.
    hop = round(max(1, round(0.02 * freq)) * SR / freq)
    rms = np.sqrt(np.convolve(y**2, np.ones(hop) / hop, mode="valid"))
    ref = np.sqrt(np.mean(x**2))
    assert rms.min() > 0.97 * ref and rms.max() < 1.03 * ref


@pytest.mark.parametrize("factor", FACTORS)
def test_starts_at_start(factor):
    x = tone(220.0)
    y = stretch(x, SR, factor)
    assert y[0] == pytest.approx(x[0], abs=1e-12)
    # The first 5 ms is the input's first 5 ms (frame 0 is not shifted, frame 1 aligns to it).
    n = SR // 200
    assert np.max(np.abs(y[:n] - x[:n])) < 0.02


@pytest.mark.parametrize("factor", [0.5, 0.6, 0.67, 0.75, 0.85])
def test_ends_at_end(factor):
    # A 220 Hz tone whose last 60 ms is 660 Hz: the output's last 20 ms must still be 660 Hz,
    # i.e. the end of the input is not dropped or faded.
    x = np.concatenate([tone(220.0, 1.0), tone(660.0, 0.06)])
    y = stretch(x, SR, factor)
    tail = y[-SR // 50 :]
    assert abs(peak_hz(np.tile(tail, 8)) - 660.0) < 15.0
    assert np.sqrt(np.mean(tail**2)) > 0.9 * np.sqrt(np.mean(x[-SR // 50 :] ** 2))


def test_identity_and_constant():
    x = np.random.default_rng(1).standard_normal(5000)
    assert np.array_equal(stretch(x, SR, 1.0), x)
    dc = np.full(24000, 0.3)
    y = stretch(dc, SR, 0.6)
    assert np.allclose(y, 0.3, atol=1e-9)  # the windows sum to exactly 1
    assert np.all(stretch(np.zeros(9000), SR, 0.7) == 0)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        stretch(np.zeros(10), SR, 0.0)
    with pytest.raises(ValueError):
        stretch(np.zeros(10), SR, float("nan"))
    with pytest.raises(ValueError):
        stretch(np.zeros((10, 2)), SR, 0.5)


def _read(path):
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768
    return a, sr


@pytest.mark.skipif(not SHORT_TAKE.exists(), reason="XTTS take not available")
@pytest.mark.parametrize("factor", [0.5, 0.67, 0.85])
def test_real_xtts_take(factor):
    import pyworld

    x, sr = _read(SHORT_TAKE)
    y = stretch(x, sr, factor)
    assert len(y) == round(len(x) * factor)
    # No clicks: steps no bigger than the take's own.
    assert np.max(np.abs(np.diff(y))) <= 1.1 * np.max(np.abs(np.diff(x)))
    # Pitch kept: F0 of each output frame vs F0 of the input frame WSOLA copied there (exact
    # map). Median |difference| < 0.3 semitone over frames voiced in both.
    _, c_out, c_in = _wsola(x, sr, factor)

    def f0_at(a, centres):
        f0, _t = pyworld.harvest(np.ascontiguousarray(a), sr, frame_period=5.0)
        return f0[np.clip(np.round(centres / sr / 0.005).astype(int), 0, len(f0) - 1)]

    fy, fx = f0_at(y, c_out), f0_at(x, c_in)
    both = (fy > 0) & (fx > 0)
    assert both.sum() > 50
    assert np.median(np.abs(12 * np.log2(fy[both] / fx[both]))) < 0.3
    # Level kept: RMS within 10 %.
    assert np.sqrt(np.mean(y**2)) == pytest.approx(np.sqrt(np.mean(x**2)), rel=0.1)
