"""WSOLA time-scale modification (waveform-similarity overlap-add), numpy only.

stretch(audio, sr, factor) -> np.ndarray
    factor < 1 shortens (0.75 -> output is 75 % as long), factor > 1 lengthens. Pitch and formants
    are untouched because no sample is resynthesised: the output is made of Hann-windowed copies of
    the input's own waveform, each shifted by up to +/-10 ms so that it continues the previous copy
    in phase (cross-correlation alignment). A vocoder (WORLD, phase vocoder) instead rebuilds every
    sample from an analysis, and leaves a buzzy or phasey footprint even when nothing is retimed.

Algorithm (Verhelst & Roelands 1993; notation as in Driedger & Mueller 2016):
    N   frame length (25 ms, even), periodic Hann window, synthesis hop Hs = N/2 (50 % overlap:
        the windows sum to exactly 1, so a constant signal passes through unchanged)
    Ha  analysis hop = Hs / factor
    Output frame k is centred at k*Hs; its nominal input centre is k*Ha. The input frame actually
    copied starts at the nominal start + d_k, with d_k in [-tol, +tol] chosen to maximise the
    cosine similarity (Hann-weighted) between the candidate and the NATURAL CONTINUATION of the
    previous copied frame -- the N samples that followed it in the input.

Edges: frame 0 is centred on input sample 0 with d_0 = 0, so the output starts exactly where the
input does. Every frame's search window is additionally clamped so that no output sample that is
kept reads outside the input (no zero padding is ever copied in, no fade at either end); near the
end this pulls the last frames back so the output ends on the input's last samples. The output is
normalised by the summed windows and cut to exactly round(len(audio) * factor) samples.
"""

from __future__ import annotations

import math

import numpy as np

FRAME_MS = 25.0
TOLERANCE_MS = 10.0


def stretch(
    audio,
    sr: int,
    factor: float,
    frame_ms: float = FRAME_MS,
    tolerance_ms: float = TOLERANCE_MS,
) -> np.ndarray:
    """Time-scale mono `audio` by `factor` without changing pitch (WSOLA).

    Args:
        audio: 1-D float array (level is preserved).
        sr: sample rate in Hz.
        factor: output length / input length. < 1 compresses, > 1 stretches; must be > 0.
            Verified for speech on 0.5-1.0 (and 1.25 as a sanity check).
        frame_ms: frame length; 20-30 ms suits speech.
        tolerance_ms: largest alignment shift either way; 10 ms covers a full pitch period down
            to 100 Hz and half a period down to 50 Hz.

    Returns:
        float64 array of exactly round(len(audio) * factor) samples.
    """
    return _wsola(audio, sr, factor, frame_ms, tolerance_ms)[0]


def _wsola(audio, sr, factor, frame_ms=FRAME_MS, tolerance_ms=TOLERANCE_MS):
    """stretch(), also returning the frame map: output centres and the input centres copied there
    (sample indices), for tests that compare the output with the input at matching times."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("stretch expects mono (1-D) audio")
    factor = float(factor)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError(f"factor must be a positive number, got {factor!r}")
    length = len(x)
    out_len = round(length * factor)
    if length == 0 or out_len == 0:
        return np.zeros(out_len), np.zeros(0, int), np.zeros(0, int)
    if abs(factor - 1.0) < 1e-9:
        return x.copy(), np.arange(length), np.arange(length)

    n = max(8, round(frame_ms * sr / 1000.0))
    n += n % 2
    hs = n // 2
    ha = hs / factor
    tol = max(1, round(tolerance_ms * sr / 1000.0))
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)  # periodic Hann
    window_sq = window * window

    # Frames centred at output 0, hs, 2hs, ... until one is centred at or past the last sample.
    frames = (out_len - 1) // hs + 2

    # Input frame starts are in unpadded coordinates; the padding only keeps slicing in range (it
    # is read by the search and by very short inputs, never by kept output of a normal input).
    pad = n + 2 * tol + math.ceil(ha) + 1
    xp = np.concatenate([np.zeros(pad), x, np.zeros(pad)])

    out = np.zeros((frames + 1) * hs + n)
    wsum = np.zeros_like(out)
    prev = None
    centres_in = np.zeros(frames, dtype=int)
    for k in range(frames):
        c_out = k * hs
        start = round(k * ha) - hs  # nominal start: frame centred on input k*Ha
        if prev is not None:
            # Kept output of this frame spans [max(0, c_out-hs), min(out_len, c_out+hs)); output
            # p reads input start + p - c_out + hs, which must stay inside [0, length).
            lo_ok = min(0, c_out - hs)
            hi_ok = length - hs + c_out - min(out_len, c_out + hs)
            lo, hi = start - tol, start + tol
            if hi_ok >= lo_ok:  # else the input is shorter than a frame: allow padding
                if hi > hi_ok:
                    lo, hi = hi_ok - 2 * tol, hi_ok
                if lo < lo_ok:
                    lo, hi = lo_ok, min(lo_ok + 2 * tol, hi_ok)
                hi = max(lo, min(hi, hi_ok))
            template = xp[pad + prev + hs : pad + prev + hs + n]
            weighted = template * window_sq
            if hi > lo and np.dot(weighted, template) > 1e-18:
                region = xp[pad + lo : pad + hi + n]
                corr = np.correlate(region, weighted, mode="valid")
                energy = np.correlate(region * region, window_sq, mode="valid")
                score = corr / np.sqrt(np.maximum(energy, 1e-18))
                start = lo + int(np.argmax(score))
            else:
                start = min(max(start, lo), hi)
        out[c_out : c_out + n] += xp[pad + start : pad + start + n] * window
        wsum[c_out : c_out + n] += window
        prev = start
        centres_in[k] = start + hs

    # Frame 0 is centred on input sample 0 and placed at out[0:n], so output time 0 is out[hs].
    kept = slice(hs, hs + out_len)
    y, w = out[kept], wsum[kept]
    y = np.where(w > 1e-3, y / np.maximum(w, 1e-3), y)
    return y, np.arange(frames) * hs, centres_in
