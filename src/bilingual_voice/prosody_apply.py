"""Impose a target pitch contour and duration on existing audio, in one WORLD pass.

    python -m bilingual_voice.prosody_apply selftest

This is the output side of the from-scratch prosody models: a model predicts a pitch level, a
duration ratio and a contour shape, and this module writes them onto audio that XTTS already
produced. It never re-synthesises speech from text, which is why the whole evaluation ladder can
run offline over the 868 wavs already on disk.

Every rule below was learned the hard way in this project. Changing any of them silently corrupts
the measurement rather than raising an error.

* ONE analysis, ONE synthesise. Round-tripping through WORLD twice compounds its artefacts, and an
  identity round-trip alone already moves measured F0 by 1.3-3.8 semitones on half of files.
* F0 frames resample by NEAREST NEIGHBOUR. WORLD marks unvoiced frames as f0 = 0, so linear
  interpolation across a voiced/unvoiced boundary invents pitch. Measured cost of getting this
  wrong: a 3.5 semitone shift on a segment that was only meant to be stretched in time.
* The spectral envelope comes from the ORIGINAL f0 and is never recomputed from the predicted one.
  CheapTrick conditions on f0, so re-running it with a modified track changes the timbre -- and
  speaker similarity is one of the things being measured.
* Pitch is only ever written onto frames WORLD already declared voiced. The voiced/unvoiced mask is
  never edited, so a predicted contour cannot put pitch into a silence.
* Everything happens in semitones, then is clipped to a plausible speech range in Hz.
"""

from __future__ import annotations

import argparse

from .prosody import F0_PLAUSIBLE_MAX, F0_PLAUSIBLE_MIN

LEVEL_CAP_ST = 3.0  # maximum pitch-level move; beyond this, identity degrades
MEDIAN_FILTER = 5  # frames, smooths the imposed track without flattening real movement
# Bounds on the span rescale. A contour that is nearly flat must not be amplified into a warble,
# and a wild one must not be crushed; both extremes sound worse than getting the span wrong.
SPAN_SCALE_MIN, SPAN_SCALE_MAX = 0.3, 3.0
MIN_SCALE, MAX_SCALE = 0.5, 2.0  # duration factor clamp, same bounds as the dubbing app
MIN_VOICED_FRAMES = 4


def _to_semitones(hz):
    import numpy as np

    return 12.0 * np.log2(np.asarray(hz, dtype=float) / 100.0)


def _to_hz(semitone):
    import numpy as np

    return 100.0 * np.power(2.0, np.asarray(semitone, dtype=float) / 12.0)


def _median_filter(values, width: int):
    """Odd-width running median, edges held. Applied to a DENSE track, never across zeros."""
    import numpy as np

    values = np.asarray(values, dtype=float)
    if width < 3 or values.size < width:
        return values
    half = width // 2
    padded = np.pad(values, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    return np.median(windows, axis=1)


def inject(
    audio,
    sr: int,
    target_level_st: float | None = None,
    target_shape=None,
    duration_factor: float | None = None,
    level_cap: float = LEVEL_CAP_ST,
    current_level_st: float | None = None,
    target_span_st: float | None = None,
):
    """Rewrite the pitch and/or timing of `audio` in a single WORLD analysis/synthesis pass.

    target_level_st   absolute pitch level to move to, in semitones re 100 Hz (None: keep)
    target_shape      normalised-time, mean-removed semitone contour to impose (None: keep shape)
    duration_factor   >1 lengthens, <1 shortens (None or ~1: keep)
    current_level_st  the audio's CURRENT level, if the caller already measured it
    target_span_st    standard deviation the imposed contour should end up with (None: leave it)

    `target_span_st` exists because a contour shape fixes the FORM of the pitch movement but not
    its amplitude. Truncating to 8 DCT coefficients, median-filtering, and a regression model's
    natural shrinkage toward the mean all flatten the result: measured on real audio, injected
    contours came out 0.3-1.0 semitones flatter than the human target, worst for the model whose
    shape correlation was best. Since pitch span is separately predicted -- and is the prediction
    with the strongest statistical support -- scaling the shape to that predicted span is what puts
    the model's best output into the audio instead of discarding it.

    `current_level_st` exists so the pitch shift is computed with one estimator on both sides. The
    models predict `f0_mean_st` -- a mean over voiced frames measured by pyin -- whereas this
    function's internal reference is WORLD's median. Mixing the two puts a systematic bias into
    every shift. When the caller has measured the level the same way the target was defined, it
    should pass it here; otherwise the internal median is used.

    Returns the resynthesised waveform. Returns the input unchanged when the audio is too short or
    too unvoiced to edit meaningfully -- silently doing nothing is correct here, because an arm of
    the evaluation ladder must still produce a file.
    """
    import numpy as np
    import pyworld

    contiguous = np.ascontiguousarray(audio, dtype=np.float64)
    if len(contiguous) < sr // 20:
        return np.asarray(audio, dtype=np.float64)

    f0, t = pyworld.dio(contiguous, sr)
    f0 = pyworld.stonemask(contiguous, f0, t, sr)
    # CheapTrick and D4C use the ORIGINAL f0. This is the one call whose f0 argument must not be
    # the predicted track.
    spectrum = pyworld.cheaptrick(contiguous, f0, t, sr)
    aperiodicity = pyworld.d4c(contiguous, f0, t, sr)

    frames = len(f0)
    if frames < MIN_VOICED_FRAMES:
        return np.asarray(audio, dtype=np.float64)

    # ---- timing: resample the frame sequence first, so the F0 edit happens on the final timeline
    factor = 1.0 if duration_factor is None else float(np.clip(duration_factor, MIN_SCALE, MAX_SCALE))
    if abs(factor - 1.0) >= 1e-3:
        index = np.linspace(0, frames - 1, max(MIN_VOICED_FRAMES, round(frames * factor)))
        source = np.arange(frames)
        f0 = f0[np.clip(np.round(index).astype(int), 0, frames - 1)]  # NEAREST, never linear
        spectrum = np.stack(
            [np.interp(index, source, spectrum[:, k]) for k in range(spectrum.shape[1])], axis=1
        )
        aperiodicity = np.stack(
            [np.interp(index, source, aperiodicity[:, k]) for k in range(aperiodicity.shape[1])],
            axis=1,
        )
        frames = len(f0)

    voiced = f0 > 0
    if voiced.sum() < MIN_VOICED_FRAMES:
        return pyworld.synthesize(
            np.ascontiguousarray(f0),
            np.ascontiguousarray(spectrum),
            np.ascontiguousarray(aperiodicity),
            sr,
        )

    # ---- pitch
    if target_level_st is not None or target_shape is not None:
        current_st = _to_semitones(np.where(voiced, f0, 1.0))
        current_level = float(np.median(current_st[voiced]))

        if target_shape is not None:
            shape = np.asarray(target_shape, dtype=float)
            # The shape's x-axis is normalised position over the whole utterance, exactly as
            # contours.resample_contour defined it, so it maps onto the frame grid directly.
            grid = np.linspace(0.0, 1.0, len(shape))
            position = (
                np.linspace(0.0, 1.0, frames) if frames > 1 else np.zeros(1)
            )
            dense_shape = np.interp(position, grid, shape)
        else:
            # Unvoiced frames are -79.7 st placeholders; left in, the median filter below would pull
            # the pitch of neighbouring voiced frames down to the floor.
            dense_shape = np.where(voiced, current_st - current_level, 0.0)

        reference = current_level if current_level_st is None else float(current_level_st)
        # Smooth BEFORE rescaling, so the span is set on the track that actually gets written --
        # filtering afterwards would flatten it again.
        dense_shape = _median_filter(dense_shape, MEDIAN_FILTER)
        if target_span_st is not None:
            observed = float(np.std(dense_shape[voiced]))
            if observed > 1e-6 and float(target_span_st) > 0:
                scale = float(
                    np.clip(float(target_span_st) / observed, SPAN_SCALE_MIN, SPAN_SCALE_MAX)
                )
                dense_shape = dense_shape * scale

        # Centre the shape on the frames that will actually be voiced: a DCT shape is mean-zero over
        # its 64 points and a measured one median-zero over the TARGET's frames, so either would
        # otherwise move the level by up to a few semitones.
        dense_shape = dense_shape - float(np.median(dense_shape[voiced]))
        level = reference if target_level_st is None else float(target_level_st)
        shift = float(np.clip(level - reference, -level_cap, level_cap))
        # The shift was measured against `reference`, so it is applied to `reference` -- adding it
        # to WORLD's own median instead (the earlier code) mixed the two estimators and wrote the
        # level 0.65-1.09 st off on median, more than 1 st off for 37-55 % of utterances.
        new_st = reference + shift + dense_shape

        new_hz = np.clip(_to_hz(new_st), F0_PLAUSIBLE_MIN, F0_PLAUSIBLE_MAX)
        # Only voiced frames are touched. The mask itself is never edited.
        f0 = np.where(voiced, new_hz, 0.0)

    return pyworld.synthesize(
        np.ascontiguousarray(f0),
        np.ascontiguousarray(spectrum),
        np.ascontiguousarray(aperiodicity),
        sr,
    )


# --------------------------------------------------------------------------------------
# Self-test on synthetic signals with known answers
# --------------------------------------------------------------------------------------


def _synth_vowel(f0_hz: float, seconds: float, sr: int = 16000, vibrato_st: float = 0.0):
    """A crude voiced signal: harmonic stack with a decaying spectrum. Enough for WORLD to track."""
    import numpy as np

    n = int(seconds * sr)
    time = np.arange(n) / sr
    track = f0_hz * np.power(2.0, vibrato_st * np.sin(2 * np.pi * 1.5 * time) / 12.0)
    phase = 2 * np.pi * np.cumsum(track) / sr
    signal = sum(np.sin(k * phase) / k for k in range(1, 25))
    envelope = np.minimum(1.0, np.minimum(time * 20, (seconds - time) * 20))
    return 0.3 * signal / np.abs(signal).max() * envelope


def _measure(audio, sr: int = 16000) -> tuple[float, float]:
    """(median F0 in Hz, duration in seconds) measured the same way the project measures."""
    import numpy as np
    import pyworld

    contiguous = np.ascontiguousarray(audio, dtype=np.float64)
    f0, t = pyworld.dio(contiguous, sr)
    f0 = pyworld.stonemask(contiguous, f0, t, sr)
    voiced = f0[f0 > 0]
    return (float(np.median(voiced)) if voiced.size else float("nan"), len(contiguous) / sr)


def selftest() -> dict:
    """Known-answer checks. A synthetic 150 Hz vowel must land where we aim it."""
    import numpy as np

    sr = 16000
    source = _synth_vowel(150.0, 1.5, sr, vibrato_st=2.0)
    base_hz, base_dur = _measure(source, sr)
    base_st = float(_to_semitones(base_hz))
    print(f"source: {base_hz:.1f} Hz ({base_st:+.2f} st), {base_dur:.3f} s")

    results = {"source_hz": base_hz, "source_st": base_st, "cases": []}
    ok = True

    # 1. Identity round trip -- the S0' arm. Establishes the vocoder's own footprint.
    got_hz, got_dur = _measure(inject(source, sr), sr)
    drift = float(_to_semitones(got_hz)) - base_st
    print(f"identity round-trip      : {got_hz:7.1f} Hz  drift {drift:+.3f} st  {got_dur:.3f} s")
    results["identity_drift_st"] = drift
    ok &= abs(drift) < 0.5

    # 2. Level shifts. Requested moves must arrive, and the cap must bind.
    for request in (-3.0, -1.5, 1.5, 3.0, 6.0):
        target = base_st + request
        got_hz, _ = _measure(inject(source, sr, target_level_st=target), sr)
        got = float(_to_semitones(got_hz)) - base_st
        expected = float(np.clip(request, -LEVEL_CAP_ST, LEVEL_CAP_ST))
        error = got - expected
        flag = "capped" if abs(request) > LEVEL_CAP_ST else ""
        print(
            f"level {request:+5.1f} st -> {got:+6.2f} st (want {expected:+5.2f}) "
            f"err {error:+.3f} {flag}"
        )
        results["cases"].append({"request_st": request, "got_st": got, "error_st": error})
        ok &= abs(error) < 0.6

    # 3. Duration. The stretch must be applied and must NOT move the pitch.
    for factor in (0.7, 1.0, 1.5):
        got_hz, got_dur = _measure(inject(source, sr, duration_factor=factor), sr)
        ratio = got_dur / base_dur
        pitch_drift = float(_to_semitones(got_hz)) - base_st
        print(
            f"duration x{factor:.2f}          : {ratio:.3f} actual  "
            f"pitch drift {pitch_drift:+.3f} st"
        )
        results["cases"].append({"factor": factor, "ratio": ratio, "pitch_drift_st": pitch_drift})
        ok &= abs(ratio - factor) < 0.06 and abs(pitch_drift) < 0.6

    # 4. Shape. Impose a rise and a fall; the imposed direction must appear in the output.
    for name, shape in (
        ("rising", np.linspace(-3.0, 3.0, 64)),
        ("falling", np.linspace(3.0, -3.0, 64)),
    ):
        out = inject(source, sr, target_level_st=base_st, target_shape=shape)
        import pyworld

        contiguous = np.ascontiguousarray(out, dtype=np.float64)
        f0, t = pyworld.dio(contiguous, sr)
        f0 = pyworld.stonemask(contiguous, f0, t, sr)
        voiced_idx = np.flatnonzero(f0 > 0)
        if voiced_idx.size >= 8:
            st = _to_semitones(f0[voiced_idx])
            half = voiced_idx.size // 2
            delta = float(np.median(st[half:]) - np.median(st[:half]))
        else:
            delta = float("nan")
        print(f"shape {name:8s}          : second half - first half = {delta:+.2f} st")
        results["cases"].append({"shape": name, "half_delta_st": delta})
        ok &= (delta > 1.0) if name == "rising" else (delta < -1.0)

    results["passed"] = bool(ok)
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bilingual_voice.prosody_apply")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    args = parser.parse_args(argv)
    if args.command == "selftest":
        raise SystemExit(0 if selftest()["passed"] else 1)


if __name__ == "__main__":
    main()
