"""Contract tests for the from-scratch prosody models and the contour injector.

These protect properties that fail SILENTLY -- each one, if broken, produces plausible numbers
rather than an error, which is exactly how this project has been bitten before.
"""

from pathlib import Path

import numpy as np
import pytest

from bilingual_voice.contours import (
    N_COEF,
    N_POINTS,
    dct_decode,
    dct_encode,
    resample_contour,
    semitones,
)
from bilingual_voice.prosody_apply import (
    LEVEL_CAP_ST,
    _median_filter,
    _to_hz,
    _to_semitones,
    inject,
)
from bilingual_voice.prosody_eval import _holm
from bilingual_voice.prosody_net import (
    AUX_TARGETS,
    DCT_OFFSET,
    INPUT_SCALARS,
    LEVEL_MED,
    N_IN_MLP,
    N_OUT,
    SCALAR_TARGETS,
    build_model,
    count_parameters,
    standardise,
)

# --------------------------------------------------------------------------------------
# The structural guarantee: a fresh model IS baseline B2
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ["mlp", "cnn"])
def test_fresh_model_outputs_exactly_zero(arch: str) -> None:
    """The zero-initialised head is the whole safety argument.

    Prediction = B2 + output * y_std. If the head is not exactly zero at initialisation, the model
    starts somewhere other than the published baseline, and the claim "it cannot start worse than
    B2" is false. A default nn.Linear init would make this pass-by-luck impossible.
    """
    import torch

    torch.manual_seed(0)
    model = build_model(arch)
    scalars = torch.randn(7, len(INPUT_SCALARS))
    dct = torch.randn(7, N_COEF)
    shape = torch.randn(7, N_POINTS)

    with torch.inference_mode():
        out = model(scalars, dct, shape)

    assert out.shape == (7, N_OUT)
    assert torch.count_nonzero(out) == 0, "a fresh model must predict the B2 residual, i.e. zero"


@pytest.mark.parametrize(("arch", "expected"), [("mlp", 6284), ("cnn", 4796)])
def test_parameter_counts_are_what_we_report(arch: str, expected: int) -> None:
    """The presentation quotes these numbers. An architecture edit must not silently change them."""
    assert count_parameters(build_model(arch)) == expected


def test_standardise_makes_zero_output_reproduce_b2_exactly() -> None:
    """End-to-end version of the guarantee above, through the real de-standardisation path."""
    rng = np.random.default_rng(498)
    n = 120
    n_targets = N_OUT
    split = np.array(["train"] * 80 + ["dev"] * 20 + ["test"] * 20)
    copy = rng.normal(size=(n, n_targets))
    data = {
        "scalars": rng.normal(size=(n, len(INPUT_SCALARS))),
        "source_dct": rng.normal(size=(n, N_COEF)),
        "source_shape": rng.normal(size=(n, N_POINTS)),
        "raw_targets": copy + rng.normal(scale=0.5, size=(n, n_targets)) + 1.25,
        "copy": copy,
        "split": split,
    }
    std = standardise(data)

    # B2 must be copy plus ONE global offset per target, fitted on train only.
    train = split == "train"
    offset = (data["raw_targets"][train] - data["copy"][train]).mean(axis=0)
    assert np.allclose(std["b2"], data["copy"] + offset)

    # Residuals are exactly centred on train, so a zero model output lands on B2 with no bias term.
    assert np.abs(std["y"][train].mean(axis=0)).max() < 1e-10

    zero_output = np.zeros((n, n_targets))
    assert np.allclose(std["b2"] + zero_output * std["y_std"], std["b2"])


def test_input_features_contain_nothing_derived_from_the_target() -> None:
    """The leak guard. `mt_words` comes from machine translation; a target-side name here would
    mean the model is being told the answer's length."""
    forbidden = ("t_", "target_", "human", "reference", "t_duration", "t_f0")
    for name in INPUT_SCALARS:
        assert name.startswith(("s_", "mt_", "log_word")), name
        assert not name.startswith(forbidden), name
    assert "mt_words" in INPUT_SCALARS
    assert N_IN_MLP == len(INPUT_SCALARS) + N_COEF


# --------------------------------------------------------------------------------------
# Contour representation
# --------------------------------------------------------------------------------------


def test_dct_round_trip_preserves_a_smooth_contour() -> None:
    """8 coefficients must describe a realistic intonation contour. The whole Model-1 input
    representation rests on this, and the week-1 gate is median r > 0.9."""
    time = np.linspace(0, 1, N_POINTS)
    contour = 3.0 * np.sin(np.pi * time) - 1.5 * time  # a rise-fall with a declination trend
    contour = contour - contour.mean()

    back = dct_decode(dct_encode(contour, N_COEF), N_POINTS)

    assert np.corrcoef(contour, back)[0, 1] > 0.99
    assert np.sqrt(np.mean((contour - back) ** 2)) < 0.1


def test_dct_decode_has_zero_mean_because_k0_is_dropped() -> None:
    """Level and shape must not both carry the offset, or the level shift is applied twice."""
    rng = np.random.default_rng(1)
    assert abs(dct_decode(rng.normal(size=N_COEF), N_POINTS).mean()) < 1e-12


def test_resample_contour_removes_the_median_and_rejects_unvoiced_tracks() -> None:
    f0 = np.full(50, np.nan)
    f0[10:40] = 120.0
    voiced = np.isfinite(f0)

    shape, level, ratio = resample_contour(f0, voiced, N_POINTS)

    assert level == pytest.approx(float(semitones(120.0)))
    assert np.abs(shape).max() < 1e-9  # a flat track has no shape
    assert ratio == pytest.approx(30 / 50)

    too_few = np.full(50, np.nan)
    too_few[0:2] = 120.0
    assert resample_contour(too_few, np.isfinite(too_few), N_POINTS) is None


def test_resample_contour_rejects_implausible_pitch() -> None:
    """A track pinned outside the speech band is a tracking failure, not a voice."""
    f0 = np.full(50, 900.0)  # inside the SEARCH band, outside the PLAUSIBILITY band
    assert resample_contour(f0, np.ones(50, dtype=bool), N_POINTS) is None


# --------------------------------------------------------------------------------------
# The injector
# --------------------------------------------------------------------------------------


def _vowel(f0_hz: float, seconds: float, sr: int = 16000) -> np.ndarray:
    n = int(seconds * sr)
    time = np.arange(n) / sr
    phase = 2 * np.pi * f0_hz * time
    signal = sum(np.sin(k * phase) / k for k in range(1, 25))
    envelope = np.minimum(1.0, np.minimum(time * 20, (seconds - time) * 20))
    return 0.3 * signal / np.abs(signal).max() * envelope


def test_semitone_helpers_are_mutual_inverses() -> None:
    """Hz-vs-semitone confusion in this project once turned a 0.820 correlation into 0.048."""
    assert _to_semitones(100.0) == pytest.approx(0.0)
    assert _to_semitones(200.0) == pytest.approx(12.0)
    for hz in (65.0, 120.0, 250.0, 400.0):
        assert _to_hz(_to_semitones(hz)) == pytest.approx(hz)


def test_median_filter_removes_a_single_frame_spike_and_preserves_length() -> None:
    values = np.zeros(21)
    values[10] = 12.0  # a one-frame octave jump
    filtered = _median_filter(values, 5)
    assert len(filtered) == len(values)
    assert filtered.max() == pytest.approx(0.0)


def test_injector_does_not_add_voicing_where_there_was_none() -> None:
    """A predicted contour must never put pitch into a silence. The voiced/unvoiced mask is
    WORLD's decision and is never edited -- only pitch on already-voiced frames is overwritten."""
    import pyworld

    sr = 16000
    silence = np.zeros(int(0.35 * sr))
    audio = np.concatenate([_vowel(150.0, 0.45, sr), silence, _vowel(150.0, 0.45, sr)])

    def voiced_frames(signal):
        contiguous = np.ascontiguousarray(signal, dtype=np.float64)
        f0, t = pyworld.dio(contiguous, sr)
        return int((pyworld.stonemask(contiguous, f0, t, sr) > 0).sum())

    before = voiced_frames(audio)
    after = voiced_frames(inject(audio, sr, target_level_st=_to_semitones(190.0)))

    assert after <= before + 2, f"injection created voicing: {before} -> {after}"


def test_injector_level_cap_binds() -> None:
    """Without the cap, a confident wrong prediction can move pitch far enough to damage speaker
    identity -- which is one of the things being measured."""
    import pyworld

    sr = 16000
    audio = _vowel(150.0, 1.0, sr)

    def median_hz(signal):
        contiguous = np.ascontiguousarray(signal, dtype=np.float64)
        f0, t = pyworld.dio(contiguous, sr)
        f0 = pyworld.stonemask(contiguous, f0, t, sr)
        return float(np.median(f0[f0 > 0]))

    base = _to_semitones(median_hz(audio))
    absurd = median_hz(inject(audio, sr, target_level_st=base + 24.0))

    assert _to_semitones(absurd) - base == pytest.approx(LEVEL_CAP_ST, abs=0.6)


def test_injector_returns_input_when_audio_is_too_short_to_edit() -> None:
    tiny = np.zeros(100)
    assert len(inject(tiny, 16000, target_level_st=0.0)) == len(tiny)


# --------------------------------------------------------------------------------------
# Multiple comparisons
# --------------------------------------------------------------------------------------


def test_holm_is_monotonic_and_matches_hand_computation() -> None:
    """Six model fits across two directions is a family; uncorrected p-values overstate the win."""
    adjusted = _holm([0.01, 0.04, 0.03, None])

    assert adjusted[3] is None
    # sorted: 0.01 (x3), 0.03 (x2), 0.04 (x1) -> 0.03, 0.06, 0.06 after monotonicity
    assert adjusted[0] == pytest.approx(0.03)
    assert adjusted[2] == pytest.approx(0.06)
    assert adjusted[1] == pytest.approx(0.06)
    finite = [p for p in adjusted if p is not None]
    assert all(p <= 1.0 for p in finite)
    # Monotonicity in rank order: a smaller raw p must never receive a larger adjusted p.
    raw = [0.01, 0.04, 0.03]
    by_rank = sorted(range(len(raw)), key=lambda i: raw[i])
    ranked = [adjusted[i] for i in by_rank]
    assert ranked == sorted(ranked)


def test_holm_handles_all_none() -> None:
    assert _holm([None, None]) == [None, None]


# --------------------------------------------------------------------------------------
# The ladder's column contract
# --------------------------------------------------------------------------------------


def test_ladder_column_indices_match_the_target_order() -> None:
    """The ladder reads the prediction vector BY POSITION. Reordering SCALAR_TARGETS without
    updating these constants would silently feed pitch span into the level slot -- every arm would
    still render, and every number would be wrong."""
    from bilingual_voice import prosody_ladder

    assert SCALAR_TARGETS[prosody_ladder.LEVEL] == "f0_mean_st"
    assert SCALAR_TARGETS[prosody_ladder.SPAN] == "f0_std_st"
    assert SCALAR_TARGETS[prosody_ladder.LOGDUR] == "log_dur_ratio"
    assert prosody_ladder.N_SCALARS_OUT == len(SCALAR_TARGETS)
    # The contour block starts after the reported targets AND the injection level.
    assert DCT_OFFSET == len(SCALAR_TARGETS) + len(AUX_TARGETS)
    assert DCT_OFFSET + N_COEF == N_OUT


def test_injection_level_is_the_robust_median_not_the_reported_mean() -> None:
    """The reported target `f0_mean_st` is a mean over every frame inside the 65-1000 Hz SEARCH
    band, so octave doublings can move it tens of semitones -- one XTTS output measured
    f0_std_st = 20.4 st. Driving injection from that computes a nonsensical shift and then quietly
    saturates the level cap. Injection must use the frame-gated median instead."""
    from bilingual_voice import prosody_ladder

    assert AUX_TARGETS[LEVEL_MED - len(SCALAR_TARGETS)] == "level_med_st"
    assert LEVEL_MED != prosody_ladder.LEVEL, "injection must not read the reported mean"
    source = Path("src/bilingual_voice/prosody_ladder.py").read_text(encoding="utf-8")
    assert "vector[position, LEVEL_MED]" in source
    assert "vector[position, LEVEL]" not in source


def test_ladder_baseline_is_the_injected_arm_not_raw_xtts() -> None:
    """S1_copy shares the vocoder path with every model arm, so its artefacts cancel. Defaulting
    the comparison to S0 would credit the models with the vocoder's damage."""
    import inspect

    from bilingual_voice import prosody_ladder

    default = inspect.signature(prosody_ladder.report).parameters["baseline"].default
    assert default == "S1_copy"
    assert "S0_identity" in prosody_ladder.ARMS, "the vocoder-footprint arm must exist"
    assert "S3_full" in prosody_ladder.ARMS, "the oracle ceiling arm must exist"


def test_injector_sets_the_requested_pitch_span() -> None:
    """A contour shape fixes the FORM of pitch movement but not its amplitude. DCT truncation, the
    median filter, and a regressor's shrinkage toward the mean all flatten it -- measured on real
    audio, injected contours landed 0.3-1.0 st flatter than the human target, worst for the model
    with the BEST shape correlation. Pitch span is separately predicted and is the prediction with
    the strongest statistical support, so it must actually reach the audio."""
    import pyworld


    sr = 16000
    audio = _vowel(150.0, 1.4, sr)
    # std ~= 1.0 semitone, so a request of 2.0 is a ~2x rescale and sits inside the clamp.
    shape = 1.4 * np.sin(2 * np.pi * np.linspace(0, 2, N_POINTS))
    baseline_std = float(np.std(shape))

    def measured_span(signal):
        contiguous = np.ascontiguousarray(signal, dtype=np.float64)
        f0, t = pyworld.dio(contiguous, sr)
        f0 = pyworld.stonemask(contiguous, f0, t, sr)
        voiced = f0[f0 > 0]
        return float(np.std(_to_semitones(voiced)))

    unset = measured_span(inject(audio, sr, target_shape=shape))
    asked = measured_span(inject(audio, sr, target_shape=shape, target_span_st=2.0))

    assert unset == pytest.approx(baseline_std, abs=0.35), (
        f"without a span request the shape's own amplitude should survive ({unset:.2f})"
    )
    assert asked == pytest.approx(2.0, abs=0.45), f"requested span must be delivered ({asked:.2f})"
    assert asked > unset, "a wider request must widen the contour"


def test_injector_refuses_an_absurd_span_amplification() -> None:
    """A nearly flat contour amplified 7x would be a warble, not speech. The clamp must bind."""
    import pyworld

    from bilingual_voice.prosody_apply import SPAN_SCALE_MAX

    sr = 16000
    audio = _vowel(150.0, 1.4, sr)
    shape = 0.4 * np.sin(2 * np.pi * np.linspace(0, 2, N_POINTS))  # std ~= 0.28

    contiguous = np.ascontiguousarray(
        inject(audio, sr, target_shape=shape, target_span_st=20.0), dtype=np.float64
    )
    f0, t = pyworld.dio(contiguous, sr)
    f0 = pyworld.stonemask(contiguous, f0, t, sr)
    got = float(np.std(_to_semitones(f0[f0 > 0])))

    ceiling = float(np.std(shape)) * SPAN_SCALE_MAX
    assert got < ceiling + 0.4, f"span scale must be clamped ({got:.2f} vs ceiling {ceiling:.2f})"
    assert got < 3.0, "an absurd request must not produce an absurd contour"
