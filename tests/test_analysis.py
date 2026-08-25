import numpy as np

from bilingual_voice.analysis import (
    DIRECTIONS,
    _semitones,
    directional_frame,
    ridge_fit,
    ridge_predict,
)


def test_ridge_recovers_known_weights() -> None:
    """Protects the headline number: a wrong solve silently changes every result."""
    rng = np.random.default_rng(498)
    X = rng.normal(size=(400, 4))
    w_true = np.array([2.0, -1.5, 0.25, 3.0])
    y = X @ w_true + 7.0

    model = ridge_fit(X, y, 1e-8)
    predicted = ridge_predict(model, X)

    assert np.allclose(predicted, y, atol=1e-6)
    # Recover unstandardised weights: w_std / sigma
    recovered = model["w"] / model["sigma"]
    assert np.allclose(recovered, w_true, atol=1e-6)


def test_semitone_and_delta_transforms() -> None:
    """A sign flip or bad reference here would silently invert the whole narrative."""
    assert _semitones(100.0) == 0.0
    assert np.isclose(_semitones(200.0), 12.0)
    assert np.isclose(_semitones(50.0), -12.0)

    import pandas as pd

    # One pair: EN 1s / 100 Hz, ES 2s / 200 Hz.
    row = {
        "pair_id": "p1", "split": "train", "same_speaker_pair": "true",
        "en_speaker": "1", "es_speaker": "1",
        "en_duration": 1.0, "en_f0_mean_st": 0.0, "en_f0_std_st": 1.0,
        "en_voiced_ratio": 0.9, "en_n_words": 3.0, "en_speaking_rate": 3.0,
        "es_duration": 2.0, "es_f0_mean_st": 12.0, "es_f0_std_st": 2.0,
        "es_voiced_ratio": 0.9, "es_n_words": 4.0, "es_speaking_rate": 2.0,
    }
    df = pd.DataFrame([row])

    en_es = directional_frame(df, "en-es")
    # Spanish is longer, so log(target/source) must be POSITIVE for en-es.
    assert np.isclose(en_es["log_dur_ratio"].iloc[0], np.log(2.0))
    assert np.isclose(en_es["target_f0_mean_st"].iloc[0], 12.0)
    assert np.isclose(en_es["s_f0_mean_st"].iloc[0], 0.0)
    assert en_es["copy_log_dur_ratio"].iloc[0] == 0.0

    es_en = directional_frame(df, "es-en")
    # Reversing the direction must flip the sign.
    assert np.isclose(es_en["log_dur_ratio"].iloc[0], -np.log(2.0))
    assert np.isclose(es_en["target_f0_mean_st"].iloc[0], 0.0)
    assert np.isclose(es_en["s_f0_mean_st"].iloc[0], 12.0)

    assert set(DIRECTIONS) == {"en-es", "es-en"}


def test_finetune_pairs_respect_direction() -> None:
    """A swapped direction here would train the model backwards and invalidate the comparison."""
    from bilingual_voice.finetune import build_pairs

    rows = [
        {"split": "train", "english_text": "hello there", "spanish_text": "hola alli"},
        {"split": "dev", "english_text": "good day", "spanish_text": "buen dia"},
        {"split": "train", "english_text": "", "spanish_text": "vacio"},
        {"split": "train", "english_text": "no target", "spanish_text": "   "},
    ]
    en_es = build_pairs(rows, "train", "en-es")
    assert en_es == [("hello there", "hola alli")]

    es_en = build_pairs(rows, "train", "es-en")
    assert es_en == [("hola alli", "hello there")]

    assert build_pairs(rows, "dev", "en-es") == [("good day", "buen dia")]
    assert build_pairs(rows, "test", "en-es") == []
