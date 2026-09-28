"""Choosing a shorter translation when a dubbed line will not fit (webapp/backend/shorten.py)."""

import numpy as np
import pytest

from webapp.backend import dubbing, shorten, speakers

# Real n-best lists from the user's Short (opus-mt-en-es, beam 8), abbreviated.
ACCENT = [
    ("Dios mío, me encanta tu acento.", -0.279),
    ("Oh, Dios mío, me encanta tu acento.", -0.341),
    ("Oh, Dios mío, amo tu acento.", -0.541),
    ("Dios mío, me encanta su acento.", -0.560),
]
WHERE = [
    ("¿De dónde eres?", -0.216),
    ("¿De dónde es usted?", -0.528),
    ("¿De dónde es?", -0.636),
    ("De dónde es.", -0.30),
]
NO_ACCENT = [("No tengo acento.", -0.181), ("Tengo acento.", -0.20), ("Yo no tengo acento.", -0.53)]


def test_spoken_chars_ignores_spaces_and_punctuation() -> None:
    assert shorten.spoken_chars("¿De dónde eres?") == 11


def test_delta_limits_how_far_down_the_list_it_goes() -> None:
    current = ACCENT[0][0]
    assert shorten.pick_shorter(ACCENT, current, "es", delta=0.2) == current  # "amo" is 0.26 below
    assert shorten.pick_shorter(ACCENT, current, "es", delta=0.3) == "Oh, Dios mío, amo tu acento."


def test_never_drops_a_question_a_negation_a_number_or_the_ending() -> None:
    assert shorten.pick_shorter(WHERE, "¿De dónde eres?", "es", delta=0.5) == "¿De dónde es?"
    assert "?" in shorten.pick_shorter(WHERE, "¿De dónde eres?", "es", delta=0.1)
    assert shorten.pick_shorter(NO_ACCENT, "No tengo acento.", "es", delta=0.5) == "No tengo acento."
    assert not shorten.keeps_meaning_markers("I have an accent.", "I don't have an accent.", "en")
    assert not shorten.keeps_meaning_markers("Nine.", "Nine 9 mm.", "en")
    assert not shorten.keeps_meaning_markers("Porque esa es la", "Porque esa es la idea.", "es")


def test_target_mode_shortens_only_as_much_as_needed() -> None:
    cands = [("Aaaa bbbb cccc dddd.", -0.1), ("Aaaa bbbb cccc.", -0.2), ("Aaaa.", -0.35)]
    current = cands[0][0]
    assert shorten.pick_shorter(cands, current, "es", 0.3, target_chars=12) == "Aaaa bbbb cccc."
    assert shorten.pick_shorter(cands, current, "es", 0.3, target_chars=2) == "Aaaa."
    assert shorten.pick_shorter(cands, current, "es", 0.3) == "Aaaa."


def test_round_trip_check_rejects_a_candidate_that_changed_meaning() -> None:
    source = "¿En qué época está..."
    cands = [("What era is he in?", -0.30), ("What time is it?", -0.58)]
    backs = {"What era is he in?": "¿En qué época está?", "What time is it?": "¿Qué hora es?"}
    assert shorten.pick_shorter(cands, cands[0][0], "en", 0.3) == "What time is it?"
    kept = shorten.pick_shorter(cands, cands[0][0], "en", 0.3, source=source, back_translations=backs)
    assert kept == "What era is he in?"


def test_empty_nbest_keeps_the_current_translation() -> None:
    assert shorten.pick_shorter([], "Hola.", "es") == "Hola."


def test_needs_shortening_counts_the_free_compression() -> None:
    line = "Dios mío, me encanta tu acento. ¿De dónde eres?"  # 35 spoken characters
    free = dubbing.FREE_COMPRESSION  # XTTS 1.25x and WSOLA to 0.85: ~1.47x
    assert shorten.needs_shortening(line, 2.0, "es", free)  # 10.8 * 2.0 * 1.47 = 31.8 < 35
    assert not shorten.needs_shortening(line, 2.5, "es", free)


class _MT:
    """Stands in for LocalModels: a fixed beam-4 translation and beam-8 candidates."""

    def __init__(self):
        self.calls = []

    def translate_many(self, texts, source, target):
        self.calls.append((source, target, list(texts)))
        table = {
            "Oh my god, I love your accent.": "Dios mío, me encanta tu acento.",
            "Where are you from?": "¿De dónde eres?",
            "Thanks.": "Gracias.",
        }
        if (source, target) == ("es", "en"):  # the back-translation used by the round-trip check
            return ["Oh my god, I love your accent." if "acento" in t else t for t in texts]
        return [table.get(t, t) for t in texts]

    def mt_model(self, source, target):
        return "tokenizer", "model"


def test_only_lines_that_cannot_fit_are_reworded(monkeypatch) -> None:
    def fake_nbest(model, tokenizer, sentences, beams=8, max_new_tokens=512):
        return [ACCENT if "love" in s else WHERE for s in sentences]

    monkeypatch.setattr(shorten, "nbest", fake_nbest)
    models = _MT()
    texts = ["Oh my god, I love your accent. Where are you from?", "Thanks."]
    current = ["Dios mío, me encanta tu acento. ¿De dónde eres?", "Gracias."]
    # line 0 gets 1.6 s -- too little for 35 characters; line 1 has room
    out = dubbing.shorten_translations(models, texts, current, [(1.6, 1.6), (3.0, 5.0)], "en", "es")
    assert out[1] == "Gracias."
    assert shorten.spoken_chars(out[0]) < shorten.spoken_chars(current[0])
    assert "?" in out[0]


def test_nothing_is_reworded_when_everything_fits() -> None:
    models = _MT()
    current = ["Dios mío, me encanta tu acento.", "Gracias."]
    out = dubbing.shorten_translations(models, ["a", "b"], current, [(9.0, 9.0), (9.0, 9.0)], "en", "es")
    assert out == current and models.calls == []


# --- pauses between lines ---------------------------------------------------------------------------


def test_lines_leave_a_pause_before_another_speaker_and_run_on_only_a_little() -> None:
    segments = [
        {"start": 0.0, "end": 3.0},
        {"start": 3.0, "end": 5.0},  # Whisper often leaves no gap at all
        {"start": 8.0, "end": 9.0},
    ]
    windows = dubbing.line_windows(segments, [0, 1, 1], total_duration=10.0)
    (_, stop0), (_, stop1), (_, stop2) = windows
    # The measured values are pinned as numbers, so a change to the constants shows up here.
    assert stop0 == pytest.approx(2.8)  # speaker change: keep 0.2 s of silence
    assert stop1 == pytest.approx(5.3)  # long silence after: run on 0.3 s at most
    assert stop2 == pytest.approx(9.3)
    same = dubbing.line_windows(segments[:2], [0, 0], total_duration=10.0)
    assert same[0][1] == pytest.approx(2.9)  # same speaker: keep 0.1 s


def test_a_tiny_line_never_runs_over_the_next_one() -> None:
    segments = [{"start": 0.0, "end": 0.1}, {"start": 0.15, "end": 1.0}]
    (window, stop), _ = dubbing.line_windows(segments, [0, 1], total_duration=2.0)
    assert window == dubbing.MIN_WINDOW_S and stop <= 0.15


# --- the reference XTTS clones from --------------------------------------------------------------


def test_spectral_gate_lowers_steady_noise_and_keeps_the_voice() -> None:
    sr = 22050
    rng = np.random.default_rng(0)
    t = np.arange(sr * 3) / sr
    voice = 0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 0.7 * t) > 0)  # on/off "turns"
    noise = 0.01 * rng.standard_normal(len(t))
    out = speakers.spectral_gate(voice + noise, sr)
    quiet = voice == 0
    assert np.std(out[quiet]) < 0.6 * np.std(noise[quiet])  # noise between turns turned down
    loud = ~quiet
    assert np.corrcoef(out[loud], voice[loud])[0, 1] > 0.99  # the voice itself untouched
